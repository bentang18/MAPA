"""Numerical parity and data-contract tests for the public preprocessing path."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mapa.data.cache_index import index_band_cache
from mapa.data.session_loader import load_v3_sessions
from mapa.preprocessing import prepare_recording, preprocessing_config, region_ids_from_names
from mapa.preprocessing.stft import _single_stft_raw_view_chunked


def fixture_voltage():
    x = (np.random.default_rng(81).normal(size=(7, 1024 * 20)) * 100).astype(np.float32)
    x[4] = 0
    return x, ["LA1", "LB1", "LA3", "LB3", "LA4", "LB4", "LA5"]


@pytest.fixture(scope="module")
def prepared():
    pytest.importorskip("mne")
    x, labels = fixture_voltage()
    return prepare_recording(
        x,
        1024,
        labels,
        [0, 1, 0, 1, 0, 1, 0],
        bad_labels=["LA4"],
        keep_labels=labels[:-1],
        chunk_frames=43,
    )


def test_matches_research_frontend_reference(prepared):
    ref = np.load(Path(__file__).parent / "data/frontend_reference.npz")
    np.testing.assert_array_equal(prepared.input_indices, ref["indices"])
    for i, (band, norm, window) in enumerate(
        zip(prepared.bands, prepared.normalizers, prepared.window(4))
    ):
        # Raw magnitudes have acquisition-dependent units. Compare in the fixed reference
        # session's MAD-sigma units so cross-platform floating-point rounding near small bins
        # is judged relative to signal variation, not an arbitrary voltage-unit tolerance.
        scale = np.maximum(ref[f"sigma{i}"].astype(np.float64), 1e-6)
        np.testing.assert_allclose(
            band.numpy().astype(np.float64) / scale,
            ref[f"band{i}"].astype(np.float64) / scale,
            rtol=2e-6, atol=2e-6,
        )
        np.testing.assert_allclose(norm.median.numpy(), ref[f"median{i}"], rtol=2e-6, atol=2e-5)
        np.testing.assert_allclose(norm.sigma.numpy(), ref[f"sigma{i}"], rtol=2e-6, atol=2e-5)
        np.testing.assert_allclose(window[0].numpy(), ref[f"window{i}"], rtol=3e-5, atol=2e-5)
    assert prepared.sidecar.labels == ("LA1", "LB1", "LA3", "LB3", "LB4")
    assert prepared.sidecar.depth.tolist() == [1, 1, 3, 3, 4]


@pytest.mark.parametrize("fft,lo,hi", [(1024, 1, 7), (256, 2, 7), (128, 4, 10)])
def test_chunk_boundaries_equal_full_recording_stft(fft, lo, hi):
    x = torch.randn(3, 32771, generator=torch.Generator().manual_seed(15))
    direct = torch.stft(x, fft, hop_length=64, window=torch.hann_window(fft), return_complex=True)
    got = _single_stft_raw_view_chunked(
        x, sample_rate=2048, nperseg=fft, hop_length=64, k0=lo, k1=hi, log_eps=1e-6, chunk_frames=31
    )
    torch.testing.assert_close(got, direct[:, lo : hi + 1].abs(), rtol=0, atol=0)


def test_window_uses_frozen_session_stats_and_original_clock(prepared):
    for a, b in zip(prepared.window(4.01), prepared.window(4.0)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert [b.shape for b in prepared.window(4, 2)] == [(1, 5, 7, 64), (1, 5, 6, 64), (1, 5, 7, 64)]
    for args in [(19.5, 1), (-1, 1), (0, 0.1), (0, float("nan"))]:
        with pytest.raises(ValueError):
            prepared.window(*args)


def test_saved_cache_loads_through_evaluation_adapter(prepared, tmp_path):
    dest = tmp_path / "recording"
    prepared.save(dest, subject_id=1, trial_id=1)
    assert next(iter(index_band_cache(str(dest / "slow")).values())).frame_rate_hz == 32
    sessions = load_v3_sessions(
        sessions=[(1, 1)],
        band_cache_dirs=[str(dest / b) for b in ("slow", "mid", "fast")],
        span_dir=str(dest / "spans"),
        region_fn=lambda s, t, labels: prepared.sidecar.region_id,
    )
    assert len(sessions) == 1
    assert json.loads((dest / "preprocessing.json").read_text())["recipe"] == preprocessing_config()
    with pytest.raises(FileExistsError):
        prepared.save(dest, subject_id=1, trial_id=1)


def test_regions_fail_on_typos_and_support_explicit_reserved_entry():
    assert region_ids_from_names(
        ["ctx-lh-superiortemporal", "ctx-rh-superiortemporal", None]
    ).tolist() == [28, 59, 74]
    with pytest.raises(KeyError):
        region_ids_from_names(["superiortemporal"])


def test_configuration_file_matches_executable_recipe():
    from importlib.resources import files

    assert (
        json.loads(files("mapa.preprocessing").joinpath("config.json").read_text())
        == preprocessing_config()
    )


@pytest.mark.parametrize("change", ["duplicate", "regions", "rate", "short", "nan"])
def test_invalid_inputs_fail_before_preprocessing(change):
    x, labels = fixture_voltage()
    regions = [0, 1, 0, 1, 0, 1, 0]
    rate = 1024
    if change == "duplicate":
        labels[1] = labels[0]
    if change == "regions":
        regions[0] = 75
    if change == "rate":
        rate = 100
    if change == "short":
        x = x[:, :1024]
    if change == "nan":
        x[0, 0] = np.nan
    with pytest.raises(ValueError):
        prepare_recording(x, rate, labels, regions)


def test_frontend_keeps_contacts_unless_caller_excludes_them():
    pytest.importorskip("mne")
    x, labels = fixture_voltage()
    recording = prepare_recording(x, 1024, labels, [0, 1, 0, 1, 0, 1, 0])
    assert recording.sidecar.labels == tuple(labels)
    np.testing.assert_array_equal(recording.input_indices, np.arange(len(labels)))


def test_normalization_does_not_clip_or_read_legacy_guard_environment(monkeypatch):
    from mapa.data.normalize import SessionRobustZNormalizer

    monkeypatch.setenv("V14_SESSION_Z_WINSOR", "15")
    norm = SessionRobustZNormalizer.from_stats(
        median=torch.zeros(1, 1, 1), sigma=torch.ones(1, 1, 1)
    )
    x = torch.tensor([[[-1000.0, 2000.0]]])
    torch.testing.assert_close(norm.transform(x), x, rtol=0, atol=0)


@pytest.mark.parametrize('selection', [{'keep_labels': ['LA1', 'LA3']}, {'bad_labels': ['ECG']}])
def test_excluded_non_neural_channel_does_not_enter_preprocessing(selection):
    pytest.importorskip('mne')
    rng = np.random.default_rng(17)
    clean = rng.normal(size=(2, 8 * 2048)).astype(np.float32)
    voltage = np.vstack([clean, np.full((1, clean.shape[1]), np.nan, dtype=np.float32)])
    got = prepare_recording(voltage, 2048, ['LA1', 'LA3', 'ECG'], [0, 0, 74], **selection)
    ref = prepare_recording(clean, 2048, ['LA1', 'LA3'], [0, 0])
    assert got.sidecar.labels == ('LA1', 'LA3')
    assert got.input_indices.tolist() == [0, 1]
    for actual, expected in zip(got.window(1), ref.window(1)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
