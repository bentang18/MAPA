"""The cache and label file parsers.

The thin disk-facing helpers the ``load_v3_sessions`` adapter composes: parse the
guard-2 bad-window spans (the the GPU cluster scan output), index a band's ``.npy`` spec cache
by scanning its ``{stem}.json`` sidecars (session ``key`` → paths + ch_names +
total_frames), and parse the LOF bad-channel report (guard-1 drop set). Pure
file-parsing, tested on synthetic JSON here. It is wired to the real BrainTreebank
anatomy calls (``voltage_electrode_order`` / ``aligned_voltage_support``) inside
``load_v3_sessions``.

Real ``bad_windows_s`` format confirmed against the the GPU cluster scan output
(btbank6_t4.json): a list of ``[start_s, end_s]`` pairs; the file also carries
``subject_id`` / ``trial_id``.
"""

from __future__ import annotations

import json

import numpy as np

from mapa.data.cache_index import (
    BandCacheEntry,
    index_bad_windows,
    index_band_cache,
    load_bad_window_spans,
    parse_key_session,
    parse_lof_report,
    parse_session_name,
    resolve_band_leaf,
)


def _real_key(subject_id: int, trial_id: int) -> str:
    """A key in the REAL producer format (nested JSON + time-range suffix)."""
    return (
        '{"cls":"Wang2024Treebank","method":"_load_raw","timeline":'
        f'{{"extra_bad":[],"subject":"btbank{subject_id}",'
        f'"subject_id":{subject_id},"trial_id":{trial_id}}}}}_0.000_6867.860'
    )


def _write(path, obj):
    path.write_text(json.dumps(obj))
    return str(path)


def test_load_bad_window_spans_reads_pairs(tmp_path) -> None:
    p = _write(tmp_path / "btbank6_t4.json", {
        "session": "btbank6_t4", "subject_id": 6, "trial_id": 4,
        "bad_windows_s": [[187.0, 188.0], [374.0, 375.0]], "frac_bad": 0.03,
    })
    spans = load_bad_window_spans(p)
    assert spans == [(187.0, 188.0), (374.0, 375.0)]


def test_load_bad_window_spans_empty(tmp_path) -> None:
    p = _write(tmp_path / "clean.json", {
        "session": "btbank3_t2", "subject_id": 3, "trial_id": 2, "bad_windows_s": [],
    })
    assert load_bad_window_spans(p) == []


def test_index_bad_windows_keys_by_subject_trial(tmp_path) -> None:
    _write(tmp_path / "a.json", {"subject_id": 1, "trial_id": 0,
                                 "bad_windows_s": [[1.0, 2.0]]})
    _write(tmp_path / "b.json", {"subject_id": 2, "trial_id": 1,
                                 "bad_windows_s": [[3.0, 4.0], [5.0, 6.0]]})
    idx = index_bad_windows(str(tmp_path))
    assert idx[(1, 0)] == [(1.0, 2.0)]
    assert idx[(2, 1)] == [(3.0, 4.0), (5.0, 6.0)]


def test_parse_session_name() -> None:
    assert parse_session_name("btbank6_t4") == (6, 4)
    assert parse_session_name("btbank1_t0") == (1, 0)


def test_parse_key_session_reads_nested_json_key() -> None:
    k = _real_key(1, 0)
    assert parse_key_session(k) == (1, 0)
    # full-integer capture: 12 never aliases 1
    assert parse_key_session(_real_key(12, 3)) == (12, 3)


def test_parse_key_session_fails_loud_on_bad_key() -> None:
    import pytest
    with pytest.raises(ValueError, match="missing subject_id"):
        parse_key_session("not-a-real-key")


def test_index_band_cache_scans_sidecars(tmp_path) -> None:
    # two sessions in one band dir; each has {stem}.json + {stem}.npy + {stem}.stats.npz
    for stem, key, ch, tf in (
        ("aaa", _real_key(1, 0), ["LA1", "LA2"], 5000),
        ("bbb", _real_key(2, 1), ["LB1"], 4000),
    ):
        _write(tmp_path / f"{stem}.json", {
            "key": key, "ch_names": ch, "total_frames": tf, "sample_rate": 2048,
        })
        np.save(str(tmp_path / f"{stem}.npy"), np.zeros((len(ch), 7, tf), np.float32))
        np.savez(str(tmp_path / f"{stem}.stats.npz"),
                 median=np.zeros((len(ch), 7, 1)), sigma=np.ones((len(ch), 7, 1)))
    idx = index_band_cache(str(tmp_path))
    e = idx[_real_key(1, 0)]
    assert e.ch_names == ("LA1", "LA2")
    assert e.total_frames == 5000
    assert e.npy_path.endswith("aaa.npy")
    assert e.stats_path.endswith("aaa.stats.npz")


def test_resolve_band_leaf_descends_and_disambiguates_stale_hop(tmp_path) -> None:
    # emulate the real exca nesting: a stale band_hop=512 leaf beside the hop=64 one
    method = tmp_path / "speech_decoding.extractors.view.MultiStftView._get_data,1"
    stale = method / "c_max=256,car=array,band_hop=512,x-aaa-111"
    live = method / "c_max=256,car=array,band_hop=64,hop_length=64,x-bbb-222"
    for d, tf in ((stale, 111), (live, 222)):
        d.mkdir(parents=True)
        _write(d / "h.json", {"key": _real_key(1, 0), "ch_names": ["LA1"],
                              "total_frames": tf, "sample_rate": 2048})
        np.save(str(d / "h.npy"), np.zeros((1, 7, tf), np.float32))
        np.savez(str(d / "h.stats.npz"),
                 median=np.zeros((1, 7, 1)), sigma=np.ones((1, 7, 1)))
    leaf = resolve_band_leaf(str(tmp_path))  # band ROOT → descend, pick hop=64
    assert leaf == str(live)
    # index_band_cache auto-resolves the same leaf (reads the live total_frames)
    idx = index_band_cache(str(tmp_path))
    assert idx[_real_key(1, 0)].total_frames == 222


def test_index_band_cache_ignores_sidecar_without_npy(tmp_path) -> None:
    # a stray .json with no matching .npy must be skipped (not half-indexed)
    _write(tmp_path / "orphan.json", {
        "key": "k", "ch_names": ["X1"], "total_frames": 1, "sample_rate": 32,
    })
    assert index_band_cache(str(tmp_path)) == {}


def test_parse_lof_report_maps_session_to_bad_channels(tmp_path) -> None:
    p = _write(tmp_path / "lof.json", {
        "threshold": 1.5, "n_neighbors": 20, "total_sessions": 2,
        "sessions": [
            {"session": "btbank1_t0", "subject_id": 1, "n_bad": 2,
             "bad_channels": ["LA3", "LB2"]},
            {"session": "btbank2_t1", "subject_id": 2, "n_bad": 0,
             "bad_channels": []},
        ],
    })
    lof = parse_lof_report(p)
    assert lof[(1, 0)] == {"LA3", "LB2"}
    assert lof[(2, 1)] == set()


def test_parse_lof_report_missing_file_is_empty() -> None:
    # LOF is optional (a run may not have fitted a report); absent → no drops.
    assert parse_lof_report(None) == {}


def test_frame_rate_hz_derives_from_sample_rate_and_band_hop() -> None:
    """The bake writes the RAW rate (2048) + the STFT band_hop, so the cache's real frame
    rate is sample_rate//band_hop. This is the quantity band_rates must be declared against
    (r6 2026-07-23 declared 4/16 Hz against three 32 Hz bakes)."""
    real = BandCacheEntry(
        npy_path="x.npy", stats_path="x.stats.npz", ch_names=("A",),
        total_frames=230501, sample_rate=2048, band_hop=64,
    )
    assert real.frame_rate_hz == 32
    native_slow = BandCacheEntry(
        npy_path="x.npy", stats_path="x.stats.npz", ch_names=("A",),
        total_frames=28813, sample_rate=2048, band_hop=512,
    )
    assert native_slow.frame_rate_hz == 4
    # sidecars with no band_hop already store the frame rate directly (synthetic test dirs)
    legacy = BandCacheEntry(
        npy_path="x.npy", stats_path="x.stats.npz", ch_names=("A",),
        total_frames=5000, sample_rate=32,
    )
    assert legacy.frame_rate_hz == 32


def test_duplicate_cache_key_is_rejected_instead_of_overwritten(tmp_path):
    import pytest

    for name in ('first', 'second'):
        np.save(tmp_path / f'{name}.npy', np.zeros((1, 1, 32)))
        _write(tmp_path / f'{name}.json', {'key': _real_key(1, 1), 'ch_names': ['LA1'],
                                          'total_frames': 32, 'sample_rate': 32})
    with pytest.raises(ValueError, match=r'duplicate.*first.npy.*second.npy'):
        index_band_cache(str(tmp_path))
