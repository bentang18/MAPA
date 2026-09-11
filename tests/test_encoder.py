"""The public API: load a checkpoint, prepare a session, encode a batch."""
import pytest
import torch

import mapa as sp
from mapa.encoder import N_REGIONS
from mapa.models.tower import MaeEncoder

LABELS = [f"{s}{i}" for s in ("LA", "LH", "RO") for i in range(1, 9)]
N = len(LABELS)
BAND_BINS = (7, 6, 7)
T = 32


def _sidecar():
    torch.manual_seed(0)
    return sp.build_sidecar(LABELS, region_id=torch.randint(0, 74, (N,)))




def _tower(**kw):
    torch.manual_seed(3)
    return MaeEncoder(n_regions=N_REGIONS, d_model=384, **kw).eval()


def _bands(b=2):
    torch.manual_seed(7)
    return [torch.randn(b, N, f, T) for f in BAND_BINS]


def _encoder(**kw):
    tower = _tower(**kw)
    return sp.MapaEncoder(
        tower, space_rope=kw.get("space_rope", True),
        region_embed=kw.get("parcel_embed", True),
    )


def test_forward_shapes():
    enc = _encoder()
    sess = enc.prepare(_sidecar(), n_time=T)
    out = enc(_bands(), sess, taps=(3, 6, 9, 12))
    assert sorted(out) == [3, 6, 9, 12]
    for v in out.values():
        assert v.shape == (2, N, sess.k_full, enc.d_model)


def test_token_grid_is_the_published_rate():
    """52 tokens per contact per second: 4 Hz slow, 16 Hz mid, 32 Hz high gamma."""
    enc = _encoder()
    assert enc.prepare(_sidecar(), n_time=32).k_full == 4 + 16 + 32


def test_vit_small_width():
    assert sum(p.numel() for p in _tower().parameters()) == pytest.approx(21.3e6, rel=0.01)


def test_tap_zero_runs_no_encoder():
    """The floor is the stem input. It cannot depend on encoder weights, which is what makes
    it a shared baseline across ablation arms rather than a per-arm quantity."""
    sess = _encoder().prepare(_sidecar(), n_time=T)
    a = _encoder()(_bands(), sess, taps=(0,))[0]
    torch.manual_seed(999)
    b = _encoder()(_bands(), sess, taps=(0,))[0]
    assert torch.equal(a, b)


def test_floor_is_identical_across_ablation_arms():
    sc = _sidecar()
    outs = []
    for kw in (dict(), dict(region_embed=False), dict(space_rope=False),
               dict(region_embed=False, space_rope=False)):
        enc = _encoder(**kw)
        outs.append(enc(_bands(), enc.prepare(sc, n_time=T), taps=(0,))[0])
    for o in outs[1:]:
        assert torch.equal(outs[0], o)


def test_priors_change_the_representation():
    """Removing a prior must change what the encoder computes, or the ablation is vacuous."""
    sc = _sidecar()
    base = _encoder()
    full = base(_bands(), base.prepare(sc, n_time=T), taps=(12,))[12]
    for kw in (dict(region_embed=False), dict(space_rope=False)):
        enc = _encoder(**kw)
        got = enc(_bands(), enc.prepare(sc, n_time=T), taps=(12,))[12]
        assert not torch.allclose(full, got)


def test_checkpoint_round_trip(tmp_path):
    """A released file carries bare keys and records the arm; a raw training checkpoint
    carries the whole autoencoder and records nothing, so the caller names the arm."""
    tower = _tower()
    saved = tower.state_dict()
    released = tmp_path / "released.pt"
    torch.save({"model": saved, "meta": {"space_rope": True}}, released)
    training = tmp_path / "training.pt"
    torch.save({"state_dict": {f"model.objective.online.{k}": v for k, v in saved.items()}},
               training)

    for path, kw in ((released, {}), (training, {"space_rope": True})):
        enc = sp.MapaEncoder.from_checkpoint(str(path), **kw)
        assert enc.d_model == 384
        loaded = enc.tower.state_dict()
        assert set(loaded) == set(saved)
        assert all(torch.equal(loaded[k], saved[k]) for k in loaded)

        # Outputs agree to float precision rather than bitwise: the attention kernel
        # accumulates in a different order across calls, which moves the last ulp.
        sess = enc.prepare(_sidecar(), n_time=T)
        a = enc(_bands(), sess, taps=(12,))[12]
        ref = sp.MapaEncoder(tower, space_rope=True, region_embed=True)
        b = ref(_bands(), ref.prepare(_sidecar(), n_time=T), taps=(12,))[12]
        assert torch.allclose(a, b, atol=1e-5, rtol=0)


def test_a_training_checkpoint_must_be_told_the_arm(tmp_path):
    path = tmp_path / "training.pt"
    torch.save({"state_dict": {"model.objective.online.encoder.blocks.0.norm1.weight":
                               torch.zeros(384)}}, path)
    with pytest.raises(ValueError, match="does not record"):
        sp.MapaEncoder.from_checkpoint(str(path))


def test_checkpoint_rejects_a_foreign_layout(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"state_dict": {"something.else": torch.zeros(3)}}, path)
    with pytest.raises(RuntimeError, match="not a released checkpoint layout"):
        sp.MapaEncoder.from_checkpoint(str(path))


def test_region_pooling_reduces_to_present_regions():
    sc = _sidecar()
    enc = _encoder()
    sess = enc.prepare(sc, n_time=T)
    feats = enc(_bands(), sess, taps=(12,))[12]
    flat = feats.reshape(feats.shape[0], feats.shape[1], -1)
    pooled = sp.pool_to_regions(flat, sess)
    assert pooled.shape[0] == flat.shape[0]
    assert pooled.shape[1] == len(torch.unique(sess.region_of_contact))
    assert pooled.shape[2] == flat.shape[2]


@pytest.fixture(scope="module")
def boundary_encoder():
    return _encoder()


@pytest.mark.parametrize("taps", [(0,), (12,)])
@pytest.mark.parametrize("contacts,frames,bins", [(N + 1, T, 7), (N, T // 2, 7), (N, T, 6)])
def test_mismatched_input_axes_fail_before_encoding(boundary_encoder, taps, contacts, frames, bins):
    session = boundary_encoder.prepare(_sidecar(), n_time=T)
    bands = [torch.zeros(1, contacts, f, frames) for f in (bins, 6, 7)]
    with pytest.raises(ValueError, match="band must have shape"):
        boundary_encoder(bands, session, taps=taps)


@pytest.mark.parametrize("taps", [(), (-1,), (13,), (1.5,), (True,)])
def test_invalid_taps_fail_before_encoding(boundary_encoder, taps):
    session = boundary_encoder.prepare(_sidecar(), n_time=T)
    with pytest.raises(ValueError, match="taps must contain"):
        boundary_encoder(_bands(), session, taps=taps)


@pytest.mark.parametrize("n_time", [0, -8, 32.5, True])
def test_invalid_window_length_is_not_coerced(boundary_encoder, n_time):
    with pytest.raises(ValueError, match="n_time must be a positive integer"):
        boundary_encoder.prepare(_sidecar(), n_time=n_time)


def test_stem_input_validates_all_three_bands(boundary_encoder):
    session = boundary_encoder.prepare(_sidecar(), n_time=T)
    with pytest.raises(ValueError, match="three bands"):
        sp.stem_input(_bands()[:2], session)


def test_guard3_matches_clipping_before_the_model_and_eval_baseline(boundary_encoder):
    from evals.neuroprobe.encode import _enc0_elec

    session = boundary_encoder.prepare(_sidecar(), n_time=T)
    raw = [x * 100 for x in _bands(b=1)]
    original = [x.clone() for x in raw]
    clipped = [x.clamp(-cap, cap) for x, cap in zip(raw, (15, 15, 20))]
    expected = boundary_encoder(clipped, session, taps=(0, 12))
    actual = boundary_encoder(raw, session, taps=(0, 12))
    for tap in (0, 12):
        torch.testing.assert_close(actual[tap], expected[tap], rtol=0, atol=0)
    # Explicit bands make the per-band caps observable in the frontend baseline.
    floor = boundary_encoder([torch.full_like(x, 100) for x in raw], session, taps=(0,))[0]
    expected_floor = torch.tensor([15.] * (7 * 4 + 6 * 16) + [20.] * (7 * 32))
    torch.testing.assert_close(floor[0, 0, 0], expected_floor)
    torch.testing.assert_close(
        _enc0_elec(raw, session.contact_order.numpy()), actual[0].squeeze(2).half(),
        rtol=0, atol=0,
    )
    for x, saved in zip(raw, original):
        torch.testing.assert_close(x, saved, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float64, torch.float16])
def test_input_dtype_mismatch_names_the_conversion(dtype):
    enc = _encoder()
    session = enc.prepare(_sidecar(), n_time=T)
    bands = [b.to(dtype) for b in _bands()]
    with pytest.raises(ValueError, match=r'encoder uses torch.float32; convert bands'):
        enc(bands, session)
    # The input-only baseline has no projection weights and can retain its input dtype.
    assert enc(bands, session, taps=(0,))[0].dtype == dtype
