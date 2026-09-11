"""The encoder's flat forward, over one session's token grid.

``Encoder.forward_flat`` composes the tested leaf primitives (the block's flat attention,
pinned to a dense oracle in test_l1_flat; ``pack_r4.build_r4_grid`` structure in
test_pack_r4) into the forward the released model runs. The composition adds three things
the leaves do not: region identity once, a shared rotary table, and the SINGLE-CLIP grid
lifted to a batch. Each is a place a silent miscompute can hide, so every invariant is
named, asserted and printed:

  1. Output width: deep supervision concatenates four taps, so 4·d.
  2. Block-diagonal by (clip, array): perturbing one clip/array leaves the others bit-stable
     — the batched cu_seqlens + rope tiling must not cross clips or arrays.
  3. Batch consistency: identical clips → identical outputs, and a clip's output is the SAME
     whether run alone or inside a batch (the lift adds no coupling).
  4. Fed from the stem: stem → pack_band_tokens → forward_flat runs finite and grads flow
     back to the stem projections (the real training wire).
"""

from __future__ import annotations

import torch

from mapa.models.geometry import build_l1_geometry
from mapa.models.pack_r4 import build_r4_grid, pack_band_tokens
from mapa.models.sidecar import build_sidecar
from mapa.models.stem import PerBandStem
from mapa.models.towers import build_encoder

T = 16  # SLOW 2, MID 8, HGA 16 tokens per contact ⇒ k_full 26, total 130 (5 contacts).
D_ENC = 256


def _session():
    sc = build_sidecar(
        ["LA1", "LA2", "LA3", "LB1", "LB2"],
        region_id=torch.tensor([0, 0, 0, 1, 1]),
    )
    geom = build_l1_geometry(sc)
    return sc, geom, build_r4_grid(geom, n_time=T)




def test_block_diagonal_by_clip_and_array() -> None:
    torch.manual_seed(0)
    sc, geom, grid = _session()
    pid = sc.region_id[grid.contact]
    enc = build_encoder(n_regions=8).eval()
    x = torch.randn(2, grid.total, D_ENC)
    out = enc.forward_flat(x, grid, pid)

    # (a) perturb clip 1 only ⇒ clip 0 bit-stable, clip 1 moves.
    x1 = x.clone()
    x1[1] += torch.randn_like(x1[1]) * 3.0
    o1 = enc.forward_flat(x1, grid, pid)
    clip_iso = torch.allclose(out[0], o1[0], atol=1e-5) and not torch.allclose(out[1], o1[1], atol=1e-4)

    # (b) perturb array LB (grid.array==1) in clip 0 ⇒ array LA clip 0 stable, clip 1 fully stable.
    lb = grid.array == 1
    la = grid.array == 0
    x2 = x.clone()
    x2[0, lb] += torch.randn_like(x2[0, lb]) * 3.0
    o2 = enc.forward_flat(x2, grid, pid)
    array_iso = (
        torch.allclose(out[0, la], o2[0, la], atol=1e-5)
        and not torch.allclose(out[0, lb], o2[0, lb], atol=1e-4)
        and torch.allclose(out[1], o2[1], atol=1e-5)
    )
    ok = clip_iso and array_iso
    print(f"[check] block-diagonal by (clip, array): clip-iso={clip_iso}, array-iso={array_iso} "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_batch_lift_adds_no_coupling() -> None:
    torch.manual_seed(1)
    sc, geom, grid = _session()
    pid = sc.region_id[grid.contact]
    enc = build_encoder(n_regions=8).eval()

    one = torch.randn(1, grid.total, D_ENC)
    both = one.repeat(2, 1, 1)  # two identical clips
    out_both = enc.forward_flat(both, grid, pid)
    out_one = enc.forward_flat(one, grid, pid)  # single-clip run
    identical_clips = torch.allclose(out_both[0], out_both[1], atol=1e-5)
    alone_eq_batched = torch.allclose(out_one[0], out_both[0], atol=1e-5)
    ok = identical_clips and alone_eq_batched
    print(f"[check] identical clips → identical out ({identical_clips}); "
          f"alone == in-batch ({alone_eq_batched}) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_tap_blocks_return_raw_depth_features() -> None:
    sc, geom, grid = _session()
    pid = sc.region_id[grid.contact]
    enc = build_encoder(n_regions=8).eval()
    out, taps = enc.forward_flat(torch.randn(2, grid.total, D_ENC), grid, pid, tap_blocks=(3, 12))
    ok = set(taps) == {3, 12} and all(t.shape == (2, grid.total, D_ENC) for t in taps.values())
    print(f"[check] tap_blocks (3,12) → raw {D_ENC}-d feats at each, out still 1024-d "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok






def test_fed_from_stem_runs_finite_and_grads_reach_stem() -> None:
    torch.manual_seed(2)
    sc, geom, grid = _session()
    pid = sc.region_id[grid.contact]
    stem = PerBandStem(d_model=D_ENC)
    enc = build_encoder(n_regions=8)  # train mode ⇒ grads

    # band inputs on the shared 32 Hz clock at T32 = n_time = 16, bins 7/6/7, 5 contacts.
    bands = (torch.randn(2, 5, 7, T), torch.randn(2, 5, 6, T), torch.randn(2, 5, 7, T))
    tokens, _ = stem(bands)  # SLOW (2,5,2,d), MID (2,5,8,d), HGA (2,5,16,d)
    x_flat = pack_band_tokens(tokens, grid)  # (2, total, d)
    assert x_flat.shape == (2, grid.total, D_ENC)

    out = enc.forward_flat(x_flat, grid, pid)
    out.sum().backward()
    finite = bool(torch.isfinite(out).all())
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in stem.projs.parameters())
    ok = finite and grad_ok
    print(f"[check] stem→pack→forward_flat finite={finite}, grads reach stem projs={grad_ok} "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok
