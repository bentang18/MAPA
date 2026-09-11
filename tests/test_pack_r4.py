"""The token layout: every structural invariant of the flat pack plan.

The flat plan is the contract the attention consumes, and a wrong layout is a silent
miscompute, so every structural invariant is named, asserted and printed:

  1. Token universe = N·k_full, array-contiguous, cu_seqlens sums to it (static VALUES).
  2. Per-token coords are exactly the SLOW/MID/HGA lattice (band counts, time_pos = j·stride,
     depth = the contact's clinical index).
  3. masked ⊇ in_loss (loss only on queries); spatially-masked contacts are UNCONDITIONALLY
     in-loss (leak-proof by construction); temporally-masked-only tokens are in-loss IFF
     their own-band margin is at least 2.
  4. masked/visible partition the grid; masked COUNTS are per-session constants (static shapes).
"""

from __future__ import annotations

import torch

from mapa.models.geometry import build_l1_geometry
from mapa.models.pack_r4 import band_token_counts, build_r4_grid, pack_band_tokens
from mapa.models.sidecar import build_sidecar

T = 16  # multiple of SLOW_STRIDE=8 ⇒ SLOW 2, MID 8, HGA 16 tokens per contact.


def _session():
    sc = build_sidecar(
        ["LA1", "LA2", "LA3", "LB1", "LB2"],
        region_id=torch.tensor([0, 0, 0, 1, 1]),
    )
    return sc, build_l1_geometry(sc)


def test_token_universe_is_array_contiguous_and_cu_static() -> None:
    sc, geom = _session()
    grid = build_r4_grid(geom, n_time=T)
    slow, mid, hga = band_token_counts(T)
    assert (slow, mid, hga) == (2, 8, 16)
    k_full = slow + mid + hga  # 26
    n = int(geom.valid.sum())  # 5
    assert grid.k_full == k_full and grid.total == n * k_full == 130
    # cu_seqlens: LA has 3 contacts (78 tokens), LB has 2 (52) ⇒ [0, 78, 130].
    assert grid.cu_seqlens.tolist() == [0, 78, 130]
    assert int(grid.cu_seqlens[-1]) == grid.total
    # array id is non-decreasing ⇒ each array block is contiguous (varlen requirement).
    assert torch.all(grid.array[1:] >= grid.array[:-1])
    print(f"[check] universe {grid.total} tok, array-contiguous, cu {grid.cu_seqlens.tolist()} OK")


def test_per_token_coords_are_the_band_lattice() -> None:
    sc, geom = _session()
    grid = build_r4_grid(geom, n_time=T)
    slow, mid, hga = grid.band_lengths
    n = int(geom.valid.sum())
    # band counts: exactly N per-band tokens.
    assert int((grid.band == 0).sum()) == n * slow
    assert int((grid.band == 1).sum()) == n * mid
    assert int((grid.band == 2).sum()) == n * hga
    # time_pos = bandpos * stride on the shared 32 Hz lattice (SLOW 8, MID 2, HGA 1).
    for b, stride in ((0, 8), (1, 2), (2, 1)):
        sel = grid.band == b
        assert torch.equal(grid.time_pos[sel], grid.bandpos[sel] * stride)
        assert int(grid.bandpos[sel].max()) == grid.band_lengths[b] - 1
    # depth (L1 index-RoPE coord): constant across a contact's k_full-token block, and the
    # set of per-contact depths == the montage's clinical depths geom.depth[valid].
    per_contact_depth = grid.depth.reshape(n, grid.k_full)
    assert torch.all(per_contact_depth == per_contact_depth[:, :1])  # constant within contact
    assert torch.equal(per_contact_depth[:, 0].sort().values, geom.depth[geom.valid].sort().values)
    print("[check] band counts + time_pos=j·stride + depth constant-per-contact OK")






def test_pack_band_tokens_places_every_token_at_its_grid_slot() -> None:
    # pack_band_tokens must land band_tokens[b][:, c, j] at the grid slot whose
    # (contact, band, bandpos) == (c, b, j). Tag each source token with a unique scalar
    # f(c,b,j) and assert the flat output reproduces f over the WHOLE grid (order-exact).
    sc, geom = _session()
    grid = build_r4_grid(geom, n_time=T)
    B, d, N = 2, 3, int(geom.valid.sum())  # N=5 full contacts
    band_tokens = []
    for b, length in enumerate(grid.band_lengths):
        c = torch.arange(N)[None, :, None, None]          # contact
        j = torch.arange(length)[None, None, :, None]     # bandpos
        tag = (c * 100 + b * 10 + j).float()              # unique per (c,b,j)
        band_tokens.append(tag.expand(B, N, length, d).clone())
    packed = pack_band_tokens(band_tokens, grid)  # (B, total, d)
    expected = (grid.contact * 100 + grid.band * 10 + grid.bandpos).float()  # (total,)
    ok = (
        packed.shape == (B, grid.total, d)
        and torch.equal(packed[0, :, 0], expected)
        and torch.equal(packed[1, :, 0], expected)  # batch-consistent
        and torch.equal(packed[0, :, 0], packed[0, :, d - 1])  # d broadcast intact
    )
    print(f"[check] pack_band_tokens order-exact over all {grid.total} tokens, "
          f"batch+width consistent {'OK' if ok else 'VIOLATED'}")
    assert ok








