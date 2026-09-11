"""The array geometry: gathering contacts into a ragged per-array grid.

The block attends JOINTLY over (contact, time) WITHIN an array, block-diagonal
by array. Rather than materialize a
global (N·T)² mask, we gather the N contacts into a ragged (n_arrays, max_c)
grid, zero-padding short arrays, and carry a validity mask + per-slot depth. The
L1 block runs one batched SDPA over the array axis (varlen → padded) — fully
vectorized, no python loop over arrays in the hot path.

This module builds that plan once per session from the sidecar (setup-time, not
the training loop).
"""

from __future__ import annotations

import torch

from mapa.models.geometry import build_l1_geometry
from mapa.models.sidecar import build_sidecar


def _sidecar(labels, regions):
    return build_sidecar(labels, region_id=torch.tensor(regions, dtype=torch.long))


def test_gather_plan_groups_contacts_by_array() -> None:
    # Two arrays: LA (3 contacts), LB (2 contacts). max_c = 3.
    sc = _sidecar(["LA1", "LA2", "LA3", "LB1", "LB2"], [0, 0, 0, 1, 1])
    geom = build_l1_geometry(sc)
    assert geom.n_arrays == 2
    assert geom.max_c == 3
    assert geom.gather_idx.shape == (2, 3)
    assert geom.valid.shape == (2, 3)
    # Array 0 = contacts [0,1,2] all valid; array 1 = contacts [3,4] + 1 pad.
    assert geom.gather_idx[0].tolist() == [0, 1, 2]
    assert geom.valid[0].tolist() == [True, True, True]
    assert geom.gather_idx[1][:2].tolist() == [3, 4]
    assert geom.valid[1].tolist() == [True, True, False]


def test_pad_slots_point_at_a_real_index_but_are_invalid() -> None:
    # Pad slots must index a legal row (so a plain x[gather_idx] never OOBs); the
    # valid mask is what excludes them from attention.
    sc = _sidecar(["LA1", "LA2", "LA3", "LB1"], [0, 0, 0, 1])
    geom = build_l1_geometry(sc)
    assert geom.gather_idx.min() >= 0
    assert geom.gather_idx.max() < 4
    # every real contact index appears exactly once among valid slots.
    valid_idx = geom.gather_idx[geom.valid].sort().values
    assert valid_idx.tolist() == [0, 1, 2, 3]


def test_depth_is_carried_per_slot_for_index_rope() -> None:
    # Depth (clinical contact number, gaps preserved) rides along each gathered
    # slot — it is the index-RoPE coordinate inside the L1 block.
    sc = _sidecar(["LA1", "LA2", "LA4", "LB7", "LB8"], [0, 0, 0, 1, 1])
    geom = build_l1_geometry(sc)
    assert geom.depth[0].tolist()[:3] == [1, 2, 4]  # gap at 3 preserved
    assert geom.depth[1][:2].tolist() == [7, 8]


def test_single_array_is_a_full_row() -> None:
    sc = _sidecar(["LA1", "LA2", "LA3"], [0, 0, 0])
    geom = build_l1_geometry(sc)
    assert geom.n_arrays == 1
    assert geom.max_c == 3
    assert geom.valid.all()


def test_scatter_inverts_gather_for_valid_slots() -> None:
    # gather then scatter must round-trip the per-contact payload.
    sc = _sidecar(["LA1", "LA2", "LA3", "LB1", "LB2"], [0, 0, 0, 1, 1])
    geom = build_l1_geometry(sc)
    x = torch.randn(2, 5, 7, 8)  # (B, N, T, d)
    gathered = x[:, geom.gather_idx]  # (B, n_arrays, max_c, T, d)
    out = torch.zeros_like(x)
    out[:, geom.gather_idx[geom.valid]] = gathered[:, geom.valid]
    assert torch.allclose(out, x)
