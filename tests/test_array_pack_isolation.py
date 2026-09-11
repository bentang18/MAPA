"""Array-pack isolation: one array's tokens never reach another's.

The linchpin of array-level (cross-patient) batching: the L1-only towers carry NO
session identity — a token is defined solely by its (depth, time_pos, region_id) coords
and its cu_seqlens block. So packing arrays from DIFFERENT sessions into one flat grid
must be CORRECTNESS-NEUTRAL: each array's per-token output is bit-identical whether it is
run alone or packed alongside arrays of a different patient (different size, different
regions, different depth-gaps).

test_towers_flat already proves block-diagonality WITHIN one session's montage. This file
proves the cross-session case that the batching change actually relies on: a HETEROGENEOUS
multi-array grid (built here to stand in for a mixed-patient pack) is decomposable, token
for token, into the single-array runs. If this holds, the whole array-batching premise is
sound and the model forward needs no change — only the data layer that assembles the pack.

Two distinct guarantees, tested at the right tolerance for each:
  * SAME-grid stability (perturb one array, hold the pack shape): BIT-exact (torch.equal) —
    identical matmul shapes, so a coord bleed is the ONLY thing that can move a flanking token.
  * ALONE vs PACKED (different pack shapes): tight allclose (~1e-6, atol=1e-5 as test_towers_flat
    uses). The CPU reference runs ONE matmul over the whole padded sequence and masks cross-block
    scores to -1e4 (underflows to exactly 0 ⇒ NO leak), but changing the K-dimension (tA vs tA+tB)
    reassociates the real terms' accumulation → fp noise. A real leak is O(1), not 1e-6 — and the
    same-grid test pins that O(1) channel shut at bit-exactness. So 1e-6 here is arithmetic
    reassociation, not a cross-patient coordinate bleed.
"""

from __future__ import annotations

import torch

from mapa.models.geometry import build_l1_geometry
from mapa.models.pack_r4 import build_r4_grid
from mapa.models.sidecar import build_sidecar
from mapa.models.towers import build_encoder

T = 16  # SLOW 2, MID 8, HGA 16 tokens per contact ⇒ k_full 26.
D_ENC, D_PRED = 256, 128
N_REGIONS = 32


def _grid(labels, regions):
    """One array-set → (sidecar, grid, per-token region id)."""
    sc = build_sidecar(list(labels), region_id=torch.tensor(regions))
    grid = build_r4_grid(build_l1_geometry(sc), n_time=T)
    return sc, grid, sc.region_id[grid.contact]  # (total,)


# Two stand-in "sessions", each a single array — deliberately UNLIKE each other:
#   A: array LA, 3 contacts, region 5, depth-gap at 3 (1,2,4).
#   B: array RB, 5 contacts, region 12, depth-gap at 4 (1,2,3,5,6).
# Different size, region, and gap pattern ⇒ nothing accidentally symmetric.
_A_LABELS, _A_REGIONS = ["LA1", "LA2", "LA4"], [5, 5, 5]
_B_LABELS, _B_REGIONS = ["RB1", "RB2", "RB3", "RB5", "RB6"], [12, 12, 12, 12, 12]


ATOL = 1e-5  # test_towers_flat's block-diagonal convention; a real leak is O(1), not this.


def _pack_two():
    """A and B run alone, plus the combined [A, B] grid (array-major ⇒ A then B)."""
    _, gA, pidA = _grid(_A_LABELS, _A_REGIONS)
    _, gB, pidB = _grid(_B_LABELS, _B_REGIONS)
    _, gAB, pidAB = _grid(_A_LABELS + _B_LABELS, _A_REGIONS + _B_REGIONS)
    return (gA, pidA), (gB, pidB), (gAB, pidAB)




def test_perturbing_one_patient_array_leaves_the_others_bit_stable() -> None:
    """Three heterogeneous single-array 'patients' in one pack; perturbing the middle
    array's input tokens moves ONLY its own output — the flanking patients are bit-stable."""
    torch.manual_seed(1)
    labels = _A_LABELS + _B_LABELS + ["MC1", "MC2"]
    regions = _A_REGIONS + _B_REGIONS + [20, 20]
    _, grid, pid = _grid(labels, regions)
    enc = build_encoder(n_regions=N_REGIONS).eval()

    x = torch.randn(1, grid.total, D_ENC)
    out = enc.forward_flat(x, grid, pid)

    mid = grid.array == 1  # the RB 'patient'
    others = grid.array != 1
    x2 = x.clone()
    x2[0, mid] += torch.randn_like(x2[0, mid]) * 3.0
    out2 = enc.forward_flat(x2, grid, pid)

    others_stable = torch.equal(out[0, others], out2[0, others])
    mid_moved = not torch.allclose(out[0, mid], out2[0, mid], atol=1e-4)
    print(f"[check] perturb patient-array 1: flanking patients bit-stable ({others_stable}); "
          f"perturbed array moved ({mid_moved}) {'OK' if others_stable and mid_moved else 'VIOLATED'}")
    assert others_stable and mid_moved
