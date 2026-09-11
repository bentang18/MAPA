"""Per-session array geometry, gathered once at setup.

Attention runs inside an array, so the contacts are pre-gathered into a ragged grid with one
row per array, short arrays padded, alongside a validity mask and the clinical contact index
each slot carries. Building this costs one pass over the labels and happens once per session,
never per window.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from mapa.models.sidecar import SensorSidecar


@dataclass(frozen=True)
class L1Geometry:
    """Padded per-array gather plan for the L1 block-diagonal attention."""

    gather_idx: Tensor  # (n_arrays, max_c) long — contact index into N; pad → 0
    valid: Tensor  # (n_arrays, max_c) bool — False on pad slots
    depth: Tensor  # (n_arrays, max_c) long — clinical contact index per slot
    array_of_contact: Tensor  # (N,) long — array id per contact (tier-split monitor)
    n_arrays: int
    max_c: int

    def to(self, device) -> L1Geometry:
        """Move the gather-plan tensors to ``device`` (for Lightning's per-batch
        device transfer — a frozen dataclass can't be moved by ``apply_to_collection``)."""
        import dataclasses

        return dataclasses.replace(
            self,
            gather_idx=self.gather_idx.to(device),
            valid=self.valid.to(device),
            depth=self.depth.to(device),
            array_of_contact=self.array_of_contact.to(device),
        )


def build_l1_geometry(sidecar: SensorSidecar) -> L1Geometry:
    """Group the sidecar's contacts into a padded ``(n_arrays, max_c)`` gather plan.

    Row s holds the contact indices of array s (order of appearance), zero-padded
    to ``max_c`` = the largest array's contact count. ``valid`` masks pad slots;
    ``depth`` carries each slot's clinical contact number (gaps preserved) for the
    index-RoPE inside the L1 block.
    """
    array_id = sidecar.array_id
    depth = sidecar.depth
    n_arrays = sidecar.n_arrays
    n = array_id.shape[0]

    members: list[list[int]] = [[] for _ in range(n_arrays)]
    for i in range(n):
        members[int(array_id[i])].append(i)
    max_c = max(len(m) for m in members)

    gather_idx = torch.zeros((n_arrays, max_c), dtype=torch.long)
    valid = torch.zeros((n_arrays, max_c), dtype=torch.bool)
    depth_pad = torch.zeros((n_arrays, max_c), dtype=torch.long)
    for s, m in enumerate(members):
        idx = torch.tensor(m, dtype=torch.long)
        gather_idx[s, : len(m)] = idx
        valid[s, : len(m)] = True
        depth_pad[s, : len(m)] = depth[idx]

    return L1Geometry(
        gather_idx=gather_idx,
        valid=valid,
        depth=depth_pad,
        array_of_contact=array_id.clone(),
        n_arrays=n_arrays,
        max_c=max_c,
    )
