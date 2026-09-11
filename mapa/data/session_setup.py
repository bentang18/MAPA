"""Per-session setup: everything fixed for the session's lifetime.

The pure core of the per-session build. One call turns a session's FULL voltage electrode order + hard region
tags + the guard-1 bad-electrode set into everything the training loop needs that
is fixed for the session's lifetime:

    labels, region_id, drop_labels ─▶ guard-1 filter ─▶ SensorSidecar
                                                     ─▶ L1Geometry
                                                     ─▶ assert_mask_feasible      
                                         keep_idx = the memmap channel-read plan

GUARD-1 (bad ELECTRODES) is voltage-domain and hop-independent, so it lands here at
sidecar/geometry build — NOT as a spec-cache recompute. The drop set is the union
of ``extra_bad`` (manual, from the spec .json ``key.timeline.extra_bad``) and the
LOF / ``drop_bads`` contacts; both name ELECTRODES, so the core drops by label.

This is the once-per-session cold path (montage is fixed for the session), so plain
python filtering is fine — the "always vectorize" rule governs the per-step hot
loop, not setup. The file-reading adapter that supplies ``labels`` / ``region_id`` /
``drop_labels`` from disk is a thin layer validated on real data at F2; the logic
here is exercised synthetically.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from mapa.models.geometry import (
    L1Geometry,
    build_l1_geometry,
)
from mapa.models.sidecar import (
    SensorSidecar,
    build_sidecar,
)


@dataclass(frozen=True)
class V3SessionSetup:
    """Everything fixed for one session's lifetime after guard-1."""

    keep_idx: Tensor  # (N,) long — survivor rows into the FULL voltage order = memmap read plan
    sidecar: SensorSidecar  # array_id / depth / region_id over survivors
    geom: L1Geometry  # padded per-array L1 gather plan
    region_id: Tensor  # (N,) long — survivor region tags (== sidecar.region_id)


def build_session_setup(
    labels: Sequence[str],
    region_id: Tensor,
    *,
    drop_labels: Iterable[str],
) -> V3SessionSetup:
    """Guard-1 filter → sidecar → geometry.

    ``labels`` is the FULL session voltage electrode order (memmap row order);
    ``region_id`` (N_full,) is the hard DKT tag per full-order electrode.
    ``drop_labels`` = extra_bad ∪ LOF/drop_bads; labels not in the montage are a
    no-op (a source may name a channel already excluded upstream).
    """
    labels = tuple(str(x) for x in labels)
    if region_id.shape != (len(labels),):
        raise ValueError(
            f"region_id shape {tuple(region_id.shape)} != ({len(labels)},) (full order)"
        )
    drop = set(drop_labels)
    keep_positions = [i for i, lab in enumerate(labels) if lab not in drop]
    if not keep_positions:
        raise ValueError("guard-1 dropped every electrode — no survivors")

    keep_idx = torch.tensor(keep_positions, dtype=torch.long)
    keep_labels = tuple(labels[i] for i in keep_positions)
    keep_region = region_id[keep_idx].long()

    sidecar = build_sidecar(keep_labels, region_id=keep_region)
    geom = build_l1_geometry(sidecar)

    return V3SessionSetup(
        keep_idx=keep_idx,
        sidecar=sidecar,
        geom=geom,
        region_id=sidecar.region_id,
    )
