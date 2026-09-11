"""DKT tags aligned to BrainTreebank cache contact labels."""
from __future__ import annotations

import typing as tp

import torch

from mapa.data.session_loader import RegionFn


def make_bt_region_fn(
    bt_root: str,
    *,
    atlas: str = "dkt",
) -> RegionFn:
    """Default ``region_fn``: DKT hard tag per electrode from BT anatomy (F2 seam).

    Maps each cache channel label to its one-hot DKT region via
    ``aligned_voltage_support`` (row-aligned to the voltage order). Electrodes with
    NO region support (all-zero row — outside every DKT region) get the RESERVED
    "unknown" id = ``len(region_labels)`` (=74), an identity distinct from every real
    region 0..73 that never collides with ``region_labels[0]``. ``_n_regions``
    reserves the matching +1 identity-table row. Atlas name threads
    ``anatomy.atlas_spec`` so the DKT CSV column and the K=74 vocabulary can never
    desync. Lazily imported so the launcher is importable (and unit-testable with a
    stub region_fn) without the BT anatomy stack.
    """
    from mapa.data.anatomy import (
        aligned_voltage_support,
        atlas_spec,
    )

    lcol, plabels = atlas_spec(atlas)  # "dkt" → ("DKT", V14_DKT_REGION_LABELS/K=74)
    unknown_id = len(plabels)  # reserved id, distinct from every real region 0..K-1

    def region_fn(subject_id: int, trial_id: int, labels: tp.Sequence[str]) -> torch.Tensor:
        hs = aligned_voltage_support(
            bt_root, subject_id, trial_id=trial_id,
            region_labels=plabels, unmapped_policy="zero", label_column=lcol,
        )
        by_label = {
            lab: (int(hs.support[c].argmax()) if bool(hs.support[c].any()) else unknown_id)
            for c, lab in enumerate(hs.electrode_labels)
        }
        missing = [lab for lab in labels if lab not in by_label]
        if missing:
            raise KeyError(
                f"subject {subject_id} trial {trial_id}: cache labels absent from the "
                f"voltage order {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        return torch.tensor([by_label[lab] for lab in labels], dtype=torch.long)

    return region_fn
