"""What the model has to be told about the electrodes, and nothing more.

Three values per contact, in the order the recording stores them:

  array_id   which array the contact belongs to. Attention is block-diagonal over this.
  depth      the clinical contact number, read off the label. Gaps are kept: if a contact is
             dropped, its neighbours stay two apart rather than being renumbered to one.
             Only the difference between two contacts on the same array is ever used, so the
             origin does not matter, but direction and gaps must be preserved.
  region_id  the DKT atlas region the contact falls in, which is the one identity that means
             the same thing in a subject the model has never seen.

These metadata describe each recording's contact layout, which lets a
model trained on one set of subjects read another.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from mapa.data.array_label import parse_array


@dataclass(frozen=True)
class SensorSidecar:
    """Row-aligned per-contact geometry for one session's surviving electrodes."""

    array_id: Tensor  # (N,) long
    depth: Tensor  # (N,) long — canonical clinical contact index, gaps preserved
    region_id: Tensor  # (N,) long — hard atlas tag
    n_arrays: int
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        n = len(self.labels)
        for name, t in (("array_id", self.array_id), ("depth", self.depth),
                        ("region_id", self.region_id)):
            if t.shape != (n,):
                raise ValueError(f"{name} shape {tuple(t.shape)} != ({n},)")
            if t.dtype != torch.long:
                raise ValueError(f"{name} dtype {t.dtype} != torch.long")


def _parse_array_depth(labels: Sequence[str]) -> tuple[Tensor, Tensor, int]:
    """labels → (array_id (N,) long, depth (N,) long, n_arrays).

    array_id: contiguous, order of first appearance (same alphabetic prefix ⇒ same
    array). depth: the trailing clinical contact number, kept verbatim (gaps
    preserved). A label with no trailing number cannot have a canonical depth and
    is rejected (it is a mislabel or a non-neural channel that should have been
    dropped upstream).
    """
    seen: dict[str, int] = {}
    array: list[int] = []
    depth: list[int] = []
    for label in labels:
        prefix, num = parse_array(str(label))
        if num is None:
            raise ValueError(
                f"label {label!r} has no trailing contact number — cannot assign a "
                "canonical depth index (drop non-neural/trigger channels upstream)"
            )
        if prefix not in seen:
            seen[prefix] = len(seen)
        array.append(seen[prefix])
        depth.append(num)
    return (
        torch.tensor(array, dtype=torch.long),
        torch.tensor(depth, dtype=torch.long),
        len(seen),
    )


def build_sidecar(
    labels: Sequence[str],
    *,
    region_id: Tensor | None = None,
    support: Tensor | None = None,
) -> SensorSidecar:
    """Assemble the sidecar for one session's surviving electrode labels.

    Provide the hard region tag either directly (``region_id``, (N,) integers as a tensor,
    array, or list) or as the one-hot ``support`` (N, K) to argmax. Exactly one is required.
    """
    if (region_id is None) == (support is None):
        raise ValueError("provide exactly one of region_id or support")
    labels = tuple(str(x) for x in labels)
    n = len(labels)
    array_id, depth, n_arrays = _parse_array_depth(labels)

    if region_id is not None:
        pid = torch.as_tensor(region_id)
        if pid.is_floating_point() or pid.is_complex() or pid.dtype == torch.bool:
            raise ValueError("region_id must contain integer DKT indices 0..74")
        pid = pid.long()
        if pid.shape != (n,):
            raise ValueError(f"region_id shape {tuple(pid.shape)} != ({n},)")
    elif support is not None:
        if support.ndim != 2 or support.shape[0] != n:
            raise ValueError(
                f"support shape {tuple(support.shape)} incompatible with N={n}"
            )
        pid = support.argmax(dim=-1).long()
    else:
        raise ValueError("provide either region_id or support")

    if not labels or len(set(labels)) != n:
        raise ValueError("provide a nonempty sequence of unique contact labels")
    if torch.any((pid < 0) | (pid > 74)):
        raise ValueError("region_id must contain integer DKT indices 0..74")

    return SensorSidecar(
        array_id=array_id,
        depth=depth,
        region_id=pid,
        n_arrays=n_arrays,
        labels=labels,
    )
