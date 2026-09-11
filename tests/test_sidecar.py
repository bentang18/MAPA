"""The sidecar: what the model has to be told about the electrodes.

The sidecar turns a session's ordered surviving electrode labels and hard atlas
support into the per-contact geometry the encoder and the masking need:

  - array_id  : contiguous group id for L1 block-diagonal attention + mask ∝ count
  - depth     : CANONICAL along-array contact index off the FULL array, drop-gaps
                PRESERVED — this is the clinical contact number (parse_array[1]),
                NOT a dense re-numbering. Keep the gap, do NOT renumber:
                a dropped LTG5 leaves LTG4→4 and LTG6→6 at distance 2.
  - region_id : DKT/DK hard tag = support.argmax(-1), the cross-subject query identity.

All three are row-aligned to the voltage electrode order (the survivors).
"""

from __future__ import annotations

import pytest
import torch

from mapa.models.sidecar import (
    SensorSidecar,
    build_sidecar,
)


def _onehot(region_ids: list[int], k: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(region_ids), num_classes=k).float()


def test_array_ids_are_contiguous_first_appearance() -> None:
    labels = ["LTA1", "LTA2", "LTB1", "LTA3", "LTB2"]
    sc = build_sidecar(labels, region_id=torch.zeros(5, dtype=torch.long))
    # LTA→0 (first seen), LTB→1 (second seen); revisits reuse the id.
    assert sc.array_id.tolist() == [0, 0, 1, 0, 1]
    assert sc.n_arrays == 2


def test_depth_preserves_drop_gaps() -> None:
    # LTG3 dropped as a bad contact → survivors keep clinical numbers 1,2,4,5.
    labels = ["LTG1", "LTG2", "LTG4", "LTG5"]
    sc = build_sidecar(labels, region_id=torch.zeros(4, dtype=torch.long))
    assert sc.depth.tolist() == [1, 2, 4, 5]
    # The crux: LTG2 ↔ LTG4 relative distance MUST be 2 (the hole), not 1.
    assert (sc.depth[2] - sc.depth[1]).item() == 2


def test_depth_is_not_densely_renumbered() -> None:
    # A dense renumber (the forbidden _bt_contact_pos behavior) would give 0,1,2,3.
    labels = ["OFa2", "OFa5", "OFa9"]
    sc = build_sidecar(labels, region_id=torch.zeros(3, dtype=torch.long))
    assert sc.depth.tolist() == [2, 5, 9]
    assert sc.depth.tolist() != [0, 1, 2]


def test_multidigit_stem_peels_only_trailing_number() -> None:
    # Stem itself contains digits ("LT3aHa"); only the trailing run is the depth.
    labels = ["LT3aHa12", "LT3aHa14"]
    sc = build_sidecar(labels, region_id=torch.zeros(2, dtype=torch.long))
    assert sc.array_id.tolist() == [0, 0]  # same array
    assert sc.depth.tolist() == [12, 14]


def test_region_id_from_support_argmax() -> None:
    labels = ["LTA1", "LTA2", "LTB1"]
    support = _onehot([3, 3, 7], k=10)
    sc = build_sidecar(labels, support=support)
    assert sc.region_id.tolist() == [3, 3, 7]
    assert sc.region_id.dtype == torch.long


def test_label_without_contact_number_is_rejected() -> None:
    # Every real sEEG/grid contact ends in a number; a bare label = mislabel → fail loud.
    with pytest.raises(ValueError):
        build_sidecar(["LTA1", "TRIG"], region_id=torch.zeros(2, dtype=torch.long))


def test_requires_support_or_region_id() -> None:
    with pytest.raises(ValueError):
        build_sidecar(["LTA1", "LTA2"])


def test_full_sidecar_shapes_dtypes_and_alignment() -> None:
    labels = ["LTA1", "LTA2", "LTA4", "LTB1", "LTB2"]
    support = _onehot([1, 1, 1, 5, 5], k=8)
    sc = build_sidecar(labels, support=support)
    n = len(labels)
    assert isinstance(sc, SensorSidecar)
    for t in (sc.array_id, sc.depth, sc.region_id):
        assert t.shape == (n,)
        assert t.dtype == torch.long
    assert sc.labels == tuple(labels)
    assert sc.n_arrays == 2
    # Row alignment: array boundary at index 3 coincides with the label prefix change.
    assert sc.array_id.tolist() == [0, 0, 0, 1, 1]
    assert sc.depth.tolist() == [1, 2, 4, 1, 2]


def test_length_mismatch_between_labels_and_region_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_sidecar(["LTA1", "LTA2"], region_id=torch.zeros(3, dtype=torch.long))


def test_ambiguous_region_sources_are_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        build_sidecar(["LA1"], region_id=[0], support=torch.ones(1, 1))


def test_region_ids_accept_a_python_list():
    assert build_sidecar(["LA1", "LA2"], region_id=[2, 3]).region_id.tolist() == [2, 3]


@pytest.mark.parametrize("ids", [[0.5, 1.5], [-1, 0], [0, 75], [True, False]])
def test_invalid_region_ids_are_not_silently_coerced(ids):
    with pytest.raises(ValueError, match="integer DKT indices"):
        build_sidecar(["LA1", "LA2"], region_id=ids)


def test_duplicate_contacts_are_rejected():
    with pytest.raises(ValueError, match="unique contact labels"):
        build_sidecar(["LA1", "LA1"], region_id=[0, 1])
