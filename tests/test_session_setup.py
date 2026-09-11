"""Per-session setup: the survivor ordering, the sidecar, the geometry.

``build_session_setup`` is the pure core of the per-session build: from the FULL session voltage
electrode order + hard region tags + the guard-1 bad-electrode set, produce the
survivor ordering the clip loader reads (``keep_idx``), the sidecar (array/depth/
region), the array geometry, and a fail-loud feasibility check.

GUARD-1 (bad ELECTRODES) is voltage-domain + hop-independent, so it lands HERE at
sidecar/geometry build, NOT as a cache recompute: drop ``extra_bad`` (manual,
travels in the spec .json ``key.timeline.extra_bad``) ∪ LOF/``drop_bads`` from the
valid set. Both sources name ELECTRODES (label strings), the stable identity
across the two sources, so the core drops by label.

The depth-gap invariant is the sharpest test: dropping a
mid-array contact must LEAVE THE GAP in ``depth`` — the survivors keep their raw
clinical index, never re-densified.
"""

from __future__ import annotations

import torch

from mapa.data.session_setup import build_session_setup


def _labels(array_sizes):
    labels = []
    for s, n in enumerate(array_sizes):
        for c in range(1, n + 1):
            labels.append(f"L{chr(65 + s)}{c}")
    return labels


def _regions(labels):
    # one region per array prefix, deterministic
    prefixes = {}
    pid = []
    for lab in labels:
        pre = lab.rstrip("0123456789")
        prefixes.setdefault(pre, len(prefixes))
        pid.append(prefixes[pre])
    return torch.tensor(pid, dtype=torch.long)


def test_no_drops_keeps_every_contact_in_order() -> None:
    labels = _labels((4, 3, 3))
    region = _regions(labels)
    setup = build_session_setup(labels, region, drop_labels=set())
    assert setup.keep_idx.tolist() == list(range(10))
    assert setup.sidecar.labels == tuple(labels)
    assert torch.equal(setup.region_id, region)
    assert setup.geom.n_arrays == 3


def test_drop_removes_named_electrodes_order_preserved() -> None:
    labels = _labels((4, 3, 3))  # LA1..4 LB1..3 LC1..3
    region = _regions(labels)
    setup = build_session_setup(labels, region, drop_labels={"LA2", "LB1"})
    kept = [labels[i] for i in setup.keep_idx.tolist()]
    assert kept == ["LA1", "LA3", "LA4", "LB2", "LB3", "LC1", "LC2", "LC3"]
    assert setup.keep_idx.tolist() == [0, 2, 3, 5, 6, 7, 8, 9]




def test_depth_gap_preserved_after_mid_array_drop() -> None:
    # LA1..LA5 → drop LA3 → survivors keep clinical depths [1,2,4,5], NOT re-densified.
    labels = ["LA1", "LA2", "LA3", "LA4", "LA5", "LB1", "LB2"]
    region = _regions(labels)
    setup = build_session_setup(labels, region, drop_labels={"LA3"})
    la = setup.sidecar.array_id == 0
    assert setup.sidecar.depth[la].tolist() == [1, 2, 4, 5]


def test_region_id_realigns_to_survivors() -> None:
    labels = _labels((3, 3))
    region = torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.long)
    setup = build_session_setup(labels, region, drop_labels={"LA2"})
    # LA2 (region 11) removed; survivors carry [10,12,20,21,22]
    assert setup.region_id.tolist() == [10, 12, 20, 21, 22]


def test_keep_idx_indexes_full_order_for_memmap_read() -> None:
    labels = _labels((4, 3))
    region = _regions(labels)
    setup = build_session_setup(
        labels, region, drop_labels={"LA1", "LB3"}
    )
    # keep_idx must be a strictly-ascending gather into the full voltage order
    ki = setup.keep_idx
    assert ki.dtype == torch.long
    assert torch.equal(ki, ki.sort().values)
    assert (ki[1:] > ki[:-1]).all()


def test_drop_label_not_in_montage_is_a_noop() -> None:
    # extra_bad may name a channel already excluded upstream; intersect, don't fail.
    labels = _labels((3, 3))
    region = _regions(labels)
    setup = build_session_setup(
        labels, region, drop_labels={"LZ9", "LA2"}
    )
    assert "LA2" not in setup.sidecar.labels
    assert len(setup.sidecar.labels) == 5




def test_feasibility_passes_for_uniform_seeg() -> None:
    # Uniform sEEG (largest array ≪ D) is always feasible — no raise.
    labels = _labels((8, 8, 8, 8, 8, 8))
    region = _regions(labels)
    setup = build_session_setup(labels, region, drop_labels=set())  # must not raise
    assert setup.geom.n_arrays == 6


