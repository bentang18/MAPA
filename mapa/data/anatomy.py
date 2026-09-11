"""BrainTreebank electrode identity, Lite montage, and hard atlas labels."""

from __future__ import annotations

import functools
import json
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SupportKind = tp.Literal["soft_surface_bna", "hard_public_bt_label", "none"]
DEFAULT_BT_LABEL_COLUMN = "DesikanKilliany"


@dataclass(frozen=True)
class HardLabelSupport:
    """One-hot electrode support over public BrainTreebank anatomy labels.

    ``support[c]`` is the one-hot region assignment for ``electrode_labels[c]``;
    ``valid[c]`` is True iff that electrode was mapped to a region in the vocab
    (i.e. ``support[c]`` is nonzero). Under ``unmapped_policy="zero"`` an
    electrode with no anatomy row, or a DK label outside ``region_labels``, gets
    a zero ``support`` row and ``valid[c] = False`` **in place** — rows are never
    re-packed, so row ``c`` always names the same physical electrode as the
    voltage / token tensors it is collated beside (C1 fix).
    """

    kind: SupportKind
    electrode_labels: tuple[str, ...]
    region_labels: tuple[str, ...]
    support: np.ndarray
    valid: np.ndarray
    label_column: str


def load_public_bt_anatomy(
    bt_root: str | Path,
    subject_id: int,
    *,
    label_column: str = DEFAULT_BT_LABEL_COLUMN,
) -> pd.DataFrame:
    """Load public `depth-wm.csv` labels for one BrainTreebank subject."""

    path = Path(bt_root) / "localization" / f"sub_{subject_id}" / "depth-wm.csv"
    if not path.exists():
        raise FileNotFoundError(f"BrainTreebank anatomy file missing: {path}")
    anatomy = pd.read_csv(path)
    required = {"Electrode", label_column}
    missing = sorted(required - set(anatomy.columns))
    if missing:
        raise KeyError(f"{path}: missing required columns {missing}")
    out = anatomy.copy()
    out["Subject"] = f"sub_{subject_id}"
    out["Electrode"] = out["Electrode"].map(clean_bt_electrode_label)
    out[label_column] = out[label_column].astype(str)
    return out


def build_hard_public_bt_label_support(
    electrode_labels: tp.Sequence[str],
    anatomy: pd.DataFrame,
    region_labels: tp.Sequence[str],
    *,
    label_column: str = DEFAULT_BT_LABEL_COLUMN,
    include_hemisphere: bool = False,
    unmapped_policy: tp.Literal["raise", "zero"] = "raise",
) -> HardLabelSupport:
    """Build `(n_electrodes, n_labels)` one-hot support in ``electrode_labels`` order.

    ``unmapped_policy``:
    - ``"raise"`` (default): raise ``KeyError`` if any electrode has no anatomy
      row, or a DK label outside ``region_labels``.
    - ``"zero"``: leave a zero support row + ``valid=False`` for such electrodes,
      **in place** (no re-pack), so row ``c`` keeps naming ``electrode_labels[c]``.
    """

    required = {"Electrode", label_column}
    if include_hemisphere:
        required.add("Hemisphere")
    _require_columns(anatomy, required)

    cleaned = tuple(clean_bt_electrode_label(label) for label in electrode_labels)
    region_labels = tuple(str(label) for label in region_labels)
    region_index = {label: idx for idx, label in enumerate(region_labels)}
    if len(region_index) != len(region_labels):
        raise ValueError("region_labels must be unique")

    rows_by_electrode = {}
    for row in anatomy.itertuples(index=False):
        electrode = clean_bt_electrode_label(str(row.Electrode))
        label = str(getattr(row, label_column))
        if include_hemisphere:
            label = f"{row.Hemisphere}:{label}"
        rows_by_electrode[electrode] = label

    support = np.zeros((len(cleaned), len(region_labels)), dtype=np.float32)
    valid = np.zeros(len(cleaned), dtype=bool)
    missing_electrodes = []
    unknown_labels = []
    for electrode_idx, electrode in enumerate(cleaned):
        label = rows_by_electrode.get(electrode)
        if label is None:
            missing_electrodes.append(electrode)
            continue
        region_idx = region_index.get(label)
        if region_idx is None:
            unknown_labels.append(label)
            continue
        support[electrode_idx, region_idx] = 1.0
        valid[electrode_idx] = True

    if unmapped_policy == "raise":
        if missing_electrodes:
            raise KeyError(
                "missing BT anatomy rows for electrodes: "
                f"{missing_electrodes[:10]}"
                + (
                    f" (+{len(missing_electrodes) - 10} more)"
                    if len(missing_electrodes) > 10
                    else ""
                )
            )
        if unknown_labels:
            unique = sorted(set(unknown_labels))
            raise KeyError(
                "BT anatomy labels absent from region vocabulary: "
                f"{unique[:10]}"
                + (f" (+{len(unique) - 10} more)" if len(unique) > 10 else "")
            )

    return HardLabelSupport(
        kind="hard_public_bt_label",
        electrode_labels=cleaned,
        region_labels=region_labels,
        support=support,
        valid=valid,
        label_column=label_column,
    )


_BT_MISSING_COORDINATE_ELECTRODES: dict[int, tuple[str, ...]] = {
    1: ("F3cId10",),
    2: (),
    3: (
        "F3c9", "F3c10", "T1aIc1", "T1aIc2", "P2a10",
        "O1aIb2", "O1aIb3", "O1aIb4", "O1aIb5", "O1aIb6", "O1aIb7", "O1aIb8",
    ),
    4: ("LT1aIb10", "LF3bIa12"),
    5: (),
    6: (),
    7: ("LF3aOFa16", "LF1cCb12"),
    8: ("F2bCb6", "F2bCb14"),
    9: ("P2a6", "P2a7", "P2a8"),
    10: ("T1aIa4", "P2cCc5"),
}
"""Per-subject electrodes upstream drops for missing coordinates when
``allow_missing_coordinates=False`` (the ``BrainTreebankSubject`` default the
v14 loader uses). Copied verbatim from the PINNED upstream
``neuroprobe.braintreebank_subject.BrainTreebankSubject._get_corrupted_electrodes``
(commit ``c7b955b0``). Drift from upstream is caught by
``test_voltage_order_matches_upstream`` (the GPU cluster-only; skipped when neuroprobe is
not importable)."""


_BT_V14_EXTRA_BAD_ELECTRODES: dict[int, tuple[str, ...]] = {}
"""Per-subject flaky-contact FALLBACK set — RETIRED + EMPTY (baked out 2026-06-18).

Was a hand-tuned set. The GUARD-1 per-session scan
replaced it with the self-calibrated, per-session source
:data:`_BT_V14_EXTRA_BAD_ELECTRODES_PER_SESSION`, which is now baked. With this
dict empty, a call with no per-session entry — ``trial_id=None`` (laptop audits /
``SimpleNamespace`` test events) or a ``(subject, trial)`` outside the scanned
v14 corpus — drops nothing. Every scanned v14 session has an explicit per-session
entry (including ``()`` for clean sessions), so the production path never reaches
this fallback. Kept as the override-with-fallback mechanism's empty default
(restoring a per-subject entry here, e.g. for a new cohort, still works)."""

_BT_V14_EXTRA_BAD_ELECTRODES_PER_SESSION: dict[tuple[int, int], tuple[str, ...]] = {
    (1, 0): ("F3dIe10",),
    (1, 1): ("F3dIe10",),
    (1, 2): ("F3dIe10",),
    (2, 0): ("LT3bHa14",),
    (2, 1): ("RT1aIa7", "RT1aIa8"),
    (2, 2): ("LT3d7", "RT1aIa7", "RT1aIa8"),
    (2, 3): (),
    (2, 4): (),
    (2, 5): (),
    (2, 6): ("LT3cHb12", "RT2aA7", "RT2aA8", "RT3bHb10", "RT3bHb11", "RT3bHb12"),
    (3, 0): ("F3d10", "F3d9", "O1bId2", "O1bId3", "O1bId4", "O1bId5", "O1bId8", "T1cIe1", "T1cIe2"),
    (3, 1): ("O1bId2", "O1bId3", "O1bId4", "O1bId5", "O1bId8", "T1cIe1", "T1cIe2"),
    (3, 2): ("O1bId2", "O1bId3", "O1bId4", "O1bId5", "O1bId8", "T1aIc5", "T1aIc6", "T1cIe1", "T1cIe2"),
    (4, 0): (),
    (4, 1): ("LF3cIc10", "LT2bHb12"),
    (4, 2): ("LF3aOFa16", "LF3bIa7", "LT2aA11", "LT2aA12"),
    (6, 0): ("T2A13", "T2A8"),
    (6, 1): ("T2A13", "T2A8"),
    (6, 4): ("T2A8",),
    (7, 0): ("LF3bOFb1", "RF1bCb11", "RF3aOFa4", "RF3aOFa5"),
    (7, 1): ("LF3bOFb1", "RF1aCaOF1", "RF1aCaOF2", "RF1bCb11", "RF3aOFa4", "RF3aOFa5"),
    (8, 0): ("F3aOFa16", "T2A13"),
    (9, 0): ("P2e5", "P2e6", "P2e7", "P2e8", "T1c5"),
    (10, 0): ("F10Fa10", "F10Fa4", "F10Fa8", "F10Fa9", "F2bCa6", "F2bCa7", "F2bCa8"),
    (10, 1): ("F10Fa10", "F10Fa8", "F10Fa9", "F2bCa6", "F2bCa7", "F2bCa8"),
}
"""Per-SESSION ``(subject_id, trial_id) -> bad-contact`` override, the active
GUARD-1 static source (per-session drop, ).

Electrode quality drifts across a subject's recording sessions, so STATIC is a
per-session decision: each session's own GUARD-1 detector output IS its drop —
there is NO cross-trial aggregation heuristic (no any/majority/pool collapse).
Populated offline by the scan and its collector; an entry is the cleaned
static set ``static_bad_mask`` flagged for that one session. An explicit entry
(even ``()``) takes precedence over the per-subject fallback; a missing entry
falls back to :data:`_BT_V14_EXTRA_BAD_ELECTRODES` (retired + empty since the bake,
so a missing entry drops nothing). Both the cache
key (``study.iter_timelines`` folds ``extra_bad_electrodes(subject_id, trial_id)``
into the per-session timeline) and the support order
(``voltage_electrode_order(..., trial_id)``) read THIS dict with the SAME
``(subject, trial)``, so per-session DP4 row-alignment holds by construction."""


def extra_bad_electrodes(
    subject_id: int, trial_id: int | None = None
) -> frozenset[str]:
    """Cleaned-label set of v14 statically-excluded flaky contacts for a session.

    THE single source both electrode-order sites consult, so
    ``voltage_electrode_order`` (support/valid_mask) and ``bt_load_raw`` (front-end
    voltage) drop exactly the same contacts for a given ``(subject, trial)``.

    ``trial_id`` selects the per-session override
    (:data:`_BT_V14_EXTRA_BAD_ELECTRODES_PER_SESSION`): an explicit entry for
    ``(subject_id, trial_id)`` is returned verbatim (per-session REPLACES, does not
    union with, the subject fallback). A ``None`` trial, or a trial with no
    override entry, falls back to the per-subject
    :data:`_BT_V14_EXTRA_BAD_ELECTRODES` (retired and empty since the bake, so
    the fallback drops nothing) — laptop audits / drift guard pass ``None`` and get
    no drop; every scanned production session has an explicit per-session entry."""
    sid = int(subject_id)
    if trial_id is not None:
        key = (sid, int(trial_id))
        if key in _BT_V14_EXTRA_BAD_ELECTRODES_PER_SESSION:
            return frozenset(
                clean_bt_electrode_label(e)
                for e in _BT_V14_EXTRA_BAD_ELECTRODES_PER_SESSION[key]
            )
    return frozenset(
        clean_bt_electrode_label(e)
        for e in _BT_V14_EXTRA_BAD_ELECTRODES.get(sid, ())
    )


def _is_trigger_label(electrode_label: str) -> bool:
    up = electrode_label.upper()
    return up.startswith("DC") or up.startswith("TRIG")


def voltage_electrode_order(
    bt_root: str | Path, subject_id: int, trial_id: int | None = None
) -> tuple[str, ...]:
    """Canonical voltage channel order — exactly the electrodes (and order) the
    v14 loader feeds the front-end.

    This reproduces ``BrainTreebankSubject(subject_id).electrode_labels``: the
    cleaned ``electrode_labels.json`` order, with corrupted electrodes
    (``corrupted_elec.json``), missing-coordinate electrodes
    (``_BT_MISSING_COORDINATE_ELECTRODES``), and trigger (``DC*`` / ``TRIG*``)
    channels removed. It is derived from the same files upstream reads so the DK
    ``support`` / ``valid_mask`` extractors align to the VOLTAGE order, not the
    independent ``depth-wm.csv`` row order (C2 fix). Reading depth-wm row order
    instead silently routes every voltage into the wrong region for any subject
    whose corrupted/trigger/unmapped contacts shift the two orders apart.

    ``trial_id`` selects the per-session STATIC override (see
    :func:`extra_bad_electrodes`): the support/valid_mask order then drops exactly
    the contacts the loader drops for the SAME session. ``None`` (laptop audits /
    pre-bake) uses the per-subject fallback — byte-identical to the legacy order.

    Replicated (rather than imported) so laptop-side audits run against the
    vendored fixtures without neuroprobe; guarded against upstream drift by
    ``test_voltage_order_matches_upstream`` (skipped when neuroprobe is absent).
    """
    root = Path(bt_root)
    sid = int(subject_id)
    labels_path = root / "electrode_labels" / f"sub_{sid}" / "electrode_labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"BrainTreebank electrode_labels file missing: {labels_path}"
        )
    raw_labels = json.loads(labels_path.read_text())
    cleaned = [clean_bt_electrode_label(str(e)) for e in raw_labels]

    drop: set[str] = set()
    corrupted_path = root / "corrupted_elec.json"
    if corrupted_path.exists():
        corrupted = json.loads(corrupted_path.read_text())
        drop.update(
            clean_bt_electrode_label(str(e))
            for e in corrupted.get(f"sub_{sid}", ())
        )
    drop.update(
        clean_bt_electrode_label(e)
        for e in _BT_MISSING_COORDINATE_ELECTRODES.get(sid, ())
    )
    # v14 flaky-contact static exclusion (per-session) — kept in lockstep with
    # bt_load_raw via the shared extra_bad_electrodes() source so support/valid_mask
    # and the front-end voltage drop the SAME contacts for this (subject, trial)
    # (DP4 row-alignment).
    drop.update(extra_bad_electrodes(sid, trial_id))

    order = tuple(
        e for e in cleaned if e not in drop and not _is_trigger_label(e)
    )
    # Fail-loud on ambiguity (L4 single-source guard). This is THE single source
    # of electrode-row identity: support / valid_mask / the front-end scatter all
    # key off this order positionally. A duplicate label here means two physical
    # contacts are indistinguishable by name, which is exactly the precondition that
    # lets a name-keyed scatter collide rows. Raise rather than silently let row c
    # name two electrodes.
    if len(set(order)) != len(order):
        seen: set[str] = set()
        dups = sorted({e for e in order if e in seen or seen.add(e)})  # type: ignore[func-returns-value]
        raise ValueError(
            f"subject {sid}: voltage_electrode_order has duplicate electrode "
            f"labels {dups[:10]} after cleaning/drop — ambiguous row identity. "
            "Two contacts share a name; positional support/valid_mask/front-end "
            "alignment is undefined."
        )
    return order


def lite_electrode_set(subject_id: int) -> frozenset[str]:
    """The subject's Neuroprobe-Lite electrodes as a cleaned label **set**, with
    NO ``bt_root`` dependency (reads only the vendored ``NEUROPROBE_LITE_ELECTRODES``
    table).

    Used by the loader (:func:`bt_load_raw`) to subset voltage rows to the Lite
    montage PRE-CAR by name membership, without re-reading ``electrode_labels.json``.
    Because the loader's post-STATIC row order already equals
    :func:`voltage_electrode_order`, filtering its rows to this set order-preserving
    yields exactly :func:`lite_voltage_order` — so the loader lands byte-aligned with
    the support / valid-mask extractors (which key off ``lite_voltage_order``) while
    staying root-free.

    Lite labels are already cleaned (no ``*``/``#``); we clean defensively so the
    set is in the same label space as ``voltage_electrode_order``.
    """
    from ._neuroprobe_lite_tables import NEUROPROBE_LITE_ELECTRODES

    sid = int(subject_id)
    key = f"btbank{sid}"
    if key not in NEUROPROBE_LITE_ELECTRODES:
        raise KeyError(
            f"subject {sid} ({key}) absent from vendored NEUROPROBE_LITE_ELECTRODES"
        )
    return frozenset(
        clean_bt_electrode_label(e) for e in NEUROPROBE_LITE_ELECTRODES[key]
    )


def lite_voltage_mask(
    bt_root: str | Path, subject_id: int, trial_id: int | None = None
) -> np.ndarray:
    """Boolean mask over :func:`voltage_electrode_order` selecting the Neuroprobe
    Lite electrodes for this subject (``trial_id`` threads the per-session STATIC
    drop so the mask aligns to the post-STATIC voltage order for that session).

    Reproduces upstream ``BrainTreebankSubjectTrialBenchmarkDataset`` Lite
    subsetting (``datasets.py``: ``[full.index(e) for e in lite if e in full]``)
    as a **set**: the realized Lite subset is ``voltage_order ∩ lite_labels``.
    Upstream keeps the subset in Lite-list order; we keep voltage order, which is
    equivalent for the v14 encoder because the block-diagonal per-region pool is
    permutation-invariant within a region — only the *set* of electrodes feeding
    each region matters, not their order. Applying this one mask in lockstep to
    the study's voltage rows AND the DK support / valid-mask extractors keeps
    ``support[c]`` ↔ ``electrode_tokens[c]`` aligned after subsetting.
    """
    lite_set = lite_electrode_set(subject_id)
    order = voltage_electrode_order(bt_root, subject_id, trial_id)
    return np.array([e in lite_set for e in order], dtype=bool)


def lite_voltage_order(
    bt_root: str | Path, subject_id: int, trial_id: int | None = None
) -> tuple[str, ...]:
    """The subject's Lite electrodes in voltage order (``voltage_order`` filtered
    to the Lite set). Set-equal to upstream's realized ``electrode_labels`` for
    the Lite Dataset; see :func:`lite_voltage_mask` for why order is free.

    ``trial_id`` threads the per-session STATIC drop so the Lite montage is the
    post-STATIC survivors FOR THAT session (per-session drop applies to eval-Lite
    too, )."""
    order = voltage_electrode_order(bt_root, subject_id, trial_id)
    mask = lite_voltage_mask(bt_root, subject_id, trial_id)
    return tuple(e for e, keep in zip(order, mask) if keep)


def clean_bt_electrode_label(electrode_label: str) -> str:
    return str(electrode_label).replace("*", "").replace("#", "")


def _require_columns(table: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns - set(table.columns))
    if missing:
        raise KeyError(f"missing required columns {missing}")


_DK_APARC_BASE_LABELS: tuple[str, ...] = (
    "bankssts",
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "cuneus",
    "entorhinal",
    "frontalpole",
    "fusiform",
    "inferiorparietal",
    "inferiortemporal",
    "insula",
    "isthmuscingulate",
    "lateraloccipital",
    "lateralorbitofrontal",
    "lingual",
    "medialorbitofrontal",
    "middletemporal",
    "paracentral",
    "parahippocampal",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "pericalcarine",
    "postcentral",
    "posteriorcingulate",
    "precentral",
    "precuneus",
    "rostralanteriorcingulate",
    "rostralmiddlefrontal",
    "superiorfrontal",
    "superiorparietal",
    "superiortemporal",
    "supramarginal",
    "temporalpole",
    "transversetemporal",
)

_DK_ASEG_BASE_LABELS: tuple[str, ...] = (
    "Hippocampus",
    "Amygdala",
    "Caudate",
    "Putamen",
    "Pallidum",
    "Thalamus-Proper",
)

V14_DK_REGION_LABELS_CORTICAL: tuple[str, ...] = tuple(
    f"ctx-{hemi}-{base}" for hemi in ("lh", "rh") for base in _DK_APARC_BASE_LABELS
)
"""68 FreeSurfer DK aparc cortical labels (hemis-distinct), in BT depth-wm.csv string format."""

V14_DK_REGION_LABELS_SUBCORTICAL: tuple[str, ...] = tuple(
    f"{prefix}-{base}" for prefix in ("Left", "Right") for base in _DK_ASEG_BASE_LABELS
)
"""12 FreeSurfer aseg subcortical labels (Hippocampus/Amygdala/Caudate/Putamen/Pallidum/Thalamus-Proper, bilateral)."""

V14_DK_REGION_LABELS: tuple[str, ...] = (
    V14_DK_REGION_LABELS_CORTICAL + V14_DK_REGION_LABELS_SUBCORTICAL
)
"""Canonical K=80 v14 DK region vocabulary. Atlas-fixed, not cohort-derived —
keeps unpopulated regions alive for cross-cohort portability."""


# --- DKT (Desikan-Killiany-Tourville) atlas -------------------------------- #
# The BT depth-wm.csv carries a NATIVE ``DKT`` column (FreeSurfer aparc.DKTatlas),
# regionlated directly — NOT derivable by dropping DK electrodes. The DKT protocol
# removes three DK gyral labels whose boundaries it could not define reliably
# (bankssts, frontalpole, temporalpole) and REASSIGNS those vertices to neighbouring
# gyri, so an electrode DK-labelled e.g. ``temporalpole`` re-appears under a DKT
# neighbour label — it is not lost. Same string format as the DK column.
_DKT_DROPPED_BASES: frozenset[str] = frozenset(
    {"bankssts", "frontalpole", "temporalpole"}
)
_DKT_APARC_BASE_LABELS: tuple[str, ...] = tuple(
    b for b in _DK_APARC_BASE_LABELS if b not in _DKT_DROPPED_BASES
)
"""31 DKT cortical base regions = the 34 DK aparc bases minus the 3 DKT drops."""

V14_DKT_REGION_LABELS_CORTICAL: tuple[str, ...] = tuple(
    f"ctx-{hemi}-{base}" for hemi in ("lh", "rh") for base in _DKT_APARC_BASE_LABELS
)
"""62 FreeSurfer DKT aparc cortical labels (hemis-distinct), BT depth-wm.csv format."""

V14_DKT_REGION_LABELS_SUBCORTICAL: tuple[str, ...] = V14_DK_REGION_LABELS_SUBCORTICAL
"""12 aseg subcortical labels — identical to DK (DKT only changes cortical gyri)."""

V14_DKT_REGION_LABELS: tuple[str, ...] = (
    V14_DKT_REGION_LABELS_CORTICAL + V14_DKT_REGION_LABELS_SUBCORTICAL
)
"""Canonical K=74 v14 DKT region vocabulary (62 cortical + 12 subcortical).
Atlas-fixed, not cohort-derived — keeps unpopulated regions alive for
cross-cohort portability (mirrors :data:`V14_DK_REGION_LABELS`)."""


# --- lobe grouping --------------------------------------------------------- #
# Regions are too fine to intersect across subjects: pairwise DKT supports run
# 3-7 regions and the 12-way Neuroprobe-Lite intersection is EMPTY, so a shared
# anatomical axis does not exist at region resolution. Lobes are the coarsest
# grouping that still separates the speech-relevant cortex, and each subject
# covers more lobes than it shares regions. Assignment follows FreeSurfer's own
# lobe convention (paracentral -> frontal; fusiform + parahippocampal ->
# temporal). Hemisphere is KEPT DISTINCT — lateralization is not a nuisance here.
_ATLAS_SPECS: dict[str, tuple[str, tuple[str, ...]]] = {
    "dk": ("DesikanKilliany", V14_DK_REGION_LABELS),
    "dkt": ("DKT", V14_DKT_REGION_LABELS),
}


def atlas_spec(atlas: str) -> tuple[str, tuple[str, ...]]:
    """Resolve an atlas name to its ``(label_column, region_labels)`` pair.

    The ONLY sanctioned way to pick an atlas — guarantees the depth-wm.csv column
    and the region vocabulary are the matched pair (``"dk"`` → DesikanKilliany /
    K=80, ``"dkt"`` → DKT / K=74). Raises on an unknown name rather than silently
    falling back to DK.
    """
    key = str(atlas).lower()
    if key not in _ATLAS_SPECS:
        raise ValueError(
            f"unknown atlas {atlas!r}; expected one of {sorted(_ATLAS_SPECS)}"
        )
    return _ATLAS_SPECS[key]


_MIN_REGION_VALID_FRACTION: float = 0.5
"""Per-subject floor on the fraction of voltage electrodes that map to an
in-vocab DK region. MEASURED 2026-06-13 across all 10 vendored BT subjects:
9/10 map at 1.000, sub_4 at 0.989 (2 ``Left-Inf-Lat-Vent`` ventricle contacts
legitimately outside the K=80 cortical+subcortical vocab), and 0 missing-anatomy
rows anywhere. The 0.5 floor sits ~18x below the worst real subject, so it never
false-fires on the live cohort but catches an S9-class silent collapse: a
per-subject ``depth-wm.csv`` namespace shift / re-rip / new subject whose DK
labels stop matching the ``electrode_labels.json`` voltage namespace would drop
most/all electrodes to ``valid=False`` (a zero support row each) with NO error
under ``unmapped_policy='zero'``, silently erasing that subject's entire
region-pool routing."""


def _assert_region_support_coverage(
    result: HardLabelSupport, subject_id: int
) -> None:
    """Fail loud if a subject's DK-region coverage collapses (ledger LG15).

    Under ``unmapped_policy='zero'`` an electrode with no anatomy row or an
    out-of-vocab DK label is silently zeroed (``valid=False``). A catastrophic
    drop — far below the measured 0.989-1.000 live range — means the electrode
    label namespace diverged for this subject, not a few stray ventricle
    contacts, so the whole subject's pooled features would be degenerate.
    """
    n_total = int(result.valid.shape[0])
    if n_total == 0:
        return
    n_valid = int(result.valid.sum())
    fraction = n_valid / n_total
    if fraction < _MIN_REGION_VALID_FRACTION:
        raise ValueError(
            f"subject {subject_id}: only {n_valid}/{n_total} voltage electrodes "
            f"({fraction:.3f}) mapped to an in-vocab {result.label_column} region "
            f"— below the {_MIN_REGION_VALID_FRACTION:.2f} floor (DK cohort is "
            "0.989-1.000; DKT 0.888-1.000, measured 2026-06-13). "
            f"The depth-wm.csv {result.label_column} labels likely stopped matching "
            "the voltage electrode namespace for this subject (S9-class); its "
            "region-pool routing would be silently zeroed under "
            "unmapped_policy='zero'. "
            f"Rebuild/verify localization/sub_{subject_id}/depth-wm.csv. "
            "See data_pipeline_bug_ledger.md LG15."
        )


@functools.lru_cache(maxsize=256)
def aligned_voltage_support(
    bt_root: str | Path,
    subject_id: int,
    *,
    trial_id: int | None = None,
    region_labels: tuple[str, ...] = V14_DK_REGION_LABELS,
    unmapped_policy: tp.Literal["raise", "zero"] = "raise",
    label_column: str = DEFAULT_BT_LABEL_COLUMN,
    electrode_set: tp.Literal["all", "lite"] = "all",
) -> HardLabelSupport:
    """Atlas support aligned to voltage contact identity; callers must not mutate cached arrays.

    Use atlas_spec to pair the label column and vocabulary. Unmapped contacts
    retain zero rows under unmapped_policy="zero"; coverage is checked before return.
    """
    order = (
        lite_voltage_order(bt_root, subject_id, trial_id)
        if electrode_set == "lite"
        else voltage_electrode_order(bt_root, subject_id, trial_id)
    )
    anatomy = load_public_bt_anatomy(
        bt_root, int(subject_id), label_column=label_column
    )
    result = build_hard_public_bt_label_support(
        order,
        anatomy,
        region_labels,
        label_column=label_column,
        unmapped_policy=unmapped_policy,
    )
    _assert_region_support_coverage(result, int(subject_id))
    return result
