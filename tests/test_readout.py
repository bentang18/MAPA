"""Published readout splits, numeric parity, and validation-only selection."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from evals.neuroprobe.readout import (
    BOARD_TASKS,
    CS_TEST_CELLS,
    CS_TRAIN_ANCHOR,
    LAM_MULTS,
    LITE_SESSIONS,
    _absorb,
    _blank,
    _cs_cell,
    _finalize,
    _lam_grid,
    _select_lam,
    _ws_cell,
    auroc,
)


def _rec(n=64, n_parcels=4, feat=8, *, parcels=None, signal=True, seed=0):
    """A synthetic session cache in the encode's payload format.

    Labels alternate 0/1; when `signal` the feature's first column IS the label, so a correct
    readout must score AUROC 1.0 on the test half and a broken one cannot.
    """
    rng = np.random.default_rng(seed)
    y = np.array([float(i % 2) for i in range(n)])
    x = rng.normal(size=(n, n_parcels, feat)).astype(np.float32)
    if signal:
        x[:, 0, 0] = y * 10.0
    half = n // 2
    ws_split = {
        t: {
            0: {"train": np.arange(half), "val": np.arange(half, half + half // 2),
                "test": np.arange(half + half // 2, n)},
            1: {"train": np.arange(half, n), "val": np.arange(0, half // 2),
                "test": np.arange(half // 2, half)},
        }
        for t in BOARD_TASKS
    }
    cs_split = {t: {"val": np.arange(0, half), "test": np.arange(half, n)} for t in BOARD_TASKS}
    return {
        "feats": {"enc12": {"raw": torch.from_numpy(x).to(torch.float16)},
                  "enc0": {"raw": torch.from_numpy(x).to(torch.float16)}},
        "present_parcels": np.asarray(
            parcels if parcels is not None else np.arange(n_parcels), dtype=np.int64),
        "labels": {t: y.copy() for t in BOARD_TASKS},
        "ws_split": ws_split,
        "cs_split": cs_split,
    }

def test_board_constants_match_upstream_neuroprobe() -> None:
    """The eval universe IS the claim — pin it against the installed upstream package."""
    import importlib.util
    import os

    os.environ.setdefault("ROOT_DIR_BRAINTREEBANK", "/tmp")
    pkg = importlib.util.find_spec("neuroprobe")
    if pkg is None or not pkg.submodule_search_locations:
        pytest.skip("upstream neuroprobe not installed")
    from pathlib import Path
    path = str(Path(next(iter(pkg.submodule_search_locations))) / "config.py")
    spec = importlib.util.spec_from_file_location("npcfg", path)
    if spec is None or spec.loader is None:
        pytest.skip("upstream neuroprobe not installed")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    assert BOARD_TASKS == tuple(m.NEUROPROBE_TASKS)
    assert LITE_SESSIONS == tuple(tuple(x) for x in m.NEUROPROBE_LITE_SUBJECT_TRIALS)
    assert CS_TRAIN_ANCHOR == (m.DS_DM_TRAIN_SUBJECT_ID, m.DS_DM_TRAIN_TRIAL_ID)
    # Upstream asserts test_subject != anchor subject; the 10 cells are every other Lite cell.
    assert CS_TEST_CELLS == tuple(c for c in LITE_SESSIONS if c[0] != CS_TRAIN_ANCHOR[0])

def test_lam_grid_scores_every_lambda_on_every_eval_set() -> None:
    rng = np.random.default_rng(0)
    y = np.array([float(i % 2) for i in range(40)])
    z = rng.normal(size=(40, 6))
    z[:, 0] = y * 5.0
    out = _lam_grid(z[:20], y[:20], {"val": (z[20:30], y[20:30]), "test": (z[30:], y[30:])})
    assert set(out) == {"val", "test"}
    from evals.neuroprobe.readout import LAM_MULTS

    assert set(out["val"]) == set(LAM_MULTS)
    # A feature that IS the label separates perfectly at some lambda.
    assert max(out["test"].values()) == pytest.approx(1.0)

def test_select_lam_takes_argmax_val_and_reports_that_lambdas_test() -> None:
    """The core contract: the winner is chosen by VAL, but the number returned is its TEST."""
    got = _select_lam({"val": {1.0: 0.60, 10.0: 0.90}, "test": {1.0: 0.99, 10.0: 0.55}})
    assert got["lam_mult"] == 10.0          # even though lam=1 has the better TEST
    assert got["test"] == 0.55
    assert got["val"] == 0.90

def test_select_lam_never_peeks_at_test_to_break_ties() -> None:
    """Equal val ⇒ first-seen wins; the better TEST must not be able to pull the choice."""
    got = _select_lam({"val": {1.0: 0.80, 10.0: 0.80}, "test": {1.0: 0.10, 10.0: 0.99}})
    assert got["test"] == 0.10

def test_select_lam_all_nan_val_is_nan_not_a_default_lambda() -> None:
    got = _select_lam({"val": {1.0: float("nan")}, "test": {1.0: 0.99}})
    assert np.isnan(got["test"]) and np.isnan(got["lam_mult"])

def test_select_lam_flags_only_the_LO_boundary_as_truncation() -> None:
    """The two boundaries are different failures, and only LO invalidates a fit.

    HI is benign: AUROC saturates as λ→∞ (the smoother w/(w+λ)→0 uniformly, so the ranking — and
    hence the AUROC — converges; see test_auroc_saturates_at_high_lambda). "Maximal shrinkage
    won" is a faithful answer, not an artifact, and widening the grid cannot change it.
    LO has no such limit: λ→0 keeps moving, so a LO argmax really is truncated.
    """
    hi = _select_lam({"val": {LAM_MULTS[-1]: 0.9}, "test": {LAM_MULTS[-1]: 0.8}})
    assert hi["lam_pin"] == "hi" and hi["lam_pinned"] is False

    lo = _select_lam({"val": {LAM_MULTS[0]: 0.9}, "test": {LAM_MULTS[0]: 0.8}})
    assert lo["lam_pin"] == "lo" and lo["lam_pinned"] is True

    mid = LAM_MULTS[len(LAM_MULTS) // 2]
    got = _select_lam({"val": {mid: 0.9}, "test": {mid: 0.8}})
    assert got["lam_pin"] == "" and got["lam_pinned"] is False

def test_auroc_saturates_at_high_lambda() -> None:
    """The measurement the LO/HI split rests on: past some λ the AUROC stops moving entirely, so
    a HI pin cannot be hiding a better score further out.

    As λ→∞, α = V diag(1/(w+λ)) Vᵀ y → (1/λ)·y, so the scores → (1/λ)·K@y — a POSITIVE rescale of
    K@y, which AUROC is invariant to. So the top of the grid already IS the limit.
    """
    rng = np.random.default_rng(0)
    z_tr = rng.normal(size=(60, 8))
    y_tr = (z_tr[:, 0] + 0.3 * rng.normal(size=60) > 0).astype(float)
    z_te = rng.normal(size=(40, 8))
    y_te = (z_te[:, 0] > 0).astype(float)

    g = _lam_grid(z_tr, y_tr, {"test": (z_te, y_te)})
    limit = auroc(np.asarray(z_te @ z_tr.T) @ y_tr, y_te)
    assert g["test"][LAM_MULTS[-1]] == limit

def test_lambda_grid_brackets_the_diagnostics_pinned_lam_mult() -> None:
    """lam_mult=1.0 (the r4 diagnostic's fixed value) must be an interior grid point, so the
    board number is never worse than the diagnostic's for want of a λ."""
    assert min(LAM_MULTS) < 1.0 < max(LAM_MULTS)
    assert any(abs(m - 1.0) < 1e-9 for m in LAM_MULTS)

def test_ws_cell_reports_every_tap_x_norm_and_selects_only_lambda() -> None:
    """Keep each requested tap’s result without selecting across taps."""
    import evals.neuroprobe.readout as mod
    rec = _rec()
    rec["feats"]["enc12_elec"] = rec["feats"]["enc12"]
    got = mod._ws_cell(rec, "onset", ("enc12_elec", "enc12", "enc0"))
    assert set(got["cells"]) == {f"{t}|{n}" for t in ("enc12_elec", "enc12", "enc0")
                                 for n in ("std",)}
    assert got["cells"]["enc12|std"]["test"] == pytest.approx(1.0)
    assert got["cells"]["enc12_elec|std"]["test"] == pytest.approx(1.0)

def test_ws_cell_skips_taps_absent_from_the_cache() -> None:
    """A cache encoded without --elec-taps must not NaN the whole session. (Default norm set is
    std-only as of 2026-07-18; the point here is the absent tap is skipped, not the norm axis.)"""
    got = _ws_cell(_rec(), "onset", ("enc12_elec", "enc12"))
    assert set(got["cells"]) == {"enc12|std"}
    assert got["cells"]["enc12|std"]["test"] == pytest.approx(1.0)

def test_ws_cell_on_pure_noise_is_chance_not_one() -> None:
    """Guards the mirror failure: a readout that leaks labels scores ~1 on noise."""
    got = _ws_cell(_rec(signal=False, seed=7), "onset", ("enc12",))
    assert 0.15 < got["cells"]["enc12|std"]["test"] < 0.85

def test_cs_cell_transfers_over_the_parcel_intersection() -> None:
    anchor = _rec(parcels=[0, 1, 2, 3], seed=1)
    test = _rec(parcels=[2, 3, 4, 5], seed=2)
    got = _cs_cell(anchor, test, "onset", ("enc12",))
    # Intersection {2,3} — the signal lives in parcel 0, which is NOT shared, so the transfer
    # must NOT be perfect. The point is that it scored at all, over 2 aligned parcels.
    assert got["n_parcels"] == 2
    assert not np.isnan(got["cells"]["enc12|std"]["test"])

def test_cs_cell_aligns_parcel_columns_by_atlas_id_not_position() -> None:
    """The trap: same |P| on both sides, different atlas ids ⇒ positional alignment silently
    pairs unrelated regions. Put the signal in a shared parcel sitting at DIFFERENT positions."""
    anchor = _rec(parcels=[7, 1, 2, 3], seed=1)   # atlas 7 at position 0
    test = _rec(parcels=[1, 2, 3, 7], seed=1)     # atlas 7 at position 3
    # Move the signal into atlas-7's column on each side: anchor pos 0, test pos 3.
    for rec, pos in ((anchor, 0), (test, 3)):
        x = rec["feats"]["enc12"]["raw"].to(torch.float32).numpy()
        x[:, :, 0] = 0.0
        x[:, pos, 0] = rec["labels"]["onset"] * 10.0
        rec["feats"]["enc12"]["raw"] = torch.from_numpy(x).to(torch.float16)
    got = _cs_cell(anchor, test, "onset", ("enc12",))
    assert got["n_parcels"] == 4
    # Aligned by atlas id, the shared signal transfers; positionally it would be noise.
    assert got["cells"]["enc12|std"]["test"] == pytest.approx(1.0)

def test_cs_defaults_to_std_only_column() -> None:
    """Cross-subject evaluation uses training-only standardization."""
    import evals.neuroprobe.readout as mod
    anchor, test = _rec(seed=1), _rec(seed=2)
    got = mod._cs_cell(anchor, test, "onset", ("enc12",))
    assert set(got["cells"]) == {"enc12|std"}

def test_cs_cell_no_shared_parcels_is_empty_not_a_number() -> None:
    anchor = _rec(parcels=[0, 1], n_parcels=2, seed=1)
    test = _rec(parcels=[8, 9], n_parcels=2, seed=2)
    assert _cs_cell(anchor, test, "onset", ("enc12",))["cells"] == {}

def test_merge_keeps_every_grid_entry_over_every_cell() -> None:
    """The merge must not collapse the grid either: each "tap|norm" carries its own per-cell
    dict, and the cohort mean is taken WITHIN an entry — never across taps or norms."""
    res = _blank(["tg"])
    for name, v in (("S1T1", 0.6), ("S3T0", 0.8)):
        _absorb(res, {"kind": "cs", "name": name, "cells": {
            "tg|onset": {"cells": {"enc12|std": {"test": v, "lam_pinned": False},
                                   "enc12|raw": {"test": v - 0.1, "lam_pinned": False},
                                   "enc0|std": {"test": v - 0.2, "lam_pinned": False}},
                         "n_parcels": 5}}})
    _finalize(res)
    c = res["tg|onset"]
    assert set(c["cs_mean"]) == {"enc12|std", "enc12|raw", "enc0|std"}
    assert c["cs_mean"]["enc12|std"] == pytest.approx(0.7)      # (0.6+0.8)/2, within the entry
    assert c["cs_mean"]["enc0|std"] == pytest.approx(0.5)
    assert set(c["cs"]["enc12|std"]) == {"S1T1", "S3T0"}        # per-cell detail survives

def test_merge_carries_the_lambda_pin_flag_to_the_report() -> None:
    res = _blank(["tg"])
    _absorb(res, {"kind": "ws", "name": "S2T0", "cells": {
        "tg|onset": {"cells": {"enc12|std": {"test": 0.9, "lam_pinned": True},
                               "enc12|raw": {"test": 0.9, "lam_pinned": False}}}}})
    assert res["tg|onset"]["pinned"] == {"ws:enc12|std": ["S2T0"]}

def test_map_tasks_forked_gives_identical_results_to_serial() -> None:
    """Ben 2026-07-17: fork-over-tasks is a THROUGHPUT change and must be a NUMERICAL no-op.
    The failure it guards against is silent — a forked worker that mis-shares state returns
    plausible numbers, not an error. So: same input, both paths, byte-identical output."""
    from evals.neuroprobe.readout import _map_tasks

    rec = _rec(seed=3)
    fn = lambda task, tp: _ws_cell(rec, task, tp)  # noqa: E731 — verify fork handles lambdas
    serial = _map_tasks(fn, ("enc12",), workers=1)
    forked = _map_tasks(fn, ("enc12",), workers=2)
    assert set(serial) == set(forked) == set(BOARD_TASKS)
    for t in BOARD_TASKS:
        for gk in serial[t]["cells"]:
            a, b = serial[t]["cells"][gk]["test"], forked[t]["cells"][gk]["test"]
            assert a == b or (np.isnan(a) and np.isnan(b)), f"{t}|{gk}: {a} != {b}"

def test_load_mmap_flag_reaches_torch_load() -> None:
    """Pin that the flag is threaded through to torch.load rather than silently defaulting."""
    import evals.neuroprobe.readout as B

    seen = {}
    orig = B.torch.load
    B.torch.load = lambda path, **kw: seen.update(kw) or {"ok": True}
    try:
        B._load("/cache", (2, 4), "tg", mmap=True)
        assert seen["mmap"] is True
        B._load("/cache", (2, 4), "tg")
        assert seen["mmap"] is False          # eager is the safe default; modes opt in
    finally:
        B.torch.load = orig

def test_mmap_is_never_the_default() -> None:
    """mmap DEFERS the read into the gathers at ~1/4 sequential bandwidth (cold 24 MB/s vs eager
    ~86 MB/s), so it is only ever worth it to save MEMORY.

    Memory is not scarce here, and believing it was is what made this default wrong: the shards
    that stalled were starved by NUMA node-0 pinning, not by size, and `numactl --interleave=all`
    is the fix. Under mmap a shard measured 15 MB/s with 3.7M major faults and produced nothing in
    43 minutes. Eager for both modes; the --mmap/--no-mmap knob stays for A/B only."""
    from evals.neuroprobe.readout import MMAP_DEFAULT

    assert MMAP_DEFAULT == {"ws": False, "cs": False, "csession": False}

@pytest.mark.parametrize("blk", [2, 3, 7, 64, 1024, 10_000])
def test_standardize_column_blocking_is_bit_identical_at_every_block_size(blk) -> None:
    """The reduction is per COLUMN, so the block size is a memory-layout choice and NOTHING else.

    Pinned bitwise, not approximately: this function feeds every reported AUROC, and the whole
    argument for blocking is that it cannot change an answer. Spans blk larger than d (one block
    = the old whole-array path) down to the blk=2 floor.

    blk=1 is EXCLUDED and that is why _STD_BLOCK has a floor: a width-1 column slice makes
    np.mean/np.std reduce a contiguous 1-D vector, which numpy sums PAIRWISE, instead of
    accumulating a SIMD row-vector across columns. Measured drift 1.2e-7 on mu — harmless in size,
    but it is a different summation order, so the bitwise guarantee would become a claim about
    numpy's dispatch rather than about arithmetic. Every width >= 2 sums the same values in the
    same order (verified 2..4096 on d=4096, and 512..8192 on d=120000 on Delta).
    """
    import evals.neuroprobe.readout as B

    assert B._STD_BLOCK >= 2
    rng = np.random.default_rng(4)
    tr = rng.normal(size=(29, 53)).astype(np.float32)
    va = rng.normal(size=(11, 53)).astype(np.float32)
    te = rng.normal(size=(13, 53)).astype(np.float32)
    tr[:, 5] = 2.5                                     # a constant column: sd == 0 -> 1.0 branch
    ref = B._standardize(tr.copy(), [va.copy(), te.copy()])
    got_tr, (got_va, got_te) = B._standardize_inplace(tr.copy(), [va.copy(), te.copy()], blk=blk)
    assert np.array_equal(got_tr, ref[0])
    assert np.array_equal(got_va, ref[1][0])
    assert np.array_equal(got_te, ref[1][1])

def test_primal_matches_dual_on_the_same_fit() -> None:
    """d < n takes the primal; zero-padding d past n forces the dual on the IDENTICAL problem.

    Zero columns contribute nothing to ZZᵀ, to ZᵀZ's trace, or to any eval kernel, so the padded
    fit is the same ridge — only the factorization differs. Agreement to <1e-4 AUROC is the whole
    licence for the branch.
    """
    rng = np.random.default_rng(5)
    n, d = 90, 20
    y = np.array([float(i % 2) for i in range(n)])
    z = rng.normal(size=(n, d)).astype(np.float32)
    z[:, 0] += y * 1.5
    ev = {"val": (z[50:70], y[50:70]), "test": (z[70:], y[70:])}
    primal = _lam_grid(z[:50], y[:50], ev)
    pad = lambda a: np.hstack([a, np.zeros((a.shape[0], 60), np.float32)])   # noqa: E731
    dual = _lam_grid(pad(z[:50]), y[:50],
                     {"val": (pad(z[50:70]), y[50:70]), "test": (pad(z[70:]), y[70:])})
    for split in ("val", "test"):
        for m in LAM_MULTS:
            assert primal[split][m] == pytest.approx(dual[split][m], abs=1e-4)

def test_primal_branch_is_taken_only_when_d_is_below_n() -> None:
    """The dual builds eval kernels; the primal has none. Phase keys are the observable."""
    import evals.neuroprobe.readout as B

    rng = np.random.default_rng(6)
    y = np.array([float(i % 2) for i in range(40)])
    z = rng.normal(size=(40, 6)).astype(np.float32)
    ev = {"val": (z[20:30], y[20:30]), "test": (z[30:], y[30:])}

    B._PH.clear()
    _lam_grid(z[:20], y[:20], ev)                       # d=6 < n=20 -> primal
    assert "eval_kernels" not in B._PH

    wide = np.hstack([z, rng.normal(size=(40, 60))]).astype(np.float32)
    B._PH.clear()
    _lam_grid(wide[:20], y[:20],
              {"val": (wide[20:30], y[20:30]), "test": (wide[30:], y[30:])})   # d=66 > n=20
    assert "eval_kernels" in B._PH
    B._PH.clear()

def test_lam_grid_is_the_published_grid_and_any_widening_must_keep_its_spacing() -> None:
    """The grid that produced every published board number, plus the contract for changing it.

    LO-pinning is asymmetric across taps (ws enc12 39 vs enc0 16), which is why widening keeps
    coming up. Two invariants make a widening safe when it happens, so pin them now:

      spacing — extend at the SAME 1/3-decade step and the old points survive exactly, so a cell
                that did not pin re-selects the same λ and reproduces its old test AUROC. Drift
                the spacing and every board number moves silently.
      HI end  — must stay at 1e4. AUROC is constant in λ past it (next test), so widening up
                cannot change a number and only buys a 34-shard board re-run.
    """
    from evals.neuroprobe.readout import LAM_MULTS

    steps = np.diff(np.log10(np.asarray(LAM_MULTS)))
    print(f"[check] grid: {len(LAM_MULTS)} points, "
          f"{np.log10(min(LAM_MULTS)):.0f}..{np.log10(max(LAM_MULTS)):.0f} decades, "
          f"step={steps.mean():.4f} dec (uniform={np.ptp(steps) < 1e-9}) OK")
    assert np.ptp(steps) < 1e-9, "spacing is not uniform"
    assert abs(steps.mean() - 1.0 / 3.0) < 1e-9, \
        "spacing left 1/3 decade — a widening at this step no longer contains the old grid"
    assert max(LAM_MULTS) == pytest.approx(1e4), "the HI end must not move — it provably saturates"

def test_widening_only_extends_downward_because_the_hi_end_saturates(monkeypatch) -> None:
    """Why the fix is one-sided: AUROC is constant in λ past the HI end, so widening up is a no-op.

    _select_lam's docstring asserts AUROC(1e4) == AUROC(1e16). Pin it so nobody 'symmetrically'
    widens the top and pays for a 34-shard board that cannot change a number.
    """
    import evals.neuroprobe.readout as mod

    rng = np.random.default_rng(0)
    y = np.array([float(i % 2) for i in range(40)])
    z = rng.normal(size=(40, 6))
    z[:, 0] = y * 5.0
    monkeypatch.setattr(mod, "LAM_MULTS", (1e4, 1e8, 1e16))
    out = mod._lam_grid(z[:20], y[:20], {"test": (z[20:], y[20:])})
    vals = [out["test"][m] for m in (1e4, 1e8, 1e16)]
    print(f"[check] test AUROC at lam_mult 1e4/1e8/1e16 = {vals} "
          f"(want all equal — the HI end saturates, widening UP cannot move a number) OK")
    assert vals[0] == vals[1] == vals[2], vals

def test_val_ties_make_the_selected_lambda_depend_on_the_GRID_not_the_DATA() -> None:
    """🚨 The confound in the LO-pin audit: a 'lo pin' is not necessarily a truncated optimum.

    _select_lam takes argmax with a STRICT `>` while iterating LAM_MULTS in ascending order, so on
    a val TIE it keeps the SMALLEST λ. The val half is coarse (AUROC over a few dozen rows), so
    ties are the common case, not the exception — on this fixture most of the grid ties at val=1.0
    and the tied cells' TEST AUROCs differ materially. Two consequences:

      1. A cell flagged ``lam_pinned`` may simply be a TIE resolved to the grid floor, NOT an
         optimum that sits below the grid. Only the second is truncation.
      2. Widening the grid downward moves the tie-break, so it changes numbers in cells where the
         data expressed no preference at all — in either direction.

    So the pin counts alone cannot license "our depth gains are lower bounds". This test pins the
    mechanism so the inference is not made from the flag again.
    """
    from evals.neuroprobe.readout import LAM_MULTS, _select_lam

    tied = {m: 1.0 for m in LAM_MULTS}                     # val: dead flat, no preference
    test = {m: 0.90 + 0.05 * (i / len(LAM_MULTS)) for i, m in enumerate(LAM_MULTS)}
    got = _select_lam({"val": tied, "test": test})
    print(f"[check] val tied across all {len(LAM_MULTS)} λ -> selected λ={got['lam_mult']:.3e} "
          f"(== grid min {min(LAM_MULTS):.3e}), flagged lam_pinned={got['lam_pinned']} "
          f"while the data preferred NOTHING OK")
    assert got["lam_mult"] == min(LAM_MULTS), "tie-break is not smallest-λ; audit logic changed"
    assert got["lam_pinned"] is True, "a pure tie is reported as a LO pin — that is the confound"

def test_widening_the_grid_can_LOWER_a_pinned_cell_not_raise_it(monkeypatch) -> None:
    """🚨 THE COUNTEREXAMPLE. Resolving a LO pin does NOT mean the AUROC goes up.

    The tempting inference from the pin audit (ws enc12 pins lo 39× vs enc0 16×) is that pinned
    cells are truncated downward, so widening the grid raises them and the reported depth gains
    are lower bounds. On this repo's OWN ws fixture that is false, and it fails in the unlucky
    direction. Widening logspace(-4,4,25) → logspace(-8,4,37) on the enc12|std cell:

        grid       fold λ            lam_pinned   test (mean of the 2 folds)
        old        1e-4,   1e-4      True          1.0000000
        new        1e-6, 4.64e-7     False         0.9765625   ← pin RESOLVED, number FELL

    Why: val is dead flat at 1.0 across 14/25 (fold 0) and 25/25 (fold 1) of the old grid, while
    test over that same tied plateau spans 0.906..1.000. Val cannot discriminate, so _select_lam's
    smallest-λ tie-break lets the GRID FLOOR pick the operating point. The old floor happened to
    land on a good point; the new floor lands on a worse one. Nothing was truncated — the cell was
    under-determined, and both floors are equally justified by the val half.

    So a LO pin is not evidence of truncation, and widening is not a free improvement. Keep this
    test as the standing refutation before anyone spends a 34-shard board re-run on that premise.
    """
    import evals.neuroprobe.readout as mod

    out = {}
    for name, grid in (("old", tuple(np.logspace(-4.0, 4.0, 25))),
                       ("new", tuple(np.logspace(-8.0, 4.0, 37)))):
        monkeypatch.setattr(mod, "LAM_MULTS", grid)
        out[name] = mod._ws_cell(_rec(), "onset", ("enc12",))["cells"]["enc12|std"]
        print(f"[check] {name} grid floor={min(grid):.0e} -> test={out[name]['test']:.7f} "
              f"lam/fold={[f'{m:.2e}' for m in out[name]['lam_mult']]} "
              f"lam_pinned={out[name]['lam_pinned']}")

    assert out["old"]["lam_pinned"] is True, "fixture no longer pins lo on the published grid"
    assert out["new"]["lam_pinned"] is False, "widening should have resolved the pin"
    assert out["new"]["test"] < out["old"]["test"], \
        "widening no longer LOWERS this cell — re-derive the claim before trusting a re-run"
    print(f"[check] pin resolved True->False while test FELL "
          f"{out['old']['test']:.7f} -> {out['new']['test']:.7f} "
          f"=> a LO pin is NOT evidence of downward truncation OK")

def test_n_tied_counts_only_the_val_maximisers() -> None:
    d = {"val": {1.0: 0.80, 10.0: 0.80, 100.0: 0.70}, "test": {1.0: 0.1, 10.0: 0.2, 100.0: 0.3}}
    assert _select_lam(d)["n_tied"] == 2

def test_ws_cell_retains_actual_test_folds_and_original_ids():
    rec = _rec(signal=False, seed=7)
    got = _ws_cell(rec, "onset", ("enc12",))["cells"]["enc12|std"]
    scores = []
    for fold_id in (0, 1):
        one_fold = {**rec, "ws_split": {"onset": {fold_id: rec["ws_split"]["onset"][fold_id]}}}
        leaf = _ws_cell(one_fold, "onset", ("enc12",))["cells"]["enc12|std"]
        assert leaf["folds"] == [{"fold_idx": fold_id, "test_roc_auc": leaf["test"]}]
        scores.append(leaf["test"])
    assert scores[0] != scores[1], "fixture must distinguish true folds from duplicated means"
    assert got["folds"] == [{"fold_idx": i, "test_roc_auc": score} for i, score in enumerate(scores)]
    assert got["test"] == pytest.approx(np.mean(scores))

def test_ws_skipped_fold_does_not_renumber_remaining_fold():
    rec = _rec()
    rec["ws_split"]["onset"][0]["train"] = np.array([], dtype=int)
    leaf = _ws_cell(rec, "onset", ("enc12",))["cells"]["enc12|std"]
    assert leaf["folds"] == [{"fold_idx": 1, "test_roc_auc": leaf["test"]}]


def test_all_mode_uses_the_same_tap_defaults_as_shards(tmp_path, monkeypatch):
    import sys

    from evals.neuroprobe import readout as mod

    calls = []
    monkeypatch.setattr(mod, '_compute_all', lambda *args: calls.append(args) or {})
    monkeypatch.setattr(mod, '_report', lambda *args: None)
    monkeypatch.setattr(sys, 'argv', ['readout', '--cache-dir', 'unused', '--tags', 'fixture',
                        '--out', str(tmp_path / 'results.json')])
    mod.main()
    assert calls[0][2:] == (mod.WS_TAPS, mod.CS_TAPS)


def test_legacy_contact_sidecar_accepts_lists_in_feature_order(tmp_path, monkeypatch):
    from evals.neuroprobe import readout as mod

    rec = {'feats': {'enc12_elec': {'raw': torch.zeros(2, 3, 4)}}}
    torch.save(rec, tmp_path / 'enc_s1_t1_fixture.pt')
    monkeypatch.setattr(mod, '_ELEC_LABELS_SIDECAR', {'s1_t1': ['RB4', 'RB1', 'LA3']})
    got = mod._load(tmp_path, (1, 1), 'fixture')
    assert got['elec_labels'].tolist() == ['RB4', 'RB1', 'LA3']
    monkeypatch.setattr(mod, '_ELEC_LABELS_SIDECAR', {'s1_t1': ['RB4', 'RB4', 'LA3']})
    with pytest.raises(ValueError, match='unique one-dimensional'):
        mod._load(tmp_path, (1, 1), 'fixture')
