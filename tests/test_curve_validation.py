"""Incomplete sweeps must not become published curves or bootstrap estimates."""
import json
import sys

import pytest

from evals.neuroprobe import sample_curve as C


def _complete(regime="cs"):
    sessions = C.CS_TEST_CELLS if regime == "cs" else C.LITE_SESSIONS
    folds = (0, 1) if regime == "ws" else (0,)
    points, anchors = [], []
    for s, t in sessions:
        for task in C.BOARD_TASKS:
            for tap in C.CURVE_TAPS[regime]:
                score = .6 if tap.startswith("enc0") else .7
                for fold in folds:
                    for n, is_full in ((16, False), (32, True)):
                        for seed in range(1 if is_full else C.SEEDS_FOR_N[n]):
                            points.append(dict(cell=f"S{s}T{t}", task=task, fold=fold, tap=tap,
                                               col="trainonly", n=n, n_is_full=is_full, seed=seed,
                                               n_bucket=C.FULL if is_full else n, test=score))
                anchors.append(dict(cell=f"S{s}T{t}", task=task, tap=tap, col="trainonly",
                                    published=score, curve=score, absdiff=0.))
    return dict(regime=regime, tags=["fixture"], contiguous=[False], points=points,
                anchor=anchors, census=[])


@pytest.mark.parametrize("regime", C.CURVE_TAPS)
def test_complete_canonical_curve_and_legacy_extra_column(regime):
    m = _complete(regime)
    C.validate_curve(m, regime)
    m["points"].append(dict(m["points"][0], col="both", test=float("nan")))
    m["anchor"].append(dict(m["anchor"][0], col="both", absdiff=float("nan")))
    C.validate_curve(m, regime)


@pytest.mark.parametrize("dimension", ("cell", "task", "fold", "tap"))
def test_missing_named_grid_dimension_is_rejected(dimension):
    m = _complete("ws")
    missing = m["points"][0][dimension]
    m["points"] = [p for p in m["points"] if p[dimension] != missing]
    with pytest.raises(ValueError, match="Incomplete full-data"):
        C.validate_curve(m, "ws")


@pytest.mark.parametrize("change", ("missing_seed", "missing_n", "duplicate", "wrong_seed",
                                   "wrong_bucket", "wrong_count", "nonfinite", "out_of_range"))
def test_invalid_subsample_grid_is_rejected(change):
    m = _complete()
    if change == "missing_seed":
        m["points"].pop(0)
    elif change == "missing_n":
        m["points"] = [p for p in m["points"] if p["n_is_full"]]
    elif change == "duplicate":
        m["points"].append(dict(m["points"][0]))
    else:
        key, value = {"wrong_seed": ("seed", 99), "wrong_bucket": ("n_bucket", 17),
                      "wrong_count": ("n", 17), "nonfinite": ("test", float("nan")),
                      "out_of_range": ("test", 1.1)}[change]
        m["points"][0][key] = value
    with pytest.raises(ValueError):
        C.validate_curve(m, "cs")


@pytest.mark.parametrize("change", ("missing", "duplicate", "nonfinite", "drift", "wrong_full"))
def test_invalid_full_data_anchors_are_rejected(change):
    m = _complete()
    if change == "missing":
        m["anchor"].pop()
    elif change == "duplicate":
        m["anchor"].append(dict(m["anchor"][0]))
    elif change == "nonfinite":
        m["anchor"][0]["absdiff"] = float("nan")
    elif change == "drift":
        m["anchor"][0]["published"] += .01
    else:
        next(p for p in m["points"] if p["n_is_full"])["test"] += .01
    with pytest.raises(ValueError):
        C.validate_curve(m, "cs")


def test_reporting_entry_points_reject_partial_curve(tmp_path, monkeypatch):
    from evals.figures.fig_r31_label_efficiency_log10 import load
    from evals.neuroprobe.label_saving_ci import run

    m = _complete()
    m["points"].pop(0)  # All named sessions/tasks remain; product-of-counts checks miss this.
    src = tmp_path / "partial.json"
    src.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Incomplete curve point"):
        run(src, "cs", 10, 0, False, True, True)
    with pytest.raises(ValueError, match="Incomplete curve point"):
        load(src, "cs", 150)
    monkeypatch.setattr(C, "_merge", lambda *args: m)
    out = tmp_path / "merged.json"
    monkeypatch.setattr(sys, "argv", ["sample_curve", "--mode", "merge", "--regime", "cs",
                                     "--shard-dir", str(tmp_path), "--out", str(out)])
    with pytest.raises(ValueError, match="Incomplete curve point"):
        C.main()
    assert not out.exists()


@pytest.mark.parametrize("legacy_extra", (False, True))
def test_complete_curve_still_reports_same_estimate(tmp_path, legacy_extra):
    from evals.figures.fig_r31_label_efficiency_log10 import load
    from evals.neuroprobe.label_saving_ci import run

    src = tmp_path / "complete.json"
    m = _complete()
    if legacy_extra:
        m["points"].append(dict(m["points"][0], col="both", test=float("nan")))
        m["anchor"].append(dict(m["anchor"][0], col="both", absdiff=1.))
    src.write_text(json.dumps(m))
    result = run(src, "cs", 10, 0, False, True, True)
    assert result["point"] == 2.
    assert result["ci"] == [2., 2.]
    assert len(result["subjects"]) == 5
    assert load(src, "cs", 150)["units"] == 150
