import pytest

from evals.neuroprobe.build_submission import (
    LITE_SESSIONS,
    TASKS,
    read_ws_curve,
    validate_folds,
    validate_grid,
)


def complete():
    sessions = [pair for pair in LITE_SESSIONS if pair[0] != 2]
    cells = [f"S{s}T{t}" for s, t in sessions]
    grid = {(task, f"btbank{s}_{t}"): 0.6 for task in TASKS for s, t in sessions}
    return grid, cells


def test_complete_grid():
    grid, cells = complete()
    validate_grid("Cross-Subject", "canon", grid, cells)


def test_empty_session_cannot_disappear_from_completeness_check():
    grid, cells = complete()
    grid = {k: v for k, v in grid.items() if k[1] != "btbank1_1"}
    with pytest.raises(SystemExit, match="15 missing"):
        validate_grid("Cross-Subject", "canon", grid, cells)


def test_wrong_session_with_correct_cell_count_is_rejected():
    grid, cells = complete()
    cells[0] = "S99T1"
    with pytest.raises(SystemExit, match="evaluation sessions differ"):
        validate_grid("Cross-Subject", "canon", grid, cells)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_scores_are_rejected(value):
    grid, cells = complete()
    grid[next(iter(grid))] = value
    with pytest.raises(SystemExit, match="invalid AUROC"):
        validate_grid("Cross-Subject", "canon", grid, cells)


def test_two_distinct_fold_scores_survive_validation():
    key = ("onset", "btbank1_1")
    folds = {key: [{"fold_idx": 1, "test_roc_auc": 0.8},
                   {"fold_idx": 0, "test_roc_auc": 0.4}]}
    validate_folds({key: 0.6}, folds)
    assert folds[key] == [{"fold_idx": 0, "test_roc_auc": 0.4},
                          {"fold_idx": 1, "test_roc_auc": 0.8}]


@pytest.mark.parametrize("folds, message", [
    ([], "fold indices"),
    ([{"fold_idx": 0, "test_roc_auc": 0.6}], "fold indices"),
    ([{"fold_idx": 0, "test_roc_auc": 0.4}, {"fold_idx": 0, "test_roc_auc": 0.8}], "fold indices"),
    ([{"fold_idx": False, "test_roc_auc": 0.4}, {"fold_idx": 1, "test_roc_auc": 0.8}], "fold indices"),
    ([{"fold_idx": 0, "test_roc_auc": float("nan")}, {"fold_idx": 1, "test_roc_auc": 0.8}], "invalid WS fold"),
    ([{"fold_idx": 0, "test_roc_auc": 1.2}, {"fold_idx": 1, "test_roc_auc": 0.0}], "invalid WS fold"),
    ([{"fold_idx": 0, "test_roc_auc": 0.5}, {"fold_idx": 1, "test_roc_auc": 0.8}], "mean differs"),
])
def test_invalid_fold_records_fail(folds, message):
    key = ("onset", "btbank1_1")
    with pytest.raises(SystemExit, match=message):
        validate_folds({key: 0.6}, {key: folds})


def curve_blob():
    return {"regime": "ws", "tags": ["canon"], "contiguous": [False], "points": [
        {"n_is_full": True, "seed": 0, "tap": "enc12_elec", "col": "trainonly",
         "cell": "S1T1", "task": "onset", "fold": fold, "test": score}
        for fold, score in [(0, 0.4), (1, 0.8)]
    ]}


def test_curve_recovery_excludes_subsets_and_other_readouts(tmp_path):
    import json
    blob = curve_blob()
    for field, value in [("n_is_full", False), ("seed", 1), ("tap", "enc0_elec"), ("col", "raw")]:
        blob["points"].append({**blob["points"][0], field: value, "test": 0.99})
    path = tmp_path / "curve.json"
    path.write_text(json.dumps(blob))
    got = read_ws_curve(path, "canon")
    validate_folds({("onset", "btbank1_1"): 0.6}, got)
    assert [f["test_roc_auc"] for f in got[("onset", "btbank1_1")]] == [0.4, 0.8]


@pytest.mark.parametrize("field,value", [("regime", "cs"), ("tags", ["other"]), ("contiguous", [True])])
def test_curve_recovery_rejects_different_experiment(tmp_path, field, value):
    import json
    blob = curve_blob()
    blob[field] = value
    path = tmp_path / "curve.json"
    path.write_text(json.dumps(blob))
    with pytest.raises(SystemExit, match="must match"):
        read_ws_curve(path, "canon")


def test_duplicate_curve_fold_is_rejected(tmp_path):
    import json
    blob = curve_blob()
    blob["points"].append(blob["points"][0])
    path = tmp_path / "curve.json"
    path.write_text(json.dumps(blob))
    with pytest.raises(SystemExit, match="duplicate"):
        read_ws_curve(path, "canon")


@pytest.mark.parametrize("regime,split,tap", [
    ("ws", "Within-Session", "enc12_elec|std"),
    ("csession", "Cross-Session", "enc12_elec|std"),
    ("cs", "Cross-Subject", "enc12|std"),
])
def test_export_round_trip(tmp_path, monkeypatch, regime, split, tap):
    import json
    import sys

    from evals.neuroprobe.build_submission import TIME_BIN, main
    shards = tmp_path / "shards"
    shards.mkdir()
    out = tmp_path / "out"
    for s, t in LITE_SESSIONS:
        if regime == "cs" and s == 2:
            continue
        leaf = {"test": 0.6}
        if regime == "ws":
            leaf["folds"] = [{"fold_idx": 0, "test_roc_auc": 0.4},
                             {"fold_idx": 1, "test_roc_auc": 0.8}]
        blob = {"cells": {f"canon|{task}": {"cells": {tap: leaf}} for task in TASKS}}
        (shards / f"{regime}_S{s}T{t}.json").write_text(json.dumps(blob))
    monkeypatch.setattr(sys, "argv", ["export", "--shards", str(shards), "--split", split, "--out", str(out)])
    main()
    for path in (out / split).glob("*.json"):
        results = json.loads(path.read_text())["evaluation_results"]
        for session in results.values():
            folds = session["population"][TIME_BIN]["folds"]
            assert [f["test_roc_auc"] for f in folds] == ([0.4, 0.8] if regime == "ws" else [0.6])
    if regime == "ws":
        # Legacy mean-only shards must fail before creating any output.
        for path in shards.glob("*.json"):
            blob = json.loads(path.read_text())
            for task in blob["cells"].values():
                del task["cells"][tap]["folds"]
            path.write_text(json.dumps(blob))
        monkeypatch.setattr(sys, "argv", ["export", "--shards", str(shards), "--split", split,
                                          "--out", str(tmp_path / "rejected")])
        with pytest.raises(SystemExit, match="incomplete WS fold grid"):
            main()
        assert not (tmp_path / "rejected").exists()
