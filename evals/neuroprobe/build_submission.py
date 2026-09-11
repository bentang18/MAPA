"""Export complete Neuroprobe Lite readout shards to leaderboard JSON.

Within-session exports preserve both test folds. Legacy shards containing only a
fold mean require --ws-curve from the matching full-data sample-efficiency run.
Cross-session and cross-subject exports each contain one test fold per session.

The published evaluation uses the one-second window after word onset. This script
writes result payloads and metadata.json. Authors must separately supply the
publication citation and signed attestations required by Neuroprobe.

Usage:
  python -m evals.neuroprobe.build_submission --shards <shard dir> \
      --split Cross-Subject --out <submission dir>
"""
import argparse
import json
import math
import os
import re
import time

# The 15 keys are the leaderboard's own task names (NEUROPROBE_TASKS_MAPPING). A task missing
# here is a submission that silently competes on fewer boards, so the set is asserted, not zipped.
TASKS = (
    "onset", "speech", "volume", "delta_volume", "pitch",
    "word_index", "word_gap", "gpt2_surprisal", "word_head_pos", "word_part_speech",
    "word_length", "global_flow", "local_flow", "frame_brightness", "face_num",
)
# Published tap per regime. cs scores region means because electrode identity does not survive a
# subject change; ws/csession score per-electrode. Getting this wrong silently submits the floor.
SPLIT_SPEC = {
    "Cross-Subject": dict(regime="cs", tap="enc12|std", cells=10),
    "Within-Session": dict(regime="ws", tap="enc12_elec|std", cells=12),
    "Cross-Session": dict(regime="csession", tap="enc12_elec|std", cells=12),
}
SHARD_RE = re.compile(r"^(ws|csession|cs)_(S(\d+)T(\d+))\.json$")
TIME_BIN = "one_second_after_onset"


def read_shards(shard_dir, regime, tap, *, fold_scores=None):
    """Read one checkpoint tag, task/session scores, and the session names present."""
    out, tags, cells = {}, set(), set()
    for fn in sorted(os.listdir(shard_dir)):
        m = SHARD_RE.match(fn)
        if not m or m.group(1) != regime:
            continue
        cells.add(m.group(2))
        session_id = f"btbank{m.group(3)}_{m.group(4)}"
        with open(os.path.join(shard_dir, fn)) as f:
            blob = json.load(f)
        for tagtask, blk in blob.get("cells", {}).items():
            tag, task = tagtask.split("|", 1)
            tags.add(tag)
            leaf = blk.get("cells", {}).get(tap)
            if leaf is not None and leaf.get("test") is not None:
                out[(task, session_id)] = float(leaf["test"])
                if fold_scores is not None and "folds" in leaf:
                    fold_scores[(task, session_id)] = leaf["folds"]
    if len(tags) > 1:
        raise SystemExit(f"[FATAL] ARM MIXING: {shard_dir} holds tags {sorted(tags)}")
    return (tags.pop() if tags else "?"), out, sorted(cells)


# The Neuroprobe Lite evaluation sessions; subject 2 is the cross-subject anchor.
LITE_SESSIONS = ((1, 1), (1, 2), (2, 0), (2, 4), (3, 0), (3, 1),
                 (4, 0), (4, 1), (7, 0), (7, 1), (10, 0), (10, 1))


def validate_grid(split, tag, grid, cells):
    """Require every published task/session pair before writing any output."""
    sessions = [pair for pair in LITE_SESSIONS
                if split != "Cross-Subject" or pair[0] != 2]
    expected_cells = {f"S{s}T{t}" for s, t in sessions}
    if set(cells) != expected_cells:
        raise SystemExit(f"[FATAL] {split}: evaluation sessions differ from Neuroprobe Lite: "
                         f"missing={sorted(expected_cells - set(cells))}, "
                         f"extra={sorted(set(cells) - expected_cells)}")
    expected = {(task, f"btbank{s}_{t}") for task in TASKS for s, t in sessions}
    missing, extra = expected - grid.keys(), grid.keys() - expected
    if missing or extra:
        raise SystemExit(f"[FATAL] incomplete task/session grid: "
                         f"{len(missing)} missing, {len(extra)} extra")
    if tag == "?":
        raise SystemExit("[FATAL] missing checkpoint tag")
    for key, value in grid.items():
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise SystemExit(f"[FATAL] invalid AUROC at {key}: {value}")


def read_ws_curve(path, tag):
    """Recover actual full-data fold scores from a matching sample-curve result."""
    with open(path) as f:
        blob = json.load(f)
    if (blob.get("regime") != "ws" or blob.get("tags") != [tag]
            or blob.get("contiguous") != [False]):
        raise SystemExit("[FATAL] WS curve must match the shard tag and random WS subsampling protocol")
    out = {}
    seen = set()
    for point in blob.get("points", []):
        if not (point.get("n_is_full") is True and point.get("seed") == 0
                and point.get("tap") == "enc12_elec" and point.get("col") == "trainonly"):
            continue
        match = re.fullmatch(r"S(\d+)T(\d+)", point.get("cell", ""))
        if match is None:
            raise SystemExit("[FATAL] invalid WS curve session")
        key = (point["task"], f"btbank{match[1]}_{match[2]}")
        identity = (*key, point["fold"])
        if identity in seen:
            raise SystemExit(f"[FATAL] duplicate WS curve fold: {identity}")
        seen.add(identity)
        out.setdefault(key, []).append({"fold_idx": point["fold"],
                                        "test_roc_auc": point["test"]})
    return out


def validate_folds(grid, fold_scores):
    """Two real WS folds must reproduce each shard's existing mean."""
    if fold_scores.keys() != grid.keys():
        raise SystemExit("[FATAL] incomplete WS fold grid. Legacy mean-only shards need "
                         "--ws-curve from the matching full-data sample-efficiency run; "
                         "a fold mean cannot reconstruct the two folds.")
    for key, folds in fold_scores.items():
        if (not isinstance(folds, list) or len(folds) != 2
                or any(not isinstance(fold, dict) for fold in folds)
                or any(type(fold.get("fold_idx")) is not int for fold in folds)
                or {fold["fold_idx"] for fold in folds} != {0, 1}):
            raise SystemExit(f"[FATAL] WS requires fold indices 0 and 1 at {key}")
        values = [fold.get("test_roc_auc") for fold in folds]
        if any(type(value) not in (float, int) or not math.isfinite(value)
               or not 0 <= value <= 1 for value in values):
            raise SystemExit(f"[FATAL] invalid WS fold AUROC at {key}")
        if not math.isclose(sum(values) / 2, grid[key], rel_tol=0, abs_tol=1e-12):
            raise SystemExit(f"[FATAL] WS fold mean differs from shard at {key}")
        fold_scores[key] = sorted(folds, key=lambda fold: fold["fold_idx"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--split", default="Cross-Subject", choices=sorted(SPLIT_SPEC))
    ap.add_argument("--out", required=True, help="submission directory to create")
    ap.add_argument("--ws-curve", help="matching merged WS sample curve for legacy mean-only shards")
    ap.add_argument("--model-name", default="MAPA frozen encoder + ridge")
    ap.add_argument("--description", default="")
    ap.add_argument("--author", default="")
    ap.add_argument("--organization", default="")
    ap.add_argument("--organization-url", default="")
    a = ap.parse_args()

    spec = SPLIT_SPEC[a.split]
    fold_scores = {}
    tag, grid, cells = read_shards(a.shards, spec["regime"], spec["tap"],
                                  fold_scores=fold_scores)

    validate_grid(a.split, tag, grid, cells)
    if a.ws_curve and a.split != "Within-Session":
        raise SystemExit("[FATAL] --ws-curve is only valid for Within-Session")
    if a.split == "Within-Session":
        if a.ws_curve:
            recovered = read_ws_curve(a.ws_curve, tag)
            validate_folds(grid, recovered)
            validate_folds({key: grid[key] for key in fold_scores}, fold_scores)
            for key, existing in fold_scores.items():
                if sorted(existing, key=lambda fold: fold["fold_idx"]) != recovered[key]:
                    raise SystemExit(f"[FATAL] stored and recovered WS folds disagree at {key}")
            fold_scores = recovered
        validate_folds(grid, fold_scores)
    else:
        fold_scores = {key: [{"test_roc_auc": score, "fold_idx": 0}]
                       for key, score in grid.items()}

    meta = dict(model_name=a.model_name, description=a.description, author=a.author,
                organization=a.organization, organization_url=a.organization_url,
                timestamp=int(time.time()))

    split_dir = os.path.join(a.out, a.split)
    os.makedirs(split_dir, exist_ok=True)
    for task in TASKS:
        results = {}
        for t, session_id in grid:
            if t != task:
                continue
            results[session_id] = {"population": {TIME_BIN: dict(
                time_bin_start=0.0, time_bin_end=1.0,
                folds=fold_scores[(t, session_id)])}}
        with open(os.path.join(split_dir, f"population_{task}.json"), "w") as f:
            json.dump(dict(meta, evaluation_results=results), f, indent=1)
    with open(os.path.join(a.out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)

    macro = sum(grid.values()) / len(grid)
    print(f"arm        {tag}")
    print(f"split      {a.split}  tap={spec['tap']}  {len(cells)} cells x {len(TASKS)} tasks "
          f"= {len(grid)} evals")
    print(f"macro      {macro:.4f}   (cross-check this against the published number)")
    print(f"wrote      {split_dir}/population_*.json  +  {a.out}/metadata.json")
    print("STILL REQUIRED, NOT GENERATED HERE: PUBLICATION.bib (must cite a real publication) "
          "and ATTESTATION.txt (two SIGNED statements).")


if __name__ == "__main__":
    main()
