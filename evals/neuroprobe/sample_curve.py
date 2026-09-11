"""Label-efficiency curves with stratified training subsets and fixed validation/test sets.

All taps share each draw. Full-data endpoints must reproduce the published
readout exactly. Shards retain sample counts and checkpoint/split provenance.
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

try:
    from evals.neuroprobe import readout as R
except ImportError:                                                             # bare-script run
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import readout as R  # type: ignore[no-redef]

BOARD_TASKS = R.BOARD_TASKS

ELEC_TAPS = R.ELEC_TAPS

LITE_SESSIONS = R.LITE_SESSIONS

CS_TEST_CELLS = R.CS_TEST_CELLS

CS_TRAIN_ANCHOR = R.CS_TRAIN_ANCHOR

_feat, _finite, _have = R._feat, R._finite, R._have

_lam_grid, _load, _select_lam = R._lam_grid, R._load, R._select_lam

_standardize_inplace, _ws_cell = R._standardize_inplace, R._ws_cell

_cs_cell, _region_cols = R._cs_cell, R._region_cols

_csession_cell, _sibling = R._csession_cell, R._sibling

CSESSION_CELLS = R.CSESSION_CELLS

CURVE_TAPS = {"ws": ("enc0_elec", "enc12_elec"), "cs": ("enc0", "enc12"),
              "csession": ("enc0_elec", "enc12_elec")}

SHARD_PREFIX = {"ws": "wscurve", "cs": "cscurve", "csession": "csessioncurve"}

TRANSFER_REGIMES = ("cs", "csession")

N_GRID = (16, 32, 64, 128, 256, 512, 1024)

N_GRID_CS = N_GRID + (2048,)

SEEDS_FOR_N = {16: 5, 32: 5, 64: 5, 128: 5, 256: 3, 512: 3, 1024: 3, 2048: 3}

FULL = "full"           # the anchor point: no subsample, 1 pass, must equal the published cell

COLUMNS = ("trainonly",)

def _x_order(k):
    """x-axis sort key. The grid mixes ints with the `full` sentinel, so a bare sorted() would
    raise TypeError comparing str to int — always order through this."""
    return (1, 0) if k == FULL else (0, int(k))

def _digest(idx) -> str:
    """Stable across processes and runs -- unlike hash(), which is PYTHONHASHSEED-salted."""
    return hashlib.blake2b(np.asarray(idx, dtype=np.int64).tobytes(), digest_size=8).hexdigest()

def _rng(session, task, fold, n, seed):
    """Draw seed derived from the CELL, not from a global counter, so any point is reproducible in
    isolation and re-running one shard cannot shift another's draws."""
    key = f"{session[0]}:{session[1]}|{task}|{fold}|{n}|{seed}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big"))

def _strat_draw(y, rows, n, rng):
    """n rows from `rows`, stratified at the parent's class balance, >=1 per class.

    Stratification is not a nicety: an unstratified draw of 16 from an unbalanced task can come
    back single-class, and `auroc` returns NaN on a single-class eval half -- which would silently
    delete the hardest points of the curve rather than reporting them.

    Returns rows in ASCENDING ORDER so the gather is a monotone index and the N=full draw is
    element-for-element the parent array.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if n >= rows.size:
        return rows
    pos = rows[y[rows] > 0]
    neg = rows[y[rows] <= 0]
    if pos.size == 0 or neg.size == 0:
        return np.sort(rng.choice(rows, size=n, replace=False))
    # Round to the parent balance, then clamp so BOTH classes survive and neither is over-drawn.
    n_pos = int(round(n * pos.size / rows.size))
    n_pos = max(1, min(n_pos, pos.size, n - 1))
    n_neg = n - n_pos
    if n_neg > neg.size:                      # rare: parent is extremely skewed toward positives
        n_neg = neg.size
        n_pos = min(n - n_neg, pos.size)
    take = np.concatenate([rng.choice(pos, size=n_pos, replace=False),
                           rng.choice(neg, size=n_neg, replace=False)])
    return np.sort(take)

def _fit_point(rec, tap, tr_s, va, te, y):
    """Fit one training subset; select lambda using the full validation half."""
    z_tr = _feat(rec, tap, tr_s)
    z_va, z_te = _feat(rec, tap, va), _feat(rec, tap, te)
    a, (b, c) = _standardize_inplace(z_tr, [z_va, z_te])
    g = _lam_grid(a, y[tr_s], {"valfull": (b, y[va]), "test": (c, y[te])})
    return {"trainonly": _select_lam({"val": g["valfull"], "test": g["test"]})}

def _ws_curve_cell(rec, session, task, taps):
    """One (session, task) → curve points over N x seed x tap x column, plus the census."""
    y = np.asarray(rec["labels"][task], dtype=np.float64)
    pts, census = [], []
    for fold, sp in sorted(rec["ws_split"][task].items()):
        tr, va, te = (_finite(y, sp["train"]), _finite(y, sp["val"]), _finite(y, sp["test"]))
        if len(tr) < 2 or len(te) < 2:
            continue
        # `full` LAST so the log reads bottom-up as "and here is the anchor".
        for n in [g for g in N_GRID if g < len(tr)] + [None]:
            is_full = n is None
            nn = len(tr) if is_full else int(n)
            for seed in range(1 if is_full else SEEDS_FOR_N[int(n)]):
                rng = _rng(session, task, fold, nn, seed)
                tr_s = _strat_draw(y, tr, nn, rng)
                d_tr = _digest(tr_s)
                for tap in taps:
                    if not _have(rec, tap):
                        continue
                    # INVARIANT 2: every tap fits the SAME rows. Structural here (the draw is made
                    # once, outside this loop) but asserted anyway -- the whole paired analysis is
                    # void if it ever stops being true, and a structural guarantee that is never
                    # checked is how it stops being true.
                    assert _digest(tr_s) == d_tr, "draw drifted across taps"
                    res = _fit_point(rec, tap, tr_s, va, te, y)
                    for col in COLUMNS:
                        pts.append({"task": task, "fold": int(fold), "tap": tap, "col": col,
                                    "n": nn, "n_is_full": is_full, "seed": seed,
                                    "test": res[col]["test"],
                                    "lam_pinned": bool(res[col]["lam_pinned"])})
                yb = (y[tr_s] > 0)
                # `n_is_full` is NOT redundant with n: after the per-task label filter a train
                # half can land exactly on a grid value, and then "N=128" and "the anchor" are
                # indistinguishable by n alone.
                census.append({"task": task, "fold": int(fold), "n": nn, "seed": seed,
                               "n_is_full": is_full,
                               "n_train": len(tr_s), "n_val": len(va), "n_test": len(te),
                               "n_pos": int(yb.sum()), "n_neg": int((~yb).sum()),
                               "parent_bal": float((y[tr] > 0).mean()),
                               "draw_bal": float(yb.mean())})
    return pts, census

def _cs_fit_point(anchor_rec, test_rec, tap, tr_s, a_idx, va, te, t_idx, y_a, y_t):
    """One (tap, N, seed) cross-subject fit → the selected-λ result for the reported `std` norm.

    Mirrors the "std" branch of ``_run_norms`` (readout.py:1238-1245) followed by
    ``_grid_cells``' ``_select_lam``: standardize on the ANCHOR's train stats, one λ grid, select on
    the test cell's val half. `grams=` is omitted for the same reason as the WS path — letting
    ``_lam_grid`` choose its own branch is what keeps the N=full anchor bit-comparable to ``_cs_cell``.
    """
    z_tr = _feat(anchor_rec, tap, tr_s, a_idx)
    z_va, z_te = _feat(test_rec, tap, va, t_idx), _feat(test_rec, tap, te, t_idx)
    a, (b, c) = _standardize_inplace(z_tr, [z_va, z_te])
    g = _lam_grid(a, y_a[tr_s], {"val": (b, y_t[va]), "test": (c, y_t[te])})
    return _select_lam({"val": g["val"], "test": g["test"]})

def _transfer_cols(train_rec, test_rec, tap):
    """Aligned (train_cols, test_cols) for one tap across two records, or None if it cannot align.

    The alignment differs by TAP, not by regime: region taps align by ATLAS ID and per-electrode taps
    align by electrode IDENTITY (readout.py:1362-1375). CS only ever asks for region taps
    and CSession only for electrode taps, but reading the tap rather than the regime is what keeps
    this from being a place a wrong pairing could hide.
    """
    if R._is_elec(tap):
        e_a, e_t, _n = R._elec_cols(train_rec, test_rec)
        return None if e_a is None else (e_a, e_t)
    p_a, p_t, common = _region_cols(train_rec, test_rec)
    return None if p_a is None or common.size == 0 else (p_a, p_t)

def _transfer_curve_cell(train_rec, test_rec, task, taps, draw_key):
    """Training subsets from the donor (CS) or sibling (CSession).

    Target validation/test remain fixed. CS uses the same donor draw for every
    target; CSession keys draws to each sibling session. There is one fold."""
    y_a = np.asarray(train_rec["labels"][task], dtype=np.float64)
    y_t = np.asarray(test_rec["labels"][task], dtype=np.float64)
    tr = _finite(y_a, np.arange(len(y_a)))
    va = _finite(y_t, test_rec["cs_split"][task]["val"])
    te = _finite(y_t, test_rec["cs_split"][task]["test"])
    pts, census = [], []
    if len(tr) < 2 or len(te) < 2:
        return pts, census
    cols = {t: _transfer_cols(train_rec, test_rec, t) for t in taps}
    if all(v is None for v in cols.values()):
        return pts, census
    for n in [g for g in N_GRID_CS if g < len(tr)] + [None]:
        is_full = n is None
        nn = len(tr) if is_full else int(n)
        for seed in range(1 if is_full else SEEDS_FOR_N[int(n)]):
            # KEYED ON THE TRAIN SESSION, NOT THE TEST CELL. In cs the donor is the SAME for all ten
            # cells, so keying on the cell would silently average over 10x more donor subsamples than
            # the claim is about -- a different, lower-variance estimand. One draw, ten patients.
            # In csession the sibling IS per-cell, so the key varies with it, as it should.
            rng = _rng(draw_key, task, 0, nn, seed)
            tr_s = _strat_draw(y_a, tr, nn, rng)
            d_tr = _digest(tr_s)
            for tap in taps:
                if cols[tap] is None or not (_have(train_rec, tap) and _have(test_rec, tap)):
                    continue
                # INVARIANT 2, as in the WS path: every tap fits the SAME train rows.
                assert _digest(tr_s) == d_tr, "draw drifted across taps"
                a_idx, t_idx = cols[tap]
                res = _cs_fit_point(train_rec, test_rec, tap, tr_s, a_idx, va, te, t_idx, y_a, y_t)
                pts.append({"task": task, "fold": 0, "tap": tap, "col": "trainonly",
                            "n": nn, "n_is_full": is_full, "seed": seed,
                            "test": res["test"], "lam_pinned": bool(res["lam_pinned"])})
            yb = (y_a[tr_s] > 0)
            census.append({"task": task, "fold": 0, "n": nn, "seed": seed, "n_is_full": is_full,
                           "n_train": len(tr_s), "n_val": len(va), "n_test": len(te),
                           "n_pos": int(yb.sum()), "n_neg": int((~yb).sum()),
                           "parent_bal": float((y_a[tr] > 0).mean()),
                           "draw_bal": float(yb.mean())})
    return pts, census

def _cs_curve_cell(train_rec, test_rec, task, taps):
    """Cross-subject: the donor is the FIXED anchor, so the draw is keyed on it."""
    return _transfer_curve_cell(train_rec, test_rec, task, taps, CS_TRAIN_ANCHOR)

def _csession_curve_cell(train_rec, test_rec, task, taps, sibling):
    """Cross-session: the train session is this cell's SIBLING, so the draw is keyed on that."""
    return _transfer_curve_cell(train_rec, test_rec, task, taps, sibling)

def _transfer_anchor_check(cell_fn, train_rec, test_rec, task, taps, pts) -> list:
    """INVARIANT 1 for a transfer regime — the N=full point must BE the published cell.

    Same licence the WS curve runs under: computed by CALLING the real cell function
    (``_cs_cell`` / ``_csession_cell``), never by restating what it does. Note both return
    ``_grid_cells`` UNFOLDED (no fold averaging), so the comparison is against a single value.
    """
    pub = cell_fn(train_rec, test_rec, task, taps)["cells"]
    rows = []
    for tap in taps:
        key = f"{tap}|std"
        if key not in pub:
            continue
        mine = [p["test"] for p in pts
                if p["task"] == task and p["tap"] == tap and p["n_is_full"]]
        if not mine:
            continue
        rows.append({"task": task, "tap": tap, "col": "trainonly",
                     "published": float(pub[key]["test"]),
                     "curve": float(np.nanmean(mine)),
                     "absdiff": float(abs(np.nanmean(mine) - pub[key]["test"]))})
    return rows

def _cs_anchor_check(train_rec, test_rec, task, taps, pts) -> list:
    return _transfer_anchor_check(_cs_cell, train_rec, test_rec, task, taps, pts)

def _csession_anchor_check(train_rec, test_rec, task, taps, pts) -> list:
    return _transfer_anchor_check(_csession_cell, train_rec, test_rec, task, taps, pts)

def _anchor_check(rec, task, taps, pts) -> list:
    """INVARIANT 1 — the N=full point must BE the published board cell.

    Computed by calling the REAL ``_ws_cell`` on this record, not by restating it. Any drift means
    the subsample harness perturbed the fit itself, and every other point on the curve inherits
    that drift, so this is the check that licenses the run.
    """
    pub = _ws_cell(rec, task, taps)["cells"]
    rows = []
    for tap in taps:
        key = f"{tap}|std"
        if key not in pub:
            continue
        for col in COLUMNS:
            mine = [p["test"] for p in pts
                    if p["task"] == task and p["tap"] == tap and p["col"] == col and p["n_is_full"]]
            if not mine:
                continue
            # _ws_cell averages the folds; the curve keeps them separate, so average to compare.
            rows.append({"task": task, "tap": tap, "col": col,
                         "published": float(pub[key]["test"]),
                         "curve": float(np.nanmean(mine)),
                         "absdiff": float(abs(np.nanmean(mine) - pub[key]["test"]))})
    return rows

ANCHOR_TOL = 1e-9

def _anchor_verdict(rows) -> list:
    """Offending anchor rows, or [] if the anchor holds. Raises on an EMPTY row list.

    The empty case is the one that has to raise rather than return []: no rows means nothing was
    compared, and a guard that reports "no violations" because it checked nothing is worse than no
    guard at all. Every other bad state shows up as a row with a large absdiff."""
    if not rows:
        raise AssertionError(
            "🔴 ANCHOR VACUOUS: zero (tap, col) comparisons were made, so the N=full point was "
            "never checked against the published _ws_cell. Nothing here is licensed.")
    return [r for r in rows if r["absdiff"] >= ANCHOR_TOL]

def _shard(cache_dir, tag, index, regime, taps, mmap=False) -> dict:
    """One array task: one WS session, or one CS test cell against the fixed anchor.

    ``mmap`` defers the cache read into the gathers. It is the only configuration that
    finishes csession at d=384: the eager load OOMs at 200 G and again at 240 G, and the
    curve touches the same taps the board readout does.
    """
    if regime == "cs":
        cell = CS_TEST_CELLS[index]
        anchor_rec = _load(cache_dir, CS_TRAIN_ANCHOR, tag, mmap=mmap)
        test_rec = _load(cache_dir, cell, tag, mmap=mmap)
        def curve_fn(task):
            return _cs_curve_cell(anchor_rec, test_rec, task, taps)

        def check_fn(task, p):
            return _cs_anchor_check(anchor_rec, test_rec, task, taps, p)

        published = "_cs_cell"
    elif regime == "csession":
        cell = CSESSION_CELLS[index]
        sib = _sibling(cell)
        train_rec = _load(cache_dir, sib, tag, mmap=mmap)
        test_rec = _load(cache_dir, cell, tag, mmap=mmap)
        if any(R._is_elec(tap) for tap in taps):
            for rec in (train_rec, test_rec):
                if rec.get("elec_labels") is None:
                    raise ValueError("Cross-session sample efficiency requires contact labels. "
                                     "Use current caches or --elec-labels-sidecar for older caches.")
        def curve_fn(task):
            return _csession_curve_cell(train_rec, test_rec, task, taps, sib)

        def check_fn(task, p):
            return _csession_anchor_check(train_rec, test_rec, task, taps, p)

        published = "_csession_cell"
    else:
        cell = LITE_SESSIONS[index]
        rec = _load(cache_dir, cell, tag, mmap=mmap)
        def curve_fn(task):
            return _ws_curve_cell(rec, cell, task, taps)

        def check_fn(task, p):
            return _anchor_check(rec, task, taps, p)

        published = "_ws_cell"

    pts, census, anchor = [], [], []
    for i, task in enumerate(BOARD_TASKS):
        p, c = curve_fn(task)
        pts += p
        census += c
        a = check_fn(task, p)
        anchor += a
        # FAIL FAST, on the FIRST task. A broken anchor invalidates every point in every shard, so
        # discovering it at the end of a 12-way array costs ~20x what discovering it here does.
        bad = _anchor_verdict(a)
        if i == 0 and bad:
            raise AssertionError(
                f"🔴 ANCHOR FAILED on the first task ({task}): the N=full point does not reproduce "
                f"the published {published}. The subsample harness has perturbed the FIT, so no point "
                f"on this curve is on the board protocol. Worst: {max(bad, key=lambda r: r['absdiff'])}")
        print(f"  [{task}] {len(p)} points  anchor max|diff| "
              f"{max(r['absdiff'] for r in a):.2e}", flush=True)
    return {"kind": SHARD_PREFIX[regime], "regime": regime,
            "name": f"S{cell[0]}T{cell[1]}", "tag": tag,
            "contiguous": False, "points": pts, "census": census, "anchor": anchor}

def _curve(pts, tap, col):
    """{N: macro AUROC} — mean over seeds, then folds, then the 15 tasks, then the cells.

    Aggregation order matters and this one matches the board's: a cell is a (session, task) and the
    macro is the mean over the 15 tasks of the cohort mean, so no task with more folds or more
    seeds can weigh more than another.
    """
    by = {}
    for p in pts:
        if p["tap"] != tap or p["col"] != col:
            continue
        by.setdefault(p["n_bucket"], {}).setdefault(p["task"], {}).setdefault(p["cell"], []).append(p["test"])
    out = {}
    for n, tasks in by.items():
        per_task = [np.nanmean([np.nanmean(v) for v in cells.values()]) for cells in tasks.values()]
        out[n] = float(np.nanmean(per_task))
    return {k: out[k] for k in sorted(out, key=_x_order)}

def _reach(curve, target):
    """Smallest N at which `curve` reaches `target`, log2-interpolated between bracketing points.

    Returns None when the curve never gets there -- reported as such, never silently clipped to the
    grid end, because "did not reach" and "reached at the last point" are different results.
    """
    ns = sorted(curve)
    for i, n in enumerate(ns):
        if curve[n] >= target:
            if i == 0:
                return float(n)
            n0, a0, a1 = ns[i - 1], curve[ns[i - 1]], curve[n]
            if a1 <= a0:
                return float(n)
            f = (target - a0) / (a1 - a0)
            return float(2.0 ** (np.log2(n0) + f * (np.log2(n) - np.log2(n0))))
    return None

def _merge(shard_dir, regime="ws") -> dict:
    pts, census, anchor, tags, contig = [], [], [], set(), set()
    paths = sorted(glob.glob(f"{shard_dir}/{SHARD_PREFIX[regime]}_*.json"))
    if not paths:
        raise SystemExit(f"🔴 no {SHARD_PREFIX[regime]}_*.json shards in {shard_dir} — "
                         f"nothing to merge for regime '{regime}'")
    for path in paths:
        with open(path) as f:
            sh = json.load(f)
        # A ws and a cs shard must NEVER land in the same merge: the taps differ, the train set
        # differs, and averaging them would produce a number of no regime at all.
        assert sh.get("regime", "ws") == regime, f"{path} is regime {sh.get('regime')}, want {regime}"
        tags.add(sh["tag"])
        contig.add(sh["contiguous"])
        for p in sh["points"]:
            # A cell is a (session, task); `n_bucket` folds `full` into a single x position so the
            # anchor is comparable across cells whose train halves differ in length after the
            # per-task label filter.
            p["cell"] = sh["name"]
            p["n_bucket"] = FULL if p["n_is_full"] else p["n"]
            pts.append(p)
        census += [dict(c, cell=sh["name"]) for c in sh["census"]]
        anchor += [dict(a, cell=sh["name"]) for a in sh["anchor"]]
    if len(tags) != 1 or len(contig) != 1:
        raise ValueError("Curve shards mix checkpoint tags or sampling protocols")
    return {"points": pts, "census": census, "anchor": anchor, "regime": regime,
            "tags": sorted(tags), "contiguous": sorted(contig)}

def validate_curve(m, regime):
    """Require the complete published training-only experiment before reporting.

    Full-data counts determine which grid sizes fit each task/fold. Legacy
    ``both`` columns may remain in the file, but never enter this validation.
    """
    if (regime not in CURVE_TAPS or m.get("regime", "ws") != regime
            or m.get("contiguous") != [False]
            or len(m.get("tags", [])) != 1 or not m["tags"][0]):
        raise ValueError("Curve must contain one checkpoint and the canonical random regime")
    sessions = CS_TEST_CELLS if regime == "cs" else LITE_SESSIONS
    cells = {f"S{s}T{t}" for s, t in sessions}
    folds = (0, 1) if regime == "ws" else (0,)
    bases = {(cell, task, fold, tap) for cell in cells for task in BOARD_TASKS
             for fold in folds for tap in CURVE_TAPS[regime]}
    points, full = {}, {}
    for p in m.get("points", []):
        if p.get("col") == "both":
            continue
        if (p.get("col") != "trainonly" or type(p.get("n_is_full")) is not bool
                or any(type(p.get(k)) is not int for k in ("fold", "n", "seed"))):
            raise ValueError("Invalid curve point identity")
        base = (p.get("cell"), p.get("task"), p["fold"], p.get("tap"))
        key = (*base, p["n_is_full"], p["n"], p["seed"])
        if base not in bases or key in points:
            raise ValueError(f"Unexpected or duplicate curve point: {key}")
        score = p.get("test")
        if (type(score) not in (int, float) or not np.isfinite(score) or not 0 <= score <= 1
                or p.get("n_bucket") != (FULL if p["n_is_full"] else p["n"])):
            raise ValueError(f"Invalid curve score or count bucket: {key}")
        points[key] = score
        if p["n_is_full"]:
            if base in full or p["seed"] != 0 or p["n"] < 2:
                raise ValueError(f"Invalid or duplicate full-data point: {key}")
            full[base] = p["n"]
    if full.keys() != bases:
        raise ValueError(f"Incomplete full-data curve grid: {len(bases - full.keys())} missing")
    grid = N_GRID if regime == "ws" else N_GRID_CS
    expected = set()
    for base, n_full in full.items():
        sibling = (*base[:3], CURVE_TAPS[regime][0])
        if n_full != full[sibling]:
            raise ValueError(f"Curve taps disagree on full-data training count: {base}")
        expected.add((*base, True, n_full, 0))
        expected.update((*base, False, n, seed) for n in grid if n < n_full
                        for seed in range(SEEDS_FOR_N[n]))
    if points.keys() != expected:
        raise ValueError(f"Incomplete curve point grid: {len(expected - points.keys())} missing, "
                         f"{len(points.keys() - expected)} extra")
    anchors = {}
    expected_anchors = {(cell, task, tap) for cell in cells for task in BOARD_TASKS
                        for tap in CURVE_TAPS[regime]}
    for a in m.get("anchor", []):
        if a.get("col") == "both":
            continue
        key = (a.get("cell"), a.get("task"), a.get("tap"))
        if a.get("col") != "trainonly" or key not in expected_anchors or key in anchors:
            raise ValueError(f"Unexpected or duplicate curve anchor: {key}")
        if any(type(a.get(k)) not in (int, float) or not np.isfinite(a[k])
               for k in ("published", "curve", "absdiff")):
            raise ValueError(f"Nonfinite curve anchor: {key}")
        cell, task, tap = key
        scores = [points[(cell, task, fold, tap, True, full[(cell, task, fold, tap)], 0)]
                  for fold in folds]
        if (not 0 <= a["published"] <= 1 or not 0 <= a["absdiff"] < ANCHOR_TOL
                or abs(a["published"] - a["curve"]) >= ANCHOR_TOL
                or abs(float(np.mean(scores)) - a["curve"]) >= ANCHOR_TOL):
            raise ValueError(f"Curve anchor drifted: {key}")
        anchors[key] = a
    if anchors.keys() != expected_anchors:
        raise ValueError("Incomplete curve anchor grid")

def _subject(cell_name: str) -> str:
    """'S3T1' -> 'S3'. The bootstrap resamples SUBJECTS, not cells: two sessions of one patient are
    not two independent draws from the population the claim generalizes over."""
    return cell_name.split("T")[0]

def _report(m) -> None:
    """Print training-only curves; label_saving_ci computes the paper's uncertainty."""
    print(f"{m['regime']}: tags={m['tags']} sampling={m['contiguous']}")
    for tap in CURVE_TAPS[m['regime']]:
        print(f"  {tap}: {_curve(m['points'], tap, 'trainonly')}")

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="ws", choices=("ws", "cs", "csession", "merge"))
    p.add_argument("--cache-dir")
    p.add_argument("--tag", default="board_iid75_vits384")
    p.add_argument("--index", type=int, default=0,
                   help="array task id: index into LITE_SESSIONS (ws), CS_TEST_CELLS (cs), "
                        "or CSESSION_CELLS (csession)")
    p.add_argument("--shard-dir", required=True)
    p.add_argument("--out", default="samplecurve.json")
    p.add_argument("--regime", default="ws", choices=("ws", "cs", "csession"),
                   help="merge mode only: which shard family to merge. A merge NEVER mixes the two.")
    p.add_argument("--taps", default="", help="default: the regime's own unit (see CURVE_TAPS)")
    p.add_argument("--elec-labels-sidecar",
                   help="Optional pickle of contact labels for older caches that lack elec_labels.")
    p.add_argument("--mmap", action="store_true",
                   help="defer the cache read into the gathers. REQUIRED for csession at d=384: "
                        "the eager load OOMs at 200 G and at 240 G.")
    args = p.parse_args()

    if args.elec_labels_sidecar:
        import pickle
        with open(args.elec_labels_sidecar, "rb") as fh:
            R._ELEC_LABELS_SIDECAR = pickle.load(fh)
        print(f"[sidecar] attaching elec_labels for {len(R._ELEC_LABELS_SIDECAR)} sessions",
              flush=True)

    if args.mode == "merge":
        m = _merge(args.shard_dir, args.regime)
        validate_curve(m, args.regime)
        _report(m)
        with open(args.out, "w") as f:
            json.dump(m, f)
        print(f"\nwrote {args.out}\nMERGE_DONE", flush=True)
        return

    taps = (tuple(t.strip() for t in args.taps.split(",") if t.strip())
            or CURVE_TAPS[args.mode])
    cell = {"cs": CS_TEST_CELLS, "csession": CSESSION_CELLS,
            "ws": LITE_SESSIONS}[args.mode][args.index]
    grid = N_GRID if args.mode == "ws" else N_GRID_CS
    t0 = time.perf_counter()
    print(f"[R31/{args.mode}] S{cell[0]}T{cell[1]} taps={taps} "
          f"N_GRID={grid} seeds={SEEDS_FOR_N}", flush=True)
    if args.mode in TRANSFER_REGIMES:
        src = CS_TRAIN_ANCHOR if args.mode == "cs" else _sibling(cell)
        print(f"  train session = S{src[0]}T{src[1]} (subsampled); "
              f"val/test = this cell's cs_split (NEVER subsampled)", flush=True)
    sh = _shard(args.cache_dir, args.tag, args.index, args.mode, taps, mmap=args.mmap)
    os.makedirs(args.shard_dir, exist_ok=True)
    out = f"{args.shard_dir}/{SHARD_PREFIX[args.mode]}_{sh['name']}.json"
    with open(out, "w") as f:
        json.dump(sh, f)
    bad = [a for a in sh["anchor"] if a["absdiff"] >= 1e-9]
    print(f"[anchor] {len(sh['anchor'])} rows, {len(bad)} with |diff| >= 1e-9 "
          f"{'✅' if not bad else '🔴 ' + str(bad[:3])}", flush=True)
    print(f"wrote {out}  ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)

if __name__ == "__main__":
    main()
