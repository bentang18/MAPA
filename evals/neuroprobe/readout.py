"""Frozen-feature linear ridge for the three Neuroprobe-Lite regimes.

Features use training-only standardization. Each fit selects regularization
on validation AUROC over the fixed 25-point grid, keeping the smallest lambda
on ties, and reports the separate test half. WS retains both fold scores.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import multiprocessing as mp
import os
import pickle
import time

import numpy as np
import torch

UNKNOWN_REGION_ID = 74

_PH: dict = {}

@contextlib.contextmanager
def _timed(name):
    t = time.perf_counter()
    try:
        yield
    finally:
        _PH[name] = _PH.get(name, 0.0) + (time.perf_counter() - t)

def _phase_report(label) -> None:
    tot = sum(_PH.values()) or 1.0
    print(f"[phase] {label} — wall budget by phase (total {tot / 60:.1f} min):", flush=True)
    for k, v in sorted(_PH.items(), key=lambda kv: -kv[1]):
        print(f"[phase]   {k:22s} {v / 60:7.2f} min  {100 * v / tot:5.1f}%", flush=True)

BOARD_TASKS = (
    "onset", "speech", "volume", "delta_volume", "pitch", "word_index",
    "word_gap", "gpt2_surprisal", "word_head_pos", "word_part_speech",
    "word_length", "global_flow", "local_flow", "frame_brightness", "face_num",
)

CS_TRAIN_ANCHOR = (2, 4)

CS_TEST_CELLS = ((1, 1), (1, 2), (3, 0), (3, 1), (4, 0), (4, 1),
                 (7, 0), (7, 1), (10, 0), (10, 1))

LITE_SESSIONS = ((1, 1), (1, 2), (2, 0), (2, 4), (3, 0), (3, 1),
                 (4, 0), (4, 1), (7, 0), (7, 1), (10, 0), (10, 1))

ENCODERS = ("enc0", "enc3", "enc6", "enc9", "enc12")  # region-mean (electrodes pooled at encode)

ELEC_TAPS = ("enc0_elec", "enc12_elec")   # per-electrode: enc0_elec = depth-0 parity FLOOR for enc12_elec

ALL_TAPS = ELEC_TAPS + ENCODERS           # universe for --taps validation

WS_TAPS = ELEC_TAPS

CS_TAPS = ENCODERS

CSESSION_TAPS = ELEC_TAPS

CSESSION_CELLS = LITE_SESSIONS            # every Lite session is a cross-session test cell, sibling-trained

def _sibling(cell):
    """The other Lite trial of the SAME subject — upstream cross_session's train trial
    (train_test_splits.py:146: the one other NEUROPROBE_LITE_SUBJECT_TRIALS entry for the subject)."""
    s, _ = cell
    sibs = [c for c in LITE_SESSIONS if c[0] == s and c != cell]
    assert len(sibs) == 1, f"subject {s} must have exactly one sibling Lite trial, got {sibs}"
    return sibs[0]

_ELEC_LABELS_SIDECAR: dict | None = None

LAM_MULTS = tuple(np.logspace(-4.0, 4.0, 25))

def auroc(scores, labels) -> float:
    """Verbatim from online_probe.auroc. NaN if the eval half is single-class."""
    from sklearn.metrics import roc_auc_score

    y = (np.asarray(labels) > 0).astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores)))

def _finite(y: np.ndarray, rows: np.ndarray) -> np.ndarray:
    r = np.asarray(rows, dtype=np.int64)
    return r[np.isfinite(y[r])]

def _standardize(z_tr, others):
    """Per-feature z-score on TRAIN stats only (never fit on val/test). σ=0 → 1.

    This is the canonical frozen-FM linear probe: the MAE/MoCo-v3/DINO lineage puts a
    BatchNorm WITHOUT affine before the linear head, and BN at eval uses statistics
    accumulated over probe TRAINING — i.e. train-set stats, which in CS means the ANCHOR's.
    """
    with _timed("standardize"):
        mu = z_tr.mean(axis=0)
        sd = z_tr.std(axis=0)
        sd[sd == 0] = 1.0
        return (z_tr - mu) / sd, [(z - mu) / sd for z in others]

_STD_BLOCK = 1024

def _standardize_inplace(z_tr, others, blk=_STD_BLOCK):
    """Apply training mean/std in place, blocking columns to bound temporary memory.

    Reductions stay on axis 0. Avoid width-one blocks, whose NumPy summation
    order differs; tests check exact agreement with whole-array standardization.
    """
    with _timed("standardize"):
        d = z_tr.shape[1]
        edges = list(range(0, d, blk)) + [d]
        if len(edges) > 2 and edges[-1] - edges[-2] == 1:
            edges.pop(-2)                      # avoid a width-one trailing block
        for lo, hi in zip(edges, edges[1:]):
            b = z_tr[:, lo:hi]
            mu = b.mean(axis=0)
            sd = b.std(axis=0)
            sd[sd == 0] = 1.0
            b -= mu
            b /= sd
            for z in others:
                zb = z[:, lo:hi]
                zb -= mu
                zb /= sd
        return z_tr, others

def _have(rec, tap) -> bool:
    return tap in rec["feats"]

def _validate_taps(taps) -> None:
    for tap in taps:
        if tap not in ALL_TAPS:
            raise SystemExit(f"unknown tap {tap!r}; choose from {ALL_TAPS}")

def _is_elec(tap) -> bool:
    return tap in ELEC_TAPS

def _feat(rec, enc, rows, col_idx=None) -> np.ndarray:
    """Gather rows and aligned contact/region columns, then flatten to fp32."""
    with _timed("gather_fp16"):
        x = rec["feats"][enc]["raw"][np.asarray(rows, dtype=np.int64)]
        if col_idx is not None:
            x = x[:, np.asarray(col_idx, dtype=np.int64)]
    with _timed("to_fp32"):
        x = x.to(torch.float32).numpy()
        return x.reshape(x.shape[0], -1)

def _linear_grams(z_tr, evals):
    """fp32 Gram products promoted to fp64 for the ridge solve."""
    with _timed("gram_gemm"):
        g = np.asarray(z_tr @ z_tr.T, dtype=np.float64)         # fp32 GEMM → fp64 Gram
    with _timed("eval_kernels"):
        kern = {name: np.asarray(z @ z_tr.T, dtype=np.float64)
                for name, (z, _) in evals.items()}
    return g, kern

def _lam_grid(z_tr, y_tr, evals):
    """Score the fixed lambda grid; use the primal only when feature width < train rows."""
    if len(y_tr) < 2:
        return {name: {m: float("nan") for m in LAM_MULTS} for name in evals}
    if z_tr.shape[1] < z_tr.shape[0]:
        return _lam_grid_primal(z_tr, y_tr, evals)
    g, kern = _linear_grams(z_tr, evals)
    n = g.shape[0]
    with _timed("eigh"):
        w, V = np.linalg.eigh(g)                                # G symmetric PSD ⇒ w >= 0
    c = V.T @ np.asarray(y_tr, dtype=np.float64)
    base = float(np.sum(w) / max(n, 1))                         # trace(G)/n — the λ scale
    out: dict = {name: {} for name in evals}
    with _timed("lam_sweep"):
        for m in LAM_MULTS:
            alpha = V @ (c / (w + m * base))
            for name, (_, y) in evals.items():
                s = kern[name] @ alpha
                out[name][m] = (auroc(s, y) if len(y) >= 2 else float("nan"))
    return out

def _lam_grid_primal(z_tr, y_tr, evals):
    """Equivalent ridge in feature space, retaining trace(Z.T @ Z) / n as the lambda scale."""
    with _timed("gram_gemm"):
        a_mat = np.asarray(z_tr.T @ z_tr, dtype=np.float64)      # (d, d)
    n = z_tr.shape[0]
    with _timed("eigh"):
        w, V = np.linalg.eigh(a_mat)
    c = V.T @ (z_tr.T @ np.asarray(y_tr, dtype=np.float64))
    base = float(np.trace(a_mat) / max(n, 1))                    # == sum(eig(G))/n, the dual's scale
    with _timed("lam_sweep"):
        lam = np.asarray(LAM_MULTS, dtype=np.float64) * base
        beta = V @ (c[:, None] / (w[:, None] + lam[None, :]))    # (d, |LAM_MULTS|)
        out: dict = {}
        for name, (z, y) in evals.items():
            if len(y) < 2:
                out[name] = {m: float("nan") for m in LAM_MULTS}
                continue
            s = np.asarray(z, dtype=np.float64) @ beta           # (n_e, |LAM_MULTS|)
            out[name] = {m: auroc(s[:, i], y) for i, m in enumerate(LAM_MULTS)}
    return out

def _select_lam(d) -> dict:
    """Select maximum validation AUROC, keeping the first (smallest) tied lambda.

    All-NaN validation returns NaN. Boundary flags describe grid position,
    not evidence that the optimum lies outside the grid."""
    finite = [(m, va) for m, va in d["val"].items() if not np.isnan(va)]
    if not finite:
        return {"val": float("nan"), "test": float("nan"), "lam_mult": float("nan"),
                "lam_pin": "", "lam_pinned": False, "n_tied": 0}
    best_val = max(va for _, va in finite)
    # Insertion order is LAM_MULTS ascending, so tied[0] is the smallest tied λ — exactly what the
    # strict-`>` loop this replaced selected. That equivalence is what keeps "argmax" byte-faithful.
    tied = [m for m, va in finite if va == best_val]
    m = tied[0]
    pin = "lo" if m == LAM_MULTS[0] else ("hi" if m == LAM_MULTS[-1] else "")
    out = {"val": best_val, "test": d["test"][m], "lam_mult": float(m),
           "lam_pin": pin, "lam_pinned": pin == "lo", "n_tied": len(tied)}
    return out

def _cell_key(tap, norm) -> str:
    return f"{tap}|{norm}"

def _grid_cells(grid) -> dict:
    """Keep a separately validation-selected result for every tap."""
    return {_cell_key(t, nm): _select_lam(d) for (t, nm), d in grid.items()}

def _region_cols(anchor_rec, test_rec):
    """Anchor∩test region columns, aligned BY ATLAS ID (not by position).

    The reserved 'unknown' id is NOT an anatomical location: two electrodes carrying it are
    not in the same place, so aligning subjects on it would be a free unearned column. It is
    a no-op on the Lite board (anchor S2T4 has no unmapped electrodes) but fires on any
    corpus that does, so it is excluded here rather than assumed away.
    """
    a_p = np.asarray(anchor_rec["present_parcels"], dtype=np.int64)
    t_p = np.asarray(test_rec["present_parcels"], dtype=np.int64)
    common = np.intersect1d(a_p, t_p)
    if UNKNOWN_REGION_ID in common:
        print(f"[check] dropping unknown region {UNKNOWN_REGION_ID} from intersection", flush=True)
        common = common[common != UNKNOWN_REGION_ID]
    if common.size == 0:
        return None, None, common
    a_idx = [int(np.where(a_p == c)[0][0]) for c in common]
    t_idx = [int(np.where(t_p == c)[0][0]) for c in common]
    return a_idx, t_idx, common

def _run_norms(grid, enc, z_tr, z_va, z_te, y_tr, y_va, y_te):
    """Standardize in place using training statistics, then score linear ridge."""
    a, (b, c) = _standardize_inplace(z_tr, [z_va, z_te])
    grid[(enc, "std")] = _lam_grid(a, y_tr, {"val": (b, y_va), "test": (c, y_te)})

def _ws_cell(rec, task, taps) -> dict:
    """Within-session: board KFold(2). Per fold fit train, λ-select on the val half, report the
    test half; average the two folds' test AUROCs, per (tap, norm)."""
    y = np.asarray(rec["labels"][task], dtype=np.float64)
    folds = []
    fold_ids = []
    for _fold, sp in sorted(rec["ws_split"][task].items()):
        tr, va, te = (_finite(y, sp["train"]), _finite(y, sp["val"]), _finite(y, sp["test"]))
        if len(tr) < 2 or len(te) < 2:
            continue
        grid = {}
        for enc in taps:
            if not _have(rec, enc):
                continue
            z_tr = _feat(rec, enc, tr)
            z_va, z_te = _feat(rec, enc, va), _feat(rec, enc, te)
            _run_norms(grid, enc, z_tr, z_va, z_te, y[tr], y[va], y[te])
        if grid:
            folds.append(_grid_cells(grid))
            fold_ids.append(int(_fold))
    if not folds:
        return {"cells": {}}
    keys = sorted({k for f in folds for k in f})
    out = {}
    for k in keys:
        vals = [f[k]["test"] for f in folds if k in f]
        out[k] = {"test": float(np.nanmean(vals)) if vals else float("nan"),
                  "folds": [{"fold_idx": idx, "test_roc_auc": float(f[k]["test"])}
                            for idx, f in zip(fold_ids, folds) if k in f],
                  "lam_pinned": bool(any(f[k]["lam_pinned"] for f in folds if k in f)),
                  "lam_sat": bool(any(f[k].get("lam_pin") == "hi" for f in folds if k in f)),
                  "lam_mult": [f[k]["lam_mult"] for f in folds if k in f]}
    return {"cells": out}

def _cs_cell(anchor_rec, test_rec, task, taps) -> dict:
    """Cross-subject: fit the anchor's finite rows, λ-select on the test cell's val half, report
    its test half. Features are the anchor∩test region intersection (atlas-id aligned)."""
    y_a = np.asarray(anchor_rec["labels"][task], dtype=np.float64)
    y_t = np.asarray(test_rec["labels"][task], dtype=np.float64)
    tr = _finite(y_a, np.arange(len(y_a)))
    va = _finite(y_t, test_rec["cs_split"][task]["val"])
    te = _finite(y_t, test_rec["cs_split"][task]["test"])
    if len(tr) < 2 or len(te) < 2:
        return {"cells": {}}
    a_idx, t_idx, common = _region_cols(anchor_rec, test_rec)
    if common.size == 0:
        return {"cells": {}}
    grid: dict = {}
    for enc in taps:
        if not (_have(anchor_rec, enc) and _have(test_rec, enc)):
            continue
        z_tr = _feat(anchor_rec, enc, tr, a_idx)
        z_va, z_te = _feat(test_rec, enc, va, t_idx), _feat(test_rec, enc, te, t_idx)
        _run_norms(grid, enc, z_tr, z_va, z_te, y_a[tr], y_t[va], y_t[te])
    if not grid:
        return {"cells": {}}
    return {"cells": _grid_cells(grid), "n_parcels": int(common.size)}

def _elec_cols(train_rec, test_rec):
    """Shared electrodes between two SAME-SUBJECT sessions, aligned BY LABEL (elec_labels).

    Sibling Lite trials drop DIFFERENT bad channels (measured: 4/6 subjects differ, e.g. subj-3
    100 vs 102), so the per-electrode axis is NOT positionally aligned across sessions —
    intersecting by identity is the only correct alignment. Returns (train_idx, test_idx,
    n_shared); (None, None, 0) if either cache lacks elec_labels (pre-edit cache) or no overlap."""
    a = train_rec.get("elec_labels")
    t = test_rec.get("elec_labels")
    if a is None or t is None:
        return None, None, 0
    a = np.asarray(a)
    t = np.asarray(t)
    common = np.intersect1d(a, t)
    if common.size == 0:
        return None, None, 0
    a_idx = [int(np.where(a == c)[0][0]) for c in common]
    t_idx = [int(np.where(t == c)[0][0]) for c in common]
    return a_idx, t_idx, int(common.size)

def _csession_cell(train_rec, test_rec, task, taps) -> dict:
    """Cross-session: train on the SIBLING trial of the SAME subject, λ-select on this cell's val
    half, report its test half.

    Upstream ``generate_splits_cross_session`` halves the test session IDENTICALLY to
    ``generate_splits_cross_subject`` (train_test_splits.py:153-156 == 66-69: val=range(size//2),
    test=range(size//2,size)), so ``cs_split``'s val/test are reused VERBATIM — only the train
    anchor differs (the sibling trial, not S2T4). Region taps align by atlas id (``_region_cols``);
    per-electrode taps align by electrode IDENTITY (``_elec_cols``), on the shared-electrode subset.
    """
    y_a = np.asarray(train_rec["labels"][task], dtype=np.float64)
    y_t = np.asarray(test_rec["labels"][task], dtype=np.float64)
    tr = _finite(y_a, np.arange(len(y_a)))
    va = _finite(y_t, test_rec["cs_split"][task]["val"])
    te = _finite(y_t, test_rec["cs_split"][task]["test"])
    if len(tr) < 2 or len(te) < 2:
        return {"cells": {}}
    p_a, p_t, p_common = _region_cols(train_rec, test_rec)
    e_a, e_t, n_elec = _elec_cols(train_rec, test_rec)
    grid: dict = {}
    for enc in taps:
        if not (_have(train_rec, enc) and _have(test_rec, enc)):
            continue
        if _is_elec(enc):
            if e_a is None:
                continue
            col_a, col_t = e_a, e_t
        else:
            if p_a is None:
                continue
            col_a, col_t = p_a, p_t
        z_tr = _feat(train_rec, enc, tr, col_a)
        z_va, z_te = _feat(test_rec, enc, va, col_t), _feat(test_rec, enc, te, col_t)
        _run_norms(grid, enc, z_tr, z_va, z_te, y_a[tr], y_t[va], y_t[te])
    if not grid:
        return {"cells": {}}
    return {"cells": _grid_cells(grid),
            "n_parcels": int(p_common.size) if p_a is not None else 0,
            "n_elec": n_elec}

MMAP_DEFAULT = {"ws": False, "cs": False, "csession": False}

def _load(cache_dir, session, tag, mmap=False):
    """Load a session cache. ``mmap=True`` defers the read into the gathers (see MMAP_DEFAULT).

    Pages arrive only where a tensor is actually indexed, so selectivity is free: a CS shard
    never gathers enc12_elec and therefore never reads those 34 GB. No tap-filter argument is
    needed — not touching IS not loading. But laziness is not a speedup, and for WS it is a
    slowdown; pick with MMAP_DEFAULT and A/B with --mmap/--no-mmap before trusting a change.
    """
    s, t = session
    rec = torch.load(f"{cache_dir}/enc_s{s}_t{t}_{tag}.pt", map_location="cpu",
                     weights_only=False, mmap=mmap)
    if _ELEC_LABELS_SIDECAR is not None and rec.get("elec_labels") is None:
        lab = _ELEC_LABELS_SIDECAR.get(f"s{s}_t{t}")
        if lab is not None:
            lab = np.asarray(lab)
            if lab.ndim != 1 or len(np.unique(lab)) != len(lab):
                raise ValueError("sidecar contact labels must be a unique one-dimensional sequence")
            if lab.shape[0] != rec["feats"]["enc12_elec"]["raw"].shape[1]:
                raise ValueError(
                    f"sidecar labels ({lab.shape[0]}) != enc12_elec electrodes "
                    f"({rec['feats']['enc12_elec']['raw'].shape[1]}) for s{s}_t{t}")
            rec["elec_labels"] = lab
    return rec

_SHARED: dict = {}

def _task_worker(task):
    """Runs in a forked child. Reads the cache from _SHARED — NEVER take it as an argument:
    Pool pickles arguments, which would serialize a multi-GB cache per task."""
    fn, taps = _SHARED["fn"], _SHARED["taps"]
    return task, fn(task, taps)

def _map_tasks(fn, taps, workers) -> dict:
    """{task: cell} over BOARD_TASKS, optionally across forked workers."""
    if workers <= 1:
        return {task: fn(task, taps) for task in BOARD_TASKS}
    _SHARED["fn"], _SHARED["taps"] = fn, taps          # set BEFORE fork ⇒ inherited, not pickled
    with mp.get_context("fork").Pool(workers) as pool:
        return dict(pool.map(_task_worker, BOARD_TASKS))

def _ws_shard(cache_dir, tag, session, taps=WS_TAPS, workers=1,
              mmap=MMAP_DEFAULT["ws"]) -> dict:
    rec = _load(cache_dir, session, tag, mmap=mmap)
    out = _map_tasks(lambda task, tp: _ws_cell(rec, task, tp), taps, workers)
    return {"kind": "ws", "name": f"S{session[0]}T{session[1]}",
            "cells": {f"{tag}|{k}": v for k, v in out.items()}}

def _cs_shard(cache_dir, tag, cell, taps=CS_TAPS, workers=1,
              mmap=MMAP_DEFAULT["cs"]) -> dict:
    taps = tuple(t for t in taps if t not in ELEC_TAPS)   # CS is region-bridged by necessity
    anchor_rec = _load(cache_dir, CS_TRAIN_ANCHOR, tag, mmap=mmap)
    test_rec = _load(cache_dir, cell, tag, mmap=mmap)
    out = _map_tasks(lambda task, tp: _cs_cell(anchor_rec, test_rec, task, tp), taps, workers)
    return {"kind": "cs", "name": f"S{cell[0]}T{cell[1]}",
            "cells": {f"{tag}|{k}": v for k, v in out.items()}}

def _csession_shard(cache_dir, tag, cell, taps=CSESSION_TAPS, workers=1,
                    mmap=MMAP_DEFAULT["csession"]) -> dict:
    """Cross-session cell: train on the sibling trial (same subject), test on this session's
    held-out half. Keeps the per-electrode taps (electrode identity IS shared within subject)."""
    train_rec = _load(cache_dir, _sibling(cell), tag, mmap=mmap)
    test_rec = _load(cache_dir, cell, tag, mmap=mmap)
    out = _map_tasks(lambda task, tp: _csession_cell(train_rec, test_rec, task, tp), taps, workers)
    return {"kind": "csession", "name": f"S{cell[0]}T{cell[1]}",
            "cells": {f"{tag}|{k}": v for k, v in out.items()}}

def _blank(tags) -> dict:
    return {f"{tag}|{t}": {"ws": {}, "cs": {}, "csession": {}, "pinned": {}, "sat": {},
                           "n_parcels": {}, "n_elec": {}}
            for tag in tags for t in BOARD_TASKS}

def _absorb(res, sh) -> None:
    """Fold one shard in. res[task][kind]["tap|norm"][cell_name] = test AUROC — the full grid,
    every entry populated over every cell. No axis is collapsed at merge time either."""
    kind = sh["kind"]
    for k, val in sh["cells"].items():
        for gk, s in (val.get("cells") or {}).items():
            res[k][kind].setdefault(gk, {})[sh["name"]] = s["test"]
            if s.get("lam_pinned"):
                res[k]["pinned"].setdefault(f"{kind}:{gk}", []).append(sh["name"])
            if s.get("lam_sat"):
                res[k]["sat"].setdefault(f"{kind}:{gk}", []).append(sh["name"])
        if val.get("n_parcels") is not None:
            res[k]["n_parcels"][sh["name"]] = val["n_parcels"]
        if val.get("n_elec") is not None:
            res[k]["n_elec"][sh["name"]] = val["n_elec"]

def _merge(tags, shard_dir) -> dict:
    res = _blank(tags)
    for kind in ("ws", "cs", "csession"):
        for path in sorted(glob.glob(f"{shard_dir}/{kind}_*.json")):
            with open(path) as f:
                _absorb(res, json.load(f))
    return _finalize(res)

def _finalize(res: dict) -> dict:
    """Cohort-mean each grid entry over its cells (12 WS sessions / 10 CS cells / 12 cross-session)."""
    for c in res.values():
        for kind in ("ws", "cs", "csession"):
            c[f"{kind}_mean"] = {gk: float(np.nanmean(list(d.values())))
                                 for gk, d in c[kind].items() if d}
    return res

def _compute_all(cache_dir, tags, ws_taps=WS_TAPS, cs_taps=CS_TAPS) -> dict:
    res = _blank(tags)
    for tag in tags:
        for session in LITE_SESSIONS:
            sh = _ws_shard(cache_dir, tag, session, ws_taps)
            _absorb(res, sh)
            print(f"[{tag}] WS done {sh['name']}", flush=True)
        for cell in CS_TEST_CELLS:
            sh = _cs_shard(cache_dir, tag, cell, cs_taps)
            _absorb(res, sh)
            print(f"[{tag}] CS done {sh['name']}", flush=True)
        for cell in CSESSION_CELLS:
            sh = _csession_shard(cache_dir, tag, cell, CSESSION_TAPS)
            _absorb(res, sh)
            print(f"[{tag}] CSession done {sh['name']}", flush=True)
    return _finalize(res)

def _macro(res, tag, kind, gk) -> float:
    """Macro over the 15 board tasks of one grid entry's cohort mean."""
    v = [res[f"{tag}|{t}"].get(f"{kind}_mean", {}).get(gk, np.nan) for t in BOARD_TASKS]
    return float(np.nanmean(v)) if not all(np.isnan(x) for x in v) else float("nan")

def _report(tags, res) -> None:
    """Print each tap’s per-task and macro test AUROC without selecting across taps."""
    for tag in tags:
        for kind, label in (("cs", "CS (anchor S2T4 → 10 cells)"),
                            ("csession", "CSession (12 cells, sibling-trained)"),
                            ("ws", "WS (12 sessions)")):
            gks = sorted({g for t in BOARD_TASKS for g in res[f"{tag}|{t}"].get(kind, {})},
                         key=lambda g: (g.split("|")[1], g.split("|")[0]))
            if not gks:
                continue
            print(f"\n=== {label} test-half AUROC — tag={tag} ===", flush=True)
            print(f"  {'task':18s}" + "".join(f"{g:>18s}" for g in gks), flush=True)
            for t in BOARD_TASKS:
                m = res[f"{tag}|{t}"].get(f"{kind}_mean", {})
                print(f"  {t:18s}" + "".join(f"{m.get(g, float('nan')):18.4f}" for g in gks),
                      flush=True)
            print(f"  {'MACRO(15)':18s}"
                  + "".join(f"{_macro(res, tag, kind, g):18.4f}" for g in gks), flush=True)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--tags", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=("all", "ws", "cs", "csession", "merge"), default="all")
    p.add_argument("--index", type=int, help="shard index (mode=ws|cs|csession)")
    p.add_argument("--shard-dir")
    p.add_argument("--workers", type=int, default=1,
                   help="fork this many task-workers per shard (memory-bound: WS ~3, CS ~8). "
                        "Set OMP_NUM_THREADS = cpus-per-task / workers.")
    p.add_argument("--mmap", dest="mmap", default=None, action="store_true",
                   help=f"force lazy paging (default per mode: {MMAP_DEFAULT}). Measured: it "
                        f"DEFERS the read into the gathers at ~1/4 sequential bandwidth — a "
                        f"win for CS (skips the 34 GB elec tap) and a loss for WS.")
    p.add_argument("--no-mmap", dest="mmap", action="store_false",
                   help="force one eager sequential read of the whole cache.")
    p.add_argument("--taps", default="",
                   help=f"comma-separated subset of {ALL_TAPS} (default: per-regime; CS drops "
                        f"{ELEC_TAPS} automatically)")
    p.add_argument("--elec-labels-sidecar",
                   help="pickle {'s{S}_t{T}': labels} to attach to records that lack elec_labels "
                        "(caches encoded before the field was stored, e.g. arm0/r4b). Validated "
                        "same-set upstream; _load re-asserts count-match before attaching.")
    args = p.parse_args()

    global _ELEC_LABELS_SIDECAR
    if args.elec_labels_sidecar:
        with open(args.elec_labels_sidecar, "rb") as fh:
            _ELEC_LABELS_SIDECAR = pickle.load(fh)
        print(f"[sidecar] attaching elec_labels for {len(_ELEC_LABELS_SIDECAR)} sessions",
              flush=True)

    tags = tuple(t.strip() for t in args.tags.split(","))
    # Default taps are per-REGIME (WS/CSession electrode-only, CS region-only); --taps overrides.
    _mode_taps = {"ws": WS_TAPS, "cs": CS_TAPS, "csession": CSESSION_TAPS}
    taps = (tuple(t.strip() for t in args.taps.split(",") if t.strip())
            or _mode_taps.get(args.mode, ALL_TAPS))
    _validate_taps(taps)

    if args.mode in ("ws", "cs", "csession"):
        cells = {"ws": LITE_SESSIONS, "cs": CS_TEST_CELLS, "csession": CSESSION_CELLS}[args.mode]
        cell = cells[args.index]
        fn = {"ws": _ws_shard, "cs": _cs_shard, "csession": _csession_shard}[args.mode]
        t0 = time.perf_counter()
        use_mmap = MMAP_DEFAULT[args.mode] if args.mmap is None else args.mmap
        sh = fn(args.cache_dir, tags[0], cell, taps, workers=args.workers, mmap=use_mmap)
        os.makedirs(args.shard_dir, exist_ok=True)
        out = f"{args.shard_dir}/{args.mode}_{sh['name']}.json"
        with open(out, "w") as f:
            json.dump(sh, f, indent=2)
        print(f"wrote {out}", flush=True)
        # Phase totals are per-PROCESS: with --workers>1 the children's timers die with them,
        # so this table is the parent's view (load + merge) only. Profile with --workers 1.
        _phase_report(f"{args.mode} {sh['name']} workers={args.workers} "
                      f"mmap={use_mmap} wall={(time.perf_counter() - t0) / 60:.1f} min")
        return

    res = _merge(tags, args.shard_dir) if args.mode == "merge" else _compute_all(
        args.cache_dir, tags,
        tuple(t for t in taps if t in ELEC_TAPS),
        tuple(t for t in taps if t not in ELEC_TAPS))
    _report(tags, res)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {args.out}\nMERGE_DONE", flush=True)

if __name__ == "__main__":
    main()
