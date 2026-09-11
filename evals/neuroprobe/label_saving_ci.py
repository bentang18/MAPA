"""Bootstrap CI and decision map for the LABEL-SAVING RATIO (the "k x fewer trials" number).

Every figure that draws the saving imports `_reach` from `sample_curve` and prints a bare
point estimate. A point estimate is not enough to license the sentence the abstract wants to make,
because the sentence is an INEQUALITY ("at least k x fewer labels") and an inequality needs a
sampling distribution, not a mean. This script supplies that distribution and nothing else: it does
not redraw, refit, or re-aggregate anything.

═══ WHAT IS RESAMPLED, AND WHY IT IS SUBJECTS ═══════════════════════════════════════════════════
The claim generalizes over PATIENTS, so the bootstrap resamples patients, via the same `_subject`
map the gain-law bootstrap already uses. Two sessions of one patient are one draw, not two. Cells
carry their subject's multiplicity, which reproduces the cell pooling `_curve` performs.

═══ THE AGGREGATION IS IMPORTED, NEVER REIMPLEMENTED ════════════════════════════════════════════
`_reach` is imported. The panel below is a fast equivalent of `_curve` (cell means, then the mean
over cells, then the mean over tasks) and `--check` asserts it reproduces `_curve` exactly on the
all-subjects draw. If that assertion ever fails, the panel is wrong and no number here is valid.

═══ THE DECISION RULE, FIXED BEFORE THE NUMBERS ═════════════════════════════════════════════════
An inequality "at least k x" is QUOTABLE iff it holds in at least 95% of draws. The point estimate
is reported separately and is quotable as a point estimate regardless, because it is the mean of
the sampling distribution and no subject is excluded from it.

═══ THE TWO WAYS TO GET A WRONG ANSWER HERE, BOTH HANDLED ═══════════════════════════════════════
1. A draw whose pretrained curve NEVER reaches the baseline's full-data accuracy has an UNDEFINED
   ratio. Dropping those draws computes the saving only over resamples where the method works,
   which inflates every probability in the map. They are counted as FAILURES of the inequality and
   their share is printed. The percentile CI is necessarily over defined draws only, and prints how
   many it dropped, so the two are never confused.
2. A draw that already clears the target at the SMALLEST grid point is CENSORED: the true reach is
   somewhere below the grid, so the ratio is a lower bound and the upper tail is truncated. The
   share of censored draws is printed. Censoring biases the map DOWNWARD, so it is conservative for
   the inequality and must not be "corrected" away.

`n_full` is held FIXED at the full-sample median by default: it is the number of trials a
BrainTreebank session contains, a property of the benchmark rather than of our estimate, and
resampling it injects dataset-size variance into a quantity that is known exactly. `--resample-nfull`
recomputes it per draw so the choice is visible instead of assumed; both are printed side by side.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from evals.neuroprobe.sample_curve import (  # noqa: E402
    CURVE_TAPS,
    FULL,
    _curve,
    _reach,
    _subject,
    _x_order,
    validate_curve,
)

COL = "trainonly"          # the only column all three regimes share
K_GRID = (5, 10, 20, 30, 50)
QUOTABLE_AT = 0.95


def _panel_with_full(pts, tap0, tap12, col):
    """`_panel`, except the FULL anchor is KEPT.

    The gain-law panel drops `n_is_full` because a regression over the N sweep must not include the
    anchor. The reach statistic needs the opposite: the anchor IS the target the sweep is measured
    against. Everything else matches, so a subject drawn twice doubles its cells' sum and count.
    """
    cellmean: dict = {}
    for p in pts:
        if p["col"] != col or p["tap"] not in (tap0, tap12):
            continue
        cellmean.setdefault(
            (_subject(p["cell"]), p["n_bucket"], p["tap"], p["task"], p["cell"]), []
        ).append(p["test"])

    subs = sorted({k[0] for k in cellmean})
    ns = sorted({k[1] for k in cellmean}, key=_x_order)
    tasks = sorted({k[3] for k in cellmean})
    si = {s: i for i, s in enumerate(subs)}
    ni = {n: i for i, n in enumerate(ns)}
    ti = {t: i for i, t in enumerate(tasks)}
    tot = np.zeros((len(subs), len(ns), 2, len(tasks)))
    cnt = np.zeros_like(tot)
    for (s, n, tap, task, _cell), vals in cellmean.items():
        v = np.nanmean(vals)
        if not np.isfinite(v):
            continue
        idx = (si[s], ni[n], 0 if tap == tap0 else 1, ti[task])
        tot[idx] += v
        cnt[idx] += 1

    # full-trial count per subject, for the optional per-draw n_full
    nfull_by_sub: dict = {}
    for p in pts:
        if p["col"] == col and p["tap"] == tap0 and p["n_is_full"]:
            nfull_by_sub.setdefault(_subject(p["cell"]), []).append(float(p["n"]))
    nfull_vec = np.array([np.median(nfull_by_sub.get(s, [np.nan])) for s in subs])

    return tot, cnt, subs, ns, nfull_vec


def _macro(tot, cnt, draw):
    """[N, 2] macro AUROC for a subject draw carrying multiplicity."""
    w = np.bincount(np.asarray(draw, dtype=np.int64), minlength=tot.shape[0]).astype(float)
    t = np.tensordot(w, tot, axes=(0, 0))
    c = np.tensordot(w, cnt, axes=(0, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        per_task = np.where(c > 0, t / np.where(c > 0, c, 1), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(per_task, axis=2)


def _ratio(macro, ns, n_full):
    """(ratio, reach, censored). ratio is None when the curve never reaches the target."""
    c0 = {n: macro[i, 0] for i, n in enumerate(ns)}
    c12 = {n: macro[i, 1] for i, n in enumerate(ns) if n != FULL}
    target = c0.get(FULL, np.nan)
    if not np.isfinite(target) or not c12:
        return None, None, False
    grid = sorted(c12, key=_x_order)
    reach = _reach({n: c12[n] for n in grid if np.isfinite(c12[n])}, target)
    if reach is None:
        return None, None, False
    censored = bool(reach <= float(grid[0]))
    return float(n_full) / reach, reach, censored


def run(src, regime, nboot, seed, resample_nfull, check, per_subject):
    m = json.load(open(src))
    validate_curve(m, regime)
    pts = m["points"]

    tap0, tap12 = CURVE_TAPS[regime]
    tot, cnt, subs, ns, nfull_vec = _panel_with_full(pts, tap0, tap12, COL)
    n_full_all = float(np.median([p["n"] for p in pts if p["tap"] == tap0 and p["n_is_full"]]))
    allsub = np.arange(len(subs))

    point, reach_hat, cens_hat = _ratio(_macro(tot, cnt, allsub), ns, n_full_all)
    if point is None:
        raise SystemExit(f"the point estimate itself never reaches the target in {regime} -- "
                         f"there is no saving to put a CI on")

    if check:
        # THE ANCHOR: the fast panel must reproduce the authoritative aggregation exactly.
        assert reach_hat is not None
        c0_ref, c12_ref = _curve(pts, tap0, COL), _curve(pts, tap12, COL)
        ref_reach = _reach({k: v for k, v in c12_ref.items() if k != FULL}, c0_ref[FULL])
        assert ref_reach is not None and abs(ref_reach - reach_hat) < 1e-9, (
            f"PANEL DISAGREES WITH _curve: reach {reach_hat} vs {ref_reach}")
        mac = _macro(tot, cnt, allsub)
        for i, n in enumerate(ns):
            assert abs(mac[i, 0] - c0_ref[n]) < 1e-12 and abs(mac[i, 1] - c12_ref[n]) < 1e-12, (
                f"PANEL DISAGREES WITH _curve at N={n}")
        print(f"  [check] panel reproduces _curve exactly at all {len(ns)} grid points")

    rng = np.random.default_rng(seed)
    ratios, n_never, n_cens = [], 0, 0
    for _ in range(nboot):
        draw = rng.integers(0, len(subs), len(subs))
        nf = float(np.median(nfull_vec[draw])) if resample_nfull else n_full_all
        r, _, cens = _ratio(_macro(tot, cnt, draw), ns, nf)
        if r is None:
            n_never += 1
        else:
            ratios.append(r)
            n_cens += int(cens)
    R = np.asarray(ratios)

    print("\n" + "=" * 92)
    print(f"LABEL-SAVING RATIO -- BOOTSTRAP OVER SUBJECTS   [{regime.upper()}]  col={COL}")
    print(f"  source     {src}")
    print(f"  subjects   {len(subs)}  ({', '.join(subs)})")
    print(f"  n_full     {n_full_all:.0f} trials" +
          ("  [RESAMPLED per draw]" if resample_nfull else "  [fixed: a protocol constant]"))
    print(f"  draws      {nboot}  seed={seed}")
    print("-" * 92)
    print(f"  POINT ESTIMATE   {point:.1f}x   (reach {reach_hat:.0f} vs {n_full_all:.0f} trials)"
          + ("   [CENSORED at the grid floor -- a LOWER BOUND]" if cens_hat else ""))
    if len(R):
        lo, hi = np.percentile(R, [2.5, 97.5])
        print(f"  95% CI           [{lo:.1f}, {hi:.1f}]   median {np.median(R):.1f}x"
              f"   (over {len(R)} defined draws, {n_never} dropped)")
    print(f"  never reaches    {n_never}/{nboot} draws ({100.0 * n_never / nboot:.1f}%)"
          "   <- counted as FAILURES below, never dropped")
    print(f"  censored at floor{n_cens:>4}/{max(len(R), 1)} defined draws "
          f"({100.0 * n_cens / max(len(R), 1):.1f}%)   <- ratio is a LOWER bound, map is conservative")
    print("-" * 92)
    print(f"  DECISION MAP -- an inequality is QUOTABLE iff it holds in >= {QUOTABLE_AT:.0%} of draws")
    best = None
    for k in K_GRID:
        p = float((R >= k).sum()) / nboot          # denominator is ALL draws
        ok = p >= QUOTABLE_AT
        if ok:
            best = k
        print(f"     P(ratio >= {k:>3}x) = {p:6.1%}   {'QUOTABLE' if ok else 'not quotable'}")
    print("-" * 92)
    if best is None:
        print(f"  => NO inequality on the grid clears {QUOTABLE_AT:.0%}. Quote the point estimate "
              f"({point:.0f}x) and the CI, never 'at least k x'.")
    else:
        print(f"  => SAFE INEQUALITY: 'at least {best}x'. The point estimate {point:.0f}x remains "
              f"quotable as a point estimate.")
    print("=" * 92)

    per_sub = {}
    if per_subject:
        # THE APPENDIX NUMBER. The headline crosses the POOLED curve; this crosses each subject's
        # own curve against that subject's own baseline-at-full. A subject that never reaches is
        # reported as such and is NEVER recoded to 1x -- "the saving is undefined here" and "the
        # saving is nil here" are different results, and only the first is true.
        print(f"  PER-SUBJECT REACH [{regime.upper()}] -- each subject against its OWN full-data "
              f"baseline\n" + "-" * 92)
        for i, s in enumerate(subs):
            r, rc, cens = _ratio(_macro(tot, cnt, np.array([i])), ns,
                                 nfull_vec[i] if resample_nfull else n_full_all)
            per_sub[s] = None if r is None else float(r)
            if r is None:
                print(f"     {s:>4}   NEVER REACHES its own full-data baseline")
            else:
                print(f"     {s:>4}   {r:5.1f}x   (reach {rc:.0f})"
                      + ("   [censored: lower bound]" if cens else ""))
        got = [v for v in per_sub.values() if v is not None]
        print("-" * 92)
        print(f"  {len(got)}/{len(subs)} subjects reach.  spread {min(got):.0f}x-{max(got):.0f}x"
              if got else "  NO subject reaches individually")
        print("=" * 92)

    return {"regime": regime, "per_subject": per_sub,
            "point": point, "reach": reach_hat, "n_full": n_full_all,
            "subjects": subs, "n_boot": nboot, "seed": seed, "n_defined": int(len(R)),
            "n_never": n_never, "n_censored": n_cens, "censored_point": cens_hat,
            "resample_nfull": bool(resample_nfull),
            "ci": [float(np.percentile(R, 2.5)), float(np.percentile(R, 97.5))] if len(R) else None,
            "median": float(np.median(R)) if len(R) else None,
            "map": {int(k): float((R >= k).sum()) / nboot for k in K_GRID},
            "quotable": best}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, nargs="+",
                    help="merged samplecurve JSON(s); regime is read from each file")
    ap.add_argument("--nboot", type=int, default=20000,
                    help="the decision map is the deliverable and 20k costs ~2 s; at 2k the "
                         "map wobbles by ~1 point, which is enough to move a borderline call")
    ap.add_argument("--seed", type=int, default=0, help="matches the gain-law bootstrap")
    ap.add_argument("--resample-nfull", action="store_true",
                    help="recompute n_full per draw instead of holding it at the full-sample median")
    ap.add_argument("--no-check", action="store_true", help="skip the _curve equivalence assertion")
    ap.add_argument("--per-subject", action="store_true",
                    help="also cross each subject's own curve (the appendix number)")
    ap.add_argument("--out", default=None, help="write the results as JSON")
    a = ap.parse_args()

    out = []
    for src in a.src:
        regime = json.load(open(src)).get("regime", "ws")
        out.append(run(src, regime, a.nboot, a.seed, a.resample_nfull,
                       not a.no_check, a.per_subject))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
