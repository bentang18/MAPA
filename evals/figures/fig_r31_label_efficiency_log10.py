"""R31 label efficiency, redrawn on a log10 / real-trial-count axis  at d=384.

Two changes from `fig_r31_label_efficiency_ladder.py`, and one addition.

1. X AXIS IS LOG10 IN REAL TRIALS, NOT log2 tick labels. The rungs are still powers of two because
   the grid is, but they are labelled with the trial count itself and a decade grid, and a second
   axis carries the same quantity in minutes of recording (a trial is exactly 1 s: Neuroprobe sets
   START_NEURAL_DATA_BEFORE_WORD_ONSET=0, END_NEURAL_DATA_AFTER_WORD_ONSET=1).

2. `pdf.fonttype=42`. The ladder figure never set it, so its PDF ships Type 3 fonts.

3. Panel B is now the GAP AGAINST N, which is the coordinate the additivity claim actually lives
   in, and it is annotated with the identity that explains the whole figure:

       ratio = 2 ** (a / s)

   with `a` the flat vertical gap and `s` the BASELINE's slope per doubling. A label-efficiency
   ratio is not a property of the gain alone. It is the gain measured in units of how fast the
   baseline climbs, so a flat baseline converts a small gain into a large ratio. Panel B prints
   both terms so the ratio in panel A can never be read as if it were the gain.

4. ALL THREE REGIMES. csession is the row that makes the other two mean something. ws and cs do
   not share a feature tap (per-electrode vs region-mean, the board's per-regime contract), so the
   ws-vs-cs contrast confounds the subject boundary with the tap. csession reads the SAME
   per-electrode tap as ws and differs only in where the training trials come from, so ws→csession
   is a one-variable step and csession→cs is the step that still carries the tap change.

⚠️ COLUMN IS `trainonly`, the only column all regimes share. Never quote against a `both` number.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # NeurIPS rejects Type 3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from evals.neuroprobe.sample_curve import CURVE_TAPS, FULL, _curve, _reach, validate_curve

GREY, ACC = "#8a8f98", "#d98324"
COL = "trainonly"

ROWS = ("ws", "csession", "cs")
SPEC = {
    "ws": dict(units=180, colour="#1f4e79", letter="A", short="within a session",
               xlabel="target-session calibration trials",
               slide_xlabel="labelled trials from the target session"),
    "csession": dict(units=180, colour="#2f6f4f", letter="B",
                     short="across sessions, same patient",
                     xlabel="OTHER-session labelled trials (same patient, another day)",
                     slide_xlabel="labelled trials from another day"),
    "cs": dict(units=150, colour="#7b2d26", letter="C", short="across patients",
               xlabel="DONOR-session labelled trials (a different patient)",
               slide_xlabel="labelled trials from a different patient"),
}


def load(src, regime, want_units):
    m = json.load(open(src))
    validate_curve(m, regime)
    pts = m["points"]
    worst = max((a for a in m["anchor"] if a["col"] == COL), key=lambda a: a["absdiff"])
    cells = sorted({p["cell"] for p in pts})
    tasks = sorted({p["task"] for p in pts})
    units = len(cells) * len(tasks)
    assert units == want_units, f"{regime}: expected {want_units} units, got {units}"
    tap0, tap12 = CURVE_TAPS[regime]
    c0, c12 = _curve(pts, tap0, COL), _curve(pts, tap12, COL)
    n_full = int(np.median([p["n"] for p in pts if p["tap"] == tap0 and p["n_is_full"]]))
    return dict(c0=c0, c12=c12, n_full=n_full, cells=cells, tasks=tasks, units=units,
                anchor=worst["absdiff"])


def mechanism(d):
    """(a, s, ratio_measured, ratio_predicted). `s` is measured over the band the crossing spans,
    because that is the stretch of baseline the saving is actually being read against."""
    grid = [n for n in d["c0"] if n != FULL]
    target, n_full = d["c0"][FULL], d["n_full"]
    reach = _reach({k: v for k, v in d["c12"].items() if k != FULL}, target)
    a = float(np.mean([d["c12"][n] - d["c0"][n] for n in grid if n in d["c12"]]))
    lo = max([n for n in grid if n <= reach], default=grid[0])
    s = (target - d["c0"][lo]) / np.log2(n_full / lo)
    return a, s, n_full / reach, 2.0 ** (a / s), reach


def panel_curves(ax, d, regime, xlim, slides=False, ylabel=True, legend=True):
    """`slides` = the 1x3 talk layout: bigger marks and type, one y label, one legend.

    Same data path and same annotations as the paper grid. Only sizes and which chrome repeats
    change, so a number can never differ between the two renders.
    """
    sp = SPEC[regime]
    grid = [n for n in d["c0"] if n != FULL]
    target, n_full = d["c0"][FULL], d["n_full"]
    a, s, ratio, _pred, reach = mechanism(d)
    lw, ms, mse = (2.4, 5.2, 8.6) if slides else (1.5, 3.4, 6.0)

    x = np.array(grid + [n_full], float)
    ax.plot(x, [d["c0"][n] for n in grid] + [target], "-o", color=GREY, lw=lw, ms=ms,
            label="untrained |STFT| frontend")
    ax.plot(x, [d["c12"][n] for n in grid] + [d["c12"][FULL]], "-o", color=sp["colour"],
            lw=lw, ms=ms, label="pretrained encoder (frozen)")
    ax.plot([n_full], [target], "o", color=GREY, ms=mse, mfc="white", mew=1.6, zorder=5)
    ax.plot([n_full], [d["c12"][FULL]], "o", color=sp["colour"], ms=mse, mfc="white", mew=1.6,
            zorder=5)
    ax.axhline(target, color=GREY, ls=":", lw=1.1 if slides else 0.9)

    ax.annotate("", xy=(reach, target), xytext=(n_full, target),
                arrowprops=dict(arrowstyle="<->", color=ACC, lw=2.0 if slides else 1.4,
                                shrinkA=0, shrinkB=0))
    ax.annotate(f"{reach:.0f} vs {n_full:,} trials   ({ratio:.1f}$\\times$ fewer)",
                xy=(np.sqrt(reach * n_full), target), xytext=(0, -17 if slides else -15),
                textcoords="offset points", ha="center", color=ACC,
                fontsize=11.0 if slides else 7.8, fontweight="bold",
                # translucent, not opaque: the label must stay readable over a curve without
                # erasing the curve it sits on
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.7))
    if not slides:
        ax.annotate(f"untrained frontend, all {n_full:,} trials", xy=(min(grid), target),
                    xytext=(1, 4), textcoords="offset points", ha="left", va="bottom",
                    color=GREY, fontsize=6.4)

    # SHARED x limits across rows. The rows are being compared on SLOPE, and a slope read off two
    # differently-scaled axes is not a comparison. ws simply ends earlier (1,750 vs 3,500).
    ax.set_xscale("log")
    ticks = [16, 64, 256, 1024, 3500]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:,}" for t in ticks])
    ax.set_xticks(grid, minor=True)
    ax.set_xticklabels([], minor=True)
    ax.set_xlim(*xlim)
    if slides:
        # The talk version says WHAT MOVED, not the panel bookkeeping. Units and tap come out of
        # the title because a slide audience cannot read them and the speaker says the tap aloud.
        ax.set_xlabel(sp["slide_xlabel"], fontsize=11)
        ax.set_title(sp["short"], loc="center", fontsize=14, pad=10)
    else:
        ax.set_xlabel(sp["xlabel"] + "   (log$_{10}$)")
        ax.set_title(f"{sp['letter']}  ·  {sp['short']}  ·  {d['units']} units  ·  "
                     f"tap {CURVE_TAPS[regime][1]}", loc="left")
    if ylabel:
        ax.set_ylabel("board macro AUROC", fontsize=12 if slides else 8)
    if legend:
        ax.legend(loc="lower right", fontsize=10 if slides else 7)

    # Minutes, at round minute values rather than decades -- a decade tick on a minutes axis is
    # exactly as unreadable as the log2 axis this figure exists to replace.
    top = ax.secondary_xaxis("top", functions=(lambda v: v / 60.0, lambda v: v * 60.0))
    mins = [m for m in (0.5, 1, 2, 5, 10, 30, 60) if xlim[0] <= m * 60 <= xlim[1]]
    top.set_xticks(mins)
    top.set_xticklabels([(f"{m:g}" if m >= 1 else f"{m}") for m in mins])
    top.minorticks_off()
    top.set_xlabel("minutes of labelled recording  (1 trial = 1 s)",
                   fontsize=10 if slides else 7, labelpad=4 if slides else 3)
    top.tick_params(labelsize=9.5 if slides else 6.5)


def panel_gap(ax, d, regime, xlim):
    sp = SPEC[regime]
    grid = [n for n in d["c0"] if n != FULL]
    a, s, ratio, pred, _reach_n = mechanism(d)
    gaps = [d["c12"][n] - d["c0"][n] for n in grid]

    ax.axhline(0, color="#ccc", lw=0.8, zorder=0)
    ax.axhline(a, color=ACC, ls="--", lw=1.4, label=f"mean gap  $a$ = {a:+.4f}")
    ax.plot(grid, gaps, "o-", color=sp["colour"], lw=1.2, ms=4.4, zorder=5,
            label="measured gap at each rung")
    ax.plot([d["n_full"]], [d["c12"][FULL] - d["c0"][FULL]], "o", color=sp["colour"],
            ms=6, mfc="white", mew=1.4, zorder=5)

    ax.set_xscale("log")
    ax.set_xticks([16, 64, 256, 1024, 3500])
    ax.set_xticklabels([f"{t:,}" for t in (16, 64, 256, 1024, 3500)])
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 0.045)
    ax.set_xlabel("labelled trials  (log$_{10}$)")
    ax.set_ylabel("gain from pretraining  ($\\Delta$AUROC)")
    # State the spread, do not assert flatness. The cs gap visibly sags at the top rungs and a
    # title that says "flat" would be overruling the panel it labels.
    span = max(grid) / min(grid)
    ax.set_title(f"{sp['letter']}$'$  ·  gain vs labels  ·  "
                 f"{min(gaps):+.4f} to {max(gaps):+.4f} over a {span:.0f}$\\times$ sweep",
                 loc="left")
    ax.legend(loc="lower left", fontsize=7)
    ax.text(0.97, 0.95,
            f"baseline slope  $s$ = {s:+.4f} / doubling\n"
            f"$a/s$ = {a/s:.2f} doublings\n"
            f"$2^{{a/s}}$ = {pred:.1f}$\\times$   (measured {ratio:.1f}$\\times$)",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.9, color="#333",
            bbox=dict(boxstyle="round,pad=0.35", fc="#f6f6f4", ec="#ddd", lw=0.6))


def render_row(D, args, xlim):
    """1x3 curves-only render for a talk.

    The gap panels and the caveat block are DROPPED, not shrunk. A slide that carries the tap
    warning in 7pt is a slide nobody reads the warning on, so the caveat moves to the speaker.
    """
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "axes.titlesize": 14,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "axes.linewidth": 1.0, "xtick.major.width": 1.0,
                         "ytick.major.width": 1.0})
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.0))
    for i, regime in enumerate(ROWS):
        panel_curves(axes[i], D[regime], regime, xlim, slides=True,
                     ylabel=(i == 0), legend=(i == 0))
        axes[i].set_ylim(0.50, 0.705)
        axes[i].axhline(0.5, color="#ccc", lw=1.0, zorder=0)
        if i:                     # one y scale, labelled once
            axes[i].set_yticklabels([])
    fig.tight_layout(w_pad=2.0)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out / f"fig_r31_label_efficiency_row_{args.tag}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print("wrote", p)
    for r in ROWS:
        _a, _s, ratio, _p, reach = mechanism(D[r])
        print(f"  {r:<9} reach {reach:6.0f}  ratio {ratio:5.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", required=True)
    ap.add_argument("--csession", required=True)
    ap.add_argument("--cs", required=True)
    ap.add_argument("--tag", default="iid75_vits384")
    ap.add_argument("--out", default="results/showcase/2_what_pretraining_does")
    ap.add_argument("--layout", choices=("grid", "row"), default="grid",
                    help="grid = the 3x2 paper figure; row = 1x3 curves only, for a talk")
    args = ap.parse_args()

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "legend.frameon": False, "axes.titlesize": 8.5, "axes.labelsize": 8,
    })

    D = {r: load(getattr(args, r), r, SPEC[r]["units"]) for r in ROWS}
    xlim = (min(n for r in D for n in D[r]["c0"] if n != FULL) * 0.8,
            max(D[r]["n_full"] for r in D) * 1.35)

    if args.layout == "row":
        return render_row(D, args, xlim)

    fig, axes = plt.subplots(3, 2, figsize=(9.4, 9.8))
    for row, regime in enumerate(ROWS):
        panel_curves(axes[row][0], D[regime], regime, xlim)
        panel_gap(axes[row][1], D[regime], regime, xlim)
    # y starts at CHANCE. A truncated AUROC axis makes any gap look like whatever the crop chooses,
    # so the axis is anchored to the only value that means something on its own.
    for row in range(len(ROWS)):
        axes[row][0].set_ylim(0.50, 0.705)
        axes[row][0].axhline(0.5, color="#ccc", lw=0.8, zorder=0)

    M = {r: mechanism(D[r]) for r in ROWS}
    (aw, sw, rw, pw, _), (an, sn, rn, pn, _), (ac, sc, rc, pc, _) = (M[r] for r in ROWS)
    fig.suptitle(
        f"R31 label efficiency · {args.tag} · column '{COL}' · anchors exact "
        f"(max |curve − published| = {max(D[r]['anchor'] for r in ROWS):.0e})\n"
        f"the ratio is the gain divided by the baseline's slope, a/s: "
        f"{aw/sw:.2f} doublings within a session, {an/sn:.2f} across sessions, "
        f"{ac/sc:.2f} across patients ({rw:.1f}×, {rn:.1f}×, {rc:.1f}× fewer trials).\n"
        f"⚠ A→B IS THE ONE-VARIABLE STEP: same {CURVE_TAPS['ws'][1]} tap, only the training "
        f"trials move off the target session. B→C ALSO CHANGES THE TAP to region-mean "
        f"(the board's per-regime contract),\n"
        f"so the {rc:.0f}× cannot be attributed to the subject boundary alone. What A→B does "
        f"establish is that the tap is not the whole story: at a FIXED tap the ratio already "
        f"moves {rw:.1f}× → {rn:.1f}×.",
        fontsize=7.0, y=1.003, color="#444")
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out / f"fig_r31_label_efficiency_log10_{args.tag}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print("wrote", p)

    print(f"\n{'regime':<9} {'tap':<11} {'a':>9} {'s/dbl':>10} {'a/s':>6} "
          f"{'2^(a/s)':>9} {'measured':>9}")
    for r in ROWS:
        a, s, ratio, pred, _ = M[r]
        print(f"{r:<9} {CURVE_TAPS[r][1]:<11} {a:+9.4f} {s:+10.5f} {a/s:6.2f} "
              f"{pred:8.2f}x {ratio:8.2f}x")

    # The a-vs-s split of a change in a/s is NOT unique: a/s = a * (1/s), so attributing the
    # difference between two regimes depends on which factor you move first. Both orders are
    # printed, and the headline number is their average (the Shapley value for two players), which
    # is the only split that does not depend on an arbitrary ordering. An earlier read of this
    # quoted one order as if it were the answer.
    for lo, hi, name in (("ws", "csession", "WS→CSESSION"), ("csession", "cs", "CSESSION→CS")):
        al, sl, *_ = M[lo]
        ah, sh, *_ = M[hi]
        extra = ah / sh - al / sl
        gain_first = (ah - al) / sl                 # move `a` first, then `s`
        slope_first = al * (1 / sh - 1 / sl)        # move `s` first, then `a`
        g = 0.5 * (gain_first + (extra - slope_first))
        print(f"\n{name}: {extra:+.2f} doublings "
              f"({'TAP HELD FIXED' if lo == 'ws' else 'TAP ALSO CHANGES — not one variable'})")
        print(f"  larger gain      {g:+.2f}  ({100*g/extra:.0f}%)   "
              f"[order-dependent: {gain_first:+.2f} to {extra-slope_first:+.2f}]")
        print(f"  flatter baseline {extra-g:+.2f}  ({100*(extra-g)/extra:.0f}%)   "
              f"[order-dependent: {extra-gain_first:+.2f} to {slope_first:+.2f}]")


if __name__ == "__main__":
    main()
