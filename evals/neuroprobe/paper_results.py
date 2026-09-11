"""Reproduce the paper's ablation statistics and held-out-subject depth results.

Inputs are the complete frozen-readout shards for all four released encoders. The
statistical unit is the subject. Tests compare each model with the shared frontend
baseline, one-sided, with BH correction over the four models within each regime.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import false_discovery_control, permutation_test

from evals.neuroprobe.build_submission import SPLIT_SPEC, read_shards, validate_grid

ARMS = ("both", "no_region", "no_relpos", "no_priors")
LABELS = ("Both encodings", "Relative position", "Region embedding", "Neither encoding")


def subject_means(grid):
    groups = {}
    for (_, session), value in grid.items():
        subject = int(session.removeprefix("btbank").split("_")[0])
        groups.setdefault(subject, []).append(value)
    return {s: float(np.mean(v)) for s, v in sorted(groups.items())}


def paired_p(model, baseline):
    def statistic(x, y, axis=-1):
        return np.mean(x - y, axis=axis)
    return float(permutation_test((model, baseline), statistic, permutation_type="samples",
                                 alternative="greater", n_resamples=np.inf,
                                 vectorized=True).pvalue)


def load(directory, split, tap):
    tag, grid, cells = read_shards(directory, SPLIT_SPEC[split]["regime"], tap + "|std")
    validate_grid(split, tag, grid, cells)
    return tag, subject_means(grid)


def summarize(directories):
    rows, heldout, provenance = [], [], {}
    for split, spec in SPLIT_SPEC.items():
        tap = spec["tap"].split("|")[0]
        _, floor = load(directories["both"], split, tap.replace("12", "0"))
        regime_rows = []
        for arm, label in zip(ARMS, LABELS):
            tag, scores = load(directories[arm], split, tap)
            previous = provenance.setdefault(arm, {"tag": tag, "shards": str(directories[arm])})
            if previous["tag"] != tag:
                raise ValueError(f"{arm}: checkpoint tag differs across regimes")
            subjects = list(floor)
            baseline = np.array([floor[s] for s in subjects])
            model = np.array([scores[s] for s in subjects])
            gain = model - baseline
            regime_rows.append(dict(regime=spec["regime"], arm=arm, label=label,
                                    frontend=float(baseline.mean()), model=float(model.mean()),
                                    gain=float(gain.mean()), subjects_improved=int((gain > 0).sum()),
                                    n_subjects=len(subjects), p=paired_p(model, baseline),
                                    subject_gain=dict(zip(subjects, gain.tolist()))))
        q = false_discovery_control([r["p"] for r in regime_rows], method="bh")
        for row, value in zip(regime_rows, q):
            row["q"] = float(value)
        rows.extend(regime_rows)
    for arm in ARMS:
        for depth in (0, 3, 6, 9, 12):
            # The input representation is shared; some ablation caches omit its duplicate.
            source = directories["both"] if depth == 0 else directories[arm]
            tag, scores = load(source, "Cross-Subject", f"enc{depth}")
            if depth and tag != provenance[arm]["tag"]:
                raise ValueError(f"{arm}: checkpoint tag differs across depths")
            for subject in (7, 10):
                heldout.append(dict(arm=arm, depth=depth, subject=subject, auroc=scores[subject]))
    return dict(provenance=provenance, ablations=rows, heldout=heldout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument('--' + arm.replace('_', '-'), required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = summarize({arm: getattr(args, arm) for arm in ARMS})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    for row in result['ablations']:
        print(f"{row['regime']:8} {row['label']:18} {row['subjects_improved']}/{row['n_subjects']} "
              f"gain={row['gain']:+.4f} p={row['p']:.4f} q={row['q']:.4f}")
    print(f"Wrote {args.out}; held-out-subject results are descriptive (two subjects).")


if __name__ == '__main__':
    main()
