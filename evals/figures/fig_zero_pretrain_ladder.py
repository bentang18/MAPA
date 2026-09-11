"""Plot the cross-subject depth results for subjects 7 and 10, excluded from pretraining.

The solid curves average the two subjects. Faint curves show each subject. This
comparison is descriptive; no significance test is run on the two subjects.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evals.neuroprobe.paper_results import ARMS, LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rows = json.loads(args.src.read_text())["heldout"]
    depths = (0, 3, 6, 9, 12)
    fig, ax = plt.subplots(figsize=(5, 4))
    for arm, label, color in zip(ARMS, LABELS, ("#1f4e79", "#6f9dc9", "#c17c48", "#888888")):
        values = []
        for subject in (7, 10):
            curve = []
            for depth in depths:
                row, = [r for r in rows if r["arm"] == arm and
                        r["subject"] == subject and r["depth"] == depth]
                curve.append(row["auroc"])
            values.append(curve)
            ax.plot(depths, curve, color=color, alpha=.2, linewidth=.8)
        ax.plot(depths, np.mean(values, axis=0), "o-", color=color, label=label)
    ax.set(xlabel="Encoder depth", ylabel="Macro AUROC", xticks=depths)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
