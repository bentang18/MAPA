"""Plot spatial-encoding ablations using the paper's subject-level permutation tests.

Create the input with evals.neuroprobe.paper_results. Stars mark q < .05 after
BH correction over the four models within each regime.
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
    rows = json.loads(args.src.read_text())["ablations"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    for ax, regime, title in zip(axes, ("ws", "csession", "cs"),
                                ("Within-session", "Cross-session", "Cross-subject")):
        for x, arm in enumerate(ARMS):
            row, = [r for r in rows if r["regime"] == regime and r["arm"] == arm]
            values = list(row["subject_gain"].values())
            ax.bar(x, row["gain"], color="#1f4e79" if x == 0 else "#bbc5cf", width=.65)
            ax.scatter(x + np.linspace(-.13, .13, len(values)), values, s=16, color="#333", zorder=3)
            mark = "*" if row["q"] < .05 else ""
            ax.text(x, max(0, max(values)) + .002,
                    f"{row['subjects_improved']}/{row['n_subjects']}{mark}", ha="center", fontsize=9)
        ax.axhline(0, color="#777", linewidth=.8)
        ax.set_xticks(range(4), LABELS, rotation=25, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=.18)
    axes[0].set_ylabel("Macro AUROC gain over frontend baseline")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
