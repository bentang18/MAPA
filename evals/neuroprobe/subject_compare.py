"""Board arm-vs-arm at the SUBJECT unit, computed DIRECTLY FROM THE SHARDS.

Two standing rules meet here:
  - "NO ARM COMPARISON UNLESS COMPUTED FROM SHARDS IN THE CURRENT TURN" (never a merged JSON,
    whose provenance and arm mixing cannot be checked by looking at it).
  - The unit of analysis is the SUBJECT (settled 2026-08-16). board_arm_compare.py signs over
    (cell, task) -- n=180/180/150 -- which is the BANNED unit.

Shard layout, read rather than assumed:
  {mode}_{cell}.json -> {kind, name, cells: {"<tag>|<task>": {cells: {"<tap>|<norm>": {test: auroc}}}}}

Cohort is derived from the shard filenames, never hardcoded, so a partial grid is visible instead
of being silently averaged over.

usage: shard_subject_compare.py <BASE_SHARD_DIR> <ARM_SHARD_DIR>
"""
import argparse
import json
import math
import os
import re
from collections import OrderedDict

PUBLISHED = OrderedDict(
    ws=("enc12_elec|std", "enc0_elec|std"),
    csession=("enc12_elec|std", "enc0_elec|std"),
    cs=("enc12|std", "enc0|std"),
)
SHARD_RE = re.compile(r"^(ws|csession|cs)_(S\d+T\d+)\.json$")


def read_shards(shard_dir, regime):
    """-> (tag, {(task, cell): {tap: auroc}})"""
    out, tags = {}, set()
    for fn in sorted(os.listdir(shard_dir)):
        m = SHARD_RE.match(fn)
        if not m or m.group(1) != regime:
            continue
        cell = m.group(2)
        with open(os.path.join(shard_dir, fn)) as f:
            d = json.load(f)
        for tagtask, blk in d.get("cells", {}).items():
            tag, task = tagtask.split("|", 1)
            tags.add(tag)
            for tapnorm, leaf in blk.get("cells", {}).items():
                v = leaf.get("test")
                if v is not None:
                    out.setdefault((task, cell), {})[tapnorm] = float(v)
    if len(tags) > 1:
        raise SystemExit(f"[FATAL] ARM MIXING: {shard_dir} holds tags {sorted(tags)}")
    return (tags.pop() if tags else "?"), out


def sign_p(deltas, eps=1e-12):
    nz = [x for x in deltas if abs(x) > eps]
    n = len(nz)
    if n == 0:
        return float("nan"), 0, 0, float("nan")
    npos = sum(1 for x in nz if x > 0)
    k = min(npos, n - npos)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return p, npos, n, min(1.0, 2.0 / 2 ** n)


def pick(grid, tap):
    return {k: v[tap] for k, v in grid.items() if tap in v}


def by_subject(g):
    acc = {}
    for (_, cell), v in g.items():
        acc.setdefault(cell.split("T")[0], []).append(v)
    return {s: sum(v) / len(v) for s, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_dir")
    ap.add_argument("arm_dir")
    a = ap.parse_args()

    print("=" * 94)
    print("COMPUTED FROM SHARDS IN THIS RUN. Unit = SUBJECT. (cell,task) is BANNED as a claim.")
    print("FLOOR = 2/2^n, the smallest p an exact paired test can return at that n.")
    print("=" * 94)

    for regime, (tap, floor_tap) in PUBLISHED.items():
        tb, gb_all = read_shards(a.base_dir, regime)
        ta, ga_all = read_shards(a.arm_dir, regime)
        gb, ga = pick(gb_all, tap), pick(ga_all, tap)
        keys = sorted(set(gb) & set(ga))
        if not keys:
            print(f"\n[{regime.upper()}] no shared cells at {tap}")
            continue
        cells = sorted({c for _, c in keys})
        tasks = sorted({t for t, _ in keys})
        mb = sum(gb[k] for k in keys) / len(keys)
        ma = sum(ga[k] for k in keys) / len(keys)
        sb, sa = by_subject({k: gb[k] for k in keys}), by_subject({k: ga[k] for k in keys})
        subs = sorted(set(sb) & set(sa), key=lambda s: int(s[1:]))
        p, npos, n, fl = sign_p([sa[s] - sb[s] for s in subs])
        verdict = ("CANNOT REACH .05 AT THIS n" if fl > .05
                   else ("significant" if p < .05 else "not significant"))
        print(f"\n[{regime.upper():<8}] {tb} -> {ta}   tap={tap}")
        print(f"  grid   {len(cells)} cells x {len(tasks)} tasks = {len(keys)} pairs")
        print(f"  macro  base {mb:.4f}  arm {ma:.4f}  delta {ma - mb:+.4f}")
        print("  subj   " + "  ".join(f"{s}{sa[s] - sb[s]:+.4f}" for s in subs))
        print(f"  SIGN   n={n} arm-higher {npos}/{n} p={p:.4f} floor={fl:.4f} -> {verdict}")

        fb, fa = pick(gb_all, floor_tap), pick(ga_all, floor_tap)
        fk = sorted(set(fb) & set(fa))
        if not fk:
            print(f"  FLOOR  {floor_tap} SKIPPED (absent) -- control unavailable, NOT passed")
            continue
        drift = [fa[k] - fb[k] for k in fk]
        worst = max(abs(x) for x in drift)
        ties = sum(1 for x in drift if x == 0.0)
        eff = abs(ma - mb)
        ok = "OK" if (worst == 0 or (sign_p(drift)[0] > .05 and eff / worst >= 10)) else "FATAL"
        print(f"  FLOOR  {floor_tap} ties {ties}/{len(fk)} max|drift| {worst:.2e} vs effect {eff:.2e}  {ok}")


if __name__ == "__main__":
    main()
