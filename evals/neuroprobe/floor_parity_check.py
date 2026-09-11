#!/usr/bin/env python
"""Does a changed encode path still produce the PUBLISHED enc0 floor?

Any change to the encode path is only interpretable if the same script, run on the same caches,
still reproduces the enc0 the board numbers were computed from. enc0 touches no weights, so the
comparison should be BIT-exact whenever both sides were encoded on the same architecture, with no
cross-architecture float reassociation to absorb. --rtol survives as a floor for a reference that
predates that, but a merely-within-tolerance result is itself a finding and the summary line
reports the bit-exact count separately.

Takes two ENCODE CACHE directories, not readout shards: it reads enc_s<S>_t<T>_<tag>.pt as
written by encode.py, and skips any session missing on either side.

Compares the taps AND the metadata a consumer slices them with -- a matching tap under a
different band layout, row count or label vector would still be a different experiment.

usage:
  python -m evals.neuroprobe.floor_parity_check \
      --published-dir <encode cache dir> --published-tag <tag> \
      --new-dir <encode cache dir> --new-tag <tag>
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

LITE_SESSIONS = ((1, 1), (1, 2), (2, 0), (2, 4), (3, 0), (3, 1),
                 (4, 0), (4, 1), (7, 0), (7, 1), (10, 0), (10, 1))
TAPS = ("enc0", "enc0_elec")


def _same(x, y) -> bool:
    """Structural equality for the record's non-tap fields.

    These are the fields the READOUT slices with, and they are not all arrays: `labels` and the
    splits are dicts keyed by task/fold. A shallow `!=` on a dict of arrays raises rather than
    answering, and `np.array_equal` on a dict silently returns False, so both failure modes would
    read as 'differs' and neither would be true. Recurse instead."""
    if isinstance(x, dict) or isinstance(y, dict):
        if not (isinstance(x, dict) and isinstance(y, dict)) or set(x) != set(y):
            return False
        return all(_same(x[k], y[k]) for k in x)
    if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
        return (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
                and x.shape == y.shape and bool(torch.equal(x, y)))
    xa, ya = np.asarray(x), np.asarray(y)
    if xa.dtype.kind in "fc" or ya.dtype.kind in "fc":
        # equal_nan is REQUIRED, not defensive. `labels` is NaN-padded: a task's label is NaN on
        # every window the task is undefined for, which is 5.5k-7.5k of 9063 on s4_t0. Plain
        # allclose is False whenever a NaN is present, so without this the gate reports every
        # session's labels as differing while max|d| over the defined entries is exactly 0 — a
        # false FAIL that hides any real one.
        return xa.shape == ya.shape and bool(np.allclose(xa, ya, equal_nan=True))
    return bool(np.array_equal(xa, ya))


def _cmp(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, bool]:
    """max|Δ|, max|Δ| relative to the reference's scale, and bit-equality."""
    exact = a.shape == b.shape and torch.equal(a, b)
    x, y = a.to(torch.float64), b.to(torch.float64)
    d = float((x - y).abs().max())
    return d, d / (float(x.abs().max()) + 1e-12), exact


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--published-dir", required=True)
    p.add_argument("--published-tag", required=True)
    p.add_argument("--new-dir", required=True)
    p.add_argument("--new-tag", required=True)
    p.add_argument("--rtol", type=float, default=1e-3,
                   help="max tolerated relative deviation; fp16 taps carry ~1e-3 of resolution "
                        "at full scale, so anything above this is a real path change, not "
                        "float reassociation")
    a = p.parse_args()

    worst, n_seen, n_exact, failures = 0.0, 0, 0, []
    for s, t in LITE_SESSIONS:
        pub = os.path.join(a.published_dir, f"enc_s{s}_t{t}_{a.published_tag}.pt")
        new = os.path.join(a.new_dir, f"enc_s{s}_t{t}_{a.new_tag}.pt")
        if not (os.path.exists(pub) and os.path.exists(new)):
            print(f"[skip] s{s}_t{t}: pub={os.path.exists(pub)} new={os.path.exists(new)}")
            continue
        # mmap: the reference may be a FULL board record (44-71 GB, ~99% encoder taps this never
        # reads). An eager load faults all of it in and OOMs — it killed an eval job at 24G.
        P = torch.load(pub, map_location="cpu", weights_only=False, mmap=True)
        N = torch.load(new, map_location="cpu", weights_only=False, mmap=True)
        n_seen += 1

        # Metadata first: a matching tap under a different layout is a different experiment.
        for k in ("band_lengths", "band_fdims", "n_windows"):
            pv, nv = tuple(np.ravel(P[k]).tolist()), tuple(np.ravel(N[k]).tolist())
            if pv != nv:
                failures.append(f"s{s}_t{t} {k}: published {pv} != new {nv}")
        pl, nl = np.asarray(P["present_parcels"]), np.asarray(N["present_parcels"])
        if not np.array_equal(pl, nl):
            failures.append(f"s{s}_t{t} present_regions differ ({len(pl)} vs {len(nl)})")
        if not np.allclose(np.asarray(P["clip_starts"]), np.asarray(N["clip_starts"])):
            failures.append(f"s{s}_t{t} clip_starts differ — different windows entirely")
        # Everything else the READOUT slices with. Bit-exact taps under a different label vector
        # or a different fold assignment would score a different experiment at full confidence,
        # which is the failure this gate exists to prevent. ckpt_tag is excluded deliberately: it
        # is the arm's NAME and is supposed to differ.
        for k in ("parcel_canon", "labels", "ws_split", "cs_split"):
            if (k in P) != (k in N):
                failures.append(f"s{s}_t{t} {k}: present in pub={k in P} new={k in N}")
            elif k in P and not _same(P[k], N[k]):
                failures.append(f"s{s}_t{t} {k} differs — same features, different experiment")

        for tap in TAPS:
            if tap not in P["feats"] or tap not in N["feats"]:
                failures.append(f"s{s}_t{t} {tap}: missing (pub={tap in P['feats']} "
                                f"new={tap in N['feats']})")
                continue
            x, y = P["feats"][tap]["raw"], N["feats"][tap]["raw"]
            if x.shape != y.shape:
                failures.append(f"s{s}_t{t} {tap}: shape {tuple(x.shape)} != {tuple(y.shape)}")
                continue
            d, rel, exact = _cmp(x, y)
            n_exact += int(exact)
            worst = max(worst, rel)
            flag = "BIT-EXACT" if exact else ("ok" if rel <= a.rtol else "DIVERGENT")
            print(f"[parity] s{s}_t{t} {tap:10s} shape={tuple(x.shape)} max|d|={d:.4g} "
                  f"rel={rel:.3g} -> {flag}", flush=True)
            if rel > a.rtol:
                failures.append(f"s{s}_t{t} {tap}: rel {rel:.3g} > rtol {a.rtol}")
        del P, N

    print(f"\n[summary] sessions={n_seen}/{len(LITE_SESSIONS)} taps_bit_exact={n_exact}/"
          f"{n_seen * len(TAPS)} worst_rel={worst:.3g} rtol={a.rtol}")
    if n_seen != len(LITE_SESSIONS):
        failures.append(f"only {n_seen}/{len(LITE_SESSIONS)} sessions compared")
    if failures:
        print("\n[FAIL] the new code path does NOT reproduce the published enc0:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("[PASS] new code path reproduces the published enc0 on every session and tap.")


if __name__ == "__main__":
    main()
