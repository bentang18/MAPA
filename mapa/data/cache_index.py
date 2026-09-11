"""Parsers for the cache and label files on disk.

Thin disk-facing helpers the ``load_v3_sessions`` adapter composes into per-session
``V3SessionSpec``s. Pure file-parsing (no braintreebank anatomy calls — those live
in the orchestrator), so unit-testable on synthetic JSON.

  * ``load_bad_window_spans`` / ``index_bad_windows`` — the guard-2 spans from the
    bad-window scan output (``bad_windows_s`` = ``[[start_s, end_s], ...]``), keyed by
    ``(subject_id, trial_id)``.
  * ``index_band_cache`` — scan a band's spec-cache dir, mapping each session's
    ``key`` (from the ``{stem}.json`` sidecar) to its ``.npy`` / ``.stats.npz``
    paths + ``ch_names`` (the npy channel/voltage order) + ``total_frames``.
  * ``parse_lof_report`` — the LOF bad-channel report → guard-1 drop set per
    ``(subject_id, trial_id)`` (optional; absent → no drops).
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass


def load_bad_window_spans(path: str) -> list[tuple[float, float]]:
    """A guard-2 span file's ``bad_windows_s`` as ``[(start_s, end_s), ...]``."""
    with open(path) as fh:
        d = json.load(fh)
    return [(float(a), float(b)) for a, b in d.get("bad_windows_s", [])]


def index_bad_windows(span_dir: str) -> dict[tuple[int, int], list[tuple[float, float]]]:
    """Scan a dir of guard-2 span files → ``(subject_id, trial_id)`` → spans."""
    out: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for path in sorted(glob.glob(os.path.join(span_dir, "*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        key = (int(d["subject_id"]), int(d["trial_id"]))
        out[key] = [(float(a), float(b)) for a, b in d.get("bad_windows_s", [])]
    return out


def parse_session_name(name: str) -> tuple[int, int]:
    """``"btbank6_t4"`` → ``(6, 4)`` (subject_id, trial_id)."""
    m = re.fullmatch(r"btbank(\d+)_t(\d+)", name)
    if not m:
        raise ValueError(f"unrecognised session name {name!r}")
    return int(m.group(1)), int(m.group(2))


def parse_key_session(key: str) -> tuple[int, int]:
    """``(subject_id, trial_id)`` from a real spec-cache ``key`` string.

    The producer writes the session uid VERBATIM, e.g.
    ``{"cls":"Wang2024Treebank","method":"_load_raw","timeline":{"extra_bad":[...],
    "subject":"btbank1","subject_id":1,"trial_id":0}}_0.000_6867.860`` — a nested
    JSON blob (subject_id/trial_id inside ``timeline``) plus a time-range suffix.
    Parse the two ids directly (full-integer capture, so 1 never aliases 12)."""
    ms = re.search(r'"subject_id"\s*:\s*(\d+)', key)
    mt = re.search(r'"trial_id"\s*:\s*(\d+)', key)
    if not ms or not mt:
        raise ValueError(f"key missing subject_id/trial_id: {key[:120]!r}")
    return int(ms.group(1)), int(mt.group(1))


def resolve_band_leaf(band_dir: str, *, band_hop: int = 64) -> str:
    """The leaf dir holding a band's ``{hash}.json/.npy/.stats.npz`` triples.

    The real cache nests the triples under exca's method-cache tree
    (``{band_root}/…MultiStftView._get_data,1/{config-hash-dir}/``), and a band root
    can carry STALE leaves from earlier regens (e.g. ``band_v3slow`` keeps a
    ``band_hop=512`` leaf beside the current ``band_hop=64`` one). Descend to the
    unique leaf whose exca config dirname carries the locked ``band_hop`` — fail loud
    on 0 or >1 so a stale cache can never be silently trained on. A dir that already
    holds ``*.json`` directly (a leaf, or a synthetic test dir) is returned as-is."""
    if glob.glob(os.path.join(band_dir, "*.json")):
        return band_dir
    leaves = sorted(
        {os.path.dirname(p) for p in glob.glob(os.path.join(band_dir, "**", "*.json"), recursive=True)}
    )
    if not leaves:
        raise ValueError(f"no spec-cache sidecars under {band_dir}")
    hop_re = re.compile(rf"band_hop={band_hop}(?![0-9])")
    pick = [d for d in leaves if hop_re.search(d)] or leaves
    if len(pick) != 1:
        raise ValueError(
            f"{band_dir}: expected exactly one band_hop={band_hop} leaf, found "
            f"{len(pick)}: {[os.path.basename(d) for d in pick]}"
        )
    return pick[0]


@dataclass(frozen=True)
class BandCacheEntry:
    npy_path: str
    stats_path: str
    ch_names: tuple[str, ...]
    total_frames: int
    sample_rate: int
    band_hop: int = 1

    @property
    def frame_rate_hz(self) -> int:
        """The cache's REAL frame rate — the rate ``band_rates`` must be declared against.

        The bake writes the RAW signal rate (``sample_rate`` 2048) plus the STFT ``band_hop``,
        so a frame lands every ``band_hop`` samples ⇒ ``2048//64 = 32 Hz``. Sidecars without a
        ``band_hop`` (synthetic test dirs) already store the frame rate directly, and the
        ``band_hop=1`` default reduces to it."""
        return self.sample_rate // self.band_hop


def index_band_cache(band_dir: str) -> dict[str, BandCacheEntry]:
    """Map each session ``key`` → its ``BandCacheEntry`` for one band's cache dir.

    Scans ``{stem}.json`` sidecars; a sidecar without a matching ``.npy`` (a
    partial/aborted write) is skipped, never half-indexed. ``ch_names`` is the npy's
    channel (voltage) order, which the sidecar/geometry build treats as the FULL
    session order.
    """
    band_dir = resolve_band_leaf(band_dir)
    out: dict[str, BandCacheEntry] = {}
    for meta_path in sorted(glob.glob(os.path.join(band_dir, "*.json"))):
        if meta_path.endswith(".stats.npz"):  # defensive; .npz never matches *.json
            continue
        stem = meta_path[: -len(".json")]
        npy_path = stem + ".npy"
        if not os.path.exists(npy_path):
            continue
        with open(meta_path) as fh:
            meta = json.load(fh)
        key = str(meta["key"])
        if key in out:
            raise ValueError(f"duplicate session cache key: {out[key].npy_path} and {npy_path}")
        out[key] = BandCacheEntry(
            npy_path=npy_path,
            stats_path=stem + ".stats.npz",
            ch_names=tuple(meta["ch_names"]),
            total_frames=int(meta["total_frames"]),
            sample_rate=int(meta.get("sample_rate", 32)),
            band_hop=int(meta.get("band_hop", 1)),
        )
    return out


def parse_lof_report(path: str | None) -> dict[tuple[int, int], set[str]]:
    """LOF report → ``(subject_id, trial_id)`` → bad-channel set (guard-1 drop).

    ``path=None`` (no report fitted) → empty map (no drops). Each ``sessions[]``
    entry carries a ``session`` name (``btbankS_tT``) and ``bad_channels``.
    """
    if path is None:
        return {}
    with open(path) as fh:
        report = json.load(fh)
    out: dict[tuple[int, int], set[str]] = {}
    for s in report.get("sessions", []):
        key = parse_session_name(str(s["session"]))
        out[key] = set(s.get("bad_channels", []))
    return out
