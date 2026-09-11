"""Load a list of sessions: the orchestrator over the parsers and the pure cores.

Composes the parsers (``cache_index``), the pure cores
(``build_session_setup`` and ``build_session_spec``) and the BrainTreebank region
lookup into the ``list[V3SessionSpec]`` the data module consumes.

For each requested ``(subject_id, trial_id)``:
  1. match its ``.npy``/``.stats.npz`` in each of the 3 band cache dirs (by the
     spec sidecar ``key``); the band ``ch_names`` is the FULL voltage order;
  2. resolve the DKT hard ``region_id`` over those labels (``region_fn`` — the real
     BT ``aligned_voltage_support`` call, injectable so the orchestration is unit-
     testable with a stub);
  3. guard-1 drop = LOF bad set (``extra_bad`` is already gone from ``ch_names`` —
     a harmless no-op if re-passed) → ``build_session_setup`` (sidecar + geom +
     feasibility assert);
  4. load each band's frozen robust-z stats and the guard-2 spans →
     ``build_session_spec`` (survivor-sliced normalizers).

The only F2-validated seam is the default ``region_fn`` (one BT-anatomy call);
everything else is exercised locally with synthetic caches + a stub region_fn.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch import Tensor

from mapa.data.cache_index import (
    BandCacheEntry,
    index_bad_windows,
    index_band_cache,
    parse_key_session,
    parse_lof_report,
)
from mapa.data.dataset import (
    V3SessionSpec,
    assert_band_rates_match_cache,
    build_session_spec,
)
from mapa.data.session_setup import build_session_setup

RegionFn = Callable[[int, int, Sequence[str]], Tensor]
KeepLabelsFn = Callable[[int, int, Sequence[str]], set[str]]


def _entry_for(index: dict[str, BandCacheEntry], subject_id: int, trial_id: int) -> BandCacheEntry:
    """The unique cache entry for a session, matched on the sidecar ``key``.

    ``key`` is the real producer uid (nested JSON with ``"subject_id":S`` /
    ``"trial_id":T`` inside ``timeline``); ``parse_key_session`` extracts the two
    ids (full-integer, so S=1 never aliases 12). Exactly one entry per session per
    band leaf (one continuous whole-movie cache), so >1 or 0 is a fail-loud."""
    hits = [e for k, e in index.items() if parse_key_session(k) == (subject_id, trial_id)]
    if len(hits) != 1:
        raise ValueError(
            f"expected exactly one cache entry for subject {subject_id} trial "
            f"{trial_id}, found {len(hits)}"
        )
    return hits[0]


def _load_stats(stats_path: str) -> tuple[Tensor, Tensor]:
    z = np.load(stats_path)
    return torch.from_numpy(z["median"]).float(), torch.from_numpy(z["sigma"]).float()


def load_v3_sessions(
    *,
    sessions: Sequence[tuple[int, int]],
    band_cache_dirs: Sequence[str],  # 3 dirs in v3 concat order (slow, mid, hga)
    span_dir: str,
    region_fn: RegionFn,
    lof_report_path: str | None = None,
    sigma_floor: float = 1e-6,
    keep_labels_fn: KeepLabelsFn | None = None,
) -> list[V3SessionSpec]:
    """Match three 32 Hz caches, select contacts, and freeze their stored normalization.

    Optional external contact exclusions are applied before building geometry
    so features, labels, atlas IDs, and statistics share one contact axis.
    """
    if len(band_cache_dirs) != 3:
        raise ValueError(f"expected 3 band_cache_dirs, got {len(band_cache_dirs)}")
    band_indexes = [index_band_cache(d) for d in band_cache_dirs]
    bad_idx = index_bad_windows(span_dir)
    lof = parse_lof_report(lof_report_path)

    specs: list[V3SessionSpec] = []
    for subject_id, trial_id in sessions:
        entries = [_entry_for(bi, subject_id, trial_id) for bi in band_indexes]
        ch0 = entries[0].ch_names
        for e in entries[1:]:
            if e.ch_names != ch0:
                raise ValueError(
                    f"band caches disagree on channel order for session "
                    f"{subject_id}/{trial_id}: {e.ch_names[:3]}... vs {ch0[:3]}..."
                )
        labels = list(ch0)
        region_id = region_fn(subject_id, trial_id, labels).long()
        drop = set(lof.get((subject_id, trial_id), set()))
        if keep_labels_fn is not None:
            keep = keep_labels_fn(subject_id, trial_id, labels)
            restricted = {lab for lab in labels if lab not in keep}
            if len(labels) - len(restricted | drop) == 0:
                raise ValueError(
                    f"session {subject_id}/{trial_id}: keep_labels_fn kept 0 of "
                    f"{len(labels)} electrodes — montage does not match this cache"
                )
            drop |= restricted
        setup = build_session_setup(
            labels, region_id, drop_labels=drop,
        )
        band_stats = [_load_stats(e.stats_path) for e in entries]
        # the declared read rates must match what was actually baked, else every per-band
        # window is a compressed, time-shifted slice at the RIGHT shape (r6 2026-07-23).
        assert_band_rates_match_cache(
            [e.frame_rate_hz for e in entries],
            where=f"session {subject_id}/{trial_id}",
        )
        n_frames_32 = min(e.total_frames for e in entries)
        specs.append(
            build_session_spec(
                session_key=(subject_id, trial_id),
                band_paths=tuple(e.npy_path for e in entries),
                band_stats=tuple(band_stats),
                setup=setup,
                n_frames=n_frames_32,
                bad_spans_s=bad_idx.get((subject_id, trial_id), []),
                sigma_floor=sigma_floor,
            )
        )
    return specs
