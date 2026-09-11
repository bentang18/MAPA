"""Cached session metadata and frozen normalization for explicit evaluation windows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor

from mapa.data.normalize import SessionRobustZNormalizer
from mapa.data.session_setup import V3SessionSetup


def assert_band_rates_match_cache(frame_rates_hz: Sequence[int], *, where: str = "") -> None:
    """Reject cache rates that would shift the model's 32 Hz evaluation windows."""
    if len(frame_rates_hz) != 3:
        raise ValueError(f"{where}: expected three band caches, got {len(frame_rates_hz)}")
    for b, rate in enumerate(frame_rates_hz):
        if rate != 32:
            raise ValueError(f"{where}: band {b} requires 32 Hz but its cache is {rate} Hz")


@dataclass(frozen=True)
class V3SessionSpec:
    """Per-session fixtures the dataset draws clips from (survivor-aligned)."""

    session_key: tuple  # (subject_id, trial_id)
    band_paths: tuple[str, ...]  # 3 × .npy path, each (C_full, F_band, T_total) fp32
    band_stats: tuple[tuple[Tensor, Tensor], ...]  # 3 × (median, sigma) sliced to (N,F,1)
    band_norms: tuple[SessionRobustZNormalizer, ...]  # frozen robust-z per band
    keep_idx: Tensor  # (N,) long — survivor rows into the .npy C dim
    setup: V3SessionSetup
    n_frames: int  # T_total of the caches
    bad_spans_s: list[tuple[float, float]]  # guard-2 spans, hop-invariant seconds


def build_session_spec(
    *,
    session_key: tuple,
    band_paths: Sequence[str],
    band_stats: Sequence[tuple[Tensor, Tensor]],
    setup: V3SessionSetup,
    n_frames: int,
    bad_spans_s: Sequence[tuple[float, float]],
    sigma_floor: float = 1e-6,
) -> V3SessionSpec:
    """Slice the full-C robust-z stats to survivors and freeze a normalizer per band.

    ``band_stats`` arrive at the caches' full ``C`` rows (as stored in ``.stats.npz``);
    ``keep_idx`` selects the survivor rows so median/σ align to the clip rows the
    mmap read returns. The denominator floor is applied at transform.
    """
    keep = setup.keep_idx
    sliced = tuple((med[keep], sig[keep]) for med, sig in band_stats)
    norms = tuple(
        SessionRobustZNormalizer.from_stats(median=med, sigma=sig, sigma_floor=sigma_floor)
        for med, sig in sliced
    )
    return V3SessionSpec(
        session_key=session_key,
        band_paths=tuple(band_paths),
        band_stats=sliced,
        band_norms=norms,
        keep_idx=keep,
        setup=setup,
        n_frames=int(n_frames),
        bad_spans_s=list(bad_spans_s),
    )
