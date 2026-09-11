"""Session-stat alignment and cache-rate validation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mapa.data.dataset import (
    assert_band_rates_match_cache,
    build_session_spec,
)
from mapa.data.session_setup import build_session_setup

F_SLOW, F_MID, F_HGA = 7, 6, 7
T_CLIP = 96
FPS = 32.0


def _write_band(tmp, name, c, f, t_total):
    arr = (np.random.RandomState(abs(hash(name)) % 2**32)
           .randn(c, f, t_total).astype(np.float32))
    path = str(tmp / f"{name}.npy")
    np.save(path, arr)
    return path, arr


def _spec(tmp, *, key=(1, 0), array_sizes=(4, 3, 3), t_total=4000,
          bad_spans=(), drop=frozenset()):
    labels, regions = [], []
    for s, n in enumerate(array_sizes):
        for c in range(1, n + 1):
            labels.append(f"L{chr(65 + s)}{c}")
            regions.append(s)
    c_full = len(labels)
    setup = build_session_setup(labels, torch.tensor(regions), drop_labels=drop)
    band_paths, band_fbins = [], (F_SLOW, F_MID, F_HGA)
    for bname, fb in zip(("v3slow", "v3mid", "hga"), band_fbins):
        p, _ = _write_band(tmp, f"{key[0]}_{key[1]}_{bname}", c_full, fb, t_total)
        band_paths.append(p)
    band_stats = [
        (torch.randn(c_full, fb, 1), torch.rand(c_full, fb, 1) + 0.5)
        for fb in band_fbins
    ]
    return build_session_spec(
        session_key=key,
        band_paths=tuple(band_paths),
        band_stats=tuple(band_stats),
        setup=setup,
        n_frames=t_total,
        bad_spans_s=list(bad_spans),
    )


def test_build_session_spec_slices_stats_to_survivors(tmp_path) -> None:
    spec = _spec(tmp_path, drop=frozenset({"LA1"}))
    n = len(spec.setup.sidecar.labels)
    for med, sig in spec.band_stats:
        assert med.shape[0] == n and sig.shape[0] == n


@pytest.mark.parametrize("rates", [(16, 32, 32), (32, 64, 32), (32, 32, 16)])
def test_cache_rate_mismatch_names_band_and_session(rates):
    with pytest.raises(ValueError, match="session 1/0: band"):
        assert_band_rates_match_cache(rates, where="session 1/0")


def test_cache_rates_match_released_frontend():
    assert_band_rates_match_cache((32, 32, 32))
