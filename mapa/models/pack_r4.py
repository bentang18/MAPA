"""The token layout for one session: which token sits where in the flat set.

Each contact carries a different number of tokens per band, because the bands run at
different rates on a shared 32 Hz clock. There is no rectangle to reshape through. This
module lays the tokens out flat and array-contiguous, and records the coordinates attention
reads: the clinical contact index, the position on the shared clock, and the bounds that
delimit each array.

Token order is array first, then contact in canonical order within the array, then band in
the order slow, mid, fast, then position within the band. Every array here is in that one
order, and the layout depends only on the session and the window length, so it is built once
and reused for every window.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from mapa.models.geometry import L1Geometry
from mapa.models.stem import PER_BAND_SPECS

# band axis order (matches stem.PER_BAND_SPECS and PerBandStem.forward): SLOW, MID, HGA.
BAND_STRIDES: tuple[int, ...] = tuple(st for _, st in PER_BAND_SPECS)
N_BANDS = len(PER_BAND_SPECS)


@dataclass(frozen=True)
class R4Grid:
    """Clip-INDEPENDENT flat full-grid geometry (one token per valid (contact,band,pos))."""

    depth: Tensor  # (total,) long — clinical contact index per token (L1 index-RoPE)
    time_pos: Tensor  # (total,) long — shared-32Hz lattice position (L1 time-RoPE)
    contact: Tensor  # (total,) long — contact index into N (for gather/scatter)
    band: Tensor  # (total,) long — 0=SLOW 1=MID 2=HGA
    bandpos: Tensor  # (total,) long — band-token index j in [0, T_b)
    array: Tensor  # (total,) long — array id per token
    cu_seqlens: Tensor  # (S+1,) int32 — per-array block bounds, TOKEN units
    cu_seqlens_drop: Tensor  # (S'+1,) int64 — cu minus zero-length arrays (njt)
    max_seqlen: int  # longest array block, tokens (static bound)
    total: int  # N * K_full — total tokens (constant per session)
    k_full: int  # tokens per contact = sum_b T_b
    band_lengths: tuple[int, ...]  # (T_slow, T_mid, T_hga)


def band_token_counts(n_time: int) -> tuple[int, ...]:
    """(T_slow, T_mid, T_hga) = n_time // stride per band."""
    for st in BAND_STRIDES:
        if n_time % st != 0:
            raise ValueError(f"n_time {n_time} not a multiple of band stride {st}")
    return tuple(n_time // st for st in BAND_STRIDES)


def _drop_offsets(cu: Tensor) -> Tensor:
    off = cu.to(torch.int64)
    keep = off[1:] != off[:-1]
    return torch.cat([off[:1], off[1:][keep]])


def build_r4_grid(
    geom: L1Geometry, *, n_time: int, max_seqlen: int | None = None
) -> R4Grid:
    """Build the clip-independent flat full-grid layout for one session.

    ``max_seqlen`` (longest array block, tokens) is a per-session CONSTANT; pass the cached
    value to skip its ``.item()`` host sync (the compiled per-step path). ``None`` ⇒ derive
    it here (one sync — the standalone/first-call fallback)."""
    device = geom.gather_idx.device
    S, max_c = geom.n_arrays, geom.max_c
    lengths = band_token_counts(n_time)  # (T_slow, T_mid, T_hga)
    k_full = int(sum(lengths))

    # canonical (array-major, slot) contact order.
    rows = torch.arange(S, device=device)[:, None].expand(S, max_c)[geom.valid]  # (N,)
    canon = geom.gather_idx[geom.valid]  # (N,) contact index into N
    depth_canon = geom.depth[geom.valid]  # (N,) clinical depth
    n = int(canon.shape[0])

    # the per-contact (band, bandpos, time_pos) block — identical for every contact.
    band_blk = torch.cat(
        [torch.full((L,), b, dtype=torch.long, device=device) for b, L in enumerate(lengths)]
    )  # (k_full,)
    pos_blk = torch.cat(
        [torch.arange(L, device=device) for L in lengths]
    )  # (k_full,) band-token index
    lat_blk = torch.cat(
        [torch.arange(L, device=device) * st for L, st in zip(lengths, BAND_STRIDES)]
    )  # (k_full,) lattice position = j * stride

    # tile over the N contacts (contact-major ⇒ array-contiguous).
    depth = depth_canon.repeat_interleave(k_full)
    contact = canon.repeat_interleave(k_full)
    array = rows.repeat_interleave(k_full)
    band = band_blk.repeat(n)
    bandpos = pos_blk.repeat(n)
    time_pos = lat_blk.repeat(n)

    # per-array token counts = n_s * k_full.
    n_per_array = geom.valid.sum(dim=1).to(torch.long)  # (S,)
    seg = (n_per_array * k_full).to(torch.int32)  # (S,)
    cu = torch.zeros(S + 1, dtype=torch.int32, device=device)
    cu[1:] = seg.cumsum(0).to(torch.int32)
    if max_seqlen is None:
        max_seqlen = int((n_per_array.max() * k_full).item())

    return R4Grid(
        depth=depth, time_pos=time_pos, contact=contact, band=band, bandpos=bandpos,
        array=array, cu_seqlens=cu, cu_seqlens_drop=_drop_offsets(cu),
        max_seqlen=max_seqlen, total=n * k_full, k_full=k_full, band_lengths=lengths,
    )


def pack_band_tokens(band_tokens: Sequence[Tensor], grid: R4Grid) -> Tensor:
    """Flatten per-band token tuples into the flat grid token order.

    ``band_tokens`` is one tensor per band, each ``(B, n_contacts, n_tokens, d)``. The result
    is ``(B, total, d)`` in the order ``build_r4_grid`` lays out, so token ``t`` is the one
    whose contact, band and band position are ``grid.contact[t]``, ``grid.band[t]`` and
    ``grid.bandpos[t]``. The layout is not restated here: the contact order is read back off
    the grid, and the bands are concatenated in their fixed order.

"""
    if len(band_tokens) != len(grid.band_lengths):
        raise ValueError(
            f"expected {len(grid.band_lengths)} band token streams, got {len(band_tokens)}"
        )
    b0 = band_tokens[0]
    B, d = b0.shape[0], b0.shape[-1]
    n = grid.total // grid.k_full
    canon = grid.contact.reshape(n, grid.k_full)[:, 0]  # (n,) contact index, canonical order
    blocks: list[Tensor] = []
    for band, (tok, length) in enumerate(zip(band_tokens, grid.band_lengths)):
        if tok.shape[2] != length:
            raise ValueError(
                f"band {band} has {tok.shape[2]} tokens, grid expects {length}"
            )
        blocks.append(tok[:, canon])  # (B, n, T_b, d) reindexed to canonical contacts
    x = torch.cat(blocks, dim=2)  # (B, n, k_full, d) — per contact [SLOW; MID; HGA]
    return x.reshape(B, grid.total, d)
