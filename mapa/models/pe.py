"""The two spatial encodings, and the initialisation everything else follows.

L1RoPE is the relative positional encoding. The head dimension is split evenly between two
axes, position along the array and time, each on the standard rotary schedule. Rotation is by
absolute position, but the query-key score depends only on the difference, which is the
property that matters: absolute position along an array is not shared across subjects, and
only the ordering is.

The released rotary bases are 8 for clinical contact indices and 64 for time slots.
Contact indices are not normalized to a common span: dropped contacts retain their
original index gaps. These are ordinal positions, not measured millimeters. The
checkpoint has no hard contact-count cutoff; longer shafts and two-dimensional
ECoG grids require separate validation. Crossing a half-turn in one rotary pair
does not establish a hard aliasing limit for the full rotary representation.

RegionIdentityEmbed is the region embedding: a learned table indexed by the DKT region a
contact falls in, added to every token of that contact. The vocabulary is fixed by the atlas
rather than learned per subject, so the table transfers even though it is learned.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def init_transformer_weights(m: nn.Module, init_std: float = 0.02) -> None:
    """V-JEPA 2 module init (``vision_transformer.py:130-141``): Linear weights
    ``trunc_normal_(std)`` + zero bias; LayerNorm weight 1 / bias 0. Applied via
    ``module.apply(...)``; Embeddings are skipped (they self-init in ``__init__``)."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=init_std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(m.weight, 1.0)


def _rotate_half(x: Tensor) -> Tensor:
    x_paired = x.unflatten(-1, (-1, 2))  # (..., hd/2, 2)
    return torch.stack([-x_paired[..., 1], x_paired[..., 0]], dim=-1).flatten(-2)


class L1RoPE(nn.Module):
    """Mixed 2-axis (contact-index, time) rotary PE, equal head_dim split."""

    def __init__(
        self,
        head_dim: int,
        *,
        base_index: float = 8.0,
        base_time: float = 64.0,
        space: bool = True,
    ) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(
                f"L1RoPE needs head_dim % 4 == 0 (equal even split across 2 axes), "
                f"got {head_dim}"
            )
        self.head_dim = head_dim
        self.space = bool(space)
        pairs = head_dim // 4  # rotation pairs per axis
        idx_freq = 1.0 / (base_index ** (torch.arange(pairs).float() / pairs))
        # ABLATION A2 (2026-07-28): space=False ZEROES the index frequencies, so
        # ang_idx == 0 ⇒ cos 1 / sin 0 ⇒ the index pairs take the IDENTITY rotation.
        # The head_dim split, pair convention and TIME half stay bit-identical; only
        # the contact-index axis stops carrying position. Since RegionIdentityEmbed is
        # region-level and band_type_emb is band-level, index-RoPE is the ONLY per-token
        # carrier of within-array contact identity ⇒ this makes L1 attention genuinely
        # permutation-invariant over contacts in a array. That is the control the
        # "physics sensor-index tokenization" claim has never had.
        if not self.space:
            idx_freq = torch.zeros_like(idx_freq)
        t_freq = 1.0 / (base_time ** (torch.arange(pairs).float() / pairs))
        self.register_buffer("idx_freq", idx_freq, persistent=False)
        self.register_buffer("t_freq", t_freq, persistent=False)

    def cos_sin(self, idx: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """The (..., seq, head_dim) fp32 cos/sin rotary table for coords (idx, t).

        Every block holds the same schedule, so the table is the same for all of them and
        for every head. It is built once per forward and passed to :meth:`rotate` in each
        block, which is the same arithmetic as rebuilding it and a good deal less of it.
        The table stays fp32 and the result is cast down afterwards, because rounding it
        first would change the numbers."""
        ang_idx = idx[..., None].float() * self.idx_freq  # (..., seq, pairs)
        ang_t = t[..., None].float() * self.t_freq
        ang = torch.cat([ang_idx, ang_t], dim=-1)  # (..., seq, head_dim/2)
        cos = ang.cos().repeat_interleave(2, dim=-1)  # (..., seq, head_dim)
        sin = ang.sin().repeat_interleave(2, dim=-1)
        return cos, sin

    @staticmethod
    def rotate(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
        """Apply a precomputed (from :meth:`cos_sin`) rotary table to q, k.

        q, k: (..., H, seq, head_dim); cos, sin: (..., seq, head_dim) — the head
        axis is broadcast in via ``unsqueeze(-3)``."""
        cos = cos.unsqueeze(-3)  # (..., 1, seq, head_dim) → broadcast over heads
        sin = sin.unsqueeze(-3)
        q_out = q * cos + _rotate_half(q) * sin
        k_out = k * cos + _rotate_half(k) * sin
        return q_out, k_out

    def forward(
        self, q: Tensor, k: Tensor, idx: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor]:
        """q, k: (..., H, seq, head_dim); idx, t: (..., seq) (no head axis)."""
        cos, sin = self.cos_sin(idx, t)  # (..., seq, head_dim)
        return self.rotate(q, k, cos, sin)


class RegionIdentityEmbed(nn.Module):
    """Learned per-region identity embedding indexed by the DKT/DK hard tag."""

    def __init__(
        self, n_regions: int, d_model: int, *, init_std: float = 1e-6, enabled: bool = True
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embed: nn.Embedding | None = None
        # enabled=False = the R2 ablation arm: the table is NOT CONSTRUCTED (no dead param to
        # weight-decay, and the checkpoint's param count is honest), and forward returns zeros.
        # Every call site adds it to the residual (towers.py:165/196/279), so "return zeros" and
        # "don't add it" are the SAME edit — the embed is purely additive, never concatenated.
        if not enabled:
            return
        embed = nn.Embedding(n_regions, d_model)
        nn.init.trunc_normal_(embed.weight, std=init_std)
        self.embed = embed
        # The table starts near zero, following V-JEPA 2.1, so the model grows into it. The
        # table is added to every token and rides every block, so at the more usual 0.02 it
        # would be about a fifth of the token signal at initialisation, which is a strong
        # per-region prior from the first step and an invitation to memorise region identity
        # rather than learn from the signal. It is weight decayed like any other 2-D table.

    def forward(self, region_id: Tensor) -> Tensor:
        """region_id: (..., seq) long → (..., seq, d_model)."""
        if self.embed is None:
            return region_id.new_zeros((*region_id.shape, self.d_model), dtype=torch.float32)
        return self.embed(region_id)
