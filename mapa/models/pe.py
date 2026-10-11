"""Rotary contact/time coordinates and learned anatomical region embeddings.

Main path: L1RoPE.cos_sin() builds angles; rotate() applies them to Q/K.

Input: Q/K features plus each token's contact/time coordinates.
Output: rotated Q/K with unchanged shapes; there are no learned RoPE parameters.
Next: return to attention.py to compute attention with these Q/K and unchanged V.

RoPE splits each head equally between clinical contact index and time on the
32 Hz clock. Within each half, adjacent feature pairs rotate at fixed frequencies.
Q and K are rotated; V is unchanged. For fixed Q/K content, a common coordinate
offset leaves their rotary dot product unchanged.

Released bases: contact 8, time 64. Contact indices preserve gaps and are ordinal,
not physical distances. These settings do not imply a hard contact-count cutoff;
generalization to longer shafts or two-dimensional grids needs validation.

RegionIdentityEmbed is a learned atlas-region lookup added to each contact's tokens.
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
    # Each adjacent feature pair (a,b) becomes (-b,a), a 90-degree rotation.
    x_paired = x.unflatten(-1, (-1, 2))  # (..., hd/2, 2)
    return torch.stack([-x_paired[..., 1], x_paired[..., 0]], dim=-1).flatten(-2)


class L1RoPE(nn.Module):
    """Two-axis axial RoPE: half of each head for contact, half for time."""

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
        pairs = head_dim // 4  # head_dim=64: 16 pairs for contact + 16 for time
        idx_freq = 1.0 / (base_index ** (torch.arange(pairs).float() / pairs))
        # Disabling space makes contact rotations identity; time rotations remain.
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
        # angle = coordinate * frequency (radians); frequencies are fixed buffers.
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
        # The disabled variant has no embedding parameters and returns zeros.
        if not enabled:
            return
        embed = nn.Embedding(n_regions, d_model)
        nn.init.trunc_normal_(embed.weight, std=init_std)
        self.embed = embed

    def forward(self, region_id: Tensor) -> Tensor:
        """region_id: (..., seq) long → (..., seq, d_model)."""
        if self.embed is None:
            return region_id.new_zeros((*region_id.shape, self.d_model), dtype=torch.float32)
        return self.embed(region_id)
