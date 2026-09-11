"""One transformer block, attending inside one array.

A token is a (contact, band, position) triple. A token attends to every other token of its
own array, over contacts and time together rather than over each axis in turn, and to
nothing outside that array. Attention across arrays is not offered, because which contacts
share an array is a fact about where the surgeon placed the electrodes and not about the
brain.

Queries and keys carry a rotary encoding on two axes: position along the array, indexed by
the clinical contact number, and time. The rotation is absolute but the score depends only
on the difference. A constant offset preserves those differences; reversing contact numbering
does not. Preserve the recording's clinical numbering and direction.

Region identity is added once at the stack input and rides the residual into every block, so
this module injects nothing of its own.
"""

from __future__ import annotations

from torch import Tensor, nn

from mapa.models.pe import L1RoPE
from mapa.models.varlen import _reference_block_diag, gpu_block_diag

LN_EPS = 1e-6


class _MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int) -> None:
        super().__init__()
        hidden = d_model * mlp_ratio
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class L1Block(nn.Module):
    """Within-array joint spatiotemporal attention block (block-diagonal, RoPE)."""

    def __init__(
        self, d_model: int, n_heads: int, *, mlp_ratio: int = 4, space_rope: bool = True
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # qkv/out biases ON to match upstream V-JEPA 2 (qkv_bias=True, proj bias;
        # vision_transformer.py factories). No QK-norm: upstream ViT-B omits it,
        # and a per-element head_dim gain rotates inconsistently across each RoPE
        # pair (breaks relative-covariance) while being WD-exempt (1-D) — worse
        # than none. Logit runaway is unlikely at d256/ViT-B scale; if it appears
        # at the 10x LR the fix is a pair-shared (not per-element) RMSNorm.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.rope = L1RoPE(self.head_dim, space=space_rope)
        self.norm1 = nn.LayerNorm(d_model, eps=LN_EPS)
        self.norm2 = nn.LayerNorm(d_model, eps=LN_EPS)
        self.mlp = _MLP(d_model, mlp_ratio)


    # qkv, then the rotary on (depth, time_pos), then block-diagonal attention over
    # cu_seqlens, then the output projection. depth is the clinical contact index and
    # time_pos the position on the shared clock; both are per token.
    def _attn_flat(
        self,
        x_flat: Tensor,
        depth: Tensor,
        time_pos: Tensor,
        cu_seqlens: Tensor,
        cu_seqlens_drop: Tensor,
        max_seqlen: int,
        *,
        rope_cs: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        # x_flat: (M, d). depth, time_pos: (M,) long. Token order = array-contiguous
        # (cu_seqlens delimits each array block).
        m, d = x_flat.shape
        qkv = self.qkv(x_flat).reshape(m, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)  # (M, H, hd)
        qh, kh = q.transpose(0, 1), k.transpose(0, 1)  # (H, M, hd)
        if rope_cs is None:
            qh, kh = self.rope(qh, kh, depth, time_pos)
        else:
            qh, kh = self.rope.rotate(qh, kh, rope_cs[0], rope_cs[1])
        q, k = qh.transpose(0, 1), kh.transpose(0, 1)  # (M, H, hd)
        q, k = q.to(v.dtype), k.to(v.dtype)  # align to v for SDPA (no-op under fp32)
        if q.is_cuda:
            ctx = gpu_block_diag(q, k, v, cu_seqlens_drop, max_seqlen)  # (M, H, hd)
        else:
            ctx = _reference_block_diag(q, k, v, cu_seqlens)  # (M, H, hd)
        return self.out(ctx.reshape(m, d))

    def forward_flat(
        self,
        x_flat: Tensor,
        depth: Tensor,
        time_pos: Tensor,
        cu_seqlens: Tensor,
        cu_seqlens_drop: Tensor,
        max_seqlen: int,
        *,
        rope_cs: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        x_flat = x_flat + self._attn_flat(
            self.norm1(x_flat), depth, time_pos, cu_seqlens, cu_seqlens_drop,
            max_seqlen, rope_cs=rope_cs,
        )
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        return x_flat
