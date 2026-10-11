"""Pre-norm transformer block with attention isolated by array and clip.

Main path: L1Block.forward_flat() -> _attn_flat().

Input/output: (M,d); neither token count nor width changes.
Flow: normalize -> QKV -> rotate Q/K -> attend within blocks -> project -> add.
Then normalize -> token-wise MLP -> add; return to the next encoder block.

Each token represents one contact, band, and time. Within its array/clip block,
it attends to all tokens across contacts, bands, and times, including itself.
There is no causal mask and no attention across block boundaries.

RoPE rotates Q and K using clinical contact indices and shared-clock time.
For fixed Q/K content, its positional contribution depends on coordinate
differences. Region embeddings are added once before the encoder stack.
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
        self.fc1 = nn.Linear(d_model, hidden)  # expand features: 384 -> 1536
        self.fc2 = nn.Linear(hidden, d_model)  # restore width: 1536 -> 384
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        # Apply the same feature transformation independently to every token.
        # Shapes below use the released encoder width d=384.
        hidden = self.fc1(x)          # (M,384) -> (M,1536)
        hidden = self.act(hidden)    # GELU; shape unchanged
        output = self.fc2(hidden)    # (M,1536) -> (M,384)
        return output


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
        # Released model: d=384, six heads of width 64; no separate Q/K norm.
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
        # 1. Project Q, K, V: (M,384) -> (M,1152) -> (M,3,6,64).
        qkv = self.qkv(x_flat)  # (M,3*d): three learned projections in one linear layer
        qkv = qkv.reshape(m, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)  # (M, H, hd)
        # 2. Rotate Q/K. RoPE expects heads before tokens; V is unchanged.
        qh, kh = q.transpose(0, 1), k.transpose(0, 1)  # (H, M, hd)
        if rope_cs is None:
            qh, kh = self.rope(qh, kh, depth, time_pos)
        else:
            qh, kh = self.rope.rotate(qh, kh, rope_cs[0], rope_cs[1])
        q, k = qh.transpose(0, 1), kh.transpose(0, 1)  # (M, H, hd)
        q, k = q.to(v.dtype), k.to(v.dtype)  # align to v for SDPA (no-op under fp32)
        # 3. Attend within each array/clip block; output remains (M,H,hd).
        # Per head: softmax(Q @ K.T / sqrt(hd), over keys) @ V; see varlen.py.
        if q.is_cuda:
            ctx = gpu_block_diag(q, k, v, cu_seqlens_drop, max_seqlen)  # (M, H, hd)
        else:
            ctx = _reference_block_diag(q, k, v, cu_seqlens)  # (M, H, hd)
        # 4. Concatenate heads, then mix them with a learned d -> d projection.
        ctx = ctx.reshape(m, d)  # (M,H,hd) -> (M,d)
        return self.out(ctx)

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
        # Block overview: keep x, learn an attention update, then learn an MLP update.
        # LayerNorm acts across features per token; both additions preserve (M,d).
        normalized = self.norm1(x_flat)
        attention_update = self._attn_flat(
            normalized, depth, time_pos, cu_seqlens, cu_seqlens_drop,
            max_seqlen, rope_cs=rope_cs,
        )
        x_flat = x_flat + attention_update
        # Normalize -> Linear(d,4d) -> GELU -> Linear(4d,d) -> add to x.
        normalized = self.norm2(x_flat)
        mlp_update = self.mlp(normalized)
        x_flat = x_flat + mlp_update
        return x_flat
