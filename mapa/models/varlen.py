"""Block-diagonal attention over a flat, ragged token set.

Tokens from every array in a session are laid end to end with no padding, and ``cu_seqlens``
marks where each array's block starts. A token attends inside its own block and nowhere
else. Two backends compute the same thing.

On GPU the packed values are viewed as a jagged nested tensor, which routes scaled dot
product attention to its ragged kernel. Nothing is padded and no mask is built, so cost
follows the true block sizes. On CPU the block-diagonal mask is materialised and handed to
dense attention, which is only tractable at test sizes and is what the tests compare
against.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# Finite rather than -inf: a row with no visible key softmaxes to uniform instead of NaN.
NEG_INF_MASK = -1e4


def _segment_ids(cu_seqlens: Tensor, total: int) -> Tensor:
    """Cumulative block bounds ``(n_block+1,)`` to a block id per token ``(total,)``.

    An empty block contributes nothing, which is exactly the number of tokens it holds.
    """
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
    seg = torch.arange(lengths.shape[0], device=cu_seqlens.device)
    return seg.repeat_interleave(lengths)


def _reference_block_diag(q: Tensor, k: Tensor, v: Tensor, cu_seqlens: Tensor) -> Tensor:
    """Dense attention under a block-diagonal mask. CPU only: the mask is ``(total, total)``.

    q, k, v are ``(total, n_heads, head_dim)``. Returns the same shape.
    """
    total = q.shape[0]
    seg = _segment_ids(cu_seqlens, total)
    same = seg[:, None] == seg[None, :]
    bias = torch.where(same, 0.0, NEG_INF_MASK).to(q.dtype)
    ctx = F.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
        attn_mask=bias[None, None],
    )
    return ctx[0].transpose(0, 1).contiguous()


def gpu_block_diag(
    q: Tensor, k: Tensor, v: Tensor, cu_seqlens_drop: Tensor, max_seqlen: int
) -> Tensor:
    """Block-diagonal attention through a jagged nested tensor. No pad and no mask.

    ``nested_tensor_from_jagged`` views the packed ``(total, n_heads, head_dim)`` values as
    ``(n_block, len_i, n_heads, head_dim)``, so attention dispatches to the ragged kernel.
    The view preserves gradients, unlike the nested tensor constructor, which detaches.

    ``cu_seqlens_drop`` is int64 and carries empty blocks already removed, because an empty
    jagged row makes the ragged backward produce NaN. Passing ``max_seqlen`` explicitly
    skips the kernel's own reduction over the offsets, which would sync the device on every
    call. The softmax scale is the default ``1/sqrt(head_dim)``, which is what the reference
    backend applies too.
    """

    def _nt(x: Tensor) -> Tensor:
        return torch.nested.nested_tensor_from_jagged(
            x, offsets=cu_seqlens_drop, min_seqlen=1, max_seqlen=max_seqlen
        ).transpose(1, 2)

    ctx = F.scaled_dot_product_attention(_nt(q), _nt(k), _nt(v))
    return ctx.transpose(1, 2).values()
