"""Twelve transformer blocks, each preserving token count and model width.

Main path: Encoder.forward_flat().

Input: flat tokens (B,total,d) and their metadata.
Output: requested width-d layer features, plus the normalized multi-layer output.
Next: attention.L1Block explains one block; encoder.py reshapes the returned taps.

Region identity is added once at the stack input. Attention stays within each
array/clip block. Blocks share the architecture and RoPE coordinates, not weights.

With deep_sup enabled, the main output concatenates separately normalized taps
3/6/9/12: width d -> 4d. Requested tap_blocks are returned separately as raw
width-d features; MapaEncoder.forward uses those for inference.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from mapa.models.pack_r4 import R4Grid

from mapa.models.attention import LN_EPS, L1Block
from mapa.models.pe import RegionIdentityEmbed, init_transformer_weights

DEPTH = 12                              # blocks; not a knob
HEAD_DIM = 64                           # held across widths, the ViT sizing convention
ENC_D_MODEL = 256                       # the width the ablation ladder was tuned at
VIT_SMALL = 384                         # the released width
ENC_SUP_TAPS: tuple[int, ...] = (3, 6, 9, 12)
N_LEVELS = len(ENC_SUP_TAPS)


def n_heads_for(d_model: int) -> int:
    """Head count for one width. Off-ladder widths fail here, not at the first matmul."""
    if d_model <= 0 or d_model % HEAD_DIM != 0:
        raise ValueError(f"d_model={d_model} must be a positive multiple of {HEAD_DIM}")
    return d_model // HEAD_DIM


class Encoder(nn.Module):
    """Pre-norm stack of within-array attention blocks."""

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_regions: int,
        deep_sup: bool = True,
        sup_taps: tuple[int, ...] = ENC_SUP_TAPS,
        region_embed: bool = True,
        space_rope: bool = True,
    ) -> None:
        super().__init__()
        self.region_embed = RegionIdentityEmbed(n_regions, d_model, enabled=region_embed)
        self.blocks = nn.ModuleList(
            [L1Block(d_model, n_heads, space_rope=space_rope) for _ in range(DEPTH)]
        )
        self.deep_sup = bool(deep_sup)
        self.sup_taps = tuple(int(b) for b in sup_taps) if self.deep_sup else ()
        if self.deep_sup:
            if not self.sup_taps:
                raise ValueError("deep_sup=True requires a non-empty sup_taps")
            # Each tap carries its own affine LayerNorm and the four are concatenated. There
            # is no separate terminal norm: the deepest tap's norm is the terminal norm.
            self.norms_block = nn.ModuleList(
                [nn.LayerNorm(d_model, eps=LN_EPS) for _ in self.sup_taps]
            )
            self.norm_out = None
        else:
            self.norms_block = None
            self.norm_out = nn.LayerNorm(d_model, eps=LN_EPS)
        # V-JEPA 2 init: Linear trunc_normal(0.02) with zero bias, LayerNorm weight 1 bias 0,
        # then the depth-scaled residual rescale below. The region embedding is an
        # nn.Embedding, which this rule does not match, so it keeps its own near-zero init.
        self.apply(init_transformer_weights)
        self._rescale_blocks()

    def _rescale_blocks(self) -> None:
        # Divide the attention output projection and the second MLP layer by sqrt(2*layer),
        # one-indexed, so residual branch variance stays flat with depth.
        for layer_id, block in enumerate(self.blocks):
            scale = math.sqrt(2.0 * (layer_id + 1))
            block.out.weight.data.div_(scale)
            block.mlp.fc2.weight.data.div_(scale)

    def forward_flat(
        self,
        x_flat: Tensor,
        grid: R4Grid,
        region_packed: Tensor,
        *,
        tap_blocks: tuple[int, ...] = (),
    ) -> Tensor | tuple[Tensor, dict[int, Tensor]]:
        """Run the stack over ``(B, total, d)`` tokens laid out by the pack plan.

        The grid coordinates are the same for every window in the batch, so they are tiled
        rather than rebuilt. Window ``b`` occupies its own set of array blocks, offset by
        ``b * total``, which makes attention block-diagonal per window and per array without
        the kernel knowing about windows at all.
        """
        b_size, m, d = x_flat.shape
        if m != grid.total:
            raise ValueError(f"x_flat has {m} tokens, grid.total={grid.total}")
        device = x_flat.device

        # 1. Add the atlas-region embedding to each token.
        region_bm = region_packed[None, :].expand(b_size, m)
        x_flat = x_flat + self.region_embed(region_bm).to(x_flat.dtype)

        # 2. Give every clip its own array blocks before flattening the batch.
        # E.g. [0,1040,2080] per clip -> [0,1040,2080,3120,4160] for two clips.
        offsets = torch.arange(b_size, device=device, dtype=torch.int64) * m
        array_starts = grid.cu_seqlens[:-1].to(torch.int64)
        batch_starts = array_starts[None, :] + offsets[:, None]  # (B,n_arrays)
        final_end = torch.tensor([b_size * m], dtype=torch.int64, device=device)
        cu_b = torch.cat([batch_starts.reshape(-1), final_end])
        keep = torch.ones_like(cu_b, dtype=torch.bool)
        keep[1:] = cu_b[1:] != cu_b[:-1]        # drop empty arrays and boundary coincidences
        cu_drop_b = cu_b[keep]
        cu_b = cu_b.to(grid.cu_seqlens.dtype)

        depth_b = grid.depth[None, :].expand(b_size, m).reshape(b_size * m)
        time_b = grid.time_pos[None, :].expand(b_size, m).reshape(b_size * m)
        # 3. Build contact/time rotations once; reuse them in all 12 blocks.
        rope_cs = self.blocks[0].rope.cos_sin(depth_b, time_b)

        # 4. Flatten to (B*total,d). Each block updates the same token positions.
        # Read attention.L1Block.forward_flat() for one block, then return here.
        xf = x_flat.reshape(b_size * m, d)
        taps: dict[int, Tensor] = {}
        levels: list[Tensor] = []
        for i, blk in enumerate(self.blocks):
            xf = blk.forward_flat(
                xf, depth_b, time_b, cu_b, cu_drop_b, grid.max_seqlen, rope_cs=rope_cs,
            )
            # Inference taps keep width d; the main multi-level output has width 4d.
            if (i + 1) in tap_blocks:
                taps[i + 1] = xf.reshape(b_size, m, d)
            if self.deep_sup and (i + 1) in self.sup_taps:
                level_index = self.sup_taps.index(i + 1)
                normalized_level = self.norms_block[level_index](xf)
                levels.append(normalized_level)
        # Inference uses raw taps (d each); pretraining used normalized taps (4*d).
        if self.deep_sup:
            out = torch.cat(levels, dim=-1)  # (B*total,4*d)
        else:
            out = self.norm_out(xf)          # (B*total,d)
        out = out.reshape(b_size, m, out.shape[-1])
        if tap_blocks:
            return out, taps
        return out


def build_encoder(
    *, n_regions: int, deep_sup: bool = True, region_embed: bool = True,
    space_rope: bool = True, d_model: int = ENC_D_MODEL,
) -> Encoder:
    return Encoder(
        d_model=d_model, n_heads=n_heads_for(d_model), n_regions=n_regions,
        deep_sup=deep_sup, sup_taps=ENC_SUP_TAPS if deep_sup else (),
        region_embed=region_embed, space_rope=space_rope,
    )
