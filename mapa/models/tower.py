"""The frozen encoder: patch embedding plus transformer blocks.

Pretraining wraps this in a masked autoencoder. The decoder reconstructs the frequency bins of
the removed patches and is discarded when pretraining ends, so the released model is this module
and nothing else. It runs the full set of patches, never a masked subset.
"""
from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from mapa.models.pack_r4 import R4Grid, pack_band_tokens
from mapa.models.stem import PerBandStem
from mapa.models.towers import build_encoder


class MaeEncoder(nn.Module):
    """Patch embedding plus the encoder blocks, over one session's token grid."""

    def __init__(
        self,
        *,
        n_regions: int,
        deep_sup: bool = True,
        region_embed: bool = True,
        space_rope: bool = True,
        d_model: int,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.stem = PerBandStem(self.d_model)
        self.encoder = build_encoder(
            n_regions=n_regions, deep_sup=deep_sup, region_embed=region_embed,
            space_rope=space_rope, d_model=self.d_model,
        )

    def forward(
        self,
        bands: Sequence[Tensor],
        grid: R4Grid,
        region_packed: Tensor,
        *,
        tap_blocks: tuple[int, ...] = (),
    ) -> Tensor | tuple[Tensor, dict[int, Tensor]]:
        tokens, _ = self.stem(bands)
        x = pack_band_tokens(tokens, grid)
        return self.encoder.forward_flat(x, grid, region_packed, tap_blocks=tap_blocks)
