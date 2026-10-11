"""Per-band token embeddings from normalized magnitude STFT inputs.

Main path: PerBandStem.forward().

Input: three normalized clip tensors (B,C,F_b,T32).
Output: three token tensors (B,C,T_b,d), plus shared-clock positions.
Next: pack_r4.pack_band_tokens() combines the bands into a flat sequence.

Input order is slow, mid, fast. All inputs use a 32 Hz frame clock; strided
slicing reduces the token rates to 4, 16, and 32 Hz. Each band's linear layer
maps its frequency bins to the model width, then adds a learned band embedding.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from mapa.models.pe import init_transformer_weights

# (frequency bins, time stride), in slow/mid/fast order.
PER_BAND_SPECS: tuple[tuple[int, int], ...] = ((7, 8), (6, 2), (7, 1))
INPUT_CLIP_Z = (15.0, 15.0, 20.0)  # Slow, Mid, Fast: published Guard 3 caps.


def clip_band(x: Tensor, band: int) -> Tensor:
    """Bound normalized model inputs without modifying the caller's tensor."""
    cap = INPUT_CLIP_Z[band]
    return x.clamp(-cap, cap)

# Initial standard deviation of the learned band embeddings.
BAND_EMB_INIT_STD: float = 1e-6


class PerBandStem(nn.Module):
    """Subsample, clip, transpose, project, and add band identity.

    For a two-second clip: (B, C, F_b, 64) -> (B, C, T_b, d),
    where F_b = 7/6/7 and T_b = 8/32/64 for slow/mid/fast.
    Projection weights are shared across contacts and times within each band.
    Returned positions use the original 32 Hz clock, not the subsampled index.
    """

    def __init__(
        self,
        d_model: int = 256,
        *,
        bands: Sequence[tuple[int, int]] = PER_BAND_SPECS,
        band_emb_std: float = BAND_EMB_INIT_STD,
    ) -> None:
        super().__init__()
        self.specs = tuple((int(nb), int(st)) for nb, st in bands)
        self.projs = nn.ModuleList(nn.Linear(nb, d_model) for nb, _ in self.specs)
        # One learned width-d vector per band, broadcast over contacts and time.
        self.band_type_emb = nn.Parameter(torch.empty(len(self.specs), d_model))
        self.projs.apply(init_transformer_weights)  # V-JEPA trunc_normal(0.02)+zero-bias
        nn.init.trunc_normal_(self.band_type_emb, std=band_emb_std)

    @staticmethod
    def decimate(x: Tensor, stride: int) -> Tensor:
        """Strided time slice ``x[..., ::stride]`` — the band's own-rate frames. Requires
        the 32 Hz length to be an exact multiple of ``stride`` (64 % {1,2,8} == 0)."""
        t32 = x.shape[-1]
        if t32 % stride != 0:
            raise ValueError(f"32 Hz length {t32} not a multiple of stride {stride}")
        return x[..., ::stride]

    def forward(
        self, band_inputs: Sequence[Tensor]
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Bands ``[(...,F_b,T32)]`` → (per-band tokens ``(..., T_b, d)``, per-band lattice
        positions ``(T_b,)`` long). Order is SLOW, MID, HGA (the spec order)."""
        if len(band_inputs) != len(self.specs):
            raise ValueError(f"expected {len(self.specs)} bands, got {len(band_inputs)}")
        tokens: list[Tensor] = []
        positions: list[Tensor] = []
        for b, (x, (n_bins, stride), proj) in enumerate(
            zip(band_inputs, self.specs, self.projs)
        ):
            if x.shape[-2] != n_bins:
                raise ValueError(
                    f"band {b} has {x.shape[-2]} freq bins, expected {n_bins}"
                )
            # 1. Subsample time and clip normalized inputs.
            xd = self.decimate(x, stride)  # (..., F_b, T_b)
            xd = clip_band(xd, b)          # same shape; cap extreme normalized values
            # 2. Put frequency last, then project F_b -> d and add band identity.
            xd = xd.transpose(-1, -2)  # (..., T_b, F_b)
            tok = proj(xd)                       # (..., T_b, F_b) -> (..., T_b, d)
            tok = tok + self.band_type_emb[b]    # add the same band vector to each token
            t_b = xd.shape[-2]
            # 3. Keep positions on the shared clock: slow 0,8,...; mid 0,2,...; fast 0,1,...
            pos = torch.arange(t_b, device=x.device, dtype=torch.long) * stride  # (T_b,)
            tokens.append(tok)
            positions.append(pos)
        return tuple(tokens), tuple(positions)
