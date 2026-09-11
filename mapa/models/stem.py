"""The frontend: three bands of magnitude STFT folded into one patch per contact and slot.

Each band arrives as ``(..., n_bins, n_frames)``, already robust z-scored per contact and
bin at load. Every band is extracted on the same 32 Hz clock, and each is decimated here by
its own stride onto the token lattice, so the slow band contributes one token where the fast
band contributes eight.

The fold is one weight-shared linear layer per band. There is no frequency embedding and no
per-band normalisation: a per-band norm would restore the low frequency dominance the robust
z-score removes.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from mapa.models.pe import init_transformer_weights

# (n_bins, hold_factor) per band, in concat order: SLOW 7, MID 6, HGA 7 = 20 ch.
# hold_factor = T32 / T_band = 32Hz / band-frame-rate. With the uniform-hop=64 fix
# (2026-07-10) every band is extracted at 32 Hz ⇒ all factors are 1 (no hold), so the
# (n_bins, stride) per band, in the order slow, mid, fast. The stride is the decimation
# step on the shared 32 Hz clock: fast 1 (32 Hz), mid 2 (16 Hz), slow 8 (4 Hz).
PER_BAND_SPECS: tuple[tuple[int, int], ...] = ((7, 8), (6, 2), (7, 1))
INPUT_CLIP_Z = (15.0, 15.0, 20.0)  # Slow, Mid, Fast: published Guard 3 caps.


def clip_band(x: Tensor, band: int) -> Tensor:
    """Bound normalized model inputs without modifying the caller's tensor."""
    cap = INPUT_CLIP_Z[band]
    return x.clamp(-cap, cap)

# Anything added to the residual starts near zero, following V-JEPA 2.1, so the model grows
# into it rather than carrying a strong prior from the first step.
BAND_EMB_INIT_STD: float = 1e-6


class PerBandStem(nn.Module):
    """Turn three bands of magnitude STFT into patch embeddings.

    Each band arrives on the shared 32 Hz clock as ``(..., n_bins, n_frames)``. The stem:

      1. decimates each band to its own token rate by a strided slice in time. The frames it
         drops are the ones the analysis windows already overlap, so consecutive tokens of a
         band overlap by exactly one half window and no more.
      2. projects each band with its own linear layer, whose separate weights identify the
         band, and adds a learned per-band embedding.
      3. reports each token's position on the shared clock, which is the coordinate the
         rotary encoding reads.
         inherit the same per-unit frequency at wider strides, so a SLOW token at lattice
         8k and an HGA token at 8k share phase ⇒ band mixing aligns them in physical time.

    Ragged BY DESIGN: bands carry different token counts (T32 / stride). Deliberately NO
    freq embed and NO per-band norm (per-band norm reintroduces within-band 1/f dominance);
    band identity rides the separate projections + the additive band embed.
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
        # additive per-band identity, one d-vector per band. NEAR-ZERO 1e-6 init, the V-JEPA 2.1 convention for anything ADDED to the residual: the
        # modality embed inits at std 1e-6 and mask_token at zero, so the model GROWS into
        # the embed rather than carrying a strong prior from step 0. This is the same
        # argument already applied to RegionIdentityEmbed (pe.py:194-202); the band embed is
        # added to every token at the stem and rides every block, so at 0.02 it was a large
        # fraction of the token signal at init. 0.02 is retained as the A/B arm via
        # band_emb_std. 🪤 NO TRACE: the init leaves nothing in a converged state_dict.
        self.band_type_emb = nn.Parameter(torch.empty(len(self.specs), d_model))
        self.projs.apply(init_transformer_weights)  # V-JEPA trunc_normal(0.02)+zero-bias
        nn.init.trunc_normal_(self.band_type_emb, std=band_emb_std)

    @staticmethod
    def decimate(x: Tensor, stride: int) -> Tensor:
        """Strided time slice ``x[..., ::stride]`` — the band's own-rate frames. Requires
        the 32 Hz length to be an exact multiple of ``stride`` (128 % {1,2,8} == 0)."""
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
            xd = clip_band(self.decimate(x, stride), b)  # (..., F_b, T_b)
            xd = xd.transpose(-1, -2)  # (..., T_b, F_b)
            tok = proj(xd) + self.band_type_emb[b]  # (..., T_b, d)
            t_b = xd.shape[-2]
            pos = torch.arange(t_b, device=x.device, dtype=torch.long) * stride  # (T_b,)
            tokens.append(tok)
            positions.append(pos)
        return tuple(tokens), tuple(positions)
