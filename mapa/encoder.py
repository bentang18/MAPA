"""Frozen encoder API for per-contact feature extraction from intracranial EEG.

Inputs are three normalized magnitude STFT tensors and a SensorSidecar with the array identity,
clinical contact number, and anatomical region of each contact. Outputs are keyed by encoder
block. Block 0 returns the frontend baseline features; block 12 returns the final encoder features.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import torch
from torch import Tensor, nn

from mapa.models.geometry import build_l1_geometry
from mapa.models.pack_r4 import BAND_STRIDES, build_r4_grid
from mapa.models.sidecar import SensorSidecar
from mapa.models.stem import clip_band
from mapa.models.tower import MaeEncoder

N_REGIONS = 75      # DKT K=74 plus one reserved id for contacts outside every region
FPS = 32.0          # the uniform frame clock every band is resampled onto
DEPTH = 12          # encoder blocks; 3/6/9/12 carry deep-supervision norms


@dataclass(frozen=True)
class SessionGrid:
    """Per-session token layout. Clip-independent, so build it once and reuse it."""

    grid: object
    region_packed: Tensor
    contact_order: Tensor   # (n,) indices into the sidecar row axis, in grid order
    region_of_contact: Tensor   # (n,) DKT id per contact, grid order
    k_full: int
    n_time: int


class MapaEncoder(nn.Module):
    """The frozen encoder that ships as the representation.

    Pretraining is masked autoencoding. The decoder that reconstructs the removed patches is
    discarded when pretraining ends, so this is the whole released model. Loading reads the
    architecture off the checkpoint rather than trusting arguments, so a mismatched shell
    fails loud.
    """

    def __init__(self, tower: MaeEncoder, *, space_rope: bool, region_embed: bool) -> None:
        super().__init__()
        self.tower = tower
        self.space_rope = bool(space_rope)
        self.region_embed = bool(region_embed)

    @property
    def d_model(self) -> int:
        return int(self.tower.d_model)

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        *,
        device: str | torch.device = "cpu",
        space_rope: bool | None = None,
    ) -> MapaEncoder:
        """Load a released checkpoint.

        A released file records whether the relative positional encoding is active, because
        that setting cannot be recovered from the weights: disabling it zeroes a buffer that
        is not persisted, so the state-dict layout alone does not identify that ablation.
        Pass ``space_rope`` only to override what the file records, or to read a raw training
        checkpoint, which records nothing.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no checkpoint at {path!r}. Pass an absolute path, or set MAPA_WEIGHTS to the "
                "directory holding the released checkpoints and load through torch.hub."
            )
        raw = torch.load(path, map_location="cpu", weights_only=True)
        sub, recorded = _tower_state(raw)
        if space_rope is None:
            if recorded is None:
                raise ValueError(
                    f"{path} does not record whether the relative positional encoding is "
                    "active, so it cannot be inferred. Pass space_rope explicitly."
                )
            space_rope = recorded

        peek = [v.shape[0] for kk, v in sub.items() if kk.endswith("region_embed.embed.weight")]
        if peek and int(peek[0]) != N_REGIONS:
            raise ValueError(f"checkpoint region table {peek[0]} != expected {N_REGIONS}")
        dkey = "encoder.blocks.0.norm1.weight"
        if dkey not in sub:
            raise RuntimeError(f"{path} has no '{dkey}'; cannot infer encoder width")

        region_embed = bool(peek)
        deep_sup = any(kk.startswith("encoder.norms_block.") for kk in sub)
        d_model = int(sub[dkey].shape[0])

        shell = MaeEncoder(
            n_regions=N_REGIONS, deep_sup=deep_sup, region_embed=region_embed,
            space_rope=space_rope, d_model=d_model,
        )
        missing, unexpected = shell.load_state_dict(sub, strict=False)
        bad = [m for m in missing if "num_batches_tracked" not in m]
        if bad or unexpected:
            raise RuntimeError(
                f"state_dict mismatch: missing={bad[:8]} unexpected={unexpected[:8]}"
            )
        shell.eval().to(device)
        for p in shell.parameters():
            p.requires_grad_(False)
        return cls(shell, space_rope=space_rope, region_embed=region_embed)

    def prepare(self, sidecar: SensorSidecar, *, n_time: int) -> SessionGrid:
        """Build the token layout for one session. ``n_time`` is the window length in frames
        on the 32 Hz clock, so a one second window is 32."""
        if isinstance(n_time, bool) or not isinstance(n_time, Integral) or n_time <= 0:
            raise ValueError("n_time must be a positive integer on the 32 Hz frame clock")
        device = next(self.tower.parameters()).device
        geom = build_l1_geometry(sidecar).to(device)
        grid = build_r4_grid(geom, n_time=int(n_time))
        region_id = sidecar.region_id.to(device)
        contact_order = grid.contact.reshape(-1, grid.k_full)[:, 0]
        return SessionGrid(
            grid=grid,
            region_packed=region_id[grid.contact],
            contact_order=contact_order.cpu(),
            region_of_contact=region_id[contact_order].cpu(),
            k_full=int(grid.k_full),
            n_time=int(n_time),
        )

    @torch.no_grad()
    def forward(
        self,
        bands: Sequence[Tensor],
        session: SessionGrid,
        *,
        taps: Sequence[int] = (DEPTH,),
    ) -> dict[int, Tensor]:
        """Encode a batch of windows.

        Returns ``{block: (B, n_contacts, k_full, d)}`` in grid contact order. The full stack
        runs once whatever is asked for, so reading several blocks costs no extra forward.
        Tap 0 is the stem input and runs no encoder at all.
        """
        taps = tuple(taps)
        if not taps or any(
            isinstance(t, bool) or not isinstance(t, Integral) or not 0 <= t <= DEPTH
            for t in taps
        ):
            raise ValueError(f"taps must contain integer block numbers from 0 to {DEPTH}")
        _validate_bands(bands, session)
        device = next(self.tower.parameters()).device
        if session.grid.contact.device != device or bands[0].device != device:
            raise ValueError(
                "encoder, session and bands must be on the same device; move the encoder first, "
                "then call prepare() and move the bands to that device"
            )
        out: dict[int, Tensor] = {}
        gpu_taps = tuple(t for t in taps if t != 0)
        dtype = next(self.tower.parameters()).dtype
        if gpu_taps and bands[0].dtype != dtype and (
            torch.float64 in (bands[0].dtype, dtype) or not torch.is_autocast_enabled(device.type)
        ):
            raise ValueError(
                f"bands use {bands[0].dtype}, encoder uses {dtype}; convert bands with "
                f"[band.to(dtype={dtype}) for band in bands]"
            )
        if 0 in taps:
            out[0] = stem_input(bands, session)
        if gpu_taps:
            _z, tapped = self.tower.forward(
                list(bands), session.grid, session.region_packed, tap_blocks=gpu_taps
            )
            n = session.contact_order.shape[0]
            for t in gpu_taps:
                x = tapped[t]
                out[t] = x.reshape(x.shape[0], n, session.k_full, x.shape[-1])
        return {t: out[t] for t in taps}


# The region embedding was called ``parcel_embed`` before the rename, and checkpoints
# written then carry that key. Both spellings load.
_LEGACY_KEYS = (("parcel_embed", "region_embed"),)


def _canonical_keys(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        for old, new in _LEGACY_KEYS:
            if old in k:
                k = k.replace(old, new)
        out[k] = v
    return out


def _tower_state(raw) -> tuple[dict, bool | None]:
    """Pull the encoder weights out of either checkpoint layout.

    A released file is ``{"model": ..., "meta": ...}`` and carries bare keys. A raw training
    checkpoint carries the whole autoencoder, of which the encoder sits under
    ``objective.online.``; the decoder is training only and is dropped here.
    """
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        meta = raw.get("meta") or {}
        rope = meta.get("space_rope")
        return _canonical_keys(raw["model"]), (None if rope is None else bool(rope))

    sd = raw["state_dict"] if "state_dict" in raw else raw
    pref = "objective.online."
    sub = {}
    for k, v in sd.items():
        kk = k.replace("_orig_mod.", "")
        if kk.startswith("model."):
            kk = kk[len("model."):]
        if kk.startswith(pref):
            sub[kk[len(pref):]] = v
    if not sub:
        raise RuntimeError("no encoder weights found; not a released checkpoint layout")
    return _canonical_keys(sub), None


def stem_input(bands: Sequence[Tensor], session: SessionGrid) -> Tensor:
    """Tap 0: the exact tensor the stem's first linear layer consumes, in grid contact order.

    Each band is decimated by its own stride onto the token lattice, made time-major, reordered
    to the grid contact axis, and concatenated. A readout on this is the input-linear floor: the
    same clips, the same contacts, the same pooling, no encoder.
    """
    _validate_bands(bands, session)
    canon = session.contact_order
    per_band = []
    for b, (x, stride) in enumerate(zip(bands, BAND_STRIDES)):
        xd = clip_band(x[..., ::stride], b).transpose(-1, -2).contiguous()
        xd = xd[:, canon]
        per_band.append(xd.reshape(xd.shape[0], xd.shape[1], -1))
    return torch.cat(per_band, dim=-1)[:, :, None, :]


def _validate_bands(bands: Sequence[Tensor], session: SessionGrid) -> None:
    """Reject mismatched axes before indexing can silently discard contacts or frames."""
    if len(bands) != 3:
        raise ValueError("provide three bands in Slow, Mid, Fast order")
    n = len(session.contact_order)
    for name, bins, band in zip(("Slow", "Mid", "Fast"), (7, 6, 7), bands):
        expected = (n, bins, session.n_time)
        if not isinstance(band, Tensor) or band.ndim != 4 or tuple(band.shape[1:]) != expected:
            raise ValueError(f"{name} band must have shape (batch, {n}, {bins}, {session.n_time})")
        if not band.is_floating_point():
            raise ValueError(f"{name} band must be a floating-point tensor")
        if band.shape[0] < 1 or band.shape[0] != bands[0].shape[0]:
            raise ValueError("all bands must have the same nonempty batch axis")
        if band.device != bands[0].device or band.dtype != bands[0].dtype:
            raise ValueError("all bands must use the same device and dtype")


def pool_to_regions(x: Tensor, session: SessionGrid) -> Tensor:
    """Average contacts within each region: ``(B, n, ...)`` to ``(B, n_regions, F)``.

    Regions are returned in sorted ID order, as given by ``session.region_of_contact.unique()``.
    Align the region IDs shared by two sessions before fitting a cross-subject readout.
    """
    regions = session.region_of_contact
    present = torch.unique(regions)
    blocks = [x[:, regions == p].mean(1).reshape(x.shape[0], -1) for p in present.tolist()]
    return torch.stack(blocks, dim=1)
