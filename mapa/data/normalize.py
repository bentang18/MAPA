"""Session-wise median/MAD normalization. No artifact detection or clipping."""

from __future__ import annotations

import torch

SCALE_TO_SIGMA = 1.4826


class SessionRobustZNormalizer:
    """Fit each contact/bin on a complete session, then reuse the frozen statistics.

    This scaler does not identify artifacts or clip normalized values.
    """

    def __init__(self, *, sigma_floor: float = 1e-6) -> None:
        self.sigma_floor = sigma_floor
        self.median: torch.Tensor | None = None
        self.sigma: torch.Tensor | None = None
        self._valid_bin_mask: torch.Tensor | None = None

    @classmethod
    def from_stats(
        cls,
        *,
        median: torch.Tensor,
        sigma: torch.Tensor,
        sigma_floor: float = 1e-6,
        valid_bin_mask: torch.Tensor | None = None,
    ) -> SessionRobustZNormalizer:
        obj = cls(sigma_floor=sigma_floor)
        obj.median, obj.sigma = median, sigma
        obj._valid_bin_mask = valid_bin_mask
        return obj

    def fit(
        self, frames: torch.Tensor, *, valid_bin_mask: torch.Tensor | None = None,
    ) -> SessionRobustZNormalizer:
        x = (
            frames
            if frames.dtype in (torch.float16, torch.float32, torch.float64)
            else frames.float()
        )
        self.median = x.median(dim=-1, keepdim=True).values
        self.sigma = SCALE_TO_SIGMA * (x - self.median).abs().median(dim=-1, keepdim=True).values
        self._valid_bin_mask = valid_bin_mask
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.median is None or self.sigma is None:
            raise RuntimeError("SessionRobustZNormalizer.transform called before fit()")
        x = x if x.dtype in (torch.float16, torch.float32, torch.float64) else x.float()
        z = (x - self.median) / self.sigma.clamp(min=self.sigma_floor)
        z = torch.where(self.sigma >= self.sigma_floor, z, torch.zeros_like(z))
        if self._valid_bin_mask is not None:
            z = torch.where(self._valid_bin_mask.to(dtype=torch.bool), z, torch.zeros_like(z))
        return z

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)
