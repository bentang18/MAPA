"""Pretrained encoders exposed through ``torch.hub``.

Four checkpoints: the released model and the three ablation arms behind the paper's Figure 4.
Each records in its own metadata whether the relative positional encoding is active, because
that setting leaves no trace in the weights.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

import torch

# The checkpoints are attached to the v0.1.0 GitHub release. torch.hub downloads them by name
# on first use and caches them under ``torch.hub.get_dir()``. An explicit ``weights=`` path or
# the MAPA_WEIGHTS environment variable takes precedence over the download.
_BASE_URL: str | None = "https://github.com/bentang18/MAPA/releases/download/v0.1.0"

_MODELS = {
    "mapa_vits384": "mapa_vits384.pt",
    "mapa_vits384_no_region": "mapa_vits384_no_region.pt",
    "mapa_vits384_no_relpos": "mapa_vits384_no_relpos.pt",
    "mapa_vits384_no_priors": "mapa_vits384_no_priors.pt",
}


def _is_url(path: str) -> bool:
    return urlparse(path).scheme in ("http", "https")


def _resolve(name: str, weights: str | None) -> str:
    env = os.environ.get("MAPA_WEIGHTS")
    if weights is not None:
        if _is_url(weights) or os.path.exists(weights):
            return weights
        # A bare filename resolves against MAPA_WEIGHTS, so `weights="mapa_vits384.pt"` works
        # from any directory once the env var points at the checkpoints.
        if env and os.path.isdir(env):
            cand = os.path.join(env, os.path.basename(weights))
            if os.path.exists(cand):
                return cand
        tried = [weights] + ([os.path.join(env, os.path.basename(weights))]
                             if env and os.path.isdir(env) else [])
        raise FileNotFoundError(
            f"no checkpoint for {name}. Tried: " + ", ".join(repr(t) for t in tried) +
            ". Pass weights='/absolute/path/to/" + _MODELS[name] + "', or set MAPA_WEIGHTS "
            "to the directory holding the released checkpoints."
        )
    if env:
        src = os.path.join(env, _MODELS[name]) if os.path.isdir(env) else env
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"MAPA_WEIGHTS is set but {src!r} does not exist. It should be the directory "
                f"holding the released checkpoints, or the path to {_MODELS[name]} itself."
            )
        return src
    if _BASE_URL:
        return f"{_BASE_URL}/{_MODELS[name]}"
    raise ValueError(
        f"no weights for {name}. Pass weights='/path/to/{_MODELS[name]}', or set MAPA_WEIGHTS "
        "to the file or to the directory holding the released checkpoints."
    )


def _build(name: str, *, pretrained: bool, weights: str | None, device, **kw):
    from mapa.encoder import MapaEncoder

    if not pretrained:
        raise ValueError(
            "This entry point requires pretrained=True. "
            "Use tap 0 for the frontend baseline."
        )
    src = _resolve(name, weights)
    if _is_url(src):
        torch.hub.load_state_dict_from_url(src, map_location="cpu", weights_only=True)
        src = os.path.join(torch.hub.get_dir(), "checkpoints",
                           os.path.basename(urlparse(src).path))
    return MapaEncoder.from_checkpoint(src, device=device, **kw)


def mapa_vits384(*, pretrained: bool = True, weights: str | None = None, device="cpu", **kw):
    """The released encoder. ViT-Small, 12 blocks, both spatial encodings."""
    return _build("mapa_vits384", pretrained=pretrained, weights=weights, device=device, **kw)


def mapa_vits384_no_region(*, pretrained: bool = True, weights: str | None = None,
                           device="cpu", **kw):
    """Ablation: no region embedding. The relative positional encoding stays."""
    return _build("mapa_vits384_no_region", pretrained=pretrained, weights=weights,
                  device=device, **kw)


def mapa_vits384_no_relpos(*, pretrained: bool = True, weights: str | None = None,
                           device="cpu", **kw):
    """Ablation: no relative positional encoding. The region embedding stays."""
    return _build("mapa_vits384_no_relpos", pretrained=pretrained, weights=weights,
                  device=device, **kw)


def mapa_vits384_no_priors(*, pretrained: bool = True, weights: str | None = None,
                           device="cpu", **kw):
    """Ablation: neither encoding. The array is an unordered set of contacts."""
    return _build("mapa_vits384_no_priors", pretrained=pretrained, weights=weights,
                  device=device, **kw)
