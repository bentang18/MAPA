"""MAPA: a masked autoencoder for intracranial EEG.

The frozen encoder uses an anatomical region embedding and a relative positional encoding.
It takes preprocessed recordings and array metadata and returns per-contact features.
"""
from mapa.encoder import (
    DEPTH,
    FPS,
    N_REGIONS,
    MapaEncoder,
    SessionGrid,
    pool_to_regions,
    stem_input,
)
from mapa.models.sidecar import SensorSidecar, build_sidecar

__all__ = [
    "DEPTH",
    "FPS",
    "N_REGIONS",
    "SensorSidecar",
    "SessionGrid",
    "MapaEncoder",
    "build_sidecar",
    "pool_to_regions",
    "stem_input",
]
__version__ = "0.1.0"
