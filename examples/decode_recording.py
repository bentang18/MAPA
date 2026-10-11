"""Synthetic raw recording -> preprocessing -> frozen features -> ridge predictions.

Run from the repository root:
  python examples/decode_recording.py
  python examples/decode_recording.py --weights /path/to/mapa_vits384.pt
This demonstrates the API, not a scientific benchmark or the paper's split.

Story: prepare the whole recording -> select/normalize a clip -> stem -> pack
-> transformer -> per-contact features -> task readout. Read the data path first;
encoder.py handles loading/metadata/output shape, and tower.py wires the stages.
This example uses 4 contacts and 1-second clips (52 tokens/contact).
"""

import argparse

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mapa.hub.backbones import mapa_vits384
from mapa.preprocessing import prepare_recording, region_ids_from_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", help="local checkpoint; otherwise download the released model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--save-preprocessing", help="new directory for caches and preprocessing.json"
    )
    args = parser.parse_args()
    rng = np.random.default_rng(7)
    fs, seconds = 2048, 32
    voltage = rng.normal(size=(4, fs * seconds)).astype(np.float32) * 100
    labels = ["LA1", "LB1", "LA2", "LB2"]
    regions = region_ids_from_names(["ctx-lh-superiortemporal", "ctx-rh-superiortemporal"] * 2)
    # Raw voltage -> cached magnitude STFT bands, session normalization stats, metadata.
    recording = prepare_recording(voltage, fs, labels, regions)
    encoder = mapa_vits384(weights=args.weights, device=args.device)
    # Metadata setup only: token coordinates and array boundaries, no signal encoding.
    # This demo uses one-second clips: 32 frontend frames, 52 tokens/contact.
    session = encoder.prepare(recording.sidecar, n_time=32)
    rows = []
    # Select 24 one-second windows, leaving margins in the synthetic recording.
    for start in range(4, 28):
        # window(): slice + normalize bands. encoder(): stem -> pack -> transformer.
        # Data-flow reading order: frontend.py -> stem.py -> pack_r4.py -> towers.py.
        feature = encoder(recording.window(start, device=args.device), session, taps=(12,))[12]
        # Layer 12: (1,4,52,384). Attention was independent per array; flatten ALL
        # arrays in this window into one ridge input (not a separate ridge per array).
        rows.append(feature.flatten(1).cpu().numpy()[0])
    x = np.stack(rows)
    y = rng.integers(0, 2, len(x))  # substitute your task labels, aligned with window starts
    # Fit both scaling and readout on training rows only. Keep later rows for testing.
    readout = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    readout.fit(x[:16], y[:16])
    scores = readout.predict(x[16:])
    print("Features:", x.shape, "held-out scores:", scores.shape)
    if args.save_preprocessing:
        recording.save(args.save_preprocessing, subject_id=1, trial_id=0)
    assert np.isfinite(scores).all()


if __name__ == "__main__":
    main()
