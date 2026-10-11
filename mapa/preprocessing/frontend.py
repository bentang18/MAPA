"""The released checkpoint's offline voltage-to-bands recipe.

Main path: prepare_recording() -> PreparedRecording.window().

Input: a full voltage recording plus contact labels/regions.
Output: cached STFT magnitudes and statistics; window() selects normalized clips.
Next: models/stem.py turns each clip's frequency bins into learned tokens.

Process a whole session, then slice windows on its 32 Hz clock. Raw arrays are
channels × samples. Supply cleaned input and any explicit contact exclusions. No task labels are used.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from mapa.data.anatomy import V14_DKT_REGION_LABELS
from mapa.data.normalize import SessionRobustZNormalizer
from mapa.models.sidecar import SensorSidecar, build_sidecar
from mapa.models.stem import INPUT_CLIP_Z
from mapa.preprocessing.stft import _single_stft_raw_view_chunked

SAMPLE_RATE = 2048
FRAME_RATE = 32
# name, FFT length, first bin, last bin (inclusive). Hz = bin * 2048 / FFT length.
# Bin centers (inclusive): slow 2-14 Hz by 2; mid 16-56 by 8; fast 64-160 by 16.
# Window lengths: 500/125/62.5 ms; all three use hop=64 samples (31.25 ms).
BANDS = (("slow", 1024, 1, 7), ("mid", 256, 2, 7), ("fast", 128, 4, 10))


def region_ids_from_names(names):
    """Map exact hemisphere-qualified DKT names; None explicitly selects ID 74.

    Unknown strings raise, so misspellings cannot silently become unassigned.
    Anatomical localization/atlas assignment must be performed upstream.
    """
    lookup = {name: i for i, name in enumerate(V14_DKT_REGION_LABELS)}
    return torch.tensor([74 if name is None else lookup[name] for name in names], dtype=torch.long)


def _filter(data, labels):
    import mne

    raw = mne.io.RawArray(
        np.asarray(data, dtype=np.float64),
        mne.create_info(list(labels), SAMPLE_RATE, "seeg"),
        verbose=False,
    )
    freqs = np.arange(60.0, min(SAMPLE_RATE / 2, 301), 60.0)
    raw.notch_filter(freqs, phase="zero", filter_length="auto", verbose=False)
    raw.filter(0.5, None, verbose=False)
    return raw.get_data()


def _reference(data, sidecar):
    data = np.array(data, copy=True)
    for array in range(sidecar.n_arrays):
        ids = np.flatnonzero(sidecar.array_id.numpy() == array)
        data[ids] -= data[ids].mean(axis=0, keepdims=True)
    return data


def preprocessing_config():
    """Return a fresh, JSON-serializable description of the released frontend."""
    return {
        "recipe": "mapa-v0.1.0",
        "sample_rate_hz": SAMPLE_RATE,
        "frame_rate_hz": FRAME_RATE,
        "resample": "scipy.signal.resample_poly; default Kaiser window; input/output float32",
        "model_reference": "mean of surviving selected contacts per shaft, before model filters",
        "model_notches_hz": [60, 120, 180, 240, 300],
        "highpass_hz": 0.5,
        "filter": "MNE default FIR; zero phase; reflect_limited padding",
        "contact_exclusions": "supplied by caller",
        "artifact_cleaning": "external contact detection and window rejection",
        "model_input_clip_z": list(INPUT_CLIP_Z),
        "bands": [
            {"name": n, "n_fft": fft, "hop": 64, "bins_inclusive": [lo, hi]}
            for n, fft, lo, hi in BANDS
        ],
        "stft": {
            "window": "periodic Hann",
            "center": True,
            "padding": "reflect",
            "normalized": False,
            "representation": "magnitude; no log",
        },
        "normalization": {
            "fit": "entire session, separately per contact and frequency bin",
            "median": "torch lower median",
            "mad_scale": 1.4826,
            "sigma_floor": 1e-6,
            "constant_bins": "zero",
        },
        "window_start": "round(seconds * 32), ties to even",
        "voltage_scaling": "none",
    }


@dataclass
class PreparedRecording:
    """Raw STFT caches plus frozen statistics; normalized clips are created on demand."""

    bands: tuple[torch.Tensor, ...]
    normalizers: tuple[SessionRobustZNormalizer, ...]
    sidecar: SensorSidecar
    input_indices: np.ndarray
    duration_s: float
    metadata: dict

    def window(self, start_s, duration_s=1.0, *, device="cpu"):
        """Return three (1, contacts, bins, frames) normalized tensors.

        Reject windows outside the recording. No artifact detection or rejection
        is performed. Window selection is the caller's responsibility.
        """
        if not math.isfinite(start_s) or start_s < 0:
            raise ValueError("start_s must be finite and nonnegative")
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("duration_s must be finite and positive")
        frames = round(duration_s * FRAME_RATE)
        if frames < 8 or frames % 8 or not math.isclose(frames / FRAME_RATE, duration_s):
            raise ValueError("duration must be a positive multiple of 0.25 seconds")
        start = round(start_s * FRAME_RATE)
        snapped = start / FRAME_RATE
        if start_s + duration_s > self.duration_s or snapped + duration_s > self.duration_s:
            raise ValueError("window extends beyond the recording")
        # Extract a clip from the cached STFT frames, not another FFT window.
        # Normalize with the full-session median/sigma fitted in prepare_recording():
        # z = (clip - median) / sigma; bins with sigma < 1e-6 become zero. No refit.
        # (C,F_b,session_frames) -> (C,F_b,frames) -> (1,C,F_b,frames).
        clip_bands = []
        for band, norm in zip(self.bands, self.normalizers):
            clip = band[..., start : start + frames]  # select cached STFT frames
            normalized = norm.transform(clip)        # use stored session statistics
            batched = normalized.unsqueeze(0)        # add the one-clip batch axis
            clip_bands.append(batched.to(device))
        return clip_bands

    def save(self, output_dir, *, subject_id, trial_id):
        """Write raw band caches/statistics compatible with the evaluation loader.

        Use a new root per recording. Existing directories are never overwritten.
        No raw voltage is written. Configuration/selection/statistics accompany the caches.
        """
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=False)
        key = json.dumps({"subject_id": int(subject_id), "trial_id": int(trial_id)})
        for (name, fft, lo, hi), band, norm in zip(BANDS, self.bands, self.normalizers):
            leaf = root / name
            leaf.mkdir()
            np.save(leaf / "session.npy", band.numpy())
            np.savez(
                leaf / "session.stats.npz", median=norm.median.numpy(), sigma=norm.sigma.numpy()
            )
            meta = {
                "key": key,
                "ch_names": list(self.sidecar.labels),
                "total_frames": band.shape[-1],
                "sample_rate": SAMPLE_RATE,
                "band_hop": 64,
                "band_nperseg": fft,
                "f_bins": hi - lo + 1,
                "n_channels": band.shape[0],
                "dtype": "float32",
            }
            (leaf / "session.json").write_text(json.dumps(meta, indent=2) + "\n")
        (root / "spans").mkdir()
        (root / "spans/session.json").write_text(
            json.dumps(
                {
                    "subject_id": int(subject_id),
                    "trial_id": int(trial_id),
                    "bad_windows_s": [],
                },
                indent=2,
            )
            + "\n"
        )
        (root / "preprocessing.json").write_text(json.dumps(self.metadata, indent=2) + "\n")


def prepare_recording(
    voltage,
    sample_rate,
    labels,
    region_ids,
    *,
    bad_labels=(),
    keep_labels=None,
    chunk_frames=960,
):
    """Prepare a full unreferenced session using the released offline recipe.

    Preserve acquisition units (no implicit conversion to MNE volts). The original
    Brain Treebank voltage scale is retained by the research loader; do not assume
    it was converted to volts. Fixed numerical floors make arbitrary rescaling
    non-equivalent for near-constant channels. See the preprocessing guide.

    bad_labels supplies contact exclusions determined upstream.
    This function does not detect bad contacts. keep_labels restricts the output
    montage before referencing.
    This is important for reproducing Neuroprobe Lite's contact budget.
    """
    if not isinstance(chunk_frames, int) or chunk_frames <= 0:
        raise ValueError("chunk_frames must be a positive integer")
    if not math.isfinite(sample_rate) or sample_rate < 2 * 160 or int(sample_rate) != sample_rate:
        raise ValueError("sample_rate must be an integer >= 320 Hz; paper rates were 1024/2048 Hz")
    x = np.asarray(voltage, dtype=np.float32)
    labels = tuple(labels)
    ids = np.asarray(region_ids)
    if x.ndim != 2 or x.shape[0] != len(labels) or ids.shape != (len(labels),):
        raise ValueError("voltage, labels and region_ids must have matching contact axes")
    if not len(labels) or len(set(labels)) != len(labels):
        raise ValueError("provide unique contact labels")
    if not np.issubdtype(ids.dtype, np.integer) or np.any((ids < 0) | (ids > 74)):
        raise ValueError("region_ids must be integer DKT indices 0..74")
    if x.shape[1] / sample_rate < 5:
        raise ValueError("supply a full session (at least 5 seconds), not an isolated trial")
    bad = set(bad_labels)
    keep = set(labels) if keep_labels is None else set(keep_labels)
    if (bad | keep) - set(labels):
        raise ValueError("bad_labels/keep_labels contain labels absent from the recording")
    # 1. Select contacts; keep the voltage rows and sidecar metadata aligned.
    original_indices = np.array([
        i for i, label in enumerate(labels) if label not in bad and label in keep
    ], dtype=np.int64)
    if not original_indices.size:
        raise ValueError("no contacts remain after supplied exclusions and montage selection")
    x = x[original_indices]
    if not np.isfinite(x).all():
        raise ValueError("selected contacts must have finite voltage")
    sidecar = build_sidecar([labels[i] for i in original_indices], region_id=ids[original_indices])
    if any(int((sidecar.array_id == i).sum()) < 2 for i in range(sidecar.n_arrays)):
        raise ValueError("a surviving shaft has fewer than two contacts for mean referencing")
    # 2. Resample to 2048 Hz when needed.
    factor = math.gcd(SAMPLE_RATE, int(sample_rate))
    if sample_rate != SAMPLE_RATE:
        x = resample_poly(x, SAMPLE_RATE // factor, int(sample_rate) // factor, axis=1).astype(
            np.float32
        )
    # 3. Subtract each shaft's mean, then apply harmonic notches and a high-pass filter.
    model_voltage = _filter(_reference(x.astype(np.float64), sidecar), sidecar.labels).astype(
        np.float32
    )
    waveform = torch.from_numpy(model_voltage)
    bands, normalizers = [], []
    # 4. Voltage (C,samples) -> magnitude STFT (C,F_b,session_frames), no log.
    # Slow: 2-14 Hz, 7 bins, 500 ms window; mid: 16-56 Hz, 6 bins, 125 ms;
    # fast: 64-160 Hz, 7 bins, 62.5 ms. Ranges are bin centers, inclusive.
    # All bands use a 31.25 ms hop (32 frames/s); the stem subsamples later.
    for _, fft, lo, hi in BANDS:
        band = _single_stft_raw_view_chunked(
            waveform,
            sample_rate=SAMPLE_RATE,
            nperseg=fft,
            hop_length=64,
            k0=lo,
            k1=hi,
            log_eps=1e-6,
            chunk_frames=chunk_frames,
        )
        # 5. Fit once over session time, separately for each contact/frequency bin:
        # median and sigma = 1.4826 * median(abs(band - median)), both (C,F_b,1).
        # Keep raw magnitudes + these statistics; window() applies normalization.
        norm = SessionRobustZNormalizer(sigma_floor=1e-6).fit(band)
        bands.append(band)
        normalizers.append(norm)
    metadata = {
        "recipe": preprocessing_config(),
        "input_sample_rate_hz": sample_rate,
        "input_units": "unchanged",
        "input_labels": list(labels),
        "output_labels": list(sidecar.labels),
        "input_indices": original_indices.tolist(),
        "region_ids": sidecar.region_id.tolist(),
        "known_exclusions": sorted(bad),
        "versions": {name: version(name) for name in ("numpy", "scipy", "torch", "mne")},
    }
    return PreparedRecording(
        tuple(bands),
        tuple(normalizers),
        sidecar,
        original_indices,
        x.shape[1] / SAMPLE_RATE,
        metadata,
    )
