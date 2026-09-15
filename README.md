# MAPA

**Pretraining for Sample-Efficient Neural Interfaces**

Ben Tang, Zachary Spalding, Gregory B. Cogan · Duke University

[![arXiv](https://img.shields.io/badge/arXiv-2609.13507-b31b1b.svg)](https://arxiv.org/abs/2609.13507)
[![Project page](https://img.shields.io/badge/project-page-1f6feb.svg)](https://bentang18.github.io/mapa-page/)
[![Checkpoints](https://img.shields.io/badge/checkpoints-v0.1.0-8250df.svg)](https://github.com/bentang18/MAPA/releases/tag/v0.1.0)
[![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey.svg)](LICENSE)

MAPA is a masked autoencoder that learns neural representations from unlabeled intracranial EEG.
An anatomical region embedding and a relative positional encoding let the model learn across
recordings with different contact placements and neuroanatomy. After pretraining, we freeze the
encoder and fit a linear probe for decoding.

MAPA achieves the highest AUROC in all three Neuroprobe evaluation regimes. In the cross-subject
regime, the probe reaches the frontend baseline's full-data accuracy with approximately 164
anchor-subject training trials rather than 3,500, a 21× saving. Target-session validation labels
are held fixed for regularization selection.

This repository provides preprocessing, the frozen encoder, four checkpoints, and evaluation code.

## Get started

Requires Python 3.10+ and PyTorch 2.6+.

```bash
git clone https://github.com/bentang18/MAPA.git
cd MAPA
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell.

Load the released encoder. The checkpoint downloads on first use and is cached locally:

```python
import torch

encoder = torch.hub.load(".", "mapa_vits384", source="local", device="cpu")
```

For a GPU, set `device="cuda"` and place the input tensors on the same device. You can also load
through `torch.hub.load("bentang18/MAPA", "mapa_vits384")` after installing the dependencies.

To use a downloaded checkpoint directly:

```python
from mapa import MapaEncoder

encoder = MapaEncoder.from_checkpoint("/path/to/mapa_vits384.pt", device="cpu")
```

The hub entry points also accept `weights="/path/to/mapa_vits384.pt"`. Set `MAPA_WEIGHTS` to a
checkpoint directory to use local files for all four models. Downloads and checksums are listed
in the [model card](MODEL_CARD.md#checkpoints).

### Start from recordings

Install the preprocessing dependency with `python -m pip install -e ".[preprocessing]"`.
Provide a continuous recording, its sampling rate, and contact metadata:

```python
import numpy as np
from mapa.preprocessing import prepare_recording, region_ids_from_names

voltage = np.load("recording.npy")  # contacts × samples, before shaft referencing
labels = ["LA1", "LB1", "LA3", "LB3"]
regions = region_ids_from_names([
    "ctx-lh-superiortemporal", "ctx-rh-superiortemporal",
    "ctx-lh-superiortemporal", "ctx-rh-superiortemporal",
])
recording = prepare_recording(voltage, 2048, labels, regions)
session = encoder.prepare(recording.sidecar, n_time=32)
features = encoder(recording.window(10.0), session, taps=(12,))[12]
```

The frontend handles resampling, filtering, shaft referencing, STFTs, and session normalization.
Supply cleaned recordings and any explicit contact exclusions through `bad_labels`. Preprocess
the whole recording once, then request windows. The Guard 1/2 detectors are not included.
For Neuroprobe, the release includes the frozen per-session Guard 1 contact exclusions;
apply them before shaft referencing as described in the evaluation guide. Guard 2 does not
reject evaluation windows. The model applies Guard 3: normalized Slow/Mid/Fast values are
clipped to ±15/±15/±20 before projection, including the frontend baseline.
See the [preprocessing guide](docs/PREPROCESSING.md) for contact selection, signal scale, atlas
mapping, and cache export. The [runnable example](examples/decode_recording.py) takes synthetic
voltage through preprocessing, feature extraction, and a fitted readout:

```bash
python examples/decode_recording.py
# Or use a checkpoint already on disk:
python examples/decode_recording.py --weights /path/to/mapa_vits384.pt
```

It prints the feature-matrix shape `(24, 79872)` and held-out-score shape `(8,)`. The example uses random
voltage and labels to demonstrate the API. For your task, supply labels aligned to the extracted
windows and fit a readout on training data. The checkpoint returns neural features; it does not
include a task-specific decoder or produce text.

**Recording geometry:** these checkpoints were trained and evaluated on sEEG shafts. The source
Brain Treebank inventory has at most 16 contacts per shaft; this is the observed data range,
not a hard model limit. Longer shafts and ECoG grids have not been validated. Preserve clinical
contact numbers, including gaps left by removed contacts.

### Use preprocessed bands

This example uses synthetic inputs for one second of recording from four contacts:

```python
from mapa import build_sidecar

# Contact labels and region IDs follow the input tensor's contact order.
labels = ["LA1", "LB1", "LA2", "LB2"]
sidecar = build_sidecar(labels, region_id=[0, 1, 0, 1])
session = encoder.prepare(sidecar, n_time=32)

# Synthetic normalized STFT bands: Slow, Mid, Fast.
bands = [torch.randn(1, 4, bins, 32) for bins in (7, 6, 7)]
features = encoder(bands, session, taps=(12,))[12]
print(features.shape)  # torch.Size([1, 4, 52, 384])

# The encoder groups contacts by array. Restore the input order if needed.
features = features[:, session.contact_order.argsort()]
```

If you already have preprocessed data, the encoder accepts:

- **Three normalized magnitude STFT bands**, each shaped `(batch, contacts, bins, time)`.
  Use `torch.float32` for the default checkpoint, including arrays converted from NumPy.
  Slow, Mid, and Fast have 7, 6, and 7 bins, all on a shared 32 Hz frame clock.
- **Clinical contact labels**, such as `LA3` for contact 3 of array LA. Preserve contact numbers
  when dropping channels.
- **Anatomical region IDs** in the order of `V14_DKT_REGION_LABELS` in
  [anatomy.py](mapa/data/anatomy.py). IDs 0–73 name the atlas regions; 74 is the reserved entry.
  Do not renumber the regions separately for each recording.

Use `prepare_recording` above to produce these inputs from voltage. Its saved configuration
and statistics record the preprocessing applied to each recording.
The [model card](MODEL_CARD.md#inputs-and-outputs) specifies band frequencies and output shapes.

Reuse `session` for windows with the same contact layout and duration. Request intermediate
features with `taps=(3, 6, 9, 12)`. Tap 0 returns the frontend baseline input features, before
projection into the encoder.

The [quickstart notebook](notebooks/quickstart.ipynb) gives a longer synthetic example. To run it
in the same activated environment:

```bash
python -m pip install notebook
python -m notebook notebooks/quickstart.ipynb
```

## Checkpoints

| Entry point | Region embedding | Relative positional encoding |
|---|---|---|
| `mapa_vits384` | Yes | Yes |
| `mapa_vits384_no_region` | No | Yes |
| `mapa_vits384_no_relpos` | Yes | No |
| `mapa_vits384_no_priors` | No | No |

Use `mapa_vits384` for feature extraction. The other checkpoints reproduce the spatial-encoding
ablations. All four have width 384, 12 blocks, and approximately 21.3M parameters. Each records
which spatial encodings are active. See the [model card](MODEL_CARD.md) for exact parameter
counts, training data, checksums, and limitations.

## Results

[Neuroprobe](https://github.com/insight-neuro/neuroprobe) evaluates 15 binary decoding tasks during
movie-watching. The table reports macro AUROC. Both entries use the same frontend and linear
probe; MAPA adds the pretrained encoder.

| Regime | Frontend baseline | MAPA |
|---|---|---|
| Within-session | .6744 | **.6953** |
| Cross-session | .6660 | **.6909** |
| Cross-subject | .5872 | **.6083** |

The cross-subject readout fits on an anchor subject, selects regularization using labeled
validation trials from each target session, and scores that session's separate test trials.
Two evaluation subjects are excluded from pretraining entirely. MAPA improves over the
frontend baseline on both.

The [evaluation guide](evals/README.md) covers:

- Frozen decoding in all three regimes.
- Sample-efficiency curves and the 21× label saving.
- Ablations of the region embedding and relative positional encoding.
- Transfer to the two subjects excluded from pretraining.

Evaluation requires Brain Treebank data and the Neuroprobe package. Follow the preprocessing
and evaluation guides for the benchmark's session/contact selection, caches, and splits.
Guard 1's frozen contact exclusions affect evaluation preprocessing. Guard 2 does not reject
evaluation windows; Guard 3 clips inputs inside the model. The Guard 1/2 detectors, pretraining
loop, and cluster infrastructure are not included.

## Tests

```bash
python -m pip install -e ".[test,preprocessing]"
python -m pytest
```

Tests use synthetic inputs and cover the encoder, frozen readout, label subsampling, and
submission export. CUDA and upstream Neuroprobe checks are skipped when unavailable.

## License

Code and checkpoints: [Apache 2.0](LICENSE).
See [NOTICE](NOTICE) for attribution and prior checkpoint licensing.

## Citation

```bibtex
@misc{tang2026pretraining,
      title={Pretraining for Sample-Efficient Neural Interfaces},
      author={Ben Tang and Zachary Spalding and Gregory B. Cogan},
      year={2026},
      eprint={2609.13507},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.13507},
}
```
