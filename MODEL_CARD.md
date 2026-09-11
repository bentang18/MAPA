# Model card

## Model

MAPA is a masked autoencoder for intracranial EEG. An anatomical region embedding and a relative
positional encoding enable the model to learn neural representations that transfer across
subjects. The encoder is a ViT-Small with width 384 and 12 blocks. The decoder is discarded after
pretraining.

## Inputs and outputs

A patch contains the frequency bins of one band, at one contact, at one time step of that band's
rate. Provide three normalized magnitude STFT tensors in Slow, Mid, Fast order. Each tensor has
shape `(batch, contacts, bins, time)`.

| Band | Window | Rate | Frequencies | Bins |
|---|---|---|---|---|
| Slow | 500 ms | 4 Hz | 2–14 Hz | 7 |
| Mid | 125 ms | 16 Hz | 16–56 Hz | 6 |
| Fast | 62.5 ms | 32 Hz | 64–160 Hz | 7 |

The encoder API accepts the three normalized magnitude STFT tensors on a shared 32 Hz frame
clock. The frontend decimates each band to the rate listed above. Each contact contributes 52
tokens per second. Features remain per-contact, in the order given by `session.contact_order`.

The region embedding uses the DKT atlas vocabulary, with 74 regions and one reserved entry.
The relative positional encoding uses the difference between clinical contact numbers along an
array. RoPE also encodes time. Neither spatial encoding uses coordinates in a template brain.

Map regions to their indices in `V14_DKT_REGION_LABELS` in
[anatomy.py](mapa/data/anatomy.py). The vocabulary distinguishes hemispheres. ID 74 is reserved
for contacts outside that vocabulary. Contact labels and region IDs must follow the input
tensor's contact order.

| Requested tap | Output for a one-second window |
|---|---|
| `0` | `(batch, contacts, 1, 348)` frontend input features |
| `3`, `6`, `9`, or `12` | `(batch, contacts, 52, 384)` encoder features |

The returned dictionary is keyed by tap number. All outputs use `session.contact_order`, which
indexes the input contacts. Encoder taps are taken before the output LayerNorm, matching the
paper's frozen readout. Tap 0 concatenates the decimated band values; it is not a 384-dimensional
encoder representation.

Use `mapa.preprocessing.prepare_recording` to produce these inputs from a full raw recording.
It includes referencing, filtering, STFTs, and frozen session normalization. Supply cleaned
recordings and explicit contact selections. The release includes the frozen per-session
Guard 1 exclusions used for Neuroprobe; these affect evaluation and must be applied before
shaft referencing. The Guard 1/2 detectors are not included, and Guard 2 does not reject
evaluation windows. The model applies Guard 3 before projection: Slow/Mid/Fast normalized
inputs are clipped to ±15/±15/±20. Tap 0 uses the same clipped inputs.
See the [preprocessing guide](docs/PREPROCESSING.md) and [configuration](mapa/preprocessing/config.json).

## Checkpoints

Download the four files from the [v0.1.0 release](https://github.com/bentang18/MAPA/releases/tag/v0.1.0).
The checkpoints are taken at step 55,000. Each records whether the region embedding and
relative positional encoding are active. The decoder and optimizer are not included.

| File | Region embedding | Relative positional encoding | Parameters |
|---|---|---|---|
| `mapa_vits384.pt` | Yes | Yes | 21,335,424 |
| `mapa_vits384_no_region.pt` | No | Yes | 21,306,624 |
| `mapa_vits384_no_relpos.pt` | Yes | No | 21,335,424 |
| `mapa_vits384_no_priors.pt` | No | No | 21,306,624 |

The region embedding contains 75 × 384 = 28,800 parameters. The release also includes
`SHA256SUMS`, `LICENSE-WEIGHTS`, `NOTICE`, and `preprocessing.json` (the shared frontend recipe). Verify downloaded files with `sha256sum -c SHA256SUMS`
(or `shasum -a 256 -c SHA256SUMS` on macOS). The expected checksums are:

```
2d236089a2f1a3cc2827e3f150c4a2ba14c51bbfaf0ce0888f84b92a6eb25a7a  mapa_vits384.pt
d1a7258ffbb164b73b1f64ec798e9fab61fa635fcca3b28cf2820dee29091393  mapa_vits384_no_region.pt
94ccd61a467cbf6c1adc23c96fa1f9f3558f8072f7842d6e514298b5f02b08f3  mapa_vits384_no_relpos.pt
e363a38e4ef923fc8bfb2b716321db9670897580e0141c8bba96677ef30a586d  mapa_vits384_no_priors.pt
```


## Training data

We pretrain on 27.9 hours of Brain Treebank recordings from 13 sessions of 7 subjects. We mask
75% of patches uniformly at random over contacts, bands, and time, and reconstruct their frequency
bins. No task labels are used during pretraining. Anatomical region assignments and clinical
contact numbers supply the spatial encodings. Two evaluation subjects, subjects 7 and 10, are
held out from pretraining entirely.

## Evaluation

The encoder is frozen for all evaluations. The readout is a linear probe. Neuroprobe scores 15
binary tasks by AUROC, and we report the mean over tasks and evaluation sessions. Trials are
one-second windows at the benchmark's word anchors, with nonverbal anchors for negative speech
and onset examples. Evaluation covers 12 sessions from 6 subjects, with
at most 3,500 trials per task.

Within-session uses two folds: each fits on one half of the task's ordered trials and divides
the other half equally into validation and test trials. Reported AUROC averages the two test folds.
Cross-session fits on one session of a subject; cross-subject fits on one anchor subject.
Both split each target session's trials into validation and test halves. Validation labels
select the ridge regularization strength; test labels are used only for scoring.

| Regime | Frontend baseline | MAPA | Evaluation sessions |
|---|---|---|---|
| Within-session | .6744 | .6953 | 12 |
| Cross-session | .6660 | .6909 | 12 |
| Cross-subject | .5872 | .6083 | 10 |

The frontend baseline and MAPA share their frontend and readout. The gain comes from the
pretrained encoder alone. The unit of statistical analysis in the paper is the subject.

In the cross-subject regime, MAPA reaches the frontend baseline's full-data accuracy with
approximately 164 anchor-subject training trials rather than 3,500. The estimated label saving
is 21.3×, with a 95% bootstrap interval of 8.4–64.0×. Target-session validation labels remain
fixed while anchor training labels are subsampled; they are not included in that count.

## Intended use and limitations

Use the encoder to extract features for downstream decoding. See the [README](README.md) for
input preprocessing and the [evaluation guide](evals/README.md) for the published readout.

Pretraining and evaluation use sEEG recordings during passive movie-watching. These results do
not establish performance on attempted speech, imagined speech, or motor decoding.

The source Brain Treebank inventory has at most 16 contacts per shaft (index span at most 15).
This is an observed data range, not a fixed architectural ceiling; longer shafts have not been
validated. The spatial RoPE coordinate is a contact number along an array. It does not encode the two spatial
axes of an ECoG grid. Performance on surface grids is not established by these experiments.

Each ablation uses one seed. Two subjects are held out from pretraining entirely, and the result
on those subjects is descriptive. Label savings vary across subjects. Subjects that never reach
the target accuracy are retained as failures in the label-saving analysis.

## License and attribution

Code: [Apache 2.0](LICENSE). Checkpoints: [CC BY 4.0](LICENSE-WEIGHTS).
The checkpoints carry the Brain Treebank attribution in [NOTICE](NOTICE).
Evaluation follows [Neuroprobe](https://github.com/insight-neuro/neuroprobe).

## Citation

```bibtex
@misc{tang2026pretraining,
  title  = {Pretraining for Sample-Efficient Neural Interfaces},
  author = {Tang, Ben and Spalding, Zachary and Cogan, Gregory B.},
  year   = {2026}
}
```
