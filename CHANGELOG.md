# Changelog

## September 14, 2026 — Checkpoint licensing update

Code and checkpoints are now distributed under [Apache 2.0](LICENSE). Brain Treebank
attribution is retained in [NOTICE](NOTICE). Earlier CC BY 4.0 grants remain valid.
Checkpoint bytes, hashes, download URLs, and the original v0.1.0 tag are unchanged.

## v0.1.0

Initial release of MAPA, a masked autoencoder for intracranial EEG.

- ViT-Small encoder pretrained on 27.9 hours of Brain Treebank recordings from 13 sessions of 7 subjects.
- Anatomical region embedding and relative positional encoding.
- Four checkpoints: both spatial encodings, each encoding alone, and neither encoding.
- Frozen feature extraction through `MapaEncoder` and `torch.hub`.
- Canonical Neuroprobe feature extraction, linear ridge readout, and leaderboard submission scripts.
- Feature caches retain contact labels in encoded order for cross-session alignment.
- Sample-efficiency curves, subject-level statistics, and held-out-subject depth curves.
- Within-session exports preserve both evaluation folds and reject incomplete fold records.
- Figure scripts and a quickstart notebook.
- Offline voltage preprocessing, atlas-name mapping, cache export, and a synthetic decoding example.
- Input validation for contact/band/window alignment, atlas IDs, and requested encoder blocks.
- Automated CPU tests on Python 3.10/PyTorch 2.6.0 and Python 3.12.
- Evaluation documentation specifies target-session validation labels and benchmark data setup.

The model applies the published per-band input clipping (Guard 3), shared by the encoder and
frontend baseline. The frozen per-session Guard 1 exclusions used for Neuroprobe are included
and affect evaluation preprocessing before shaft referencing. The Guard 1/2 detectors are not
included. Guard 2 does not reject evaluation windows. The generic frontend accepts cleaned
recordings and caller-supplied exclusions.

The pretraining decoder and training loop are not included. See the [model card](MODEL_CARD.md)
for results and limitations. At initial release, code was licensed under [Apache 2.0](LICENSE),
and checkpoints under [CC BY 4.0](https://github.com/bentang18/MAPA/blob/v0.1.0/LICENSE-WEIGHTS).
See the September 14 licensing update above for the current checkpoint license.
