# Evaluation

Encode the evaluation windows once on a GPU. Fit the frozen ridge readout on CPU for each regime.
All three regimes use the same feature cache.

Use only feature caches and legacy contact-label sidecars that you created or trust. These
evaluation files contain Python/NumPy objects and are loaded with pickle. Released model
checkpoints use restricted tensor loading.

The scripts run from the repository root after `pip install -e .`. Evaluation also requires the
[Neuroprobe](https://github.com/insight-neuro/neuroprobe) package and Brain Treebank data. Band
caches can be built with the released [preprocessing API/CLI](../docs/PREPROCESSING.md).
Apply the included, frozen per-session Guard 1 contact exclusions and select the Neuroprobe
Lite contacts before shaft referencing. These exclusions affect evaluation inputs even though
the Guard 1 detector is not rerun. The Guard 1/2 detectors are not included. Guard 2 does not
reject evaluation windows. Guard 3 clips normalized inputs inside both the encoder and frontend
baseline. Preserve the benchmark's specified trials and contact selection.
The saved `slow`, `mid`, `fast`, and `spans` directories feed the encoder below.

## Evaluation prerequisites

Use the Neuroprobe revision audited against the released split assignments (package version
0.1.8). Install it in the same environment as MAPA:

```bash
python -m pip install 'neuroprobe @ git+https://github.com/insight-neuro/neuroprobe.git@f9b084241aeeaab548ca45ec98368b70141465a8'
export ROOT_DIR_BRAINTREEBANK=/absolute/path/to/braintreebank
```

Neuroprobe reads `ROOT_DIR_BRAINTREEBANK` at import time. Pass the same directory as
`--bt-root "$ROOT_DIR_BRAINTREEBANK"`. Obtain recordings and metadata through the
[Brain Treebank dataset](https://braintreebank.dev/); neither is bundled with MAPA.
For encoding existing band caches, the root must include:

```text
braintreebank/
  corrupted_elec.json
  electrode_labels/sub_<subject>/electrode_labels.json
  localization/sub_<subject>/depth-wm.csv
  subject_timings/sub_<subject>_trial<trial:03d>_timings.csv
  transcripts/<movie>/features.csv
```

Localization uses the `Electrode` and `DKT` columns; timing files use `index` and `movie_time`.
Neuroprobe supplies the aligned word/nonverbal CSVs and pitch/volume JSON files in its package.
The evaluation sessions are `(1,1), (1,2), (2,0), (2,4), (3,0), (3,1), (4,0), (4,1),
(7,0), (7,1), (10,0), (10,1)`, where each pair is `(subject, trial)`.

For new band caches, first load a continuous recording into a NumPy array. MAPA's generic
frontend accepts this array; it does not read the dataset's HDF5 files. The dataset metadata
helpers provide the surviving contact order, frozen exclusions, and DKT mapping. The Guard 1
lists are available through `extra_bad_electrodes(subject_id, trial_id)` in
[anatomy.py](../mapa/data/anatomy.py); `voltage_electrode_order` already applies them together
with the dataset's corrupted/missing-coordinate/trigger exclusions:

```python
from mapa.data.anatomy import lite_electrode_set, voltage_electrode_order
from mapa.data.region_fn import make_bt_region_fn

labels = voltage_electrode_order(bt_root, subject_id, trial_id)
region_ids = make_bt_region_fn(bt_root)(subject_id, trial_id, labels).numpy()
keep_labels = [label for label in labels if label in lite_electrode_set(subject_id)]
```

Pass `trial_id` to select the frozen decisions for that session. Select and reorder the actual
voltage rows to `labels` before calling `prepare_recording`; assigning these labels to an
unfiltered voltage array does not remove contacts. Pass `keep_labels` to restrict the surviving
montage before shaft referencing. Contacts already removed from the voltage array should not
also be passed in `bad_labels`, which accepts only labels present in its input.

These helpers return metadata, not voltage or automatic artifact detection. The localization
CSV's row order is not the voltage channel order. Intersecting the Lite list with the available
labels avoids requesting contacts absent from the recording. The released exclusion lists
preserve the recorded decisions; reproducing them does not require rerunning the detector.

## 1. Encode

Pass three band-cache directories in Slow, Mid, Fast order. Each cache uses the shared 32 Hz
frame clock. Use the Neuroprobe Lite contacts and all 15 tasks for the published evaluation.

```bash
python -m evals.neuroprobe.encode \
    --ckpt /path/to/mapa_vits384.pt --tag board_iid75_vits384 \
    --out-dir /path/to/features \
    --band-cache-dir /path/to/slow \
    --band-cache-dir /path/to/mid \
    --band-cache-dir /path/to/fast \
    --span-dir /path/to/spans --bt-root "$ROOT_DIR_BRAINTREEBANK" \
    --sessions board --tasks board15 --electrode-set lite \
    --elec-taps 0,12
```

The cache contains region-pooled features at blocks 0, 3, 6, 9, and 12. `--elec-taps 0,12` also
stores the per-contact features used for within-session and cross-session evaluation. The
cross-subject readout uses the region-pooled features. Block 0 is the frontend baseline.

Released checkpoints record whether the relative positional encoding is active. The script
reads that setting and prints the selected model configuration.
New caches also record the checkpoint SHA256 and effective spatial settings. Output files must
be new; existing files are never silently accepted or overwritten. After an interrupted batch,
use `--session-index` to run only missing sessions, or choose a new output directory/tag.

## 2. Fit the readout

Run one process per evaluation session: indices 0–11 for within-session and cross-session,
and 0–9 for cross-subject. The example runs one within-session shard. Replace `--mode ws`
with `csession` or `cs` for the other regimes.

```bash
python -m evals.neuroprobe.readout \
    --cache-dir /path/to/features --tags board_iid75_vits384 \
    --mode ws --index 0 --shard-dir /path/to/shards \
    --out /path/to/results.json
```

The shard command writes to `--shard-dir`. The `--out` argument is required by the command-line
parser and is used by the merge step below.

Cross-session evaluation aligns contacts by their clinical labels. Current feature caches carry
those labels. Older caches need `--elec-labels-sidecar /path/to/labels.pkl`. The file maps keys
such as `s1_t1` to labels in the exact **cached feature-column order**, not the original voltage
order. For the matching original session, that order is
`np.asarray(sidecar.labels)[session.contact_order.numpy()]` after `encoder.prepare(...)`.
Use the same surviving montage as the old cache; re-encode if its order is unknown. Without
correct contact labels, the cross-session readout cannot produce the published per-contact result.

The readout uses training-only feature standardization and linear ridge. It selects the
regularization with highest validation AUROC, keeping the smallest value on ties.

Within-session uses two folds with approximately half the ordered task trials for training,
one quarter for validation, and one quarter for testing. Cross-session and cross-subject
readouts use labeled validation trials from the target session to select regularization;
the target's separate test trials are reserved for scoring. Cross-subject training trials
come from the anchor subject.

## 3. Merge and validate

```bash
python -m evals.neuroprobe.readout \
    --cache-dir /path/to/features --tags board_iid75_vits384 \
    --mode merge --shard-dir /path/to/shards --out /path/to/results.json
```

The merge summarizes the shards that are present. A merged result alone does not establish that
the evaluation is complete. Validate each regime with the submission builder, which requires
every task/session pair and rejects mixed checkpoint tags and invalid AUROC values.

```bash
python -m evals.neuroprobe.build_submission \
    --shards /path/to/shards --split Cross-Subject --out /path/to/submission \
    --model-name MAPA --author 'Ben Tang' \
    --organization Duke --organization-url https://duke.edu \
    --description 'A masked autoencoder for intracranial EEG, evaluated with a frozen encoder and a linear probe.'
```

Repeat with `--split Within-Session` and `--split Cross-Session`. The builder writes results and
metadata. Add the publication citation and signed attestations according to Neuroprobe's
submission instructions. Do not submit a preparation directory before those are complete.

Within-session exports require both test folds, with indices 0 and 1. New readout shards retain
these scores, and the builder checks that their mean matches the shard's reported AUROC. For
legacy shards that contain only the mean, provide the matching full-data WS sample curve:

```bash
python -m evals.neuroprobe.build_submission \
    --shards /path/to/shards --split Within-Session --out /path/to/submission \
    --ws-curve /path/to/ws-curve.json \
    --model-name MAPA --author 'Ben Tang' \
    --organization Duke --organization-url https://duke.edu \
    --description 'A masked autoencoder for intracranial EEG, evaluated with a frozen encoder and a linear probe.'
```

The curve must use the same checkpoint tag and the default random subsampling protocol. The
builder uses only full-data, seed-0, block-12 per-contact scores with training-only feature
normalization. It rejects missing folds or a mismatch with the stored mean. A fold mean alone
cannot reconstruct the two scores.

## Ablations and label savings

Use a separate feature-cache tag and shard directory for each checkpoint. Check that the
frontend baseline agrees across the runs before comparing encoder features:

```bash
python -m evals.neuroprobe.floor_parity_check \
    --published-dir /path/to/both-shards --published-tag board_iid75_vits384 \
    --new-dir /path/to/ablation-shards --new-tag board_iid75_vits384_noregion
```

`subject_compare` reports descriptive differences and a two-sided sign test between two shard
directories. It is not the one-sided paired permutation test reported in the paper. Each input
directory must contain one checkpoint tag. Different tags across the two directories are expected.

## Reproduce the sample-efficiency experiment

This experiment refits the readout on subsets of the training trials. Both models use the same
subsets. Test trials are never subsampled, and regularization is selected on the full validation
set. In cross-subject evaluation, the subsampled count is the number of anchor-subject training
labels; target-session validation labels remain fixed and are not included in that count.
The full-data fit is checked against the original readout before a curve shard is accepted.

Run all ten cross-subject evaluation sessions:

```bash
for index in 0 1 2 3 4 5 6 7 8 9; do
    python -m evals.neuroprobe.sample_curve \
        --mode cs --cache-dir /path/to/features --tag board_iid75_vits384 \
        --index "$index" --mmap --shard-dir /path/to/cs-curve
done
python -m evals.neuroprobe.sample_curve \
    --mode merge --regime cs --shard-dir /path/to/cs-curve --out /path/to/cs-curve.json
python -m evals.neuroprobe.label_saving_ci \
    --src /path/to/cs-curve.json --nboot 20000 --seed 0 --per-subject \
    --out /path/to/cs-label-savings.json
```

Repeat the loop with `--mode ws` and `--mode csession`, using indices 0–11 and separate shard
and output directories. Set the corresponding `--regime` when merging. Current caches include
contact labels; cross-session runs on older caches additionally need `--elec-labels-sidecar`.
Keep the default taps, random sampling, label-count grid, and subset counts for the paper's result.

The reported crossing is interpolated in log label count. The cross-subject result in the paper
is approximately 164 of 3,500 trials, a 21× saving. The subject bootstrap gives the uncertainty;
subjects that do not reach their full-data baseline remain reported as failures.

Plot all three regimes after completing their runs:

```bash
python -m evals.figures.fig_r31_label_efficiency_log10 \
    --ws /path/to/ws-curve.json --csession /path/to/csession-curve.json \
    --cs /path/to/cs-curve.json --layout row --out /path/to/figures
```

## Reproduce the spatial-encoding and held-out-subject results

Encode and read out all four checkpoints with separate tags and shard directories. The two
ablations with one encoding retain either relative position or the region embedding:

| Checkpoint | Retained encoding | Argument below |
|---|---|---|
| `mapa_vits384.pt` | Both | `--both` |
| `mapa_vits384_no_region.pt` | Relative position | `--no-region` |
| `mapa_vits384_no_relpos.pt` | Region embedding | `--no-relpos` |
| `mapa_vits384_no_priors.pt` | Neither | `--no-priors` |

```bash
python -m evals.neuroprobe.paper_results \
    --both /path/to/both-shards --no-region /path/to/no-region-shards \
    --no-relpos /path/to/no-relpos-shards --no-priors /path/to/no-priors-shards \
    --out /path/to/paper-results.json
python -m evals.figures.fig_r30_geometry_ablation \
    --src /path/to/paper-results.json --out /path/to/figures/ablations.pdf
python -m evals.figures.fig_zero_pretrain_ladder \
    --src /path/to/paper-results.json --out /path/to/figures/heldout.pdf
```

`paper_results` checks the complete task/session grid and checkpoint tags. It reports the
frontend and model macro AUROC, each subject's gain, the number of subjects improved, and the
paper's one-sided exact paired permutation test. BH correction covers the four models separately
within each regime. The ablation plot marks corrected q < .05.

The same JSON records each model's cross-subject AUROC at blocks 0, 3, 6, 9, and 12 for subjects
7 and 10, both excluded from pretraining. These two-subject results are descriptive. The plots
reproduce the analyses; their layout differs from the paper's typeset figures.
