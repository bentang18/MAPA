# From recordings to MAPA features

Install the preprocessing dependency with `python -m pip install -e '.[preprocessing]'`.
The frozen encoder alone does not need MNE. Preprocessing runs on CPU; encoder inference can
run on CPU or CUDA.

## Prepare a recording

Provide a cleaned, continuous, unreferenced recording as a NumPy array `(contacts, samples)`, its native
sampling rate, and row-aligned contact metadata. Process the full session once, then extract
windows. The filters are zero-phase, STFT windows are centered, and normalization uses the
whole session: this is an offline pipeline, not a causal streaming decoder.

```python
import numpy as np
from mapa.preprocessing import prepare_recording, region_ids_from_names

voltage = np.load("recording.npy")
labels = ["LA1", "LB1", "LA3", "LB3"]
regions = region_ids_from_names([
    "ctx-lh-superiortemporal", "ctx-rh-superiortemporal",
    "ctx-lh-superiortemporal", "ctx-rh-superiortemporal",
])
recording = prepare_recording(voltage, 2048, labels, regions)
bands = recording.window(start_s=10.0, duration_s=1.0)
```

Here LA and LB are distinct shafts. The missing contact 2 stays missing; do not rename LA3 to
LA2. Atlas names are exact entries in `V14_DKT_REGION_LABELS` in
[anatomy.py](../mapa/data/anatomy.py). The helper distinguishes hemispheres, raises on unknown
strings, and maps an explicit `None` to reserved ID 74. It converts existing atlas assignments
to IDs; it does not localize electrodes or register a brain to an atlas.

Then run the encoder:

```python
from mapa import MapaEncoder

encoder = MapaEncoder.from_checkpoint("/path/to/mapa_vits384.pt", device="cpu")
session = encoder.prepare(recording.sidecar, n_time=32)
features = encoder(bands, session, taps=(12,))[12]
```

Use `recording.sidecar`, because supplied exclusions can change the contact axis. `recording.input_indices`
maps surviving contacts back to the original voltage rows. The encoder further groups those
contacts by shaft; `features[:, session.contact_order.argsort()]` restores the surviving input
order. No feature is produced for a removed contact.

A complete runnable synthetic example includes a fitted readout:

```bash
python examples/decode_recording.py --weights /path/to/mapa_vits384.pt
```

It uses random voltage and labels to exercise the interface. Its ridge example is not the
paper's evaluation or a performance claim; use the [evaluation guide](../evals/README.md) for
Neuroprobe's trials, validation-selected regularization, and splits.

## Contact selection and signal scale

Supply already-cleaned recordings. Optional `bad_labels` excludes contacts chosen upstream;
`keep_labels` selects the allowed model montage before its shaft mean is computed. The frontend
does not detect bad contacts or reject artifact windows. It returns unclipped normalized bands;
the model applies the value caps described below. A surviving
shaft with fewer than two contacts is rejected because mean referencing would produce zero signal.

The input must use a consistent acquisition scale across contacts. This API does not convert
values to MNE's conventional volts. The research loader retains Brain Treebank's stored voltage
scale; the source metadata does not establish an absolute voltage-unit conversion. Preserve that
scale when reproducing Brain Treebank. Do not claim exact equivalence after arbitrary rescaling:
normalization contains a fixed numerical floor. For a new acquisition,
record its units and gain, and check the fitted scales and contact exclusions. Anatomical
assignment, acquisition calibration, and identifying non-neural channels remain input metadata.

`recording.metadata` records selected/dropped labels, original indices, atlas IDs, native rate,
recipe, and library versions. Normalization statistics remain in `recording.normalizers`.

## The released frontend

The executable configuration is also shipped as
[config.json](../mapa/preprocessing/config.json). All four checkpoints share these frontend operations.
Guard 3 is part of the model input. The Guard 1/2 detectors are not shipped here, but the
frozen per-session Guard 1 contact exclusions used for Neuroprobe are included in
[anatomy.py](../mapa/data/anatomy.py). The generic frontend does not look up these exclusions
by subject/session; supply the selected voltage rows or explicit exclusions yourself.

1. Preserve the loader's float32 voltage convention; polyphase-resample to 2048 Hz when needed.
2. Apply only the contact exclusions supplied by the caller.
3. Select the requested surviving montage. Subtract the surviving mean separately on each
   shaft. Apply the model's notches at 60, 120, 180, 240, and 300 Hz, followed by the 0.5 Hz
   high-pass. Filtering uses MNE's zero-phase FIR implementation; return to float32 afterward.
4. Compute whole-session magnitude STFTs with periodic Hann windows, centered reflect padding,
   no log transform and no FFT normalization. All three caches use a 64-sample hop (32 Hz).

| Band | FFT/window samples | Inclusive FFT bins | Frequencies | Model rate |
|---|---|---|---|---|
| Slow | 1024 | 1–7 | 2–14 Hz | 4 Hz |
| Mid | 256 | 2–7 | 16–56 Hz | 16 Hz |
| Fast | 128 | 4–10 | 64–160 Hz | 32 Hz |

5. Fit each contact/bin's median and 1.4826 × MAD over the entire session, including the
   centered endpoint frame. Use PyTorch's lower median, a scale floor of 1e-6, and zero output
   for bins below that floor. Slice windows from these continuous caches, then normalize with
   the frozen session statistics. These cached/window values remain unclipped.
6. The model applies Guard 3: clip normalized Slow/Mid/Fast inputs to ±15/±15/±20 before
   projection. The frontend-baseline tap uses these same capped inputs. The model decimates
   Slow/Mid/Fast by 8/2/1. A one-second window yields 52 tokens per
   contact. Window starts round to the nearest 32 Hz frame, with ties to even.

Guard 1 affects evaluation through the contact exclusions applied before shaft referencing
and cache construction. The detector is not rerun during inference. Guard 2 does not reject
evaluation windows. Guard 3 clips the normalized inputs inside the model.

For Neuroprobe, `voltage_electrode_order(bt_root, subject_id, trial_id)` returns the surviving
contact labels after the dataset's exclusions and the frozen Guard 1 exclusions. The
`trial_id` argument is required to select the recorded per-session decisions. Select and
reorder the actual voltage rows to those labels, then apply the Lite montage before shaft
referencing. See the [evaluation guide](../evals/README.md#evaluation-prerequisites).
Preserve the paper's cache construction, normalization, and split conventions as well.

## Save caches for evaluation

```python
recording.save("prepared/session_1_1", subject_id=1, trial_id=1)
```

Or provide a `.npy` recording and JSON metadata on the command line:

```json
{
  "labels": ["LA1", "LB1", "LA3", "LB3"],
  "region_names": ["ctx-lh-superiortemporal", "ctx-rh-superiortemporal",
                   "ctx-lh-superiortemporal", "ctx-rh-superiortemporal"],
  "bad_labels": []
}
```

```bash
python -m mapa.preprocessing --voltage recording.npy --contacts contacts.json \
    --sample-rate 2048 --subject-id 1 --trial-id 1 --out prepared/session_1_1
```

Use either `region_names` or integer `region_ids`. Optional `keep_labels` restricts the model
montage. The output directory must be new:

```text
session_1_1/
  preprocessing.json
  slow/session.npy + session.json + session.stats.npz
  mid/session.npy  + session.json + session.stats.npz
  fast/session.npy + session.json + session.stats.npz
  spans/session.json
```

The span JSON is empty; no artifact-window detector runs. The `.npy` files contain unnormalized
float32 STFT values; the statistics files hold frozen
median/sigma. These directories can be read by `load_v3_sessions` and the evaluation encoder.
For multiple sessions, collect each recording's triples under common `slow`, `mid`, `fast`
directories with distinct stems such as `s1_t1`; collect its empty compatibility span JSON under a common `spans`
directory. Preserve the matching basename of each `.npy`, `.json`, and `.stats.npz` triple.
For the published benchmark, use its exact session/contact lists and original trial metadata;
this generic adapter does not choose those on your behalf.

The implementation bounds temporary FFT work with `chunk_frames`, preserving the full-session
frame grid across chunks. It still holds the voltage and output band caches in CPU memory.
Plan memory for a whole recording; it is not a disk-streaming raw-data loader.

## Supported geometry and windows

The released checkpoints were trained and evaluated on sEEG shafts. The archived Brain Treebank
contact inventory spans at most 16 contacts per shaft (clinical index span at most 15); removing
contacts preserves the original indices. This describes the data range, not a hard architectural
maximum. The encoder groups a variable number of contacts per shaft and has no fixed contact-count
ceiling. Longer shafts have not been validated, and flattening a two-dimensional ECoG grid into
one sequence does not supply its spatial geometry. Do not retune the released RoPE bases at
inference and assume the checkpoint's behavior is preserved.

Published decoding uses one-second windows. The API accepts durations in multiples of 0.25 seconds
(the common band-stride grid); shape compatibility for a different duration is not an evaluation
result. Whole-session normalization requires a full recording rather than separate per-trial fits.
The five-second minimum accepted by this helper is an input check, not a claim that five seconds
is enough to calibrate a real session.

## Verification

[Frontend tests](../tests/test_frontend.py) compare with a synthetic reference generated by the
research implementation, including its actual CAR and NeuralSet harmonic-notch helper. Tests
also cover chunk boundaries, contact identity, normalization/window timing, cached-data loading,
and the absence of automatic contact detection. Encoder tests cover the shared model/baseline
clipping boundary. The reference uses no participant recording.
