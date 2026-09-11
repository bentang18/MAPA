"""Prepare a recording from .npy voltage and row-aligned JSON contact metadata."""

import argparse
import json

import numpy as np

from mapa.preprocessing import prepare_recording, region_ids_from_names


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--voltage", required=True, help="channels x samples .npy, acquisition scale")
    p.add_argument("--contacts", required=True, help="JSON: labels, region_names or region_ids")
    p.add_argument("--sample-rate", required=True, type=int)
    p.add_argument("--subject-id", required=True, type=int)
    p.add_argument("--trial-id", required=True, type=int)
    p.add_argument("--out", required=True, help="new output directory")
    a = p.parse_args()
    with open(a.contacts) as f:
        contacts = json.load(f)
    if ("region_names" in contacts) == ("region_ids" in contacts):
        p.error("provide exactly one of region_names or region_ids")
    regions = (
        region_ids_from_names(contacts["region_names"])
        if "region_names" in contacts
        else contacts["region_ids"]
    )
    recording = prepare_recording(
        np.load(a.voltage, allow_pickle=False),
        a.sample_rate,
        contacts["labels"],
        regions,
        bad_labels=contacts.get("bad_labels", []),
        keep_labels=contacts.get("keep_labels"),
    )
    recording.save(a.out, subject_id=a.subject_id, trial_id=a.trial_id)
    print(
        f"Prepared {len(recording.sidecar.labels)} contacts, {recording.duration_s:.2f}s -> {a.out}"
    )


if __name__ == "__main__":
    main()
