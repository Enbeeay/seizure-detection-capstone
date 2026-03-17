from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENDOR_DIR = PROJECT_ROOT / ".vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

from eeg_demo.dataset import filter_records, index_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Kaggle-style EEG dataset layout")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("D:/seizure-detection-data"),
        help="Root directory containing .mat files",
    )
    args = parser.parse_args()

    records = index_dataset(args.dataset_root)
    summary: dict[str, object] = {
        "dataset_root": str(args.dataset_root),
        "total_files": len(records),
        "patients": {},
    }

    for patient_id in sorted({record.patient_id for record in records}):
        patient_records = filter_records(records, patient_id=patient_id)
        train_records = filter_records(patient_records, split="train")
        test_records = filter_records(patient_records, split="test")
        summary["patients"][str(patient_id)] = {
            "train_interictal": len(filter_records(train_records, label_name="interictal")),
            "train_preictal": len(filter_records(train_records, label_name="preictal")),
            "test_unlabeled": len(test_records),
        }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
