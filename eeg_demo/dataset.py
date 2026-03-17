from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Optional


TRAIN_PATTERN = re.compile(r"^(?P<patient>\d+)_(?P<segment>\d+)_(?P<label>[01])\.mat$", re.IGNORECASE)
TEST_PATTERN = re.compile(r"^(?P<patient>\d+)_(?P<segment>\d+)\.mat$", re.IGNORECASE)
PAT_TRAIN_PATTERN = re.compile(
    r"^Pat(?P<patient>\d+)Train_(?P<segment>\d+)_(?P<label>[01])\.mat$",
    re.IGNORECASE,
)
PAT_TEST_PATTERN = re.compile(
    r"^Pat(?P<patient>\d+)Test_(?P<segment>\d+)(?:_(?P<label>[01]))?\.mat$",
    re.IGNORECASE,
)

LABEL_MAP = {
    0: "interictal",
    1: "preictal",
}


@dataclass(frozen=True)
class ClipRecord:
    path: Path
    patient_id: int
    segment_id: int
    split: str
    label_id: Optional[int]
    label_name: Optional[str]


def parse_clip_filename(path: str | Path) -> ClipRecord:
    file_path = Path(path)
    name = file_path.name

    pat_train_match = PAT_TRAIN_PATTERN.match(name)
    if pat_train_match:
        label_id = int(pat_train_match.group("label"))
        return ClipRecord(
            path=file_path,
            patient_id=int(pat_train_match.group("patient")),
            segment_id=int(pat_train_match.group("segment")),
            split="train",
            label_id=label_id,
            label_name=LABEL_MAP[label_id],
        )

    pat_test_match = PAT_TEST_PATTERN.match(name)
    if pat_test_match:
        label_group = pat_test_match.group("label")
        label_id = int(label_group) if label_group is not None else None
        return ClipRecord(
            path=file_path,
            patient_id=int(pat_test_match.group("patient")),
            segment_id=int(pat_test_match.group("segment")),
            split="test" if label_id is None else "train",
            label_id=label_id,
            label_name=LABEL_MAP[label_id] if label_id is not None else None,
        )

    train_match = TRAIN_PATTERN.match(name)
    if train_match:
        label_id = int(train_match.group("label"))
        return ClipRecord(
            path=file_path,
            patient_id=int(train_match.group("patient")),
            segment_id=int(train_match.group("segment")),
            split="train",
            label_id=label_id,
            label_name=LABEL_MAP[label_id],
        )

    test_match = TEST_PATTERN.match(name)
    if test_match:
        return ClipRecord(
            path=file_path,
            patient_id=int(test_match.group("patient")),
            segment_id=int(test_match.group("segment")),
            split="test",
            label_id=None,
            label_name=None,
        )

    raise ValueError(
        f"Filename {name} does not match expected train pattern I_J_K.mat or test pattern I_J.mat"
    )


def index_dataset(root: str | Path, recursive: bool = True) -> list[ClipRecord]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {root_path}")

    glob_pattern = "**/*.mat" if recursive else "*.mat"
    records: list[ClipRecord] = []
    for path in sorted(root_path.glob(glob_pattern)):
        try:
            records.append(parse_clip_filename(path))
        except ValueError:
            continue
    return records


def filter_records(
    records: Iterable[ClipRecord],
    *,
    patient_id: Optional[int] = None,
    split: Optional[str] = None,
    label_name: Optional[str] = None,
) -> list[ClipRecord]:
    results = []
    for record in records:
        if patient_id is not None and record.patient_id != patient_id:
            continue
        if split is not None and record.split != split:
            continue
        if label_name is not None and record.label_name != label_name:
            continue
        results.append(record)
    return results
