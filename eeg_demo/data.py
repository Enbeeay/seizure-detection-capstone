from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as sio

from eeg_demo.dataset import parse_clip_filename


DEFAULT_SAMPLING_RATE = 400.0


@dataclass
class EEGClip:
    path: Path
    data: np.ndarray
    sampling_rate: float
    channel_names: list[str]
    label: Optional[str] = None
    sequence: Optional[int] = None


def infer_label_from_path(path: Path) -> Optional[str]:
    try:
        return parse_clip_filename(path).label_name
    except ValueError:
        stem = path.stem.lower()
        if stem.endswith("_1"):
            return "preictal"
        if stem.endswith("_0"):
            return "interictal"
        return None


def _channel_names(count: int, channel_indices: Optional[np.ndarray]) -> list[str]:
    if channel_indices is None:
        return [f"Ch {idx + 1}" for idx in range(count)]
    flat = np.ravel(channel_indices)
    return [f"Ch {int(idx)}" for idx in flat]


def load_eeg_clip(path: str | Path) -> EEGClip:
    file_path = Path(path)
    mat = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)

    data = None
    sampling_rate = None
    channel_indices = None
    sequence = None

    if "dataStruct" in mat:
        ds = mat["dataStruct"]
        if hasattr(ds, "data"):
            data = np.asarray(ds.data, dtype=np.float32)
        if hasattr(ds, "iEEGsamplingRate"):
            sampling_rate = float(ds.iEEGsamplingRate)
        if hasattr(ds, "channelIndices"):
            channel_indices = np.asarray(ds.channelIndices)
        if hasattr(ds, "sequence"):
            sequence = int(ds.sequence)

    if data is None and "data" in mat:
        data = np.asarray(mat["data"], dtype=np.float32)
        if "iEEGsamplingRate" in mat:
            sampling_rate = float(np.asarray(mat["iEEGsamplingRate"]).squeeze())
        if "channelIndices" in mat:
            channel_indices = np.asarray(mat["channelIndices"])

    if data is None:
        raise ValueError(f"Could not find EEG data in {file_path}")
    if data.ndim != 2:
        raise ValueError(f"Expected 2D EEG data, got shape {data.shape}")

    if sampling_rate is None:
        sampling_rate = DEFAULT_SAMPLING_RATE

    return EEGClip(
        path=file_path,
        data=data,
        sampling_rate=sampling_rate,
        channel_names=_channel_names(data.shape[1], channel_indices),
        label=infer_label_from_path(file_path),
        sequence=sequence,
    )


def remove_dropout_rows(data: np.ndarray) -> np.ndarray:
    keep_mask = ~np.all(data == 0, axis=1)
    if not np.any(keep_mask):
        raise ValueError("Clip contains only dropout rows")
    return data[keep_mask]


def normalize_per_channel(data: np.ndarray) -> np.ndarray:
    center = np.median(data, axis=0, keepdims=True)
    scale = np.std(data, axis=0, keepdims=True)
    scale[scale == 0] = 1.0
    return (data - center) / scale
