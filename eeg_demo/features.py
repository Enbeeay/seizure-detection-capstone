from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch


BANDS = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 80.0),
)


@dataclass
class WindowedFeatures:
    matrix: np.ndarray
    start_times: np.ndarray
    feature_names: list[str]


def make_windows(
    data: np.ndarray,
    sampling_rate: float,
    window_seconds: float = 5.0,
    stride_seconds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    window_size = int(window_seconds * sampling_rate)
    stride_size = int(stride_seconds * sampling_rate)
    if window_size <= 0 or stride_size <= 0:
        raise ValueError("Window and stride must be positive")
    if data.shape[0] < window_size:
        raise ValueError("Clip is shorter than one analysis window")

    windows = []
    start_times = []
    for start in range(0, data.shape[0] - window_size + 1, stride_size):
        end = start + window_size
        windows.append(data[start:end])
        start_times.append(start / sampling_rate)
    return np.asarray(windows), np.asarray(start_times)


def _bandpower(signal: np.ndarray, sampling_rate: float) -> list[float]:
    freqs, power = welch(signal, fs=sampling_rate, nperseg=min(len(signal), 512))
    features = []
    for _, low, high in BANDS:
        mask = (freqs >= low) & (freqs < high)
        value = np.trapz(power[mask], freqs[mask]) if np.any(mask) else 0.0
        value = np.log1p(value)
        features.append(float(value))
    return features


def extract_window_features(windows: np.ndarray, sampling_rate: float) -> WindowedFeatures:
    rows: list[list[float]] = []
    feature_names: list[str] = []
    channel_count = windows.shape[2]

    if not feature_names:
        for channel_idx in range(channel_count):
            prefix = f"ch{channel_idx + 1}"
            feature_names.extend(
                [
                    f"{prefix}_mean",
                    f"{prefix}_std",
                    f"{prefix}_line_length",
                    f"{prefix}_rms",
                ]
            )
            for band_name, _, _ in BANDS:
                feature_names.append(f"{prefix}_{band_name}_power")

    for window in windows:
        row: list[float] = []
        for channel_idx in range(channel_count):
            signal = window[:, channel_idx]
            diffs = np.diff(signal)
            row.extend(
                [
                    float(np.mean(signal)),
                    float(np.std(signal)),
                    float(np.mean(np.abs(diffs))),
                    float(np.sqrt(np.mean(signal ** 2))),
                ]
            )
            row.extend(_bandpower(signal, sampling_rate))
        rows.append(row)

    return WindowedFeatures(
        matrix=np.asarray(rows, dtype=np.float32),
        start_times=np.array([], dtype=np.float32),
        feature_names=feature_names,
    )
