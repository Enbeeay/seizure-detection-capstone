from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from scipy.signal import butter, hilbert, sosfiltfilt, welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from eeg_demo.advanced_eeg_features import AdvancedFeatureConfig, extract_combined_features, validate_feature_matrix
from eeg_demo.dataset import ClipRecord, index_dataset


def load_raw_clip(path: Path) -> tuple[np.ndarray, float]:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    data = None
    sampling_rate = 400.0

    if "dataStruct" in mat:
        ds = mat["dataStruct"]
        if hasattr(ds, "data"):
            data = np.asarray(ds.data, dtype=np.float32)
        if hasattr(ds, "iEEGsamplingRate"):
            sampling_rate = float(ds.iEEGsamplingRate)

    if data is None and "data" in mat:
        data = np.asarray(mat["data"], dtype=np.float32)
        if "iEEGsamplingRate" in mat:
            sampling_rate = float(np.asarray(mat["iEEGsamplingRate"]).squeeze())

    if data is None:
        raise ValueError(f"Could not load EEG clip from {path}")

    keep_mask = ~np.all(data == 0, axis=1)
    if np.any(keep_mask):
        data = data[keep_mask]

    center = np.median(data, axis=0, keepdims=True)
    scale = np.std(data, axis=0, keepdims=True)
    scale[scale == 0] = 1.0
    data = (data - center) / scale

    return data.astype(np.float32), sampling_rate


def downsample_mean(data: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return data
    usable = (data.shape[0] // factor) * factor
    data = data[:usable]
    return data.reshape(usable // factor, factor, data.shape[1]).mean(axis=1)


def spectral_entropy(power: np.ndarray) -> float:
    total = float(np.sum(power))
    if total <= 0:
        return 0.0
    probs = power / total
    probs = np.clip(probs, 1e-12, None)
    return float(-np.sum(probs * np.log(probs)))


def _time_vector_for_window(window: np.ndarray, pool_bins: int) -> np.ndarray:
    """window: (time, channels) after downsampling."""
    pooled = np.array_split(window, pool_bins, axis=0)
    pooled_means = np.stack([chunk.mean(axis=0) for chunk in pooled], axis=0)
    return pooled_means.reshape(-1).astype(np.float32)


def build_time_vectors(data: np.ndarray, window_size: int, stride_size: int, pool_bins: int) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for start in range(0, data.shape[0] - window_size + 1, stride_size):
        window = data[start : start + window_size]
        vectors.append(_time_vector_for_window(window, pool_bins))
    return np.stack(vectors, axis=0)


_SPECTRAL_BANDS = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
)

_PLV_SINGLE_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 45.0),
}


def _spectral_vector_for_window(window: np.ndarray, effective_rate: float) -> np.ndarray:
    """window: (time, channels)."""
    features: list[float] = []
    for channel_idx in range(window.shape[1]):
        signal = window[:, channel_idx]
        freqs, power = welch(signal, fs=effective_rate, nperseg=min(len(signal), 256))
        total_power = float(np.trapezoid(power, freqs)) if len(freqs) else 0.0
        features.append(np.log1p(total_power))
        features.append(spectral_entropy(power))
        if len(freqs):
            features.append(float(freqs[int(np.argmax(power))]))
        else:
            features.append(0.0)
        band_powers: list[float] = []
        for _, low, high in _SPECTRAL_BANDS:
            mask = (freqs >= low) & (freqs < high)
            band_power = float(np.trapezoid(power[mask], freqs[mask])) if np.any(mask) else 0.0
            band_powers.append(band_power)
            features.append(np.log1p(band_power))
        alpha = band_powers[2] + 1e-6
        theta = band_powers[1] + 1e-6
        features.append(float(np.log1p(band_powers[3] / alpha)))
        features.append(float(np.log1p(band_powers[4] / theta)))
    return np.asarray(features, dtype=np.float32)


def build_spectral_vectors(data: np.ndarray, window_size: int, stride_size: int, effective_rate: float) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for start in range(0, data.shape[0] - window_size + 1, stride_size):
        window = data[start : start + window_size]
        vectors.append(_spectral_vector_for_window(window, effective_rate))
    return np.stack(vectors, axis=0)


def build_window_vectors(
    data: np.ndarray,
    sampling_rate: float,
    target_rate: int = 100,
    window_seconds: int = 10,
    stride_seconds: int = 10,
    pool_bins: int = 50,
    feature_mode: str = "time",
    advanced_config: AdvancedFeatureConfig | None = None,
    *,
    log_feature_progress: bool = False,
) -> np.ndarray:
    """
    Per-window feature matrix. Concatenates, in order:
    time pool (optional), spectral stats (optional), PLV block (optional), Riemannian block (optional).
    Controlled by ``feature_mode`` and ``advanced_config`` (``AdvancedFeatureConfig``).
    """
    adv = advanced_config or AdvancedFeatureConfig()
    if not adv.use_existing_features and not adv.use_plv and not adv.use_riemannian:
        raise ValueError("At least one of use_existing_features, use_plv, or use_riemannian must be True")

    factor = max(1, int(round(sampling_rate / target_rate)))
    data = downsample_mean(data, factor)
    effective_rate = sampling_rate / factor

    window_size = int(window_seconds * effective_rate)
    stride_size = int(stride_seconds * effective_rate)
    if data.shape[0] < window_size:
        raise ValueError("Clip shorter than one window")

    if adv.use_existing_features and feature_mode not in {"time", "spectral", "both"}:
        raise ValueError("feature_mode must be one of: time, spectral, both")

    warned_plv_bands: set[str] = set()
    vectors: list[np.ndarray] = []
    n_windows = 0
    for start in range(0, data.shape[0] - window_size + 1, stride_size):
        window = data[start : start + window_size]
        parts: list[np.ndarray] = []

        if adv.use_existing_features:
            if feature_mode in {"time", "both"}:
                parts.append(_time_vector_for_window(window, pool_bins))
            if feature_mode in {"spectral", "both"}:
                parts.append(_spectral_vector_for_window(window, effective_rate))

        if adv.use_plv or adv.use_riemannian:
            window_ct = np.ascontiguousarray(window.T, dtype=np.float32)
            parts.append(
                extract_combined_features(
                    window_ct,
                    float(effective_rate),
                    adv,
                    warned_plv_bands=warned_plv_bands,
                )
            )

        vectors.append(np.concatenate(parts, axis=0).astype(np.float32))
        n_windows += 1

    out = np.stack(vectors, axis=0)
    if log_feature_progress:
        tqdm.write(
            f"[features] windows={n_windows} dim={out.shape[1]} "
            f"existing={adv.use_existing_features} plv={adv.use_plv} riemannian={adv.use_riemannian}"
        )
    return out


def _plv_upper_triangle_for_window(
    window_ct: np.ndarray,
    sfreq: float,
    band_name: str,
    *,
    filter_order: int = 4,
) -> np.ndarray:
    """
    Build one PLV vector for a single window.

    window_ct: (channels, samples)
    Returns: upper-triangle (k=1) PLV values in [0, 1], shape=(n_channels*(n_channels-1)/2,)
    """
    if band_name not in _PLV_SINGLE_BANDS:
        raise ValueError(f"Unknown PLV band {band_name!r}; choose from {sorted(_PLV_SINGLE_BANDS)}")
    low_hz, high_hz = _PLV_SINGLE_BANDS[band_name]
    nyq = 0.5 * float(sfreq)
    if high_hz >= nyq:
        raise ValueError(
            f"Band {band_name} ({low_hz}-{high_hz} Hz) is not valid for sfreq={sfreq:.2f} Hz (Nyquist={nyq:.2f})"
        )

    x = np.asarray(window_ct, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("window_ct must be 2D (channels, samples)")
    n_ch, _ = x.shape
    if n_ch < 2:
        return np.zeros(0, dtype=np.float32)

    sos = butter(filter_order, [low_hz, high_hz], btype="band", fs=float(sfreq), output="sos")
    phases = np.empty_like(x, dtype=np.float64)
    for c in range(n_ch):
        band_sig = sosfiltfilt(sos, x[c])
        phases[c, :] = np.angle(hilbert(band_sig))

    z = np.exp(1j * phases)
    prod = z[:, np.newaxis, :] * np.conj(z[np.newaxis, :, :])
    plv = np.abs(np.mean(prod, axis=2))
    np.fill_diagonal(plv, 0.0)
    plv = np.clip(plv, 0.0, 1.0)
    iu = np.triu_indices(n_ch, k=1)
    return plv[iu].astype(np.float32)


def build_plv_sequence_vectors(
    data: np.ndarray,
    sampling_rate: float,
    *,
    target_rate: int = 100,
    window_seconds: float = 5.0,
    stride_seconds: float = 2.0,
    band_name: str = "beta",
    filter_order: int = 4,
    log_feature_progress: bool = False,
) -> np.ndarray:
    """
    Build a full clip sequence of single-band PLV vectors.

    Returns matrix shaped (n_windows, n_plv_pairs).
    """
    factor = max(1, int(round(sampling_rate / target_rate)))
    data = downsample_mean(data, factor)
    effective_rate = sampling_rate / factor

    window_size = int(window_seconds * effective_rate)
    stride_size = int(stride_seconds * effective_rate)
    if window_size <= 0 or stride_size <= 0:
        raise ValueError("Window and stride must be positive")
    if data.shape[0] < window_size:
        raise ValueError("Clip shorter than one window")

    vectors: list[np.ndarray] = []
    for start in range(0, data.shape[0] - window_size + 1, stride_size):
        window = data[start : start + window_size]
        vectors.append(
            _plv_upper_triangle_for_window(
                np.ascontiguousarray(window.T, dtype=np.float32),
                float(effective_rate),
                band_name,
                filter_order=filter_order,
            )
        )
    out = np.stack(vectors, axis=0)
    if log_feature_progress:
        tqdm.write(
            f"[plv-seq] windows={out.shape[0]} dim={out.shape[1]} band={band_name} "
            f"window={window_seconds}s stride={stride_seconds}s effective_rate={effective_rate:.2f}Hz"
        )
    return out


def prepare_plv_sequence_clip(
    record: ClipRecord,
    *,
    target_rate: int,
    window_seconds: float,
    stride_seconds: float,
    band_name: str,
    filter_order: int = 4,
    log_feature_progress: bool = False,
) -> PreparedClip:
    raw, sampling_rate = load_raw_clip(record.path)
    vectors = build_plv_sequence_vectors(
        raw,
        sampling_rate=sampling_rate,
        target_rate=target_rate,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        band_name=band_name,
        filter_order=filter_order,
        log_feature_progress=log_feature_progress,
    )
    return PreparedClip(record=record, vectors=vectors)


@dataclass
class PreparedClip:
    record: ClipRecord
    vectors: np.ndarray


def prepare_clip(
    record: ClipRecord,
    target_rate: int,
    window_seconds: int,
    stride_seconds: int,
    pool_bins: int,
    feature_mode: str,
    advanced_config: AdvancedFeatureConfig | None = None,
    *,
    log_feature_progress: bool = False,
    feature_debug: bool = False,
) -> PreparedClip:
    raw, sampling_rate = load_raw_clip(record.path)
    vectors = build_window_vectors(
        raw,
        sampling_rate=sampling_rate,
        target_rate=target_rate,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        pool_bins=pool_bins,
        feature_mode=feature_mode,
        advanced_config=advanced_config,
        log_feature_progress=log_feature_progress,
    )
    if feature_debug:
        validate_feature_matrix(vectors, name=f"{record.path.name} windows")
    return PreparedClip(record=record, vectors=vectors)


class WindowSequenceDataset(Dataset):
    def __init__(self, clips: list[PreparedClip], history_steps: int):
        self.clips = clips
        self.history_steps = history_steps
        self.index: list[tuple[int, int]] = []
        for clip_idx, clip in enumerate(clips):
            for target_idx in range(history_steps, len(clip.vectors)):
                self.index.append((clip_idx, target_idx))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        clip_idx, target_idx = self.index[idx]
        vectors = self.clips[clip_idx].vectors
        history = vectors[target_idx - self.history_steps : target_idx]
        target = vectors[target_idx]
        return torch.from_numpy(history), torch.from_numpy(target)


class ClipSequenceDataset(Dataset):
    """One training example per clip: variable-length sequence -> clip label."""

    def __init__(self, clips: list[PreparedClip], *, require_labels: bool = True):
        self.clips = clips
        self.require_labels = require_labels

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        clip = self.clips[idx]
        if self.require_labels and clip.record.label_id is None:
            raise ValueError(f"Clip {clip.record.path.name} has no label_id")
        seq = torch.from_numpy(clip.vectors.astype(np.float32))
        label_id = -1 if clip.record.label_id is None else int(clip.record.label_id)
        label = torch.tensor(label_id, dtype=torch.float32)
        length = torch.tensor(int(clip.vectors.shape[0]), dtype=torch.long)
        return seq, label, length


def clip_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    Pad variable-length clip sequences for batched LSTM classification.

    Returns padded sequences sorted by descending length for packed LSTM.
    """
    if not batch:
        raise ValueError("Empty batch")
    sequences, labels, lengths = zip(*batch)
    lengths_t = torch.stack(list(lengths))
    labels_t = torch.stack(list(labels))
    order = torch.argsort(lengths_t, descending=True)
    sorted_sequences = [sequences[int(i)] for i in order]
    sorted_lengths = lengths_t[order]
    sorted_labels = labels_t[order]
    padded = pad_sequence(sorted_sequences, batch_first=True)
    return padded, sorted_labels, sorted_lengths


class SequenceForecaster(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, input_dim),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        batch_size, history_steps, input_dim = history.shape
        encoded = self.encoder(history.reshape(batch_size * history_steps, input_dim))
        encoded = encoded.reshape(batch_size, history_steps, -1)
        _, hidden = self.gru(encoded)
        return self.decoder(hidden[-1])


class LSTMClipClassifier(nn.Module):
    """Clip-level binary classifier from variable-length PLV sequences."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, padded_sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            padded_sequences,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        _, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]
        logits = self.classifier(self.dropout(last_hidden)).squeeze(1)
        return logits


def train_model(
    model: SequenceForecaster,
    dataset: WindowSequenceDataset,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    *,
    show_progress: bool = True,
    shuffle_seed: int | None = None,
) -> list[float]:
    generator = None
    if shuffle_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(shuffle_seed))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    losses: list[float] = []

    model.to(device)
    n_batches = len(loader)
    if n_batches == 0:
        return []
    total_steps = epochs * n_batches
    pbar = tqdm(
        total=total_steps,
        desc="Training",
        unit="batch",
        disable=not show_progress,
        smoothing=0.05,
    )
    try:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            total_examples = 0
            for history, target in loader:
                history = history.to(device=device, dtype=torch.float32)
                target = target.to(device=device, dtype=torch.float32)
                optimizer.zero_grad()
                prediction = model(history)
                loss = criterion(prediction, target)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(history)
                total_examples += len(history)
                pbar.set_postfix(epoch=f"{epoch + 1}/{epochs}", loss=f"{float(loss.item()):.4f}")
                pbar.update(1)
            epoch_mean = total_loss / max(total_examples, 1)
            losses.append(epoch_mean)
            pbar.set_postfix(epoch=f"{epoch + 1}/{epochs}", epoch_loss=f"{epoch_mean:.4f}")
    finally:
        pbar.close()
    return losses


def train_lstm_classifier(
    model: LSTMClipClassifier,
    dataset: ClipSequenceDataset,
    *,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    show_progress: bool = True,
    shuffle_seed: int | None = None,
    pos_weight: float | None = None,
    print_shapes: bool = False,
    verbosity: int = 0,
) -> list[float]:
    generator = None
    if shuffle_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(shuffle_seed))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=clip_collate_fn,
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=None if pos_weight is None else torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    losses: list[float] = []
    n_batches = len(loader)
    if n_batches == 0:
        return losses
    pbar = tqdm(
        total=epochs * n_batches,
        desc="Train LSTM classifier",
        unit="batch",
        disable=not show_progress,
        smoothing=0.05,
    )
    if verbosity >= 1:
        tqdm.write(
            f"[train-lstm] clips={len(dataset)} batches={n_batches} epochs={epochs} "
            f"batch_size={batch_size} pos_weight={pos_weight}"
        )
    shape_printed = False
    try:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            total_examples = 0
            for padded, labels, lengths in loader:
                padded = padded.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.float32)
                lengths = lengths.to(device=device, dtype=torch.long)
                if print_shapes and not shape_printed:
                    print(
                        "[shapes] train_batch "
                        f"padded={tuple(padded.shape)} labels={tuple(labels.shape)} lengths={tuple(lengths.shape)}"
                    )
                    shape_printed = True
                optimizer.zero_grad()
                logits = model(padded, lengths)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                bsz = int(labels.shape[0])
                total_loss += float(loss.item()) * bsz
                total_examples += bsz
                pbar.set_postfix(epoch=f"{epoch + 1}/{epochs}", loss=f"{float(loss.item()):.4f}")
                pbar.update(1)
            epoch_mean = total_loss / max(total_examples, 1)
            losses.append(epoch_mean)
            pbar.set_postfix(epoch=f"{epoch + 1}/{epochs}", epoch_loss=f"{epoch_mean:.4f}")
            if verbosity >= 2:
                tqdm.write(f"[train-lstm] epoch={epoch + 1} mean_loss={epoch_mean:.6f}")
    finally:
        pbar.close()
    return losses


def score_clip(model: SequenceForecaster, clip: PreparedClip, history_steps: int, device: torch.device) -> np.ndarray:
    model.eval()
    vectors = clip.vectors
    scores: list[float] = []
    with torch.no_grad():
        for target_idx in range(history_steps, len(vectors)):
            history = torch.from_numpy(vectors[target_idx - history_steps : target_idx]).unsqueeze(0)
            history = history.to(device=device, dtype=torch.float32)
            target = vectors[target_idx]
            prediction = model(history).cpu().numpy()[0]
            scores.append(float(np.mean((prediction - target) ** 2)))
    return np.asarray(scores, dtype=np.float32)


def predict_clip_probabilities(
    model: LSTMClipClassifier,
    clips: list[PreparedClip],
    *,
    device: torch.device,
    batch_size: int = 64,
    show_progress: bool = True,
    print_shapes: bool = False,
    verbosity: int = 0,
) -> np.ndarray:
    dataset = ClipSequenceDataset(clips, require_labels=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=clip_collate_fn)
    model.eval()
    if verbosity >= 1:
        tqdm.write(f"[predict-lstm] clips={len(clips)} batches={len(loader)} batch_size={batch_size}")
    all_probs: list[np.ndarray] = []
    shape_printed = False
    with torch.no_grad():
        iterator = tqdm(loader, desc="Predict clips", unit="batch", disable=not show_progress, smoothing=0.05)
        for padded, _, lengths in iterator:
            padded = padded.to(device=device, dtype=torch.float32)
            lengths = lengths.to(device=device, dtype=torch.long)
            if print_shapes and not shape_printed:
                print(f"[shapes] predict_batch padded={tuple(padded.shape)} lengths={tuple(lengths.shape)}")
                shape_printed = True
            logits = model(padded, lengths)
            probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
            all_probs.append(probs)
    if not all_probs:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(all_probs, axis=0)


def summarize_clip_probabilities(
    clips: list[PreparedClip],
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    if len(clips) != len(probabilities):
        raise ValueError(f"Length mismatch: clips={len(clips)} probabilities={len(probabilities)}")
    rows = []
    for clip, prob in zip(clips, probabilities):
        rows.append(
            {
                "path": str(clip.record.path),
                "filename": clip.record.path.name,
                "patient_id": clip.record.patient_id,
                "segment_id": clip.record.segment_id,
                "label": clip.record.label_id,
                "probability": float(prob),
                "predicted_label": int(float(prob) >= float(threshold)),
                "n_windows": int(clip.vectors.shape[0]),
                "plv_feature_dim": int(clip.vectors.shape[1]),
            }
        )
    return pd.DataFrame(rows)


def summarize_scores(scores: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            float(np.mean(scores)),
            float(np.std(scores)),
            float(np.max(scores)),
            float(np.quantile(scores, 0.9)),
            float(np.quantile(scores, 0.95)),
        ],
        dtype=np.float32,
    )


def stratified_three_way_split(
    records: list[ClipRecord],
    random_state: int,
    calibration_fraction: float,
    validation_fraction: float,
) -> tuple[list[ClipRecord], list[ClipRecord], list[ClipRecord]]:
    labels = [record.label_id for record in records]
    train_records, holdout_records = train_test_split(
        records,
        test_size=calibration_fraction + validation_fraction,
        stratify=labels,
        random_state=random_state,
    )

    holdout_labels = [record.label_id for record in holdout_records]
    validation_share = validation_fraction / (calibration_fraction + validation_fraction)
    calibration_records, validation_records = train_test_split(
        holdout_records,
        test_size=validation_share,
        stratify=holdout_labels,
        random_state=random_state,
    )
    return train_records, calibration_records, validation_records


@dataclass
class EvaluationResult:
    probabilities: np.ndarray
    predictions: np.ndarray
    metrics: dict[str, float | int | None]


def find_best_f1_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
) -> dict[str, float | None]:
    """
    Choose a decision threshold by scanning [0, 1] that maximizes F1 (binary, positive class = 1).

    Threshold is selected on validation to balance precision and recall in an imbalanced setting;
    do not call this on test labels when reporting generalization.

    Uses ``zero_division=0`` so undefined precision/recall (e.g. no predicted positives) yield 0
    instead of raising.
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs, dtype=np.float64)
    if y_true.size == 0:
        raise ValueError("y_true must be non-empty")
    y_int = y_true.astype(np.int32, copy=False)
    if len(np.unique(y_int)) < 2:
        return {
            "best_threshold": 0.5,
            "best_f1": None,
            "precision": None,
            "recall": None,
        }

    thresholds = np.linspace(0.0, 1.0, 101)
    best_f1 = -1.0
    best_threshold = 0.5
    best_precision: float | None = None
    best_recall: float | None = None

    for t in thresholds:
        y_pred = (y_probs >= float(t)).astype(np.int32)
        f1 = float(f1_score(y_int, y_pred, pos_label=1, average="binary", zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = round(float(t), 4)
            best_precision = float(precision_score(y_int, y_pred, pos_label=1, average="binary", zero_division=0))
            best_recall = float(recall_score(y_int, y_pred, pos_label=1, average="binary", zero_division=0))

    return {
        "best_threshold": best_threshold,
        "best_f1": float(best_f1),
        "precision": best_precision,
        "recall": best_recall,
    }


def evaluate_probabilities(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> EvaluationResult:
    predictions = (probabilities >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics: dict[str, float | int | None] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "threshold": float(threshold),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
        metrics["pr_auc"] = float(average_precision_score(labels, probabilities))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    sens, spec, prec, rec, f1 = _binary_rates_from_counts(int(tn), int(fp), int(fn), int(tp))
    metrics["sensitivity"] = sens
    metrics["specificity"] = spec
    metrics["precision"] = prec
    metrics["recall"] = rec
    metrics["f1"] = f1

    return EvaluationResult(probabilities=probabilities, predictions=predictions, metrics=metrics)


def _binary_rates_from_counts(
    tn: int, fp: int, fn: int, tp: int
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Sensitivity (=recall), specificity, precision, recall, F1; None if denominator is zero."""
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else None
    recall = sensitivity
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = float(2 * precision * recall / (precision + recall))
    else:
        f1 = None
    return sensitivity, specificity, precision, recall, f1


def collect_records(root: Path) -> list[ClipRecord]:
    records = [record for record in index_dataset(root) if record.label_id is not None]
    if not records:
        raise ValueError(f"No labeled records found in {root}")
    return sorted(records, key=lambda record: record.path.name)


def fit_calibrator(summary_features: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(summary_features, labels)
    return model


def collect_test_records(root: Path) -> list[ClipRecord]:
    records = index_dataset(root)
    if not records:
        raise ValueError(f"No records found in {root}")
    return sorted(records, key=lambda record: record.path.name)


def apply_public_test_labels(records: list[ClipRecord], csv_path: Path) -> tuple[list[ClipRecord], list[ClipRecord]]:
    labels = pd.read_csv(csv_path)
    labels["image"] = labels["image"].astype(str)
    label_map = {
        row.image: (None if pd.isna(row["class"]) else int(row["class"]), str(row.usage))
        for _, row in labels.iterrows()
    }

    enriched: list[ClipRecord] = []
    public_records: list[ClipRecord] = []
    for record in records:
        stem = record.path.stem
        label_id, usage = label_map.get(stem, (record.label_id, "Missing"))
        label_name = None if label_id is None else ("interictal" if label_id == 0 else "preictal")
        updated = ClipRecord(
            path=record.path,
            patient_id=record.patient_id,
            segment_id=record.segment_id,
            split=record.split,
            label_id=label_id,
            label_name=label_name,
        )
        enriched.append(updated)
        if usage == "Public" and label_id is not None:
            public_records.append(updated)

    return enriched, public_records
