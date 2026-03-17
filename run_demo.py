from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENDOR_DIR = PROJECT_ROOT / ".vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from eeg_demo.data import load_eeg_clip, normalize_per_channel, remove_dropout_rows
from eeg_demo.dataset import filter_records, index_dataset
from eeg_demo.features import extract_window_features, make_windows
from eeg_demo.model import FeatureForecaster
from eeg_demo.plots import save_distribution_plot, save_score_plot


def prepare_feature_matrix(path: Path, window_seconds: float, stride_seconds: float):
    clip = load_eeg_clip(path)
    cleaned = remove_dropout_rows(clip.data)
    normalized = normalize_per_channel(cleaned)
    windows, start_times = make_windows(
        normalized,
        sampling_rate=clip.sampling_rate,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
    )
    features = extract_window_features(windows, clip.sampling_rate)
    return clip, features.matrix, start_times


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal EEG anomaly forecasting demo")
    parser.add_argument("--train", type=Path, help="Interictal clip used for training")
    parser.add_argument("--eval", type=Path, help="Preictal clip used for evaluation")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("D:/seizure-detection-data"),
        help="Root directory containing Kaggle-style .mat files",
    )
    parser.add_argument("--patient", type=int, help="Patient id to auto-select clips for")
    parser.add_argument(
        "--train-segment",
        type=int,
        help="Specific interictal training segment id to use when auto-selecting",
    )
    parser.add_argument(
        "--eval-segment",
        type=int,
        help="Specific preictal training segment id to use when auto-selecting",
    )
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--history-steps", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.train is None or args.eval is None:
        if args.patient is None:
            raise ValueError("Provide --train and --eval, or provide --dataset-root with --patient")
        records = index_dataset(args.dataset_root)
        patient_records = filter_records(records, patient_id=args.patient, split="train")
        interictal_records = filter_records(patient_records, label_name="interictal")
        preictal_records = filter_records(patient_records, label_name="preictal")
        if args.train_segment is not None:
            interictal_records = [r for r in interictal_records if r.segment_id == args.train_segment]
        if args.eval_segment is not None:
            preictal_records = [r for r in preictal_records if r.segment_id == args.eval_segment]
        if not interictal_records:
            raise ValueError("No matching interictal training clip found")
        if not preictal_records:
            raise ValueError("No matching preictal training clip found")
        args.train = interictal_records[0].path
        args.eval = preictal_records[0].path

    train_clip, train_features, train_times = prepare_feature_matrix(
        args.train, args.window_seconds, args.stride_seconds
    )
    eval_clip, eval_features, eval_times = prepare_feature_matrix(
        args.eval, args.window_seconds, args.stride_seconds
    )

    model = FeatureForecaster(history_steps=args.history_steps)
    model.fit(train_features)
    train_result = model.score(train_features)
    eval_result = model.score(eval_features)

    score_labels = np.concatenate(
        [
            np.zeros_like(train_result.scores, dtype=np.int32),
            np.ones_like(eval_result.scores, dtype=np.int32),
        ]
    )
    score_values = np.concatenate([train_result.scores, eval_result.scores])

    metrics = {
        "train_path": str(train_clip.path),
        "train_label": train_clip.label,
        "eval_path": str(eval_clip.path),
        "eval_label": eval_clip.label,
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "history_steps": args.history_steps,
        "train_windows": int(len(train_result.scores)),
        "eval_windows": int(len(eval_result.scores)),
        "train_score_mean": float(np.mean(train_result.scores)),
        "eval_score_mean": float(np.mean(eval_result.scores)),
        "train_score_std": float(np.std(train_result.scores)),
        "eval_score_std": float(np.std(eval_result.scores)),
        "roc_auc": float(roc_auc_score(score_labels, score_values)),
        "pr_auc": float(average_precision_score(score_labels, score_values)),
    }

    train_score_times = train_times[args.history_steps :]
    eval_score_times = eval_times[args.history_steps :]

    save_score_plot(
        args.output_dir / "forecast_error_timeseries.png",
        train_score_times,
        train_result.scores,
        eval_score_times,
        eval_result.scores,
    )
    save_distribution_plot(
        args.output_dir / "forecast_error_distribution.png",
        train_result.scores,
        eval_result.scores,
    )

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
