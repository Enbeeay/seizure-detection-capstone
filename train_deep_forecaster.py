from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from eeg_demo.deep_forecaster import (
    PreparedClip,
    SequenceForecaster,
    apply_public_test_labels,
    collect_test_records,
    WindowSequenceDataset,
    collect_records,
    evaluate_probabilities,
    fit_calibrator,
    prepare_clip,
    score_clip,
    stratified_three_way_split,
    summarize_scores,
    train_model,
)


def prepare_many(records, target_rate, window_seconds, stride_seconds, pool_bins, feature_mode):
    clips: list[PreparedClip] = []
    for idx, record in enumerate(records, start=1):
        print(f"[prepare] {idx}/{len(records)} {record.path.name}")
        try:
            clips.append(
                prepare_clip(
                    record,
                    target_rate=target_rate,
                    window_seconds=window_seconds,
                    stride_seconds=stride_seconds,
                    pool_bins=pool_bins,
                    feature_mode=feature_mode,
                )
            )
        except Exception as exc:
            print(f"[skip] {record.path.name}: {exc}")
    return clips


def score_many(model, clips, history_steps, device):
    rows = []
    for idx, clip in enumerate(clips, start=1):
        print(f"[score] {idx}/{len(clips)} {clip.record.path.name}")
        scores = score_clip(model, clip, history_steps=history_steps, device=device)
        summary = summarize_scores(scores)
        rows.append(
            {
                "path": str(clip.record.path),
                "filename": clip.record.path.name,
                "patient_id": clip.record.patient_id,
                "segment_id": clip.record.segment_id,
                "label": clip.record.label_id,
                "score_mean": float(summary[0]),
                "score_std": float(summary[1]),
                "score_max": float(summary[2]),
                "score_p90": float(summary[3]),
                "score_p95": float(summary[4]),
            }
        )
    return pd.DataFrame(rows)


def save_histogram(output_path: Path, frame: pd.DataFrame, title: str) -> None:
    plt.figure(figsize=(8, 5))
    for label_value, label_name in [(0, "Interictal"), (1, "Preictal")]:
        subset = frame.loc[frame["label"] == label_value, "probability"]
        if len(subset) > 0:
            plt.hist(subset, bins=20, alpha=0.6, label=label_name, density=True)
    plt.xlabel("Predicted preictal probability")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep sequence forecaster for file-level preictal probability")
    parser.add_argument("--train-root", type=Path, default=Path("D:/seizure-detection-data-train"))
    parser.add_argument("--test-root", type=Path, default=Path("D:/seizure-detection-data-test"))
    parser.add_argument("--test-label-csv", type=Path, default=Path("D:/contest_test_data_labels_public.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/deep_forecaster"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--history-steps", type=int, default=6)
    parser.add_argument("--window-seconds", type=int, default=10)
    parser.add_argument("--stride-seconds", type=int, default=10)
    parser.add_argument("--target-rate", type=int, default=100)
    parser.add_argument("--pool-bins", type=int, default=50)
    parser.add_argument("--feature-mode", choices=["time", "spectral", "both"], default="time")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-files", type=int)
    parser.add_argument("--max-test-files", type=int)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_records = collect_records(args.train_root)
    all_test_records, public_test_records = apply_public_test_labels(
        collect_test_records(args.test_root),
        args.test_label_csv,
    )

    if args.max_train_files:
        labels = [record.label_id for record in train_records]
        train_records, _ = train_test_split(
            train_records,
            train_size=args.max_train_files,
            stratify=labels,
            random_state=args.random_state,
        )
    if args.max_test_files:
        labels = [record.label_id for record in public_test_records]
        if len(set(labels)) > 1 and args.max_test_files < len(public_test_records):
            public_test_records, _ = train_test_split(
                public_test_records,
                train_size=args.max_test_files,
                stratify=labels,
                random_state=args.random_state,
            )
        else:
            public_test_records = public_test_records[: args.max_test_files]

    forecaster_records, calibration_records, validation_records = stratified_three_way_split(
        train_records,
        random_state=args.random_state,
        calibration_fraction=args.calibration_fraction,
        validation_fraction=args.validation_fraction,
    )

    print(
        f"[split] forecaster={len(forecaster_records)} calibration={len(calibration_records)} validation={len(validation_records)} public_test={len(public_test_records)} all_test={len(all_test_records)}"
    )

    forecaster_clips = prepare_many(
        forecaster_records,
        args.target_rate,
        args.window_seconds,
        args.stride_seconds,
        args.pool_bins,
        args.feature_mode,
    )
    calibration_clips = prepare_many(
        calibration_records,
        args.target_rate,
        args.window_seconds,
        args.stride_seconds,
        args.pool_bins,
        args.feature_mode,
    )
    validation_clips = prepare_many(
        validation_records,
        args.target_rate,
        args.window_seconds,
        args.stride_seconds,
        args.pool_bins,
        args.feature_mode,
    )
    public_test_clips = prepare_many(
        public_test_records,
        args.target_rate,
        args.window_seconds,
        args.stride_seconds,
        args.pool_bins,
        args.feature_mode,
    )
    all_test_clips = prepare_many(
        all_test_records,
        args.target_rate,
        args.window_seconds,
        args.stride_seconds,
        args.pool_bins,
        args.feature_mode,
    )

    input_dim = forecaster_clips[0].vectors.shape[1]
    dataset = WindowSequenceDataset(forecaster_clips, history_steps=args.history_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[dataset] sequence_examples={len(dataset)} input_dim={input_dim}")

    model = SequenceForecaster(input_dim=input_dim)
    losses = train_model(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )

    calibration_frame = score_many(model, calibration_clips, args.history_steps, device)
    validation_frame = score_many(model, validation_clips, args.history_steps, device)
    public_test_frame = score_many(model, public_test_clips, args.history_steps, device)
    all_test_frame = score_many(model, all_test_clips, args.history_steps, device)

    summary_columns = ["score_mean", "score_std", "score_max", "score_p90", "score_p95"]
    calibrator = fit_calibrator(calibration_frame[summary_columns].to_numpy(), calibration_frame["label"].to_numpy())

    validation_probabilities = calibrator.predict_proba(validation_frame[summary_columns].to_numpy())[:, 1]
    validation_eval = evaluate_probabilities(validation_frame["label"].to_numpy(), validation_probabilities)
    validation_frame["probability"] = validation_eval.probabilities
    validation_frame["predicted_label"] = validation_eval.predictions

    public_test_probabilities = calibrator.predict_proba(public_test_frame[summary_columns].to_numpy())[:, 1]
    test_eval = evaluate_probabilities(public_test_frame["label"].to_numpy(), public_test_probabilities)
    public_test_frame["probability"] = test_eval.probabilities
    public_test_frame["predicted_label"] = test_eval.predictions

    all_test_probabilities = calibrator.predict_proba(all_test_frame[summary_columns].to_numpy())[:, 1]
    all_test_frame["probability"] = all_test_probabilities
    all_test_frame["predicted_label"] = (all_test_probabilities >= 0.5).astype(np.int32)

    validation_frame.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    public_test_frame.to_csv(args.output_dir / "public_test_predictions.csv", index=False)
    all_test_frame.to_csv(args.output_dir / "all_test_predictions.csv", index=False)

    torch.save(model.state_dict(), args.output_dir / "sequence_forecaster.pt")
    joblib.dump(calibrator, args.output_dir / "probability_calibrator.joblib")

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Sequence Forecaster Training Loss")
    plt.tight_layout()
    plt.savefig(args.output_dir / "training_loss.png", dpi=150)
    plt.close()

    save_histogram(args.output_dir / "validation_probability_histogram.png", validation_frame, "Validation Probability Separation")
    save_histogram(
        args.output_dir / "public_test_probability_histogram.png",
        public_test_frame,
        "Public Test Probability Separation",
    )

    metrics = {
        "device": str(device),
        "forecaster_train_files": len(forecaster_clips),
        "calibration_files": len(calibration_clips),
        "validation_files": len(validation_clips),
        "public_test_files": len(public_test_clips),
        "all_test_files": len(all_test_clips),
        "sequence_examples": len(dataset),
        "epochs": args.epochs,
        "history_steps": args.history_steps,
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "target_rate": args.target_rate,
        "pool_bins": args.pool_bins,
        "feature_mode": args.feature_mode,
        "training_losses": losses,
        "validation_metrics": validation_eval.metrics,
        "public_test_metrics": test_eval.metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"[saved] {args.output_dir}")


if __name__ == "__main__":
    main()
