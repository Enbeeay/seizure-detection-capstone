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
from tqdm.auto import tqdm

from eeg_demo.advanced_eeg_features import AdvancedFeatureConfig
from eeg_demo.dataset import filter_records
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


def prepare_many(
    records,
    target_rate,
    window_seconds,
    stride_seconds,
    pool_bins,
    feature_mode,
    advanced_config: AdvancedFeatureConfig | None = None,
    *,
    desc: str = "Prepare clips",
    show_progress: bool = True,
    feature_debug_first: bool = False,
    log_feature_progress: bool = False,
):
    clips: list[PreparedClip] = []
    adv = advanced_config or AdvancedFeatureConfig()
    iterator = tqdm(
        records,
        desc=desc,
        unit="file",
        disable=not show_progress,
        smoothing=0.05,
    )
    for clip_idx, record in enumerate(iterator):
        name = record.path.name
        iterator.set_postfix_str(name[:48] + ("…" if len(name) > 48 else ""))
        try:
            clips.append(
                prepare_clip(
                    record,
                    target_rate=target_rate,
                    window_seconds=window_seconds,
                    stride_seconds=stride_seconds,
                    pool_bins=pool_bins,
                    feature_mode=feature_mode,
                    advanced_config=adv,
                    log_feature_progress=log_feature_progress,
                    feature_debug=feature_debug_first and clip_idx == 0,
                )
            )
        except Exception as exc:
            tqdm.write(f"[skip] {record.path.name}: {exc}")
    return clips


def score_many(
    model,
    clips,
    history_steps,
    device,
    *,
    desc: str = "Score clips",
    show_progress: bool = True,
):
    rows = []
    iterator = tqdm(
        clips,
        desc=desc,
        unit="file",
        disable=not show_progress,
        smoothing=0.05,
    )
    for clip in iterator:
        name = clip.record.path.name
        iterator.set_postfix_str(name[:48] + ("…" if len(name) > 48 else ""))
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


def three_way_split_records(
    records: list,
    random_state: int,
    calibration_fraction: float,
    validation_fraction: float,
) -> tuple[list, list, list]:
    if len(records) < 3:
        raise ValueError(f"Need at least 3 labeled train clips for split, got {len(records)}")
    labels = [record.label_id for record in records]
    n0 = sum(1 for lab in labels if lab == 0)
    n1 = sum(1 for lab in labels if lab == 1)
    use_stratify = n0 >= 2 and n1 >= 2
    if use_stratify:
        try:
            return stratified_three_way_split(
                records,
                random_state=random_state,
                calibration_fraction=calibration_fraction,
                validation_fraction=validation_fraction,
            )
        except ValueError:
            tqdm.write("[warn] stratified three-way split failed; using random split")
    holdout_frac = calibration_fraction + validation_fraction
    train_part, holdout = train_test_split(
        records,
        test_size=holdout_frac,
        random_state=random_state,
        shuffle=True,
    )
    if len(holdout) < 2:
        raise ValueError("Holdout split too small; need more train clips or lower calibration/validation fractions")
    val_share = validation_fraction / holdout_frac
    calib_part, val_part = train_test_split(
        holdout,
        test_size=val_share,
        random_state=random_state,
        shuffle=True,
    )
    return train_part, calib_part, val_part


def save_histogram(output_path: Path, frame: pd.DataFrame, title: str) -> None:
    plt.figure(figsize=(8, 5))
    if len(frame) == 0 or "probability" not in frame.columns or "label" not in frame.columns:
        plt.text(0.5, 0.5, "No data", ha="center", va="center", transform=plt.gca().transAxes)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return
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


def _subsample_train_records(records, max_files: int | None, random_state: int) -> list:
    if not max_files or len(records) <= max_files:
        return list(records)
    labels = [record.label_id for record in records]
    if len(set(labels)) > 1:
        subset, _ = train_test_split(
            records,
            train_size=max_files,
            stratify=labels,
            random_state=random_state,
        )
        return list(subset)
    return list(records[:max_files])


def advanced_config_from_args(args) -> AdvancedFeatureConfig:
    return AdvancedFeatureConfig(
        use_existing_features=not args.no_existing_features,
        use_plv=args.use_plv,
        use_riemannian=args.use_riemannian,
        plv_feature_mode=args.plv_feature_mode,
        riemannian_mode=args.riemannian_mode,
        covariance_regularization=args.covariance_regularization,
        include_low_gamma_plv=not args.plv_no_low_gamma,
    )


def _subsample_public_test_records(records, max_files: int | None, random_state: int) -> list:
    if not max_files or len(records) <= max_files:
        return list(records)
    labels = [record.label_id for record in records]
    if len(set(labels)) > 1 and max_files < len(records):
        subset, _ = train_test_split(
            records,
            train_size=max_files,
            stratify=labels,
            random_state=random_state,
        )
        return list(subset)
    return list(records[:max_files])


def run_single_training(
    args,
    output_dir: Path,
    train_records: list,
    public_test_records: list,
    all_test_records: list,
    *,
    patient_id: int | None = None,
    show_progress: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    adv = advanced_config_from_args(args)
    if not adv.use_existing_features and not adv.use_plv and not adv.use_riemannian:
        raise ValueError(
            "At least one of existing (time/spectral), PLV, or Riemannian features must be enabled."
        )

    train_records = _subsample_train_records(train_records, args.max_train_files, args.random_state)
    public_test_records = _subsample_public_test_records(
        public_test_records, args.max_test_files, args.random_state
    )

    forecaster_records, calibration_records, validation_records = three_way_split_records(
        train_records,
        random_state=args.random_state,
        calibration_fraction=args.calibration_fraction,
        validation_fraction=args.validation_fraction,
    )

    pid_note = f" patient={patient_id}" if patient_id is not None else ""
    print(
        f"[split{pid_note}] forecaster={len(forecaster_records)} calibration={len(calibration_records)} "
        f"validation={len(validation_records)} public_test={len(public_test_records)} all_test={len(all_test_records)}"
    )

    _prep_kw = {
        "show_progress": show_progress,
        "advanced_config": adv,
        "log_feature_progress": args.feature_log,
    }
    forecaster_clips = prepare_many(
        forecaster_records,
        args.target_rate,
        args.window_seconds,
        args.stride_seconds,
        args.pool_bins,
        args.feature_mode,
        desc="Prepare forecaster",
        feature_debug_first=args.feature_debug,
        **_prep_kw,
    )
    _prep_rest = {**_prep_kw, "feature_debug_first": False}
    calibration_clips = prepare_many(
        calibration_records,
        args.target_rate,
        args.window_seconds,
        args.stride_seconds,
        args.pool_bins,
        args.feature_mode,
        desc="Prepare calibration",
        **_prep_rest,
    )
    validation_clips = prepare_many(
        validation_records,
        args.target_rate,
        args.window_seconds,
        args.stride_seconds,
        args.pool_bins,
        args.feature_mode,
        desc="Prepare validation",
        **_prep_rest,
    )
    public_test_clips = prepare_many(
        public_test_records,
        args.target_rate,
        args.window_seconds,
        args.stride_seconds,
        args.pool_bins,
        args.feature_mode,
        desc="Prepare public test",
        **_prep_rest,
    )
    all_test_clips = prepare_many(
        all_test_records,
        args.target_rate,
        args.window_seconds,
        args.stride_seconds,
        args.pool_bins,
        args.feature_mode,
        desc="Prepare all test",
        **_prep_rest,
    )

    input_dim = forecaster_clips[0].vectors.shape[1]
    dataset = WindowSequenceDataset(forecaster_clips, history_steps=args.history_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(
        f"[dataset] sequence_examples={len(dataset)} input_dim={input_dim} "
        f"windows_per_first_clip={forecaster_clips[0].vectors.shape[0]}"
    )

    model = SequenceForecaster(input_dim=input_dim)
    losses = train_model(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        show_progress=show_progress,
    )

    calibration_frame = score_many(
        model, calibration_clips, args.history_steps, device, desc="Score calibration", show_progress=show_progress
    )
    validation_frame = score_many(
        model, validation_clips, args.history_steps, device, desc="Score validation", show_progress=show_progress
    )
    public_test_frame = score_many(
        model, public_test_clips, args.history_steps, device, desc="Score public test", show_progress=show_progress
    )
    all_test_frame = score_many(
        model, all_test_clips, args.history_steps, device, desc="Score all test", show_progress=show_progress
    )

    summary_columns = ["score_mean", "score_std", "score_max", "score_p90", "score_p95"]
    calib_X = calibration_frame[summary_columns].to_numpy()
    calib_y = calibration_frame["label"].to_numpy()
    if len(np.unique(calib_y)) < 2:
        raise ValueError(
            "Calibration set has only one class; increase train clips, relax --max-train-files, "
            "or lower calibration/validation fractions."
        )
    calibrator = fit_calibrator(calib_X, calib_y)

    validation_probabilities = calibrator.predict_proba(validation_frame[summary_columns].to_numpy())[:, 1]
    validation_eval = evaluate_probabilities(validation_frame["label"].to_numpy(), validation_probabilities)
    validation_frame["probability"] = validation_eval.probabilities
    validation_frame["predicted_label"] = validation_eval.predictions

    if len(public_test_frame) == 0:
        tqdm.write("[warn] no public test clips; public_test_metrics omitted")
        public_test_frame = pd.DataFrame(
            columns=[
                "path",
                "filename",
                "patient_id",
                "segment_id",
                "label",
                "score_mean",
                "score_std",
                "score_max",
                "score_p90",
                "score_p95",
                "probability",
                "predicted_label",
            ]
        )
        test_eval_metrics: dict = {}
    else:
        public_test_probabilities = calibrator.predict_proba(public_test_frame[summary_columns].to_numpy())[:, 1]
        test_eval = evaluate_probabilities(public_test_frame["label"].to_numpy(), public_test_probabilities)
        public_test_frame["probability"] = test_eval.probabilities
        public_test_frame["predicted_label"] = test_eval.predictions
        test_eval_metrics = test_eval.metrics

    if len(all_test_frame) > 0:
        all_test_probabilities = calibrator.predict_proba(all_test_frame[summary_columns].to_numpy())[:, 1]
        all_test_frame["probability"] = all_test_probabilities
        all_test_frame["predicted_label"] = (all_test_probabilities >= 0.5).astype(np.int32)
    else:
        all_test_frame = pd.DataFrame(
            columns=[
                "path",
                "filename",
                "patient_id",
                "segment_id",
                "label",
                "score_mean",
                "score_std",
                "score_max",
                "score_p90",
                "score_p95",
                "probability",
                "predicted_label",
            ]
        )

    validation_frame.to_csv(output_dir / "validation_predictions.csv", index=False)
    public_test_frame.to_csv(output_dir / "public_test_predictions.csv", index=False)
    all_test_frame.to_csv(output_dir / "all_test_predictions.csv", index=False)

    torch.save(model.state_dict(), output_dir / "sequence_forecaster.pt")
    joblib.dump(calibrator, output_dir / "probability_calibrator.joblib")

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Sequence Forecaster Training Loss")
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=150)
    plt.close()

    hist_val_title = "Validation Probability Separation"
    if patient_id is not None:
        hist_val_title = f"Patient {patient_id} — {hist_val_title}"
    save_histogram(output_dir / "validation_probability_histogram.png", validation_frame, hist_val_title)
    hist_pub_title = "Public Test Probability Separation"
    if patient_id is not None:
        hist_pub_title = f"Patient {patient_id} — {hist_pub_title}"
    save_histogram(
        output_dir / "public_test_probability_histogram.png",
        public_test_frame,
        hist_pub_title,
    )

    metrics = {
        "patient_id": patient_id,
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
        "use_existing_features": adv.use_existing_features,
        "use_plv": adv.use_plv,
        "use_riemannian": adv.use_riemannian,
        "plv_feature_mode": adv.plv_feature_mode,
        "riemannian_mode": adv.riemannian_mode,
        "covariance_regularization": adv.covariance_regularization,
        "include_low_gamma_plv": adv.include_low_gamma_plv,
        "training_losses": losses,
        "validation_metrics": validation_eval.metrics,
        "public_test_metrics": test_eval_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"[saved] {output_dir}")


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
    parser.add_argument(
        "--use-plv",
        action="store_true",
        help="Append per-window PLV (phase locking) features (bandpass + Hilbert phase)",
    )
    parser.add_argument(
        "--use-riemannian",
        action="store_true",
        help="Append per-window Riemannian / SPD covariance embedding features",
    )
    parser.add_argument(
        "--no-existing-features",
        action="store_true",
        help="Disable pooled time + Welch spectral features (for PLV-only / Riemannian-only ablations)",
    )
    parser.add_argument(
        "--plv-feature-mode",
        choices=["summary", "full_matrix"],
        default="summary",
        help="PLV: per-band mean/std/max of upper triangle, or flattened upper triangle per band",
    )
    parser.add_argument(
        "--riemannian-mode",
        choices=["tangent_space", "log_euclidean"],
        default="tangent_space",
        help="Covariance embedding: pyriemann tangent map at I if installed; else SPD matrix log fallback",
    )
    parser.add_argument(
        "--covariance-regularization",
        type=float,
        default=1e-6,
        help="Diagonal jitter added to sample covariance before log / tangent map",
    )
    parser.add_argument(
        "--plv-no-low-gamma",
        action="store_true",
        help="Exclude 30-45 Hz band from PLV (when sampling rate allows)",
    )
    parser.add_argument(
        "--feature-debug",
        action="store_true",
        help="On first forecaster clip: assert finite features and print shape",
    )
    parser.add_argument(
        "--feature-log",
        action="store_true",
        help="Per file: log window count and concatenated feature dimension during featurization",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-files", type=int)
    parser.add_argument("--max-test-files", type=int)
    parser.add_argument(
        "--per-patient",
        action="store_true",
        help="Train one model per patient (train clips for patient i only; test clips for patient i only)",
    )
    parser.add_argument(
        "--patient",
        type=int,
        default=None,
        metavar="ID",
        help="With --per-patient: only this patient. Without --per-patient: single run using only this patient's clips",
    )
    parser.add_argument(
        "--smoke-patient",
        type=int,
        nargs="?",
        const=1,
        default=None,
        metavar="ID",
        help="Smoke test: same as --per-patient --patient ID (default ID=1); caps to 32 train / 24 public test files if those flags unset",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm bars (prepare, train, score)",
    )
    args = parser.parse_args()

    if args.smoke_patient is not None:
        args.per_patient = True
        args.patient = args.smoke_patient
        if args.max_train_files is None:
            args.max_train_files = 32
        if args.max_test_files is None:
            args.max_test_files = 24

    show_progress = not args.no_progress

    train_records = collect_records(args.train_root)
    all_test_records, public_test_records = apply_public_test_labels(
        collect_test_records(args.test_root),
        args.test_label_csv,
    )

    if args.per_patient:
        train_ids = sorted({r.patient_id for r in train_records})
        if args.patient is not None:
            if args.patient not in train_ids:
                raise ValueError(
                    f"No train clips for patient {args.patient}; available train patient ids: {train_ids[:20]}"
                    + (" ..." if len(train_ids) > 20 else "")
                )
            patient_ids = [args.patient]
        else:
            patient_ids = train_ids

        args.output_dir.mkdir(parents=True, exist_ok=True)
        for pid in patient_ids:
            tr = filter_records(train_records, patient_id=pid)
            pub = filter_records(public_test_records, patient_id=pid)
            all_t = filter_records(all_test_records, patient_id=pid)
            if len(tr) < 3:
                tqdm.write(f"[skip] patient {pid}: need >=3 train clips, got {len(tr)}")
                continue
            out = args.output_dir / f"patient_{pid}"
            tqdm.write(f"[patient {pid}] train={len(tr)} public_test={len(pub)} all_test={len(all_t)} -> {out}")
            try:
                run_single_training(
                    args,
                    out,
                    tr,
                    pub,
                    all_t,
                    patient_id=pid,
                    show_progress=show_progress,
                )
            except Exception as exc:
                tqdm.write(f"[skip] patient {pid}: {exc}")
        return

    if args.patient is not None:
        train_records = filter_records(train_records, patient_id=args.patient)
        public_test_records = filter_records(public_test_records, patient_id=args.patient)
        all_test_records = filter_records(all_test_records, patient_id=args.patient)
        if len(train_records) < 3:
            raise ValueError(f"Need at least 3 train clips for patient {args.patient}, got {len(train_records)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_single_training(
        args,
        args.output_dir,
        train_records,
        public_test_records,
        all_test_records,
        patient_id=args.patient,
        show_progress=show_progress,
    )


if __name__ == "__main__":
    main()
