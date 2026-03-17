from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_score_plot(
    output_path: str | Path,
    interictal_times: np.ndarray,
    interictal_scores: np.ndarray,
    preictal_times: np.ndarray,
    preictal_scores: np.ndarray,
) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(interictal_times, interictal_scores, label="Interictal anomaly score", linewidth=1.5)
    plt.plot(preictal_times, preictal_scores, label="Preictal anomaly score", linewidth=1.5)
    plt.xlabel("Window start time (s)")
    plt.ylabel("Forecast error")
    plt.title("Forecasting Error by Window")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_distribution_plot(
    output_path: str | Path,
    interictal_scores: np.ndarray,
    preictal_scores: np.ndarray,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(interictal_scores, bins=20, alpha=0.6, label="Interictal", density=True)
    plt.hist(preictal_scores, bins=20, alpha=0.6, label="Preictal", density=True)
    plt.xlabel("Forecast error")
    plt.ylabel("Density")
    plt.title("Anomaly Score Separation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
