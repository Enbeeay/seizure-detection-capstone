from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ForecastResult:
    scores: np.ndarray
    predicted: np.ndarray
    actual: np.ndarray


class FeatureForecaster:
    """Predicts the next feature vector from a short history of prior windows."""

    def __init__(self, history_steps: int = 3, alpha: float = 1.0):
        self.history_steps = history_steps
        self.pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )

    def _make_supervised(self, feature_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_rows = []
        y_rows = []
        for idx in range(self.history_steps, len(feature_matrix)):
            x_rows.append(feature_matrix[idx - self.history_steps : idx].reshape(-1))
            y_rows.append(feature_matrix[idx])
        if not x_rows:
            raise ValueError("Not enough windows to build a forecasting dataset")
        return np.asarray(x_rows), np.asarray(y_rows)

    def fit(self, feature_matrix: np.ndarray) -> "FeatureForecaster":
        x_train, y_train = self._make_supervised(feature_matrix)
        self.pipeline.fit(x_train, y_train)
        return self

    def score(self, feature_matrix: np.ndarray) -> ForecastResult:
        x_eval, y_eval = self._make_supervised(feature_matrix)
        predicted = self.pipeline.predict(x_eval)
        errors = (predicted - y_eval) ** 2
        scores = np.mean(errors, axis=1)
        return ForecastResult(scores=scores, predicted=predicted, actual=y_eval)
