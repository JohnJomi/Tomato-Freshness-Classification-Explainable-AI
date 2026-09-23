"""Loads the trained ML artifacts once at startup and serves predictions.

All ML logic lives in ml/src; this module only loads what run_pipeline.py
produced and reuses the same preprocessing/split code, so the backend can
never drift from the experiment.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ML_DIR = ROOT / "ml"
ARTIFACTS_DIR = ML_DIR / "artifacts"
DATA_PATH = ROOT / "data" / "window_statistical_features.csv"

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src.data_loader import FEATURE_COLUMNS, load_dataset  # noqa: E402
from src.preprocessing import (  # noqa: E402
    CLASS_ORDER,
    encode_target,
    make_train_test_split,
    map_target,
    split_features_target,
)

MODEL_FILES = {
    "Random Forest": "random_forest.joblib",
    "SVM": "svm.joblib",
    "XGBoost": "xgboost.joblib",
}


class ArtifactsMissingError(RuntimeError):
    """Raised when run_pipeline.py hasn't been run yet."""


def _read_json(name: str) -> Any:
    with open(ARTIFACTS_DIR / name) as f:
        return json.load(f)


@dataclass
class ModelService:
    models: dict[str, Any]
    metrics: list[dict[str, Any]]
    confusion_matrices: dict[str, Any]
    feature_importance: dict[str, Any]
    model_selection: dict[str, Any]
    dataset_summary: dict[str, Any]
    feature_stats: dict[str, dict[str, float]]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.Series

    @classmethod
    def load(cls) -> "ModelService":
        missing = [f for f in MODEL_FILES.values() if not (ARTIFACTS_DIR / f).exists()]
        if missing:
            raise ArtifactsMissingError(
                f"Missing model artifacts {missing} in {ARTIFACTS_DIR}. "
                "Run `python ml/run_pipeline.py` first."
            )

        models = {name: joblib.load(ARTIFACTS_DIR / f) for name, f in MODEL_FILES.items()}

        # Rebuild the exact same train/test split the pipeline used (same
        # function, same seed) so the UI can browse the real held-out samples.
        df = map_target(load_dataset(DATA_PATH))
        X, y_str = split_features_target(df)
        X_train, X_test, _, y_test = make_train_test_split(X, encode_target(y_str))

        stats = X.describe().T
        feature_stats = {
            feat: {
                "mean": float(stats.loc[feat, "mean"]),
                "std": float(stats.loc[feat, "std"]),
                "min": float(stats.loc[feat, "min"]),
                "median": float(stats.loc[feat, "50%"]),
                "max": float(stats.loc[feat, "max"]),
            }
            for feat in FEATURE_COLUMNS
        }

        return cls(
            models=models,
            metrics=_read_json("metrics.json"),
            confusion_matrices=_read_json("confusion_matrices.json"),
            feature_importance=_read_json("feature_importance.json"),
            model_selection=_read_json("model_selection.json"),
            dataset_summary=_read_json("dataset_summary.json"),
            feature_stats=feature_stats,
            X_train=X_train,
            X_test=X_test.reset_index(drop=True),
            y_test=y_test.reset_index(drop=True),
        )

    # ------------------------------------------------------------------ #

    @property
    def xai_model_name(self) -> str:
        return self.model_selection["xai_selected_model"]

    def resolve_model(self, name: str | None) -> tuple[str, Any]:
        name = name or self.xai_model_name
        if name not in self.models:
            raise KeyError(name)
        return name, self.models[name]

    @staticmethod
    def to_frame(features: dict[str, float]) -> pd.DataFrame:
        return pd.DataFrame([[features[f] for f in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    def predict(self, features: dict[str, float], model_name: str | None = None) -> dict[str, Any]:
        name, model = self.resolve_model(model_name)
        proba = model.predict_proba(self.to_frame(features))[0]
        idx = int(np.argmax(proba))
        return {
            "predicted_class": CLASS_ORDER[idx],
            "confidence": float(proba[idx]),
            "probabilities": {c: float(p) for c, p in zip(CLASS_ORDER, proba)},
            "model": name,
        }

    def test_sample(self, index: int) -> tuple[dict[str, float], str]:
        if not 0 <= index < len(self.X_test):
            raise IndexError(index)
        row = self.X_test.iloc[index]
        return {f: float(row[f]) for f in FEATURE_COLUMNS}, CLASS_ORDER[int(self.y_test.iloc[index])]
