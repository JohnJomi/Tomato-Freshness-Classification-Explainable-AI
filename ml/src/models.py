"""Define the three supervised classifiers as sklearn Pipelines
(preprocessor -> estimator), so preprocessing is always performed correctly
inside cross-validation."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .preprocessing import RANDOM_STATE, make_preprocessor

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # XGBoost is one of the three required models; no substitute.
    raise ImportError(
        "XGBoost is required for this experiment.\n"
        "Install it with:\n\n    pip install xgboost\n\n"
        "(On macOS, XGBoost also needs the OpenMP runtime: brew install libomp)"
    ) from exc


def build_random_forest() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(scale=False)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def build_svm() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(scale=True)),
            (
                "model",
                SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_xgboost() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(scale=False)),
            (
                "model",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=3,
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_STATE,
                    eval_metric="mlogloss",
                ),
            ),
        ]
    )


def get_models() -> dict[str, Pipeline]:
    """Return the three model pipelines, keyed by display name."""
    return {
        "Random Forest": build_random_forest(),
        "SVM": build_svm(),
        "XGBoost": build_xgboost(),
    }
