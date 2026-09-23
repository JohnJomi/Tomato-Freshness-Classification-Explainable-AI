"""Define the three supervised classifiers as sklearn Pipelines
(preprocessor -> estimator), so preprocessing is always performed correctly
inside cross-validation."""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .preprocessing import RANDOM_STATE, make_preprocessor

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent fallback
    XGBOOST_AVAILABLE = False


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
    """XGBoost multiclass classifier. Falls back to GradientBoostingClassifier
    (documented) if xgboost is unavailable in the environment."""
    if XGBOOST_AVAILABLE:
        estimator = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
        )
    else:
        print(
            "[models.py] WARNING: xgboost is not available in this environment. "
            "Falling back to GradientBoostingClassifier."
        )
        estimator = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=RANDOM_STATE,
        )

    return Pipeline(
        [
            ("preprocessor", make_preprocessor(scale=False)),
            ("model", estimator),
        ]
    )


def get_models() -> dict[str, Pipeline]:
    """Return the three model pipelines, keyed by display name."""
    return {
        "Random Forest": build_random_forest(),
        "SVM": build_svm(),
        "XGBoost": build_xgboost(),
    }
