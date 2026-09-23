"""CV-based model selection for XAI, and feature importance for the selected model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from .data_loader import FEATURE_COLUMNS

# Model selection uses training-set CV scores only. Test-set metrics are never
# consulted, so the held-out test set stays a final, unbiased evaluation.
SELECTION_METRIC = "cv_f1_weighted_mean"
TIE_BREAKER_METRIC = "cv_accuracy_mean"


def select_model_for_xai(results: dict[str, dict[str, Any]]) -> str:
    """Pick the model for feature importance/LIME/SHAP by mean 5-fold CV weighted
    F1 on the training set, tie-broken by mean CV accuracy. Never hard-coded,
    and never based on test-set performance."""
    return max(
        results,
        key=lambda name: (
            results[name][SELECTION_METRIC],
            results[name][TIE_BREAKER_METRIC],
        ),
    )


def get_feature_importance(
    fitted_pipeline, X_train: pd.DataFrame | None = None, y_train: pd.Series | None = None
) -> tuple[pd.DataFrame, str]:
    """Feature importance for the selected model. Returns (ranked DataFrame, method).

    Tree-based models use native feature_importances_. Other models (e.g. SVM)
    fall back to permutation importance computed on the *training* set, so the
    test set is not used here either.
    """
    model = fitted_pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        method = "feature_importances_"
    else:
        if X_train is None or y_train is None:
            raise ValueError("Permutation importance needs X_train and y_train.")
        perm = permutation_importance(
            fitted_pipeline,
            X_train,
            y_train,
            scoring="f1_weighted",
            n_repeats=10,
            random_state=42,
        )
        importances = perm.importances_mean
        method = "permutation_importance (train set, f1_weighted)"

    df = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True), method


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: str | Path,
    model_name: str,
    top_n: int = 15,
) -> None:
    top = importance_df.head(top_n).iloc[::-1]  # reverse so highest bar is on top

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(x="importance", y="feature", data=top, hue="feature", legend=False, ax=ax)
    ax.set_title("Top Features Influencing Tomato Freshness Classification")
    ax.set_xlabel(f"Feature Importance ({model_name})")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")
