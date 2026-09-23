"""Feature importance for the model selected for XAI (native feature_importances_)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data_loader import FEATURE_COLUMNS


def select_model_for_xai(results: dict[str, dict[str, Any]]) -> str:
    """Pick the model for feature importance/LIME/SHAP based on measured test
    performance (test F1-weighted, tie-broken by test accuracy) — never hard-coded."""
    return max(
        results,
        key=lambda name: (
            results[name]["metrics"]["test_f1_weighted"],
            results[name]["metrics"]["test_accuracy"],
        ),
    )


def get_feature_importance(fitted_pipeline) -> pd.DataFrame:
    """Extract feature_importances_ from the fitted model step. Requires a
    tree-based model (RandomForest/XGBoost/GradientBoosting)."""
    model = fitted_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            f"{type(model).__name__} has no feature_importances_; "
            "pick a tree-based model for feature importance."
        )
    importances = model.feature_importances_
    df = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


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
