"""SHAP global + local explanations for the tree-based model selected for XAI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .data_loader import FEATURE_COLUMNS
from .preprocessing import CLASS_ORDER


def _transform(fitted_pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply just the fitted pipeline's preprocessing step (Imputer[, Scaler]),
    matching what the underlying tree model actually saw during training."""
    transformed = fitted_pipeline.named_steps["preprocessor"].transform(X)
    return pd.DataFrame(transformed, columns=FEATURE_COLUMNS, index=X.index)


def build_explanation(
    fitted_pipeline, X_eval: pd.DataFrame, X_background: pd.DataFrame | None = None
):
    """Return a shap.Explanation (samples x features x classes) for X_eval.

    Tree-based models use TreeExplainer on the model step, with the pipeline's
    preprocessing applied first. Other models (e.g. SVM) use a model-agnostic
    explainer over the full pipeline's predict_proba, with a training-set
    background sample.
    """
    model = fitted_pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        X_eval_t = _transform(fitted_pipeline, X_eval)
        # tree_path_dependent perturbation doesn't need a background dataset and
        # avoids XGBoost's "categorical split not supported" error under the
        # (default) interventional perturbation mode.
        explainer = shap.TreeExplainer(
            model, feature_names=FEATURE_COLUMNS, feature_perturbation="tree_path_dependent"
        )
        return explainer(X_eval_t), X_eval_t

    if X_background is None:
        raise ValueError("Non-tree models need X_background (training data) for SHAP.")
    masker = shap.maskers.Independent(X_background, max_samples=100)
    explainer = shap.Explainer(
        fitted_pipeline.predict_proba, masker, feature_names=FEATURE_COLUMNS, seed=42
    )
    return explainer(X_eval), X_eval


def plot_global_bar(explanation, output_path: str | Path, top_n: int = 15) -> None:
    """Global SHAP bar plot: mean |SHAP value| per feature, averaged over samples
    and classes. Built manually (rather than shap.plots.bar) since that helper
    errors on 3-D multiclass Explanation objects in this shap version."""
    values = np.asarray(explanation.values)  # (n_samples, n_features, n_classes)
    mean_abs = np.abs(values).mean(axis=(0, 2))

    order = np.argsort(mean_abs)[::-1][:top_n]
    features = [FEATURE_COLUMNS[i] for i in order][::-1]
    scores = mean_abs[order][::-1]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(features, scores, color="#4c72b0")
    ax.set_title("Global SHAP Feature Importance (mean |SHAP value| across classes)")
    ax.set_xlabel("Mean |SHAP value|")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_summary(explanation, class_index: int, class_name: str, output_path: str | Path) -> None:
    """SHAP summary (beeswarm) plot for one class — shows importance, direction and spread."""
    fig = plt.figure(figsize=(9, 7))
    class_explanation = explanation[:, :, class_index]
    shap.plots.beeswarm(class_explanation, show=False, max_display=15)
    fig.suptitle(f"SHAP Summary — class: {class_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_local_explanation(
    explanation, sample_index: int, class_index: int, class_name: str, output_path: str | Path
) -> None:
    """Local SHAP waterfall plot for one sample/class, for direct comparison with LIME."""
    fig = plt.figure(figsize=(9, 7))
    shap.plots.waterfall(explanation[sample_index, :, class_index], show=False, max_display=12)
    fig.suptitle(f"SHAP Local Explanation — sample #{sample_index} ({class_name})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def save_shap_values(explanation, output_path: str | Path) -> None:
    """Save raw SHAP values (samples x features x classes) + base values as JSON."""
    import json

    payload = {
        "feature_names": FEATURE_COLUMNS,
        "class_names": CLASS_ORDER,
        "values": np.asarray(explanation.values).tolist(),
        "base_values": np.asarray(explanation.base_values).tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(payload, f)
    print(f"Saved {output_path}")
