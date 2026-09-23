"""LIME local explanation for one test-set prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from .data_loader import FEATURE_COLUMNS
from .preprocessing import CLASS_ORDER


def pick_representative_sample(
    fitted_pipeline, X_test: pd.DataFrame, y_test: pd.Series, min_confidence: float = 0.7
) -> int:
    """Pick a correctly-classified, reasonably-confident test sample (positional index
    into X_test/y_test) so the LIME/SHAP explanation is easy to discuss."""
    proba = fitted_pipeline.predict_proba(X_test)
    y_pred = np.argmax(proba, axis=1)
    confidence = proba.max(axis=1)

    y_test_arr = np.asarray(y_test)
    correct = y_pred == y_test_arr
    confident = confidence >= min_confidence
    candidates = np.where(correct & confident)[0]

    if len(candidates) == 0:
        # relax: any correct prediction, most confident first
        candidates = np.where(correct)[0]

    best = candidates[np.argmax(confidence[candidates])]
    return int(best)


def explain_instance(
    fitted_pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sample_index: int,
) -> dict[str, Any]:
    """Build a LIME explainer on the training data and explain one test sample."""
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=FEATURE_COLUMNS,
        class_names=CLASS_ORDER,
        mode="classification",
        random_state=42,
    )

    x_instance = X_test.iloc[sample_index].values
    proba = fitted_pipeline.predict_proba(x_instance.reshape(1, -1))[0]
    predicted_class = CLASS_ORDER[int(np.argmax(proba))]
    actual_class = CLASS_ORDER[int(np.asarray(y_test)[sample_index])]

    explanation = explainer.explain_instance(
        x_instance,
        fitted_pipeline.predict_proba,
        num_features=10,
        labels=(int(np.argmax(proba)),),
    )

    print(f"\nLIME explanation — test sample #{sample_index}")
    print(f"Actual class:    {actual_class}")
    print(f"Predicted class: {predicted_class}")
    print(f"Probabilities:   {dict(zip(CLASS_ORDER, np.round(proba, 4)))}")

    return {
        "explanation": explanation,
        "sample_index": sample_index,
        "actual_class": actual_class,
        "predicted_class": predicted_class,
        "probabilities": dict(zip(CLASS_ORDER, proba.tolist())),
    }


def plot_lime_explanation(result: dict[str, Any], output_path: str | Path) -> None:
    """Save LIME's feature-contribution bar chart as a static figure."""
    explanation = result["explanation"]
    predicted_label = CLASS_ORDER.index(result["predicted_class"])
    contributions = explanation.as_list(label=predicted_label)

    features = [c[0] for c in contributions][::-1]
    values = [c[1] for c in contributions][::-1]
    colors = ["#2e7d32" if v > 0 else "#c62828" for v in values]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(features, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(
        f"LIME Local Explanation — sample #{result['sample_index']}\n"
        f"Actual: {result['actual_class']} | Predicted: {result['predicted_class']}"
    )
    ax.set_xlabel(f"Contribution to '{result['predicted_class']}'")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def save_lime_html(result: dict[str, Any], output_path: str | Path) -> None:
    result["explanation"].save_to_file(str(output_path))
    print(f"Saved {output_path}")
