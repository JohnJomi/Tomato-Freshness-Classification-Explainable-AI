"""Cross-validation, final fit/predict, and metrics for all models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from .preprocessing import CLASS_ORDER

# Integer codes 0/1/2, in CLASS_ORDER, matching preprocessing.encode_target().
CLASS_CODES = list(range(len(CLASS_ORDER)))

# Scoring used for cross-validation (weighted, per doc; macro added for the
# minority-class visibility noted in the plan).
CV_SCORING = {
    "accuracy": "accuracy",
    "precision_weighted": "precision_weighted",
    "recall_weighted": "recall_weighted",
    "f1_weighted": "f1_weighted",
    "f1_macro": "f1_macro",
}


def cross_validate_model(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, cv
) -> dict[str, float]:
    """Run stratified k-fold CV on the training set only. Returns mean/std per metric."""
    scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=CV_SCORING)
    result = {}
    for metric in CV_SCORING:
        values = scores[f"test_{metric}"]
        result[f"cv_{metric}_mean"] = float(np.mean(values))
        result[f"cv_{metric}_std"] = float(np.std(values))
    return result


def evaluate_on_test(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Fit the pipeline on the full training set and evaluate once on the untouched test set."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision_weighted": precision_score(y_test, y_pred, average="weighted"),
        "test_recall_weighted": recall_score(y_test, y_pred, average="weighted"),
        "test_f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "test_f1_macro": f1_score(y_test, y_pred, average="macro"),
    }

    report = classification_report(
        y_test, y_pred, labels=CLASS_CODES, target_names=CLASS_ORDER, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_CODES)

    return {
        "fitted_pipeline": pipeline,
        "y_pred": y_pred,
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def evaluate_all_models(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv,
) -> dict[str, dict[str, Any]]:
    """Cross-validate then fit/test each model. Never touches the test set for model selection."""
    results: dict[str, dict[str, Any]] = {}
    for name, pipeline in models.items():
        print(f"\n--- Evaluating {name} ---")
        cv_scores = cross_validate_model(pipeline, X_train, y_train, cv)
        test_result = evaluate_on_test(pipeline, X_train, y_train, X_test, y_test)

        print(
            f"CV accuracy: {cv_scores['cv_accuracy_mean']:.4f} "
            f"(+/- {cv_scores['cv_accuracy_std']:.4f})"
        )
        print(
            f"Test accuracy: {test_result['metrics']['test_accuracy']:.4f} | "
            f"Test F1 (weighted): {test_result['metrics']['test_f1_weighted']:.4f}"
        )
        print(test_result["classification_report"])

        results[name] = {**cv_scores, **test_result}
    return results


def build_results_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build the model-comparison DataFrame (CV mean/std + final test scores)."""
    rows = []
    for name, r in results.items():
        rows.append(
            {
                "Model": name,
                "Accuracy": r["metrics"]["test_accuracy"],
                "Precision": r["metrics"]["test_precision_weighted"],
                "Recall": r["metrics"]["test_recall_weighted"],
                "F1-score": r["metrics"]["test_f1_weighted"],
                "F1-macro": r["metrics"]["test_f1_macro"],
                "CV Accuracy Mean": r["cv_accuracy_mean"],
                "CV Accuracy Std": r["cv_accuracy_std"],
                "CV F1-weighted Mean": r["cv_f1_weighted_mean"],
                "CV F1-weighted Std": r["cv_f1_weighted_std"],
            }
        )
    return pd.DataFrame(rows)


def save_results(
    results: dict[str, dict[str, Any]],
    results_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Save the comparison table and per-model classification reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "model_metrics.csv", index=False)

    with open(output_dir / "classification_reports.txt", "w") as f:
        for name, r in results.items():
            f.write(f"{'=' * 70}\n{name}\n{'=' * 70}\n")
            f.write(r["classification_report"])
            f.write("\n\nConfusion matrix (rows=actual, cols=predicted):\n")
            f.write(f"Class order: {CLASS_ORDER}\n")
            f.write(f"{r['confusion_matrix']}\n\n")

    print(f"\nSaved results to {output_dir}/model_metrics.csv and classification_reports.txt")
