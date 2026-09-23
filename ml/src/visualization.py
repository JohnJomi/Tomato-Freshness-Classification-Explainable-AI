"""Required visualizations: class distribution, confusion matrices, model comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .preprocessing import CLASS_ORDER

matplotlib.use("Agg")  # headless/non-interactive backend, safe for scripts

sns.set_theme(style="whitegrid")


def plot_class_distribution(y: pd.Series, output_path: str | Path) -> None:
    """Bar chart of the three target classes after mapping."""
    counts = y.value_counts().reindex(CLASS_ORDER)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, ax=ax, legend=False)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 3, str(int(v)), ha="center", fontweight="bold")
    ax.set_title("Class Distribution (after Stale + Spoiled merge)")
    ax.set_xlabel("Freshness Class")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_confusion_matrices(
    results: dict[str, dict[str, Any]], output_path: str | Path
) -> None:
    """1x3 grid of confusion matrices, one per model, fixed class order."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, r) in zip(axes, results.items()):
        cm = r["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_ORDER,
            yticklabels=CLASS_ORDER,
            cbar=False,
            ax=ax,
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices by Model")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_model_comparison(results_df: pd.DataFrame, output_path: str | Path) -> None:
    """Grouped bar chart: Accuracy/Precision/Recall/F1-score across models."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    x = np.arange(len(results_df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, results_df[metric], width, label=metric)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df["Model"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison (Test Set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")
