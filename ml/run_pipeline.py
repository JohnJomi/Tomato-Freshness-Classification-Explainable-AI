"""Single entry point for the full tomato-freshness ML/XAI pipeline.

Run from the project root:

    source .venv/bin/activate
    python ml/run_pipeline.py

Regenerates everything under ml/outputs/ and ml/artifacts/ from scratch.
Reproducible: random_state=42 is used everywhere it's supported.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.data_loader import FEATURE_COLUMNS, TARGET_COLUMN, inspect_dataset, load_dataset
from src.evaluation import build_results_table, evaluate_all_models, save_results
from src.feature_importance import (
    get_feature_importance,
    plot_feature_importance,
    select_model_for_xai,
)
from src.lime_explainer import explain_instance, pick_representative_sample, plot_lime_explanation, save_lime_html
from src.models import get_models
from src.preprocessing import (
    CLASS_ORDER,
    encode_target,
    make_cv,
    make_train_test_split,
    map_target,
    split_features_target,
)
from src.shap_explainer import (
    build_explanation,
    plot_global_bar,
    plot_local_explanation,
    plot_summary,
    save_shap_values,
)
from src.visualization import plot_class_distribution, plot_confusion_matrices, plot_model_comparison

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "window_statistical_features.csv"
FIGURES_DIR = ROOT / "ml" / "outputs" / "figures"
RESULTS_DIR = ROOT / "ml" / "outputs" / "results"
ARTIFACTS_DIR = ROOT / "ml" / "artifacts"

MODEL_FILENAMES = {
    "Random Forest": "random_forest.joblib",
    "SVM": "svm.joblib",
    "XGBoost": "xgboost.joblib",
}


def main() -> None:
    for d in (FIGURES_DIR, RESULTS_DIR, ARTIFACTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load & inspect ---------------------------------------------------
    df = load_dataset(DATA_PATH)
    inspect_dataset(df)

    raw_distribution = df[TARGET_COLUMN].value_counts().to_dict()

    # 2. Target mapping + X/y split ---------------------------------------
    df = map_target(df)
    X, y_str = split_features_target(df)
    y = encode_target(y_str)

    mapped_distribution = y_str.value_counts().reindex(CLASS_ORDER).to_dict()

    # 3. Train/test split + CV strategy ------------------------------------
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    cv = make_cv()

    # 4. Class distribution figure (post-mapping) --------------------------
    plot_class_distribution(y_str, FIGURES_DIR / "class_distribution.png")

    # 5. Models: CV + final test evaluation ---------------------------------
    models = get_models()
    results = evaluate_all_models(models, X_train, y_train, X_test, y_test, cv)
    results_df = build_results_table(results)
    print("\nFinal results table:\n", results_df.to_string(index=False))
    save_results(results, results_df, RESULTS_DIR)

    # 6. Confusion matrices + model comparison figures ----------------------
    plot_confusion_matrices(results, FIGURES_DIR / "confusion_matrices.png")
    plot_model_comparison(results_df, FIGURES_DIR / "model_comparison.png")

    # 7. Select model for XAI (based on evaluation, not hard-coded) ---------
    xai_model_name = select_model_for_xai(results)
    print(f"\nModel selected for XAI: {xai_model_name}")
    fitted = results[xai_model_name]["fitted_pipeline"]

    # 8. Feature importance ---------------------------------------------------
    importance_df = get_feature_importance(fitted)
    plot_feature_importance(importance_df, FIGURES_DIR / "feature_importance.png", xai_model_name)

    # 9. LIME -------------------------------------------------------------
    sample_index = pick_representative_sample(fitted, X_test, y_test)
    lime_result = explain_instance(fitted, X_train, X_test, y_test, sample_index)
    plot_lime_explanation(lime_result, FIGURES_DIR / "lime_explanation.png")
    save_lime_html(lime_result, FIGURES_DIR / "lime_explanation.html")

    # 10. SHAP --------------------------------------------------------------
    explanation, _ = build_explanation(fitted, X_test)
    plot_global_bar(explanation, FIGURES_DIR / "shap_bar.png")
    predicted_class_idx = CLASS_ORDER.index(lime_result["predicted_class"])
    plot_summary(explanation, predicted_class_idx, lime_result["predicted_class"], FIGURES_DIR / "shap_summary.png")
    plot_local_explanation(
        explanation, sample_index, predicted_class_idx, lime_result["predicted_class"], FIGURES_DIR / "shap_local.png"
    )
    save_shap_values(explanation, ARTIFACTS_DIR / "shap_values.json")

    # 11. Save artifacts for future reuse (e.g. a backend service) -----------
    for name, pipeline in models.items():
        joblib.dump(results[name]["fitted_pipeline"], ARTIFACTS_DIR / MODEL_FILENAMES[name])
    joblib.dump(fitted.named_steps["preprocessor"], ARTIFACTS_DIR / "preprocessor.joblib")

    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(results_df.to_dict(orient="records"), f, indent=2)

    with open(ARTIFACTS_DIR / "feature_importance.json", "w") as f:
        json.dump(
            {"model": xai_model_name, "features": importance_df.to_dict(orient="records")}, f, indent=2
        )

    with open(ARTIFACTS_DIR / "confusion_matrices.json", "w") as f:
        json.dump(
            {
                "class_order": CLASS_ORDER,
                "matrices": {name: r["confusion_matrix"].tolist() for name, r in results.items()},
            },
            f,
            indent=2,
        )

    with open(ARTIFACTS_DIR / "dataset_summary.json", "w") as f:
        json.dump(
            {
                "samples": int(df.shape[0]),
                "features": FEATURE_COLUMNS,
                "n_features": len(FEATURE_COLUMNS),
                "classes": CLASS_ORDER,
                "raw_distribution": {str(k): int(v) for k, v in raw_distribution.items()},
                "mapped_distribution": {str(k): int(v) for k, v in mapped_distribution.items()},
                "xai_model": xai_model_name,
            },
            f,
            indent=2,
        )

    print(f"\nAll artifacts saved to {ARTIFACTS_DIR}")
    print(f"All figures saved to {FIGURES_DIR}")
    print(f"All results saved to {RESULTS_DIR}")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
