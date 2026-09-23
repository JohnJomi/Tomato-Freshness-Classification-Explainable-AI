# Tomato Freshness Classification + Explainable AI

Supervised ML system that classifies tomato freshness from sensor-derived
statistical features into three classes: **Pure Fresh**, **Good**, **Stale to
Spoiled**. Trains and compares three classifiers (Random Forest, SVM, XGBoost),
evaluates them with standard classification metrics, and explains predictions
with feature importance, LIME, and SHAP.

See `architecture.md` for the full architecture spec.

## Project layout

```
data/                   dataset CSV
ml/src/                 pipeline modules (data loading, preprocessing, models,
                         evaluation, feature importance, LIME, SHAP, visualization)
ml/run_pipeline.py      single entry point that runs the full pipeline
ml/outputs/figures/     generated plots
ml/outputs/results/     generated metrics/reports
ml/artifacts/           saved trained models + preprocessing artifacts (joblib/json)
report/                 final written report
tests/                  pytest regression tests (e.g. CV-only model selection)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **macOS only:** XGBoost requires the OpenMP runtime, which isn't bundled.
> Install it once with `brew install libomp` (needed for `import xgboost` to
> succeed on Apple Silicon/Intel Macs). Not required on Linux/Windows.

## Run the tests

```bash
source .venv/bin/activate
python -m pytest
```

## Run the full pipeline

```bash
source .venv/bin/activate
python ml/run_pipeline.py
```

This regenerates everything under `ml/outputs/` and `ml/artifacts/` from
scratch and is safe to rerun at any time (fixed `random_state=42` throughout).
