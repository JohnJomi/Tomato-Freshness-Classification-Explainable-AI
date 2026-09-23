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
backend/                FastAPI service over the trained artifacts
  main.py               app, CORS, startup loading, error handling
  api/                  routers: dataset, models, predict, explanations
  services/             model_service (artifacts + predictions), lime_service, shap_service
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

## Run the backend (FastAPI)

Requires the artifacts from `python ml/run_pipeline.py` (the `.joblib` models
are gitignored, so run the pipeline once after cloning).

```bash
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Interactive API docs: http://localhost:8000/docs. If port 8000 is taken, pass
another `--port`. CORS allows `http://localhost:3000` by default; override with
`FRONTEND_ORIGINS=http://host:port,...`.

Models and explainers are loaded once at startup; nothing is retrained per
request. If the artifacts are missing, the server still starts, `/health`
reports `"degraded"`, and the other endpoints return a 503 that says to run the
pipeline.

| Method | Endpoint | Returns |
|---|---|---|
| GET | `/health` | status + CV-selected XAI model |
| GET | `/dataset` | sizes, classes, class distribution, per-feature stats |
| GET | `/dataset/samples` | held-out test samples (index + actual class) |
| GET | `/dataset/samples/{index}` | one test sample's 26 feature values |
| GET | `/models` | test metrics, CV mean/std, confusion matrix per model, selection metadata |
| POST | `/predict` | `{features, model?}` → predicted class, confidence, probabilities |
| GET | `/feature-importance` | ranked features for the CV-selected model |
| POST | `/explain/lime` | `{features}` or `{sample_index}` → local LIME contributions |
| POST | `/explain/shap` | `{features}` or `{sample_index}` → per-class SHAP values + base values |
| GET | `/explain/shap/global?class_name=` | global mean \|SHAP\| + beeswarm points |

`/predict` validates that exactly the 26 expected features are present and
finite, and defaults to the CV-selected model. The explanation endpoints
always use the CV-selected model and label each response `local` or `global`.
