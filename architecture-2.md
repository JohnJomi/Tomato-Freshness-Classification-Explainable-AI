# High-Level Architecture — Tomato Freshness Classification + Explainable AI

## 1. Project Goal

Build a supervised machine learning system that classifies tomato freshness from sensor-derived statistical features into three required classes:

- **Pure Fresh**
- **Good**
- **Stale to Spoiled**

The system must train and compare three classification models, evaluate them using standard classification metrics, analyze feature importance, and explain individual/model predictions using **LIME** and **SHAP**.

> **Dataset observation:** The provided CSV contains 555 samples, 26 numeric input features, and currently has four target values: `Pure Fresh`, `Good`, `Stale`, and `Spoiled`.
>
> **Required 3-class mapping:** `Stale` and `Spoiled` should be merged into a single target class called **`Stale to Spoiled`**. Do this once during preprocessing and use the resulting three-class target throughout the project.

---

# 2. High-Level System Architecture

```text
                    ┌──────────────────────────┐
                    │      CSV Dataset         │
                    │ window_statistical_      │
                    │ features.csv              │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Data Loading & Inspection │
                    │ - shape / dtypes         │
                    │ - class distribution     │
                    │ - missing values         │
                    │ - duplicate check         │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      Preprocessing        │
                    │ - target mapping          │
                    │ - missing value handling  │
                    │ - numeric feature setup   │
                    │ - train/test split        │
                    │ - scaling where required  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Stratified K-Fold CV      │
                    │      (5 folds)            │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
     ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
     │ Model 1        │ │ Model 2        │ │ Model 3        │
     │ Random Forest  │ │ SVM (RBF)      │ │ XGBoost        │
     │ Classifier     │ │ Classifier     │ │ Classifier     │
     └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
             │                  │                  │
             └──────────────────┼──────────────────┘
                                ▼
                    ┌──────────────────────────┐
                    │ Model Evaluation         │
                    │ - Accuracy               │
                    │ - Precision              │
                    │ - Recall                 │
                    │ - F1-score               │
                    │ - Confusion matrices     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Select Model for XAI      │
                    │ Based on evaluation      │
                    │ (do not hard-code winner) │
                    └────────────┬─────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
                  ▼                             ▼
        ┌──────────────────┐          ┌──────────────────┐
        │ Feature          │          │ Explainable AI   │
        │ Importance       │          │                  │
        │ - built-in       │          │ ┌──────────────┐ │
        │   importance     │          │ │ LIME         │ │
        │ - permutation    │          │ └──────────────┘ │
        └────────┬─────────┘          │ ┌──────────────┐ │
                 │                    │ │ SHAP         │ │
                 │                    │ └──────────────┘ │
                 │                    └────────┬─────────┘
                 └──────────────┬─────────────┘
                                ▼
                    ┌──────────────────────────┐
                    │ Visualizations & Report  │
                    │ - class distribution     │
                    │ - model comparison       │
                    │ - confusion matrices     │
                    │ - feature importance     │
                    │ - LIME explanation       │
                    │ - SHAP explanation       │
                    └──────────────────────────┘
```

---

# 3. Dataset Specification

## Input File

`window_statistical_features.csv`

## Observed Dataset

- Rows: **555**
- Target column: `Freshness_Level`
- Input features: **26**
- Input feature types: numeric (`float64`)
- Missing values: currently **none observed**
- Categorical input features: **none**
- Target values observed:
  - `Pure Fresh`
  - `Good`
  - `Stale`
  - `Spoiled`

## Target Transformation

Convert:

```text
Pure Fresh  → Pure Fresh
Good        → Good
Stale       → Stale to Spoiled
Spoiled     → Stale to Spoiled
```

Final target classes:

```text
Pure Fresh
Good
Stale to Spoiled
```

The implementation should print the class distribution before and after this transformation.

---

# 4. Input Features

The dataset contains sensor statistics and environmental statistics.

### MQ Gas Sensor Features

For each sensor, there are mean, standard deviation, and slope features:

- `MQ2_mean`
- `MQ2_std`
- `MQ2_slope`
- `MQ3_mean`
- `MQ3_std`
- `MQ3_slope`
- `MQ9_mean`
- `MQ9_std`
- `MQ9_slope`
- `MQ135_mean`
- `MQ135_std`
- `MQ135_slope`
- `MQ136_mean`
- `MQ136_std`
- `MQ136_slope`
- `MQ138_mean`
- `MQ138_std`
- `MQ138_slope`

### Environmental Features

- `Temperature_mean`
- `Temperature_std`
- `Temperature_slope`
- `Humidity_mean`
- `Humidity_std`
- `Humidity_slope`

### Derived Ratio Features

- `MQ3_over_MQ2_end`
- `MQ136_over_MQ138_end`

These 26 columns are the model input features.

---

# 5. Data Preprocessing Pipeline

Use a reproducible preprocessing pipeline.

## Step 1 — Load

Use pandas to load the CSV.

## Step 2 — Inspect

Generate:

- dataset shape
- column names
- data types
- missing-value counts
- duplicate count
- target/class distribution
- basic descriptive statistics

## Step 3 — Target Mapping

Merge `Stale` and `Spoiled` into `Stale to Spoiled`.

## Step 4 — Separate X and y

```text
X = 26 numeric features
y = 3-class Freshness_Level
```

Do not include the target in X.

## Step 5 — Missing Values

Although the current dataset contains no missing values, implement a robust numeric imputation step using:

```text
SimpleImputer(strategy="median")
```

This keeps the pipeline safe if missing values appear later.

## Step 6 — Train/Test Split

Use:

```text
train_test_split(
    test_size=0.20,
    stratify=y,
    random_state=42
)
```

This gives an approximately 80/20 split while preserving class proportions.

## Step 7 — Scaling

Scaling is required for models such as SVM.

Use:

```text
StandardScaler()
```

For consistency, implement models through sklearn Pipelines.

Recommended:

```text
Imputer → StandardScaler → Model
```

for scale-sensitive models.

For Random Forest, scaling is not mathematically required, but using a consistent preprocessing pipeline is acceptable.

---

# 6. Cross-Validation Strategy

Use **Stratified 5-Fold Cross Validation**.

Reason:

- Classification problem
- Three target classes
- Preserves class proportions across folds

Configuration:

```text
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Use cross-validation on the training data.

Do not use the test set during model selection.

The final test set should remain untouched until final evaluation.

---

# 7. Classification Models

Use exactly three supervised classification models to satisfy the assignment.

## Model 1 — Random Forest

```text
RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
```

Why:

- Handles nonlinear relationships
- Works well with numeric sensor features
- Provides native feature importance
- Requires little preprocessing

---

## Model 2 — Support Vector Machine

Use an RBF-kernel SVM:

```text
SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced",
    random_state=42
)
```

Preprocessing:

```text
SimpleImputer → StandardScaler → SVC
```

Why:

- Effective for medium-sized datasets
- Can model nonlinear class boundaries
- Good fit for a dataset of only 555 samples

---

## Model 3 — XGBoost

Use XGBoost multiclass classification:

```text
XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss"
)
```

If XGBoost is unavailable in the environment, use a clearly documented fallback such as `GradientBoostingClassifier`.

Do not spend significant project time tuning hyperparameters. The goal is a complete, reproducible comparison rather than an exhaustive optimization study.

---

# 8. Evaluation Architecture

For each model:

1. Fit on training data.
2. Run 5-fold stratified cross-validation on the training set.
3. Calculate cross-validation metrics.
4. Fit the final model on the complete training set.
5. Predict the untouched test set.
6. Calculate test metrics.

Required metrics:

- Accuracy
- Precision
- Recall
- F1-score

For multiclass metrics use:

```text
average="weighted"
```

Also generate a classification report containing per-class precision, recall, and F1.

## Results Table

Create a DataFrame similar to:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest | ... | ... | ... | ... |
| SVM | ... | ... | ... | ... |
| XGBoost | ... | ... | ... | ... |

If useful, also include mean 5-fold CV scores separately from final test scores.

---

# 9. Visualization Architecture

The project must contain appropriate visualizations but should avoid unnecessary plots because the project has a short implementation deadline.

## Visualization 1 — Class Distribution

Recommended first visualization:

**Count plot / bar chart of the three target classes after target mapping.**

Purpose:

- Shows whether the dataset is balanced.
- Helps explain why stratified splitting/CV is used.

---

## Visualization 2 — Confusion Matrices

Create one confusion matrix per model.

Recommended layout:

```text
Random Forest      SVM      XGBoost
[ matrix ]       [ matrix ] [ matrix ]
```

Use the same class order:

```text
Pure Fresh
Good
Stale to Spoiled
```

---

## Visualization 3 — Model Comparison

Create a grouped bar chart comparing:

- Accuracy
- Precision
- Recall
- F1-score

across the three models.

---

# 10. Feature Importance

Use the selected tree-based model, preferably Random Forest or XGBoost.

Primary implementation:

```text
feature_importances_
```

Generate a horizontal bar chart showing the top 10 or top 15 features.

Recommended title:

```text
Top Features Influencing Tomato Freshness Classification
```

## Interpretation

Discuss which sensor statistics appear most informative.

Examples of possible interpretations:

- MQ gas sensor measurements may capture volatile compounds associated with freshness changes.
- Mean values may capture overall gas concentration.
- Standard deviation may capture variability.
- Slope may capture the change in sensor response over the observation window.
- Temperature and humidity may contribute as environmental/contextual variables.
- Ratio features may capture relative sensor behavior.

Do not claim a feature causes spoilage. Feature importance indicates predictive contribution, not causation.

---

# 11. LIME Architecture

Use:

```text
lime.lime_tabular.LimeTabularExplainer
```

LIME should explain **one individual test-set prediction**.

## Input

- Training data
- Feature names
- Class names
- Model prediction function

For a selected test sample:

```text
x_test[index]
```

obtain:

```text
predicted class
actual class
prediction probabilities
```

Then generate the LIME explanation.

## LIME Output

Save:

1. LIME feature-contribution visualization
2. HTML explanation if useful

Recommended visualization:

```text
Feature contribution
────────────────────────────────
MQ3_mean > threshold       ++++++
MQ136_slope < threshold    ++++
Humidity_mean              ---
...
```

Use a representative test sample.

Preferably choose:

- a correctly classified sample
- with reasonably confident prediction

This makes the explanation easier to discuss.

## LIME Interpretation

Explain:

- Which features pushed the prediction toward the predicted class.
- Which features pushed it away from competing classes.
- LIME is a **local approximation**, so the explanation applies to that specific prediction rather than the entire model.

---

# 12. SHAP Architecture

Use SHAP to explain model predictions.

Preferred model:

```text
XGBoost
```

If XGBoost is unavailable, use a tree-based model such as Random Forest.

For XGBoost:

```text
shap.TreeExplainer(model)
```

## Required SHAP Visualizations

At minimum:

### Global Feature Importance

Use:

```text
SHAP bar plot
```

This shows the overall importance of features.

### SHAP Summary Plot

Use:

```text
shap.summary_plot(...)
```

This shows:

- feature importance
- direction of influence
- distribution of SHAP values

### Optional — Individual Prediction

For the same sample used for LIME, generate a local SHAP explanation if practical.

This allows direct comparison:

```text
LIME → local explanation
SHAP → local + global explanation
```

---

# 13. XAI Interpretation Strategy

The report should distinguish between:

### Global explanations

Questions:

- Which features matter most across predictions?
- Which sensor statistics consistently influence classification?

Tools:

- Random Forest/XGBoost feature importance
- SHAP summary/bar plots

### Local explanations

Question:

- Why did the model classify this particular tomato sample as this freshness class?

Tools:

- LIME
- SHAP local explanation

Avoid saying:

> "The model knows the tomato is spoiled because MQ3 causes spoilage."

Instead say:

> "The model associated the observed MQ3-related feature values with a higher likelihood of the Stale to Spoiled class."

This distinction is important for explainable AI.

---

# 14. Recommended Project Structure

```text
tomato-freshness-ml/
│
├── data/
│   └── window_statistical_features.csv
│
├── notebooks/
│   └── tomato_freshness_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── feature_importance.py
│   ├── lime_explainer.py
│   ├── shap_explainer.py
│   └── visualization.py
│
├── outputs/
│   ├── figures/
│   │   ├── class_distribution.png
│   │   ├── confusion_matrix_random_forest.png
│   │   ├── confusion_matrix_svm.png
│   │   ├── confusion_matrix_xgboost.png
│   │   ├── model_comparison.png
│   │   ├── feature_importance.png
│   │   ├── lime_explanation.png
│   │   ├── shap_bar.png
│   │   └── shap_summary.png
│   │
│   └── results/
│       ├── model_metrics.csv
│       └── classification_reports.txt
│
├── report/
│   └── report.md
│
├── requirements.txt
├── README.md
└── architecture.md
```

For a 3-hour project, this can be simplified if necessary. The architecture should prioritize working code over excessive abstraction.

---

# 15. Suggested Module Responsibilities

## `data_loader.py`

Responsibilities:

- Load CSV
- Validate required columns
- Print basic dataset information
- Return DataFrame

## `preprocessing.py`

Responsibilities:

- Map target labels
- Separate X/y
- Train/test split
- Create preprocessing pipelines
- Create cross-validation strategy

## `models.py`

Responsibilities:

- Define the three models
- Return sklearn-compatible pipelines

## `evaluation.py`

Responsibilities:

- Cross-validation
- Model fitting
- Predictions
- Accuracy/precision/recall/F1
- Classification reports
- Confusion matrices
- Results DataFrame

## `feature_importance.py`

Responsibilities:

- Extract feature importance
- Rank features
- Generate importance visualization

## `lime_explainer.py`

Responsibilities:

- Create LIME explainer
- Explain selected test instance
- Save LIME visualization/HTML

## `shap_explainer.py`

Responsibilities:

- Create SHAP explainer
- Generate global SHAP plots
- Generate local SHAP explanation if feasible

## `visualization.py`

Responsibilities:

- Class distribution
- Model comparison
- Confusion matrices
- General plotting utilities

---

# 16. Main Execution Flow

A single script/notebook should make the complete experiment reproducible.

```text
START
  │
  ▼
Load CSV
  │
  ▼
Inspect dataset
  │
  ▼
Map 4 original labels → 3 required labels
  │
  ▼
Visualize class distribution
  │
  ▼
Split X/y
  │
  ▼
Stratified 80/20 train-test split
  │
  ▼
Create 5-fold Stratified CV
  │
  ▼
Train/evaluate Random Forest
  │
  ▼
Train/evaluate SVM
  │
  ▼
Train/evaluate XGBoost
  │
  ▼
Generate comparison table
  │
  ▼
Generate confusion matrices
  │
  ▼
Select model for XAI
  │
  ├──────────────┐
  ▼              ▼
Feature        LIME
Importance      │
  │              ▼
  │            SHAP
  │              │
  └──────┬───────┘
         ▼
Generate all figures
         │
         ▼
Generate report
         │
         ▼
       END
```

---

# 17. Reproducibility Requirements

Use:

```text
random_state = 42
```

where supported.

All experiments should be reproducible.

Save:

- model metrics
- confusion matrices
- feature importance
- LIME output
- SHAP output

Do not rely only on screenshots from notebook output.

---

# 18. Report Architecture

The final report must contain exactly these major sections:

## 1. Introduction

Explain:

- tomato freshness classification problem
- motivation for automated classification
- role of machine learning
- role of explainable AI

## 2. Dataset and Preprocessing

Include:

- dataset size
- features
- target classes
- original four labels
- Stale + Spoiled → Stale to Spoiled mapping
- missing-value analysis
- train/test split
- scaling
- cross-validation

## 3. Methodology

Explain:

- Random Forest
- SVM
- XGBoost
- 5-fold stratified cross-validation
- evaluation metrics
- XAI methodology

## 4. Results and Discussion

Include:

- comparison table
- accuracy
- precision
- recall
- F1-score
- confusion matrices
- observations about misclassification

Do not fabricate numerical results. Generate them from the actual experiment.

## 5. Feature Importance

Include:

- feature importance plot
- top features
- interpretation

## 6. LIME Analysis

Include:

- selected sample
- actual class
- predicted class
- probability distribution
- LIME visualization
- interpretation

## 7. SHAP Analysis

Include:

- SHAP feature importance
- SHAP summary plot
- local explanation if generated
- interpretation

## 8. Conclusion

Summarize:

- model comparison
- important predictive features
- usefulness of XAI
- limitations
- possible future improvements

---

# 19. Time-Constrained Implementation Plan

Because the project has approximately **3 hours**, prioritize in this order:

### Phase 1 — 0 to 30 minutes

- Environment setup
- Install dependencies
- Load dataset
- Verify target mapping
- Build preprocessing pipeline

### Phase 2 — 30 to 75 minutes

- Implement Random Forest
- Implement SVM
- Implement XGBoost
- Run 5-fold CV
- Run final test evaluation

### Phase 3 — 75 to 110 minutes

- Generate comparison table
- Generate confusion matrices
- Generate class distribution
- Generate model comparison plot

### Phase 4 — 110 to 145 minutes

- Feature importance
- LIME
- SHAP
- Save all figures

### Phase 5 — 145 to 180 minutes

- Generate report
- Verify figures
- Verify metrics
- Run complete pipeline from start to finish
- Clean README/project structure

---

# 20. Dependency Requirements

Expected Python packages:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
lime
shap
jupyter
```

Optional:

```text
joblib
```

for saving trained models.

---

# 21. Important Implementation Rules

1. **Never leak the test set into training or model selection.**
2. Use stratification because this is a multiclass classification problem.
3. Use pipelines so preprocessing is performed correctly inside cross-validation.
4. Do not manually scale the entire dataset before cross-validation.
5. Do not fabricate evaluation results.
6. Use the same test set for all three final model comparisons.
7. Keep class names consistent everywhere:
   - `Pure Fresh`
   - `Good`
   - `Stale to Spoiled`
8. Use weighted precision, recall, and F1 for the overall multiclass comparison.
9. Preserve per-class metrics in the classification report.
10. Use a fixed random seed (`42`) for reproducibility.
11. Avoid excessive hyperparameter tuning because the project has a strict time limit.
12. Feature importance indicates predictive contribution, not causal influence.
13. LIME explains a local prediction; SHAP can provide both global and local explanations.
14. All visualizations should be saved to `outputs/figures/`.
15. All numerical results should be saved to `outputs/results/`.
16. The final report should reference the generated figures rather than manually recreating results.

---

# 22. Definition of Done

The project is complete when:

- [ ] Dataset loads successfully.
- [ ] Dataset inspection is included.
- [ ] Four original target labels are mapped to three required classes.
- [ ] Missing-value handling exists.
- [ ] Train/test split is stratified.
- [ ] 5-fold stratified CV is implemented.
- [ ] Three supervised classifiers are implemented.
- [ ] All three models produce predictions.
- [ ] Accuracy is calculated.
- [ ] Precision is calculated.
- [ ] Recall is calculated.
- [ ] F1-score is calculated.
- [ ] Comparison table is generated.
- [ ] Confusion matrix exists for each model.
- [ ] Class distribution visualization exists.
- [ ] Feature importance is calculated and visualized.
- [ ] LIME explanation is generated for at least one prediction.
- [ ] SHAP analysis is generated.
- [ ] SHAP visualization(s) are saved.
- [ ] Report contains all required sections.
- [ ] Complete pipeline can be rerun from scratch.
- [ ] No results are manually fabricated.

---

# 23. Recommended Implementation Philosophy

Keep the project **simple, reproducible, and demonstrable**.

The goal is not to build a production-grade MLOps platform. The goal is to demonstrate the complete machine-learning lifecycle:

```text
Data
  ↓
Preprocessing
  ↓
Cross-Validation
  ↓
Three Classifiers
  ↓
Evaluation
  ↓
Feature Importance
  ↓
LIME
  ↓
SHAP
  ↓
Interpretation
  ↓
Report
```

Claude Code should implement this architecture without introducing unnecessary frameworks, databases, APIs, frontend components, or deployment infrastructure.


---

# 24. Full-Stack Web Application Architecture

The project should include a **full-stack web UI built with Next.js** so the machine-learning pipeline can be demonstrated through an interactive frontend rather than only through a notebook.

The web application is a presentation and interaction layer around the Python ML/XAI pipeline.

## Full-Stack Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         NEXT.JS WEB APP                             │
│                                                                     │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────────┐  │
│  │ Dashboard      │  │ Prediction     │  │ Model Comparison    │  │
│  │                │  │ Playground     │  │                     │  │
│  └───────┬────────┘  └───────┬────────┘  └──────────┬──────────┘  │
│          │                   │                      │              │
│  ┌───────▼────────┐  ┌───────▼────────┐  ┌─────────▼──────────┐  │
│  │ Feature        │  │ LIME           │  │ SHAP               │  │
│  │ Importance     │  │ Explanation    │  │ Explanation        │  │
│  └───────┬────────┘  └───────┬────────┘  └─────────┬──────────┘  │
│          │                   │                      │              │
│          └───────────────────┼──────────────────────┘              │
│                              ▼                                     │
│                    Next.js API / Server Layer                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP / JSON
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PYTHON ML BACKEND                               │
│                                                                     │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────────┐ │
│  │ Prediction API │  │ Model Service  │  │ XAI Service          │ │
│  │                │  │                │  │                      │ │
│  │ /predict       │  │ RF / SVM / XGB │  │ LIME / SHAP          │ │
│  └───────┬────────┘  └───────┬────────┘  └──────────┬───────────┘ │
│          │                   │                      │             │
│          └───────────────────┼──────────────────────┘             │
│                              ▼                                    │
│                     Trained ML Artifacts                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │ Dataset / Artifacts   │
                    │                       │
                    │ CSV                   │
                    │ Models                │
                    │ Metrics               │
                    │ XAI results            │
                    └───────────────────────┘
```

---

# 25. Frontend Technology Stack

## Core

```text
Next.js
React
TypeScript
```

Use the **Next.js App Router**.

Recommended project:

```text
frontend/
```

## Styling

Use:

```text
Tailwind CSS
```

## UI Components

Use a lightweight component system such as:

```text
shadcn/ui
```

Only use components that improve the dashboard. Do not spend significant time building a large design system.

## Data Visualization

Use:

```text
Recharts
```

for:

- model comparison charts
- class distribution
- feature importance
- metric charts

For SHAP/LIME visualizations that originate from Python, the backend can either:

1. return structured JSON values for the frontend to render, or
2. return generated image/HTML artifacts.

Prefer **structured JSON** where practical because it gives the UI better interactivity.

## Icons

Use:

```text
Lucide React
```

---

# 26. Frontend Design Direction

The UI should look like a modern **AI/ML analytics dashboard**, not a generic CRUD application.

Recommended visual style:

```text
Clean
Technical
Minimal
Data-focused
Modern
Responsive
```

Use a dark-first dashboard if convenient, with high-contrast cards and charts.

Suggested visual hierarchy:

```text
┌───────────────────────────────────────────────────────────────┐
│ TomatoAI                                      Dataset Status  │
├──────────────┬────────────────────────────────────────────────┤
│              │                                                │
│ Dashboard    │  Model Performance                             │
│ Prediction   │  ┌────────┐ ┌────────┐ ┌────────┐            │
│ Models       │  │ 92.4%  │ │ 91.8%  │ │ 94.1%  │            │
│ Features     │  │   RF   │ │  SVM    │ │ XGBoost│            │
│ LIME         │  └────────┘ └────────┘ └────────┘            │
│ SHAP         │                                                │
│ Dataset      │  Charts / confusion matrices / insights       │
│              │                                                │
└──────────────┴────────────────────────────────────────────────┘
```

---

# 27. Required Frontend Pages

Keep the frontend focused. The following pages are sufficient.

## 27.1 Dashboard

Route:

```text
/
```

Purpose:

Give an overview of the entire experiment.

Display:

- Dataset size
- Number of features
- Number of freshness classes
- Best-performing model based on measured test metrics
- Accuracy / Precision / Recall / F1 cards
- Model comparison chart
- Class distribution chart
- Short methodology summary

Example:

```text
Tomato Freshness AI

555 Samples
26 Features
3 Classes
3 Models

┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Accuracy     │ │ Precision    │ │ F1 Score     │
│ 94.1%        │ │ 93.8%        │ │ 93.9%        │
└──────────────┘ └──────────────┘ └──────────────┘

Model Comparison
████████████████████ XGBoost
██████████████████   Random Forest
████████████████     SVM
```

**Important:** these numbers are placeholders for the UI design only. The actual application must load real experiment results.

---

# 28. Prediction Playground

Route:

```text
/predict
```

This is the main interactive feature.

The user should be able to enter or adjust the 26 sensor features and request a prediction.

## UI

Group features logically.

### MQ2

```text
MQ2_mean
MQ2_std
MQ2_slope
```

### MQ3

```text
MQ3_mean
MQ3_std
MQ3_slope
```

Repeat for:

- MQ9
- MQ135
- MQ136
- MQ138

### Environment

```text
Temperature_mean
Temperature_std
Temperature_slope

Humidity_mean
Humidity_std
Humidity_slope
```

### Ratios

```text
MQ3_over_MQ2_end
MQ136_over_MQ138_end
```

Provide:

```text
[Predict Freshness]
```

Result:

```text
Prediction

┌────────────────────────────────┐
│        PURE FRESH               │
│        Confidence: 94.2%        │
└────────────────────────────────┘

Pure Fresh       █████████████ 94.2%
Good             ██             4.1%
Stale to Spoiled █              1.7%
```

Then provide:

```text
[Explain with LIME]
[Explain with SHAP]
```

---

# 29. Model Comparison Page

Route:

```text
/models
```

Display all three models:

```text
Random Forest
SVM
XGBoost
```

For each model display:

- Accuracy
- Precision
- Recall
- F1-score
- CV mean score
- CV standard deviation

Use:

- comparison bar chart
- metric cards
- sortable table

Also show confusion matrices.

Example:

```text
Model Performance

              Accuracy Precision Recall F1
Random Forest    92.1     91.8    92.0  91.8
SVM              90.4     90.1    90.3  90.0
XGBoost          94.1     93.8    94.0  93.9
```

The application must populate these values from actual generated results.

---

# 30. Feature Importance Page

Route:

```text
/features
```

Display:

- top 10/15 features
- horizontal feature importance chart
- feature ranking
- model selector

Example:

```text
Feature Importance

MQ3_mean             █████████████████
MQ136_slope          █████████████
MQ2_mean             ███████████
Humidity_mean        █████████
MQ138_std            ███████
...
```

Include a short explanation:

> Feature importance indicates how strongly a feature contributes to model predictions. It should not be interpreted as proof of causal influence.

---

# 31. LIME Explanation Page

Route:

```text
/explain/lime
```

Allow the user to select:

- a test sample
- or submit a prediction from the Prediction Playground

Display:

```text
Actual Class: Good
Predicted Class: Good
Confidence: 88.4%
```

Then display local feature contributions.

Use a diverging horizontal bar chart:

```text
Features supporting prediction
──────────────────────────────
MQ3_mean             ++++++++
MQ136_slope          +++++
Temperature_mean     +++

Features opposing prediction
──────────────────────────────
Humidity_std         ---
MQ2_std              --
```

Include an explanation card:

```text
LIME explains this individual prediction by approximating
the model locally around the selected sample.
```

---

# 32. SHAP Explanation Page

Route:

```text
/explain/shap
```

Display:

### Global SHAP Importance

```text
Feature          Mean |SHAP|
MQ3_mean             ███████████
MQ136_slope          █████████
MQ2_mean             ████████
...
```

### SHAP Summary

Display a beeswarm-style visualization if the backend supplies the required SHAP values.

### Individual Explanation

Allow selection of a test sample.

Show:

```text
Base prediction
       +
Feature contributions
       =
Final prediction
```

Include a simple explanation:

> SHAP values describe how individual features move a prediction away from the model's baseline for the selected class.

---

# 33. Dataset Page

Route:

```text
/dataset
```

Display:

- number of samples
- number of features
- target classes
- class counts
- missing-value status
- feature list
- basic descriptive statistics

Include the class distribution visualization.

Do not expose unnecessary raw dataset details if they make the UI cluttered.

---

# 34. Frontend Navigation

Use a persistent sidebar.

```text
TomatoAI
────────────────────

Overview
Prediction
Models
Features
LIME
SHAP
Dataset

────────────────────
Experiment
v1.0
```

On mobile, collapse the sidebar into a navigation drawer.

---

# 35. Frontend Components

Suggested component structure:

```text
frontend/
├── app/
│   ├── page.tsx
│   ├── predict/
│   │   └── page.tsx
│   ├── models/
│   │   └── page.tsx
│   ├── features/
│   │   └── page.tsx
│   ├── explain/
│   │   ├── lime/
│   │   │   └── page.tsx
│   │   └── shap/
│   │       └── page.tsx
│   └── dataset/
│       └── page.tsx
│
├── components/
│   ├── layout/
│   │   ├── Sidebar.tsx
│   │   ├── Header.tsx
│   │   └── PageContainer.tsx
│   │
│   ├── dashboard/
│   │   ├── MetricCard.tsx
│   │   ├── ModelComparison.tsx
│   │   └── ClassDistribution.tsx
│   │
│   ├── prediction/
│   │   ├── FeatureInputForm.tsx
│   │   ├── PredictionResult.tsx
│   │   └── ProbabilityChart.tsx
│   │
│   ├── models/
│   │   ├── ModelTable.tsx
│   │   └── ConfusionMatrix.tsx
│   │
│   ├── xai/
│   │   ├── LimeExplanation.tsx
│   │   ├── ShapSummary.tsx
│   │   └── ShapLocalExplanation.tsx
│   │
│   └── charts/
│       ├── MetricChart.tsx
│       ├── FeatureImportanceChart.tsx
│       └── ClassDistributionChart.tsx
│
├── lib/
│   ├── api.ts
│   ├── types.ts
│   └── utils.ts
│
└── public/
```

---

# 36. Backend API Architecture

The frontend needs a small Python API service.

Recommended backend:

```text
FastAPI
```

Project:

```text
backend/
```

Architecture:

```text
Next.js
    │
    │ HTTP/JSON
    ▼
FastAPI
    │
    ├── /health
    ├── /dataset
    ├── /models
    ├── /predict
    ├── /feature-importance
    ├── /explain/lime
    └── /explain/shap
    │
    ▼
ML Pipeline
    │
    ├── preprocessing
    ├── Random Forest
    ├── SVM
    └── XGBoost
```

---

# 37. API Endpoints

## GET `/health`

Returns:

```json
{
  "status": "ok"
}
```

## GET `/dataset`

Returns:

```json
{
  "samples": 555,
  "features": 26,
  "classes": [
    "Pure Fresh",
    "Good",
    "Stale to Spoiled"
  ]
}
```

## GET `/models`

Returns model metrics:

```json
{
  "models": [
    {
      "name": "Random Forest",
      "accuracy": 0.92,
      "precision": 0.91,
      "recall": 0.92,
      "f1": 0.91
    }
  ]
}
```

## POST `/predict`

Request:

```json
{
  "features": {
    "MQ2_mean": 1.2,
    "MQ2_std": 0.3
  }
}
```

Response:

```json
{
  "predicted_class": "Pure Fresh",
  "probabilities": {
    "Pure Fresh": 0.94,
    "Good": 0.04,
    "Stale to Spoiled": 0.02
  },
  "model": "XGBoost"
}
```

## GET `/feature-importance`

Returns ranked features:

```json
{
  "features": [
    {
      "name": "MQ3_mean",
      "importance": 0.18
    }
  ]
}
```

## POST `/explain/lime`

Request:

```json
{
  "features": {
    "...": "..."
  }
}
```

Response should contain structured local contributions:

```json
{
  "prediction": "Good",
  "contributions": [
    {
      "feature": "MQ3_mean",
      "value": 1.23,
      "contribution": 0.31
    }
  ]
}
```

## POST `/explain/shap`

Return:

- prediction
- class probabilities
- SHAP values
- feature names
- base values

This allows the frontend to render the visualization itself.

---

# 38. Model Artifact Strategy

Do not retrain the models every time a user opens the dashboard.

The training pipeline should generate saved artifacts:

```text
artifacts/
├── random_forest.joblib
├── svm.joblib
├── xgboost.joblib
├── preprocessor.joblib
├── metrics.json
├── feature_importance.json
├── confusion_matrices.json
├── shap_values.json
└── dataset_summary.json
```

The FastAPI service loads these artifacts at startup.

For the 3-hour project, this is much simpler and faster than building a database.

---

# 39. Data Flow for an Interactive Prediction

```text
User enters 26 features
          │
          ▼
Next.js Prediction Form
          │
          ▼
POST /predict
          │
          ▼
FastAPI
          │
          ▼
Preprocessing Pipeline
          │
          ▼
Selected Trained Model
          │
          ▼
Prediction + Probabilities
          │
          ▼
JSON Response
          │
          ▼
Next.js
          │
          ▼
Prediction Result Card
```

For explanations:

```text
Prediction
    │
    ├──────────────► LIME endpoint
    │                     │
    │                     ▼
    │               Local explanation
    │
    └──────────────► SHAP endpoint
                          │
                          ▼
                    SHAP values
                          │
                          ▼
                    Interactive UI
```

---

# 40. Backend/Frontend Separation

Keep responsibilities clearly separated.

## Python Backend

Responsible for:

- ML models
- preprocessing
- predictions
- evaluation results
- feature importance
- LIME
- SHAP
- dataset statistics

## Next.js Frontend

Responsible for:

- user interface
- routing
- forms
- charts
- displaying predictions
- displaying explanations
- loading/error states
- responsive design

Do not implement ML logic in TypeScript.

---

# 41. Full Project Structure

The final project can use:

```text
tomato-freshness-ml/
│
├── data/
│   └── window_statistical_features.csv
│
├── ml/
│   ├── src/
│   │   ├── data_loader.py
│   │   ├── preprocessing.py
│   │   ├── models.py
│   │   ├── evaluation.py
│   │   ├── feature_importance.py
│   │   ├── lime_explainer.py
│   │   ├── shap_explainer.py
│   │   └── visualization.py
│   │
│   ├── notebooks/
│   │   └── tomato_freshness_analysis.ipynb
│   │
│   ├── outputs/
│   │   ├── figures/
│   │   └── results/
│   │
│   └── artifacts/
│
├── backend/
│   ├── main.py
│   ├── api/
│   │   ├── predict.py
│   │   ├── models.py
│   │   ├── explanations.py
│   │   └── dataset.py
│   ├── services/
│   │   ├── model_service.py
│   │   ├── lime_service.py
│   │   └── shap_service.py
│   └── requirements.txt
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── lib/
│   ├── public/
│   ├── package.json
│   └── next.config.ts
│
├── report/
│   └── report.md
│
├── README.md
└── architecture.md
```

---

# 42. Recommended Full-Stack Development Order

Because only approximately **3 hours** are available, build in this order.

## Phase 1 — ML Core

```text
Dataset
  ↓
Preprocessing
  ↓
Three Models
  ↓
Evaluation
  ↓
Saved Artifacts
```

Do not start the frontend before the prediction/evaluation pipeline works.

## Phase 2 — XAI

```text
Feature Importance
  ↓
LIME
  ↓
SHAP
```

Verify that actual explanation values are being generated.

## Phase 3 — FastAPI

Implement only:

```text
GET  /health
GET  /dataset
GET  /models
POST /predict
GET  /feature-importance
POST /explain/lime
POST /explain/shap
```

## Phase 4 — Next.js

Build:

1. Dashboard
2. Prediction Playground
3. Models
4. Feature Importance
5. LIME
6. SHAP
7. Dataset

Do not spend time implementing authentication, databases, user accounts, or deployment infrastructure.

## Phase 5 — Polish

Add:

- loading states
- error states
- responsive layout
- empty states
- hover effects
- subtle animations
- consistent typography
- clean charts

---

# 43. Frontend UX Requirements

The frontend should feel like a finished ML product.

## Loading

While waiting for the backend:

```text
Analyzing tomato...
Generating explanation...
```

Use skeletons/spinners rather than freezing the UI.

## Error Handling

If the backend is unavailable:

```text
Unable to connect to the ML service.
Please make sure the backend is running.
```

Do not expose raw Python stack traces to users.

## Prediction Result

Always show:

- predicted class
- confidence
- all class probabilities
- model used

## Explanation Result

Always identify:

- what is being explained
- which model generated the explanation
- whether the explanation is local or global

---

# 44. Frontend Visualization Requirements

Minimum frontend visualizations:

### Dashboard

- Class distribution
- Model metric comparison

### Models

- Metric comparison
- Three confusion matrices

### Features

- Feature importance chart

### LIME

- Local feature contribution chart

### SHAP

- Global SHAP importance
- SHAP summary
- Optional local SHAP waterfall/bar chart

Avoid chart overload. Every visualization should answer a clear question.

---

# 45. Full-Stack Definition of Done

In addition to the ML checklist above:

- [ ] Next.js frontend is created.
- [ ] FastAPI backend is created.
- [ ] Frontend successfully communicates with backend.
- [ ] Dashboard displays real experiment metrics.
- [ ] Dataset page displays real dataset statistics.
- [ ] Prediction page accepts all 26 features.
- [ ] Prediction endpoint returns real model predictions.
- [ ] Prediction probabilities are displayed.
- [ ] Model comparison page displays all three models.
- [ ] Confusion matrices are displayed.
- [ ] Feature importance page works.
- [ ] LIME explanation is accessible from the UI.
- [ ] SHAP explanation is accessible from the UI.
- [ ] Loading states work.
- [ ] Backend errors are handled gracefully.
- [ ] UI is responsive.
- [ ] No fake metrics or hard-coded prediction results are used.
- [ ] ML logic remains in Python.
- [ ] The application can be started locally with documented commands.

---

# 46. Final Full-Stack Architecture

The completed project should demonstrate the following end-to-end system:

```text
                         USER
                           │
                           ▼
                 ┌───────────────────┐
                 │   NEXT.JS UI      │
                 │                   │
                 │ Dashboard         │
                 │ Prediction        │
                 │ Models            │
                 │ Features          │
                 │ LIME              │
                 │ SHAP              │
                 └─────────┬─────────┘
                           │
                      HTTP / JSON
                           │
                           ▼
                 ┌───────────────────┐
                 │     FASTAPI       │
                 │    ML Backend     │
                 └─────────┬─────────┘
                           │
          ┌────────────────┼─────────────────┐
          │                │                 │
          ▼                ▼                 ▼
    ┌──────────┐    ┌─────────────┐   ┌─────────────┐
    │Prediction│    │ Evaluation  │   │ XAI Engine  │
    │ Service  │    │ Service     │   │             │
    └────┬─────┘    └──────┬──────┘   │ LIME / SHAP │
         │                 │           └──────┬──────┘
         └─────────────────┼──────────────────┘
                           ▼
                 ┌───────────────────┐
                 │ Trained Models    │
                 │                   │
                 │ Random Forest     │
                 │ SVM               │
                 │ XGBoost           │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Dataset + Model  │
                 │ Artifacts         │
                 └───────────────────┘
```

The final product therefore has two connected layers:

```text
ML / XAI ENGINE
      +
FULL-STACK INTERACTIVE UI
```

The UI is a demonstration and interaction layer over the actual trained models; it must never replace or duplicate the Python ML pipeline.
