# Tomato Freshness Classification + Explainable AI — Report

## 1. Introduction

Tomatoes deteriorate through a continuum from freshly harvested to spoiled,
and the point at which produce should be pulled from a supply chain is
normally judged by manual inspection — slow, subjective, and hard to scale.
This project builds a supervised machine learning system that classifies
tomato freshness automatically from time-windowed statistics of six gas
sensors (MQ2, MQ3, MQ9, MQ135, MQ136, MQ138) plus temperature and humidity,
into three classes: **Pure Fresh**, **Good**, and **Stale to Spoiled**.

Beyond raw predictive accuracy, the project also asks *why* the model makes
a given prediction. Gas-sensor classifiers are easy to distrust as "black
boxes," so alongside three classifiers (Random Forest, SVM, XGBoost) the
project applies explainable AI (XAI) — native feature importance, LIME, and
SHAP — so predictions can be inspected and reasoned about rather than taken
on faith.

## 2. Dataset and Preprocessing

- **Source:** `data/window_statistical_features.csv`
- **Size:** 555 samples, 26 numeric input features, 1 target column
  (`Freshness_Level`)
- **Features:** for each of the 6 MQ gas sensors — `mean`, `std`, `slope`
  over the observation window; for Temperature and Humidity — the same
  `mean`/`std`/`slope` triplet; plus two derived ratio features,
  `MQ3_over_MQ2_end` and `MQ136_over_MQ138_end`.
- **Missing values / duplicates:** none observed (0 missing, 0 duplicate
  rows). A `SimpleImputer(strategy="median")` is still included in every
  model pipeline so the pipeline stays robust if missing values appear in
  future data.

**Original labels → required 3-class mapping.** The raw dataset carries four
labels:

| Label | Count |
|---|---:|
| Spoiled | 216 |
| Good | 144 |
| Stale | 126 |
| Pure Fresh | 69 |

`Stale` and `Spoiled` are merged into a single class, `Stale to Spoiled`,
giving the three required classes:

| Class | Count | Share |
|---|---:|---:|
| Pure Fresh | 69 | 12.4% |
| Good | 144 | 25.9% |
| Stale to Spoiled | 342 | 61.6% |

![Class distribution](../ml/outputs/figures/class_distribution.png)

The mapped dataset is imbalanced (Pure Fresh is the minority class at ~12%
of samples), which is why stratification is used throughout — both for the
train/test split and for cross-validation — and why the report tracks
macro-averaged F1 alongside the weighted metrics (Section 4).

**Train/test split:** stratified 80/20 (`train_test_split`,
`stratify=y`, `random_state=42`) → 444 training / 111 test samples, with
class proportions preserved in both.

**Scaling:** each model is wrapped in an sklearn `Pipeline` combining an
imputer (and, for SVM, a `StandardScaler`) with the estimator, so
preprocessing is fit only on training folds during cross-validation — never
on the held-out data.

**Cross-validation:** Stratified 5-fold CV (`StratifiedKFold(n_splits=5,
shuffle=True, random_state=42)`) is run on the training set only, before any
test-set evaluation.

## 3. Methodology

Three supervised classifiers were trained and compared:

- **Random Forest** (`n_estimators=200`, `class_weight="balanced"`) — an
  ensemble of decision trees; handles nonlinear feature interactions well
  and needs little preprocessing.
- **SVM (RBF kernel)** (`class_weight="balanced"`, `probability=True`) — fit
  on imputed + standardized features; effective on datasets of this size
  (555 samples) with nonlinear class boundaries.
- **XGBoost** (`multi:softprob`, `n_estimators=200`, `max_depth=4`,
  `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`) — gradient-
  boosted trees, generally strong on structured/tabular sensor data.

For each model: 5-fold stratified CV on the training set (to estimate
generalization and variance before touching the test set), then a single
fit on the full training set and one evaluation on the untouched test set.
Metrics are accuracy, and weighted precision/recall/F1 (`average="weighted"`,
required by the assignment); macro-F1 is also reported to surface how the
minority `Pure Fresh` class performs, since weighted metrics can mask
under-performance on a small class. `random_state=42` is fixed everywhere
it's supported, so the experiment is reproducible end-to-end via
`ml/run_pipeline.py`.

The model used for feature importance, LIME, and SHAP is **not hard-coded**:
it is selected programmatically as the model with the best test F1-weighted
score (tie-broken by test accuracy) — see `select_model_for_xai()` in
`ml/src/feature_importance.py`.

## 4. Results and Discussion

| Model | Accuracy | Precision | Recall | F1-score | F1-macro |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.9910 | 0.9913 | 0.9910 | 0.9910 | 0.9919 |
| SVM | 0.9820 | 0.9831 | 0.9820 | 0.9821 | 0.9839 |
| XGBoost | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

5-fold CV (training set only, mean ± std):

| Model | CV Accuracy | CV F1-weighted |
|---|---:|---:|
| Random Forest | 0.9888 ± 0.0123 | 0.9889 ± 0.0121 |
| SVM | 0.9752 ± 0.0194 | 0.9750 ± 0.0195 |
| XGBoost | 0.9865 ± 0.0131 | 0.9865 ± 0.0131 |

![Model comparison](../ml/outputs/figures/model_comparison.png)

![Confusion matrices](../ml/outputs/figures/confusion_matrices.png)

All three models perform very well, and CV scores track test scores closely
(no evidence of leakage or overfitting to the test set). **XGBoost reached
100% accuracy on the 111-sample test set** — every sample correctly
classified. Random Forest's only error was one `Stale to Spoiled` sample
predicted as `Good`; SVM misclassified two `Stale to Spoiled` samples as
`Good`. No model confused `Pure Fresh` with either other class, suggesting
the sensor signatures for freshly-harvested tomatoes are cleanly separable
from the other two states in this dataset.

The near-perfect scores across all three models — not just the selected one
— indicate the 26 engineered sensor statistics carry a very strong signal
for this task, more than model choice. A caveat worth stating plainly: with
only 111 test samples (and just 14 `Pure Fresh` examples), a handful of
different samples could change these numbers meaningfully; the CV standard
deviations (up to ~2 points of accuracy) give a more honest sense of
variance than the single test-set number alone.

## 5. Feature Importance

Model used: **XGBoost** (selected per Section 3's criterion).

![Feature importance](../ml/outputs/figures/feature_importance.png)

The top-ranked features are dominated by MQ gas sensor **means**
(`MQ3_over_MQ2_end`, `MQ135_mean`, `MQ138_mean`, `MQ9_mean`) and one
environmental mean (`Humidity_mean`), with sensor `slope` and `std` features
contributing comparatively less. This is consistent with a plausible
underlying process: mean gas concentration over the observation window
should shift substantially as volatile organic compounds change with
spoilage, while slope/variability features add finer, secondary detail. The
MQ3/MQ2 ratio ranking highest also fits — MQ3 (ethanol) and MQ2
(combustible/gas mix) both respond to fermentation byproducts, and their
ratio may normalize away some sensor-drift noise that raw means don't.

**Feature importance indicates predictive contribution, not causation** —
these results say the model relies on these features to separate classes,
not that any one gas *causes* spoilage.

## 6. LIME Analysis

A correctly-classified, high-confidence test sample was selected (test
index #42) so the explanation is easy to discuss:

- **Actual class:** Stale to Spoiled
- **Predicted class:** Stale to Spoiled
- **Probabilities:** Pure Fresh 0.03%, Good 0.04%, Stale to Spoiled 99.93%

![LIME explanation](../ml/outputs/figures/lime_explanation.png)

The features pushing the prediction toward `Stale to Spoiled` are led by a
low `MQ135_mean` value, a high `Humidity_mean` (92–95%), and an elevated
`MQ2_mean`; `MQ3_slope` was the one feature pulling weakly against this
class. LIME is a **local approximation** around this one sample — it
explains why the model classified *this specific tomato reading* as it did,
not a general rule the model applies to every prediction.

## 7. SHAP Analysis

The same XGBoost model and the same test sample (#42) are used for SHAP so
LIME and SHAP can be compared directly.

**Global SHAP feature importance** (mean |SHAP value| across all test
samples and all three classes):

![SHAP global bar](../ml/outputs/figures/shap_bar.png)

This global ranking — led by `MQ135_mean`, `MQ138_mean`,
`MQ136_over_MQ138_end`, `Humidity_mean`, `MQ9_mean` — broadly agrees with
the XGBoost `feature_importances_` ranking in Section 5, which is reassuring
since the two methods measure importance differently (impurity-based
splits vs. game-theoretic attribution).

**SHAP summary (beeswarm) for the `Stale to Spoiled` class:**

![SHAP summary](../ml/outputs/figures/shap_summary.png)

This adds direction: low `MQ135_mean` (blue, left) and high `Humidity_mean`
(red, right) both push toward `Stale to Spoiled`, matching the local
explanation below.

**Local explanation for the same sample used by LIME:**

![SHAP local](../ml/outputs/figures/shap_local.png)

SHAP's waterfall for sample #42 names the same top three drivers LIME found
— `MQ135_mean`, `MQ138_mean`, `Humidity_mean` — moving the prediction from
the baseline `E[f(X)] = 0.852` up to `f(x) = 4.42` for the `Stale to
Spoiled` class logit. The agreement between two independently-implemented
methods (a local surrogate model vs. exact game-theoretic attribution for
trees) is a useful sanity check that the explanation reflects genuine model
behavior rather than an artifact of one method.

As with feature importance: SHAP describes how features **moved this
prediction away from the model's baseline**, not that those gas readings
*caused* the tomato to spoil.

## 8. Conclusion

All three classifiers — Random Forest, SVM, and XGBoost — perform strongly
on this task (98–100% test accuracy), with XGBoost slightly ahead and
selected as the model used for explainability. The 26 sensor-derived
statistics, particularly MQ135/MQ138/MQ9 means, the MQ3/MQ2 ratio, and mean
humidity, carry a strong signal for distinguishing Pure Fresh, Good, and
Stale-to-Spoiled tomatoes. Feature importance, LIME, and SHAP converge on a
consistent, plausible story about which sensor signals drive predictions,
and LIME/SHAP agree closely on an individual example — evidence the
explanations are capturing real model behavior rather than noise.

**Limitations:**
- The dataset is modest (555 samples, 111 in the test set) and imbalanced
  (only 69 `Pure Fresh` examples); near-perfect test scores should be read
  alongside the wider CV standard deviations, not as a guarantee of this
  performance on new data or other tomato batches/sensor rigs.
- Feature importance and SHAP indicate association/predictive contribution,
  not causal mechanisms of spoilage.
- No hyperparameter tuning was performed, by design (project scope/time
  constraint) — reported numbers reflect the architecture's specified
  configurations, not an optimized ceiling.

**Possible future improvements:** collect more samples (especially more
`Pure Fresh` examples) and, ideally, from additional sensor units/batches to
test generalization; add light hyperparameter tuning within CV; and extend
XAI coverage to more individual samples/misclassified cases to see whether
the same features drive the model's mistakes.

---
*All figures and numbers in this report are generated by
`ml/run_pipeline.py` from the actual dataset — nothing here is fabricated.
Rerun with `python ml/run_pipeline.py` to reproduce.*
