"""Target mapping, train/test split, and preprocessing pipelines."""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_loader import FEATURE_COLUMNS, TARGET_COLUMN

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

# Required 3-class mapping (Stale + Spoiled merge into one class).
TARGET_MAPPING = {
    "Pure Fresh": "Pure Fresh",
    "Good": "Good",
    "Stale": "Stale to Spoiled",
    "Spoiled": "Stale to Spoiled",
}

CLASS_ORDER = ["Pure Fresh", "Good", "Stale to Spoiled"]


def map_target(df: pd.DataFrame) -> pd.DataFrame:
    """Merge Stale/Spoiled into 'Stale to Spoiled'. Prints distribution before/after."""
    print("\nClass distribution BEFORE mapping:")
    print(df[TARGET_COLUMN].value_counts())

    unmapped = set(df[TARGET_COLUMN].unique()) - set(TARGET_MAPPING)
    if unmapped:
        raise ValueError(f"Unexpected target values found: {sorted(unmapped)}")

    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(TARGET_MAPPING)

    print("\nClass distribution AFTER mapping (Stale + Spoiled -> Stale to Spoiled):")
    print(df[TARGET_COLUMN].value_counts().reindex(CLASS_ORDER))

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate X (26 numeric features) and y (3-class target). Target is excluded from X."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def make_train_test_split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train/test split with a fixed random seed."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def make_preprocessor(scale: bool) -> Pipeline:
    """Build an Imputer(-> Scaler) pipeline.

    scale=True:  Imputer -> StandardScaler   (for scale-sensitive models, e.g. SVM)
    scale=False: Imputer only                (for tree-based models, e.g. RF/XGBoost)
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def make_cv() -> StratifiedKFold:
    """Stratified 5-fold CV, shuffled, fixed random seed."""
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
