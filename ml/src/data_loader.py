"""Load and inspect the tomato freshness sensor dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET_COLUMN = "Freshness_Level"

FEATURE_COLUMNS = [
    "MQ2_mean", "MQ2_std", "MQ2_slope",
    "MQ3_mean", "MQ3_std", "MQ3_slope",
    "MQ9_mean", "MQ9_std", "MQ9_slope",
    "MQ135_mean", "MQ135_std", "MQ135_slope",
    "MQ136_mean", "MQ136_std", "MQ136_slope",
    "MQ138_mean", "MQ138_std", "MQ138_slope",
    "Temperature_mean", "Temperature_std", "Temperature_slope",
    "Humidity_mean", "Humidity_std", "Humidity_slope",
    "MQ3_over_MQ2_end", "MQ136_over_MQ138_end",
]

EXPECTED_COLUMNS = [TARGET_COLUMN, *FEATURE_COLUMNS]


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the CSV and validate that all expected columns are present."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    """Print dataset shape, dtypes, missing values, duplicates and class distribution."""
    print("=" * 70)
    print("DATASET INSPECTION")
    print("=" * 70)

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\nColumn dtypes:")
    print(df.dtypes)

    missing_counts = df.isna().sum()
    total_missing = int(missing_counts.sum())
    print(f"\nMissing values (total = {total_missing}):")
    if total_missing:
        print(missing_counts[missing_counts > 0])
    else:
        print("None observed.")

    n_duplicates = int(df.duplicated().sum())
    print(f"\nDuplicate rows: {n_duplicates}")

    print(f"\nClass distribution ({TARGET_COLUMN}, raw):")
    print(df[TARGET_COLUMN].value_counts())

    print("\nDescriptive statistics (numeric features):")
    print(df[FEATURE_COLUMNS].describe().T)
    print("=" * 70)
