"""
Load real patient data (from original xlsx), merge with synthetic data,
and provide train/test split.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import REAL_DATA_PATH, SYNTHETIC_DATA_PATH, TARGET_COL, TEST_SIZE, RANDOM_STATE


# Columns shared between real and synthetic data (used for merging)
MERGE_COLS = [
    "Sex", "SN_GoGn", "Occ_Plane", "Ramus_Ht", "Cond_Ht",
    "Bigonial", "ANB", "Right_AEI", "Left_AEI", "Mean_AEI",
]


def load_real_data(path: str = REAL_DATA_PATH) -> pd.DataFrame:
    """
    Load the 19 real patients from the original xlsx file.

    Fixes applied:
    - ANB recalculated as SNA - SNB for all patients
    - Mean_AEI computed from original Right/Left values
    - Skeletal Class and Patient Name dropped
    """
    df = pd.read_excel(path)

    out = pd.DataFrame()
    out["Age"] = df["Age"]
    out["Sex"] = df["Sex"]
    out["SNA"] = df["SNA"]
    out["SNB"] = df["SNB"]
    out["ANB"] = (df["SNA"] - df["SNB"]).round(1)  # Recalculated
    out["SN_GoGn"] = df["GoGn-SN"]
    out["Occ_Plane"] = df["Occlusal Plane angle"]
    out["Ramus_Ht"] = df["Ramus Height "]
    out["Cond_Ht"] = df["Condylar height"]
    out["Bigonial"] = df["Bigonial width(mm)"]
    out["Right_AEI"] = df["(Right)Articular eminence inclination"]
    out["Left_AEI"] = df["(Left) Articular eminence inclination "]
    out["Mean_AEI"] = ((out["Right_AEI"] + out["Left_AEI"]) / 2).round(1)
    out["Source"] = "real"

    return out


def load_synthetic_data(path: str = SYNTHETIC_DATA_PATH) -> pd.DataFrame:
    """Load the 81 synthetic patients."""
    df = pd.read_csv(path)
    df = df[MERGE_COLS].copy()
    df["Source"] = "synthetic"
    return df


def load_merged_dataset() -> pd.DataFrame:
    """Load and merge real + synthetic into a single 100-patient dataset."""
    df_real = load_real_data()
    df_synth = load_synthetic_data()
    # Only keep columns present in both datasets for the merge
    df_real_trimmed = df_real[MERGE_COLS + ["Source"]].copy()
    df = pd.concat([df_real_trimmed, df_synth], ignore_index=True)
    return df


def get_train_test_split(df: pd.DataFrame = None):
    """
    Return X_train, X_test, y_train, y_test from the merged 100-patient dataset.

    Uses stratified split on Sex to maintain sex ratio in both sets.
    The 'Source' column is dropped before splitting.
    """
    if df is None:
        df = load_merged_dataset()

    feature_cols = [c for c in MERGE_COLS if c not in [TARGET_COL, "Right_AEI", "Left_AEI"]]
    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["Sex"],
    )

    return X_train, X_test, y_train, y_test


def print_data_audit(df: pd.DataFrame) -> None:
    """Print a data quality audit report."""
    print(f"Total patients: {len(df)}")
    print(f"  Real: {(df['Source'] == 'real').sum()}")
    print(f"  Synthetic: {(df['Source'] == 'synthetic').sum()}")
    print(f"\nSex distribution: {df['Sex'].value_counts().to_dict()} (0=F, 1=M)")

    print(f"\nDescriptive statistics:")
    cols = [c for c in MERGE_COLS if c != "Right_AEI" and c != "Left_AEI"]
    print(df[cols].describe().round(2).to_string())


if __name__ == "__main__":
    df = load_merged_dataset()
    print_data_audit(df)

    X_train, X_test, y_train, y_test = get_train_test_split(df)
    print(f"\nTrain/test split: {len(X_train)} train, {len(X_test)} test")
    print(f"Train sex dist: {X_train['Sex'].value_counts().to_dict()}")
    print(f"Test sex dist:  {X_test['Sex'].value_counts().to_dict()}")
