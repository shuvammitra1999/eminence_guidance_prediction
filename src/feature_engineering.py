"""
Feature set definitions and StandardScaler pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import FEATURE_SETS, TARGET_COL


def get_feature_matrix(X: pd.DataFrame, feature_set_name: str) -> pd.DataFrame:
    """Select columns for the given feature set."""
    cols = FEATURE_SETS[feature_set_name]
    return X[cols]


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler on training data only."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame, feature_set_name: str):
    """
    Select features for the given set, fit scaler on train, transform both.

    Returns:
        X_train_scaled (np.ndarray), X_test_scaled (np.ndarray), scaler, feature_names
    """
    cols = FEATURE_SETS[feature_set_name]
    X_tr = X_train[cols]
    X_te = X_test[cols]

    scaler = fit_scaler(X_tr)
    X_tr_scaled = scaler.transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    return X_tr_scaled, X_te_scaled, scaler, cols
