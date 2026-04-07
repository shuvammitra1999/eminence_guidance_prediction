"""Configuration constants, feature lists, and hyperparameter grids."""

import os

RANDOM_STATE = 42

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DATA_PATH = os.path.join(PROJECT_ROOT, "Recorded condylar guidance.xlsx")
SYNTHETIC_DATA_PATH = os.path.join(PROJECT_ROOT, "synthetic_81.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Target variable
TARGET_COL = "Mean_AEI"

# Feature sets
FEATURE_SETS = {
    "base_anb": [
        "Age", "Sex", "ANB", "SN_GoGn", "Occ_Plane",
        "Ramus_Ht", "Cond_Ht", "Bigonial",
    ],
    "base_snb": [
        "Age", "Sex", "SNB", "SN_GoGn", "Occ_Plane",
        "Ramus_Ht", "Cond_Ht", "Bigonial",
    ],
    "full_regularized": [
        "Age", "Sex", "SNA", "SNB", "ANB", "SN_GoGn", "Occ_Plane",
        "Ramus_Ht", "Cond_Ht", "Bigonial",
    ],
}

# Train/test split
TEST_SIZE = 0.20

# Hyperparameter grids (kept small to avoid heavy compute)
PARAM_GRIDS = {
    "Ridge": {
        "alpha": [0.1, 1.0, 10.0],
    },
    "ElasticNet": {
        "alpha": [0.1, 1.0],
        "l1_ratio": [0.3, 0.7],
    },
    "GradientBoosting": {
        "n_estimators": [50, 100],
        "max_depth": [2, 3],
        "min_samples_leaf": [10],
        "learning_rate": [0.1],
    },
    "SVR": {
        "C": [1.0, 10.0],
        "epsilon": [0.1, 0.5],
        "gamma": ["scale"],
    },
}

# Cross-validation folds for hyperparameter tuning
CV_FOLDS = 3
