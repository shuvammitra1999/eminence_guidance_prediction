"""
Evaluation metrics, comparison table, and clinical plots (lightweight).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import RESULTS_DIR, RANDOM_STATE


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Max_Error": np.max(np.abs(y_true - y_pred)),
    }


def build_comparison_table(all_results: dict) -> pd.DataFrame:
    rows = []
    for fs_name, models in all_results.items():
        for model_name, info in models.items():
            m = info["metrics"]
            rows.append({
                "Feature Set": fs_name,
                "Model": model_name,
                "CV MAE": abs(info["cv_mae"]),
                "Test MAE": m["MAE"],
                "Test RMSE": m["RMSE"],
                "Test R2": m["R2"],
                "Max Error": m["Max_Error"],
            })
    return pd.DataFrame(rows).sort_values("Test MAE")


def plot_predicted_vs_actual(y_true, y_pred, model_name, feature_set, save_dir=RESULTS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", linewidth=0.5)
    lims = [min(min(y_true), min(y_pred)) - 2, max(max(y_true), max(y_pred)) + 2]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual Mean AEI (deg)")
    ax.set_ylabel("Predicted Mean AEI (deg)")
    ax.set_title(f"{model_name} ({feature_set})")
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    fig.tight_layout()
    fname = os.path.join(save_dir, f"pred_vs_actual_{model_name}_{feature_set}.png")
    fig.savefig(fname, dpi=100)
    plt.close(fig)
    return fname


def plot_bland_altman(y_true, y_pred, model_name, feature_set, save_dir=RESULTS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mean_vals = (y_true + y_pred) / 2
    diff = y_pred - y_true
    mean_diff = diff.mean()
    std_diff = diff.std()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(mean_vals, diff, alpha=0.7, edgecolors="k", linewidth=0.5)
    ax.axhline(mean_diff, color="blue", linestyle="-", label=f"Mean bias: {mean_diff:.2f}")
    ax.axhline(mean_diff + 1.96 * std_diff, color="red", linestyle="--",
               label=f"+1.96 SD: {mean_diff + 1.96 * std_diff:.2f}")
    ax.axhline(mean_diff - 1.96 * std_diff, color="red", linestyle="--",
               label=f"-1.96 SD: {mean_diff - 1.96 * std_diff:.2f}")
    ax.set_xlabel("Mean of Actual and Predicted (deg)")
    ax.set_ylabel("Predicted - Actual (deg)")
    ax.set_title(f"Bland-Altman: {model_name} ({feature_set})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fname = os.path.join(save_dir, f"bland_altman_{model_name}_{feature_set}.png")
    fig.savefig(fname, dpi=100)
    plt.close(fig)
    return fname


def plot_feature_importance(model, feature_names, model_name, feature_set, save_dir=RESULTS_DIR):
    """Feature importance: coefficients for linear models, built-in for GB."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))

    if hasattr(model, "coef_"):
        importances = model.coef_
        ax.barh(feature_names, importances, color="steelblue")
        ax.set_xlabel("Standardized Coefficient")
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = importances.argsort()
        ax.barh([feature_names[i] for i in sorted_idx],
                importances[sorted_idx], color="steelblue")
        ax.set_xlabel("Feature Importance (impurity-based)")
    else:
        plt.close(fig)
        return None

    ax.set_title(f"Feature Importance: {model_name} ({feature_set})")
    fig.tight_layout()
    fname = os.path.join(save_dir, f"feat_importance_{model_name}_{feature_set}.png")
    fig.savefig(fname, dpi=100)
    plt.close(fig)
    return fname
