"""
Eminence Guidance Prediction Model
===================================

Predicts Mean Articular Eminence Inclination (degrees) from routine
cephalometric and panoramic radiographic measurements. Intended to support
individualized sagittal condylar guidance programming in prosthodontics.

Model:   Ridge Regression (alpha=10)
Features: Sex, SN-GoGn, Occlusal Plane Angle, Ramus Height,
          Condylar Height, Bigonial Width, ANB
Target:  Mean Articular Eminence Inclination (degrees)

Performance (80/20 train-test split):
    MAE  = 3.15 degrees
    RMSE = 3.96 degrees
    R²   = 0.33

Usage:
    python eminence_model.py              # Train, evaluate, save model, generate plots
    python eminence_model.py --predict    # Interactive prediction mode
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================================
# Configuration
# ============================================================================

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_model.pkl")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

FEATURE_COLS = ["Sex", "SN_GoGn", "Occ_Plane", "Ramus_Ht", "Cond_Ht", "Bigonial", "ANB"]
TARGET_COL = "Mean_AEI"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Feature descriptions (for display)
FEATURE_INFO = {
    "Sex":       ("Sex", "0 = Female, 1 = Male"),
    "SN_GoGn":   ("SN-GoGn", "degrees"),
    "Occ_Plane": ("Occlusal Plane Angle", "degrees"),
    "Ramus_Ht":  ("Ramus Height", "mm"),
    "Cond_Ht":   ("Condylar Height", "mm"),
    "Bigonial":  ("Bigonial Width", "mm"),
    "ANB":       ("ANB Angle", "degrees"),
}

# Plot styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Color palette
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "success": "#2ECC71",
    "danger": "#E74C3C",
    "dark": "#2C3E50",
    "light_bg": "#F8F9FA",
    "ridge": "#2E86AB",
    "elasticnet": "#A23B72",
    "gb": "#F18F01",
    "svr": "#2ECC71",
}


# ============================================================================
# 1. Data Loading
# ============================================================================

def load_dataset():
    """Load the 100-patient dataset from CSV."""
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(df)} patients from {DATASET_PATH}")
    print(f"  Features: {', '.join(FEATURE_COLS)}")
    print(f"  Target: {TARGET_COL}")
    return df


# ============================================================================
# 2. Model Training
# ============================================================================

def train_model(df):
    """
    Train the Ridge Regression model on an 80/20 split.

    Also trains 3 comparison models (ElasticNet, GradientBoosting, SVR)
    for the model comparison plot.

    Returns:
        model: trained Ridge model
        scaler: fitted StandardScaler
        X_train, X_test, y_train, y_test: split data
        comparison: dict of model_name -> (test_mae, test_r2, cv_mae)
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Stratified split on Sex to maintain sex ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["Sex"]
    )

    # Standardize features (fit on training data only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- Primary model: Ridge Regression ---
    ridge_gs = GridSearchCV(
        Ridge(random_state=RANDOM_STATE),
        {"alpha": [0.1, 1.0, 10.0]},
        cv=3, scoring="neg_mean_absolute_error", n_jobs=1,
    )
    ridge_gs.fit(X_train_s, y_train)
    ridge = ridge_gs.best_estimator_
    ridge_cv_mae = -ridge_gs.best_score_

    y_pred = ridge.predict(X_test_s)
    ridge_test_mae = mean_absolute_error(y_test, y_pred)
    ridge_test_r2 = r2_score(y_test, y_pred)

    print(f"\n  Ridge (alpha={ridge_gs.best_params_['alpha']})")
    print(f"    CV MAE:   {ridge_cv_mae:.2f} deg")
    print(f"    Test MAE: {ridge_test_mae:.2f} deg")
    print(f"    Test R²:  {ridge_test_r2:.3f}")

    # --- Comparison models (for the comparison plot) ---
    comparison = {
        "Ridge": (ridge_test_mae, ridge_test_r2, ridge_cv_mae),
    }

    models = {
        "ElasticNet": GridSearchCV(
            ElasticNet(random_state=RANDOM_STATE, max_iter=10000),
            {"alpha": [0.1, 1.0], "l1_ratio": [0.3, 0.7]},
            cv=3, scoring="neg_mean_absolute_error", n_jobs=1,
        ),
        "Gradient Boosting": GridSearchCV(
            GradientBoostingRegressor(random_state=RANDOM_STATE, min_samples_leaf=10, learning_rate=0.1),
            {"n_estimators": [50, 100], "max_depth": [2, 3]},
            cv=3, scoring="neg_mean_absolute_error", n_jobs=1,
        ),
        "SVR": GridSearchCV(
            SVR(kernel="rbf", gamma="scale"),
            {"C": [1.0, 10.0], "epsilon": [0.1, 0.5]},
            cv=3, scoring="neg_mean_absolute_error", n_jobs=1,
        ),
    }

    for name, gs in models.items():
        gs.fit(X_train_s, y_train)
        preds = gs.best_estimator_.predict(X_test_s)
        test_mae = mean_absolute_error(y_test, preds)
        test_r2 = r2_score(y_test, preds)
        cv_mae = -gs.best_score_
        comparison[name] = (test_mae, test_r2, cv_mae)
        print(f"  {name}: Test MAE={test_mae:.2f}, Test R²={test_r2:.3f}")

    return ridge, scaler, X_train_s, X_test_s, y_train, y_test, comparison


# ============================================================================
# 3. Save / Load Model
# ============================================================================

def save_model(model, scaler, path=MODEL_PATH):
    """Save the trained model and scaler to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "features": FEATURE_COLS}, f)
    print(f"\n  Model saved to {path}")


def load_model(path=MODEL_PATH):
    """Load a previously saved model and scaler."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"], data["features"]


# ============================================================================
# 4. Prediction
# ============================================================================

def predict_single(model, scaler, patient: dict) -> float:
    """
    Predict Mean AEI for a single patient.

    Args:
        patient: dict with keys matching FEATURE_COLS
                 e.g. {"Sex": 0, "SN_GoGn": 25.0, "Occ_Plane": 13.0,
                        "Ramus_Ht": 61.0, "Cond_Ht": 12.0,
                        "Bigonial": 183.0, "ANB": 2.0}
    Returns:
        Predicted Mean AEI in degrees
    """
    X = np.array([[patient[f] for f in FEATURE_COLS]])
    X_scaled = scaler.transform(X)
    return float(model.predict(X_scaled)[0])


def interactive_predict():
    """Interactive command-line prediction mode."""
    model, scaler, features = load_model()

    print("\n" + "=" * 50)
    print("  Eminence Inclination Predictor")
    print("=" * 50)
    print("  Enter patient measurements to predict Mean AEI.")
    print("  Type 'quit' to exit.\n")

    while True:
        try:
            patient = {}
            for feat in features:
                label, unit = FEATURE_INFO[feat]
                raw = input(f"  {label} ({unit}): ")
                if raw.lower() == "quit":
                    print("  Exiting.")
                    return
                patient[feat] = float(raw)

            prediction = predict_single(model, scaler, patient)
            print(f"\n  >> Predicted Mean AEI: {prediction:.1f} degrees\n")
            print("-" * 50)

        except (ValueError, KeyError) as e:
            print(f"  Error: {e}. Please try again.\n")
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            return


# ============================================================================
# 5. Plots (presentation-quality)
# ============================================================================

def plot_predicted_vs_actual(y_true, y_pred):
    """Scatter plot: predicted vs actual Mean AEI."""
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true, y_pred, c=COLORS["primary"], s=60, alpha=0.7,
               edgecolors="white", linewidth=0.8, zorder=3)

    lims = [min(min(y_true), min(y_pred)) - 2, max(max(y_true), max(y_pred)) + 2]
    ax.plot(lims, lims, color=COLORS["danger"], linestyle="--", linewidth=1.5,
            alpha=0.7, label="Perfect prediction", zorder=2)

    ax.set_xlabel("Actual Mean AEI (degrees)")
    ax.set_ylabel("Predicted Mean AEI (degrees)")
    ax.set_title("Predicted vs Actual — Ridge Regression")
    ax.legend(framealpha=0.9)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.92, f"MAE = {mae:.2f}°\nR² = {r2:.3f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "predicted_vs_actual.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_bland_altman(y_true, y_pred):
    """Bland-Altman agreement plot."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    means = (y_true + y_pred) / 2
    diffs = y_pred - y_true
    mean_d = diffs.mean()
    sd_d = diffs.std()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(means, diffs, c=COLORS["primary"], s=60, alpha=0.7,
               edgecolors="white", linewidth=0.8, zorder=3)

    ax.axhline(mean_d, color=COLORS["dark"], linewidth=1.5,
               label=f"Mean bias: {mean_d:+.2f}°")
    ax.axhline(mean_d + 1.96 * sd_d, color=COLORS["danger"], linewidth=1.2,
               linestyle="--", label=f"+1.96 SD: {mean_d + 1.96 * sd_d:+.2f}°")
    ax.axhline(mean_d - 1.96 * sd_d, color=COLORS["danger"], linewidth=1.2,
               linestyle="--", label=f"−1.96 SD: {mean_d - 1.96 * sd_d:+.2f}°")

    ax.axhspan(mean_d - 1.96 * sd_d, mean_d + 1.96 * sd_d,
               alpha=0.06, color=COLORS["danger"])

    ax.set_xlabel("Mean of Actual and Predicted (degrees)")
    ax.set_ylabel("Predicted − Actual (degrees)")
    ax.set_title("Bland-Altman Agreement Plot")
    ax.legend(fontsize=9, framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "bland_altman.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(model, feature_names):
    """Horizontal bar chart of standardized Ridge coefficients."""
    coefs = model.coef_
    labels = [FEATURE_INFO[f][0] for f in feature_names]

    # Sort by absolute value
    order = np.argsort(np.abs(coefs))
    coefs_sorted = coefs[order]
    labels_sorted = [labels[i] for i in order]

    # Color: positive = blue, negative = coral
    bar_colors = [COLORS["primary"] if c >= 0 else COLORS["secondary"] for c in coefs_sorted]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels_sorted, coefs_sorted, color=bar_colors, edgecolor="white",
                   linewidth=0.8, height=0.6)

    ax.axvline(0, color=COLORS["dark"], linewidth=0.8)
    ax.set_xlabel("Standardized Coefficient")
    ax.set_title("Feature Importance — Ridge Regression")

    # Add value labels
    for bar, val in zip(bars, coefs_sorted):
        x_pos = bar.get_width() + 0.02 if val >= 0 else bar.get_width() - 0.02
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.2f}", va="center", ha=ha, fontsize=9, color=COLORS["dark"])

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "feature_importance.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_comparison(comparison):
    """Bar chart comparing all 4 models on test MAE."""
    names = list(comparison.keys())
    test_maes = [comparison[n][0] for n in names]
    cv_maes = [comparison[n][2] for n in names]

    model_colors = [COLORS["ridge"], COLORS["elasticnet"], COLORS["gb"], COLORS["svr"]]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(x - width / 2, cv_maes, width, label="CV MAE (training)",
                   color=model_colors, alpha=0.45, edgecolor="white", linewidth=1.2)
    bars2 = ax.bar(x + width / 2, test_maes, width, label="Test MAE",
                   color=model_colors, alpha=0.85, edgecolor="white", linewidth=1.2)

    # Value labels on test bars
    for bar, val in zip(bars2, test_maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}°", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Mean Absolute Error (degrees)")
    ax.set_title("Model Comparison — Predicting Eminence Inclination")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, max(test_maes) * 1.3)

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "model_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_residual_distribution(y_true, y_pred):
    """Histogram of prediction residuals."""
    residuals = np.array(y_pred) - np.array(y_true)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(residuals, bins=10, color=COLORS["primary"], alpha=0.7,
            edgecolor="white", linewidth=1.2)

    ax.axvline(0, color=COLORS["danger"], linestyle="--", linewidth=1.5, label="Zero error")
    ax.axvline(residuals.mean(), color=COLORS["accent"], linestyle="-", linewidth=1.5,
               label=f"Mean: {residuals.mean():+.2f}°")

    ax.set_xlabel("Prediction Error (degrees)")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Distribution of Prediction Errors")
    ax.legend(framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "residual_distribution.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_correlation(df):
    """Heatmap of feature correlations with the target."""
    cols = FEATURE_COLS + [TARGET_COL]
    labels = [FEATURE_INFO.get(c, (c, ""))[0] for c in FEATURE_COLS] + ["Mean AEI"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.5 else COLORS["dark"]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson Correlation")
    ax.set_title("Feature Correlation Matrix")

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "correlation_matrix.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================================
# 6. Main
# ============================================================================

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # --- Load data ---
    print("=" * 50)
    print("  Eminence Guidance Prediction")
    print("=" * 50)
    df = load_dataset()

    # --- Train model ---
    print("\nTraining models...")
    model, scaler, X_train_s, X_test_s, y_train, y_test, comparison = train_model(df)

    # --- Save model ---
    save_model(model, scaler)

    # --- Test predictions ---
    y_pred = model.predict(X_test_s)

    # --- Generate all plots ---
    print("\nGenerating plots...")
    p1 = plot_predicted_vs_actual(y_test, y_pred)
    p2 = plot_bland_altman(y_test, y_pred)
    p3 = plot_feature_importance(model, FEATURE_COLS)
    p4 = plot_model_comparison(comparison)
    p5 = plot_residual_distribution(y_test, y_pred)
    p6 = plot_feature_correlation(df)

    for p in [p1, p2, p3, p4, p5, p6]:
        print(f"  Saved: {os.path.basename(p)}")

    # --- Summary ---
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{'=' * 50}")
    print(f"  FINAL MODEL PERFORMANCE")
    print(f"{'=' * 50}")
    print(f"  Model:     Ridge Regression (alpha={model.alpha})")
    print(f"  Features:  {len(FEATURE_COLS)} ({', '.join(FEATURE_COLS)})")
    print(f"  Train/Test: {len(y_train)}/{len(y_test)} patients")
    print(f"  MAE:       {mae:.2f} degrees")
    print(f"  RMSE:      {rmse:.2f} degrees")
    print(f"  R²:        {r2:.3f}")
    print(f"\n  Model saved to: {MODEL_PATH}")
    print(f"  Plots saved to: {PLOTS_DIR}/")
    print(f"\n  To make predictions: python eminence_model.py --predict")


if __name__ == "__main__":
    if "--predict" in sys.argv:
        interactive_predict()
    else:
        main()
