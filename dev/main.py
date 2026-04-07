"""
End-to-end pipeline: Eminence Guidance Prediction (lightweight).
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
from config import FEATURE_SETS, RESULTS_DIR
from data_loader import load_merged_dataset, get_train_test_split, print_data_audit
from feature_engineering import scale_data
from model_training import tune_and_train
from evaluation import (
    compute_metrics,
    build_comparison_table,
    plot_predicted_vs_actual,
    plot_bland_altman,
    plot_feature_importance,
)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Step 1: Load data
    print("=" * 50)
    print("STEP 1: Data Loading")
    print("=" * 50)
    df = load_merged_dataset()
    print_data_audit(df)

    X_train, X_test, y_train, y_test = get_train_test_split(df)
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # Step 2: Train across feature sets x models
    all_results = {}

    for fs_name in FEATURE_SETS:
        print(f"\n--- Feature Set: {fs_name} ---")
        X_tr, X_te, scaler, feat_names = scale_data(X_train, X_test, fs_name)

        model_results = tune_and_train(X_tr, y_train.values)
        all_results[fs_name] = {}

        for model_name, info in model_results.items():
            model = info["best_estimator"]
            y_pred = model.predict(X_te)

            all_results[fs_name][model_name] = {
                "metrics": compute_metrics(y_test.values, y_pred),
                "cv_mae": info["best_cv_score"],
                "y_pred": y_pred,
                "y_true": y_test.values,
                "model": model,
                "feature_names": feat_names,
            }

    # Step 3: Comparison
    print(f"\n{'=' * 50}")
    print("RESULTS")
    print("=" * 50)
    comparison = build_comparison_table(all_results)
    print(comparison.to_string(index=False))
    comparison.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

    # Step 4: Best model details + plots
    best = comparison.iloc[0]
    best_info = all_results[best["Feature Set"]][best["Model"]]

    print(f"\nBEST: {best['Model']} ({best['Feature Set']})")
    print(f"  MAE:  {best['Test MAE']:.2f} deg")
    print(f"  RMSE: {best['Test RMSE']:.2f} deg")
    print(f"  R2:   {best['Test R2']:.3f}")

    plot_predicted_vs_actual(best_info["y_true"], best_info["y_pred"],
                            best["Model"], best["Feature Set"])
    plot_bland_altman(best_info["y_true"], best_info["y_pred"],
                      best["Model"], best["Feature Set"])
    plot_feature_importance(best_info["model"], best_info["feature_names"],
                           best["Model"], best["Feature Set"])

    # Step 5: Per-patient predictions
    print(f"\nPer-patient predictions ({best['Model']}):")
    pred_df = pd.DataFrame({
        "Actual": best_info["y_true"],
        "Predicted": np.round(best_info["y_pred"], 1),
        "Abs_Error": np.round(np.abs(best_info["y_pred"] - best_info["y_true"]), 1),
    })
    print(pred_df.to_string())
    print(f"\nPlots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
