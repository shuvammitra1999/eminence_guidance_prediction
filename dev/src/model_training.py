"""
Model definitions, hyperparameter tuning via GridSearchCV, and training.
"""

import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from config import PARAM_GRIDS, CV_FOLDS, RANDOM_STATE


def get_models():
    """Return dict of model name -> base estimator."""
    return {
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "ElasticNet": ElasticNet(random_state=RANDOM_STATE, max_iter=10000),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "SVR": SVR(kernel="rbf"),
    }


def tune_and_train(X_train: np.ndarray, y_train: np.ndarray):
    """
    Tune all 4 models via GridSearchCV and return best estimators.

    Returns:
        dict of model_name -> {
            "best_estimator": fitted model,
            "best_params": dict,
            "best_cv_score": float (negative MAE),
        }
    """
    models = get_models()
    results = {}

    for name, estimator in models.items():
        grid = PARAM_GRIDS[name]

        gs = GridSearchCV(
            estimator,
            grid,
            cv=CV_FOLDS,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
            refit=True,
        )
        gs.fit(X_train, y_train)

        results[name] = {
            "best_estimator": gs.best_estimator_,
            "best_params": gs.best_params_,
            "best_cv_score": gs.best_score_,
        }

        print(f"  {name:20s} | CV MAE: {-gs.best_score_:.2f} | Params: {gs.best_params_}")

    return results
