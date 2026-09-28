"""Day-1 baseline: engineered/embedded features -> LightGBM + XGBoost + CatBoost,
with out-of-fold predictions ready for stacking. Get this running within the
first few hours of the sprint before touching any fine-tuning.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.model_selection import KFold


def train_gbm_cv(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    task: Literal["regression", "classification"] = "regression",
    n_splits: int = 5,
    seed: int = 42,
    model_names: tuple[str, ...] = ("lgbm", "xgb", "cat"),
) -> dict:
    """K-fold CV training three GBM families. Returns a dict:
        {model_name: {"oof": array[n_train], "test": array[n_test], "models": [...]}}
    Feed the "oof" arrays straight into `ensemble.py` for blending/stacking.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = {name: {"oof": np.zeros(len(y)), "test": np.zeros(len(X_test)), "models": []} for name in model_names}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if "lgbm" in model_names:
            model = _fit_lgbm(X_tr, y_tr, X_val, y_val, task, seed)
            results["lgbm"]["oof"][val_idx] = _predict(model, X_val, task)
            results["lgbm"]["test"] += _predict(model, X_test, task) / n_splits
            results["lgbm"]["models"].append(model)

        if "xgb" in model_names:
            model = _fit_xgb(X_tr, y_tr, X_val, y_val, task, seed)
            results["xgb"]["oof"][val_idx] = _predict(model, X_val, task)
            results["xgb"]["test"] += _predict(model, X_test, task) / n_splits
            results["xgb"]["models"].append(model)

        if "cat" in model_names:
            model = _fit_cat(X_tr, y_tr, X_val, y_val, task, seed)
            results["cat"]["oof"][val_idx] = _predict(model, X_val, task)
            results["cat"]["test"] += _predict(model, X_test, task) / n_splits
            results["cat"]["models"].append(model)

        print(f"fold {fold + 1}/{n_splits} done")

    return results


def _fit_lgbm(X_tr, y_tr, X_val, y_val, task, seed):
    import lightgbm as lgb

    Model = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
    model = Model(n_estimators=2000, learning_rate=0.03, num_leaves=63, random_state=seed)
    model.fit(
        X_tr, y_tr, eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def _fit_xgb(X_tr, y_tr, X_val, y_val, task, seed):
    import xgboost as xgb

    Model = xgb.XGBRegressor if task == "regression" else xgb.XGBClassifier
    model = Model(
        n_estimators=2000, learning_rate=0.03, max_depth=8,
        random_state=seed, early_stopping_rounds=100, eval_metric="mae" if task == "regression" else "logloss",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def _fit_cat(X_tr, y_tr, X_val, y_val, task, seed):
    from catboost import CatBoostClassifier, CatBoostRegressor

    Model = CatBoostRegressor if task == "regression" else CatBoostClassifier
    model = Model(
        iterations=2000, learning_rate=0.03, depth=8,
        random_seed=seed, verbose=False, early_stopping_rounds=100,
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    return model


def _predict(model, X, task):
    if task == "classification" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] == 2 else proba.argmax(axis=1)
    return model.predict(X)
