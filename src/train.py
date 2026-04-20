"""Model training with XGBoost + LightGBM, Optuna HPO, and MLflow tracking."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_full_pipeline, prepare_features

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as fh:
        return yaml.safe_load(fh)


def _temporal_split(
    df: pd.DataFrame,
    train_end: int,
    test_start: int,
    period_col: str = "periodo",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe on periodo boundary."""
    train = df[df[period_col] <= train_end].copy()
    test = df[df[period_col] >= test_start].copy()
    logger.info("Temporal split — train: %d | test: %d", len(train), len(test))
    return train, test


# ---- XGBoost ---------------------------------------------------------------

def _xgb_objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> float:
    """Optuna objective for XGBoost hyperparameter search."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5.0, 20.0),
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }
    scores = []
    for fold_train_idx, fold_val_idx in cv.split(X, y):
        model = XGBClassifier(**params)
        model.fit(
            X[fold_train_idx],
            y[fold_train_idx],
            eval_set=[(X[fold_val_idx], y[fold_val_idx])],
            verbose=False,
        )
        preds = model.predict_proba(X[fold_val_idx])[:, 1]
        scores.append(roc_auc_score(y[fold_val_idx], preds))
    return float(np.mean(scores))


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train XGBoost with Optuna HPO.

    Returns:
        Tuple of (fitted pipeline, best_params dict).
    """
    n_trials = params["model"]["n_trials"]
    n_folds = params["model"]["cv_folds"]
    seed = params["data"]["random_seed"]

    from sklearn.pipeline import Pipeline as _SKPipeline
    from src.features import build_preprocessor, LogTransformer, DelinquencyRatioTransformer

    pre_pipeline = _SKPipeline([
        ("log_transform", LogTransformer()),
        ("delinquency", DelinquencyRatioTransformer()),
        ("preprocessor", build_preprocessor()),
    ])
    X_arr = pre_pipeline.fit_transform(X_train)
    y_arr = y_train.values

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(
        lambda trial: _xgb_objective(trial, X_arr, y_arr, cv),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_params = study.best_params
    logger.info("XGB best CV AUC=%.4f | params=%s", study.best_value, best_params)

    best_params.update(
        {
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": seed,
            "n_jobs": -1,
        }
    )

    pipeline = build_full_pipeline(XGBClassifier(**best_params))
    pipeline.fit(X_train, y_train)
    return pipeline, best_params


# ---- LightGBM --------------------------------------------------------------

def _lgbm_objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> float:
    """Optuna objective for LightGBM hyperparameter search."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5.0, 20.0),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    scores = []
    for fold_train_idx, fold_val_idx in cv.split(X, y):
        model = lgb.LGBMClassifier(**params)
        model.fit(X[fold_train_idx], y[fold_train_idx])
        preds = model.predict_proba(X[fold_val_idx])[:, 1]
        scores.append(roc_auc_score(y[fold_val_idx], preds))
    return float(np.mean(scores))


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train LightGBM with Optuna HPO.

    Returns:
        Tuple of (fitted pipeline, best_params dict).
    """
    n_trials = params["model"]["n_trials"]
    n_folds = params["model"]["cv_folds"]
    seed = params["data"]["random_seed"]

    from sklearn.pipeline import Pipeline as _SKPipeline
    from src.features import build_preprocessor, LogTransformer, DelinquencyRatioTransformer

    pre_pipeline = _SKPipeline([
        ("log_transform", LogTransformer()),
        ("delinquency", DelinquencyRatioTransformer()),
        ("preprocessor", build_preprocessor()),
    ])
    X_arr = pre_pipeline.fit_transform(X_train)
    y_arr = y_train.values

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(
        lambda trial: _lgbm_objective(trial, X_arr, y_arr, cv),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_params = study.best_params
    logger.info("LGBM best CV AUC=%.4f | params=%s", study.best_value, best_params)

    best_params.update({"random_state": seed, "n_jobs": -1, "verbose": -1})
    pipeline = build_full_pipeline(lgb.LGBMClassifier(**best_params))
    pipeline.fit(X_train, y_train)
    return pipeline, best_params


# ---- Main ------------------------------------------------------------------

def main() -> None:
    """CLI entry-point: train XGB + LGBM, log to MLflow, register best model."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train credit risk models")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = _load_params(args.params)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    train_path = params["data"]["train_path"]
    test_path = params["data"]["test_path"]

    logger.info("Loading train data from %s", train_path)
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    X_train, y_train = prepare_features(train_df, args.params)
    X_test, y_test = prepare_features(test_df, args.params)

    best_auc = 0.0
    best_run_id: str | None = None

    for model_type, train_fn, model_name_key in [
        ("xgboost", train_xgboost, "xgb_model_name"),
        ("lightgbm", train_lightgbm, "lgbm_model_name"),
    ]:
        logger.info("Training %s ...", model_type)
        with mlflow.start_run(run_name=model_type) as run:
            pipeline, best_hp = train_fn(X_train, y_train, params)

            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_pred_proba)
            logger.info("%s test AUC=%.4f", model_type, test_auc)

            mlflow.log_params(best_hp)
            mlflow.log_metric("test_roc_auc", test_auc)
            mlflow.log_metric("train_size", len(X_train))
            mlflow.log_metric("test_size", len(X_test))

            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                registered_model_name=params["mlflow"]["model_name"],
            )

            if test_auc > best_auc:
                best_auc = test_auc
                best_run_id = run.info.run_id

    logger.info("Best model: run_id=%s AUC=%.4f", best_run_id, best_auc)

    auc_threshold = params["model"]["auc_threshold"]
    if best_auc < auc_threshold:
        raise RuntimeError(
            f"Best AUC {best_auc:.4f} < threshold {auc_threshold}. Training failed gate."
        )

    logger.info("AUC gate passed (%.4f >= %.4f)", best_auc, auc_threshold)

    # Promote best model version to Production stage
    model_name = params["mlflow"]["model_name"]
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    best_version = next((v for v in versions if v.run_id == best_run_id), None)
    if best_version:
        client.transition_model_version_stage(
            name=model_name,
            version=best_version.version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info("Model v%s promoted to Production (AUC=%.4f)", best_version.version, best_auc)


if __name__ == "__main__":
    main()
