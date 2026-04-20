"""Model evaluation: AUC, KS statistic, SHAP explanations, and model card."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict

import mlflow
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as fh:
        return yaml.safe_load(fh)


def compute_ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Kolmogorov–Smirnov statistic for binary classification.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted positive-class probabilities.

    Returns:
        KS statistic in [0, 1].
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute a comprehensive set of binary classification metrics.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted positive-class probabilities.
        threshold: Decision boundary for precision/recall/F1.

    Returns:
        Dict mapping metric name → value.
    """
    y_pred = (y_score >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "avg_precision": average_precision_score(y_true, y_score),
        "ks_statistic": compute_ks_statistic(y_true, y_score),
        "brier_score": brier_score_loss(y_true, y_score),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def compute_shap_values(
    pipeline: Pipeline,
    X: pd.DataFrame,
    max_samples: int = 1000,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute SHAP values using the model step of the pipeline.

    Args:
        pipeline: Fitted sklearn Pipeline with a 'model' and 'preprocessor' step.
        X: Raw feature dataframe.
        max_samples: Cap for SHAP background/foreground samples.

    Returns:
        Tuple of (shap_values array, feature_names list).
    """
    import shap

    # Apply all pre-model steps to get the array the model sees
    X_sample = X.sample(min(max_samples, len(X)), random_state=42)

    # Transform through all steps except the final 'model'
    X_transformed = X_sample.copy()
    for name, step in pipeline.steps[:-1]:
        X_transformed = step.transform(X_transformed)

    model = pipeline.named_steps["model"]

    if hasattr(model, "get_booster"):  # XGBoost
        explainer = shap.TreeExplainer(model)
    elif hasattr(model, "booster_"):  # LightGBM
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, X_transformed)

    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Feature names from preprocessor
    try:
        feature_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    except AttributeError:
        feature_names = [f"f{i}" for i in range(shap_values.shape[1])]

    return shap_values, feature_names


def build_model_card(
    metrics: Dict[str, float],
    params: dict,
    run_id: str,
    output_path: str,
) -> None:
    """
    Write a JSON model card with evaluation metrics and training config.

    Args:
        metrics: Computed metrics dict.
        params: Full pipeline params.
        run_id: MLflow run ID.
        output_path: Where to save the JSON file.
    """
    card = {
        "model_name": params["mlflow"]["model_name"],
        "mlflow_run_id": run_id,
        "dataset": {
            "n_train_rows": params["data"].get("n_train_rows", "N/A"),
            "n_test_rows": params["data"].get("n_test_rows", "N/A"),
            "train_end_periodo": params["data"]["train_end_periodo"],
            "test_start_periodo": params["data"]["test_start_periodo"],
            "target": params["features"]["target"],
            "default_rate": params["data"]["default_rate"],
        },
        "metrics": {k: round(v, 5) for k, v in metrics.items()},
        "auc_gate": {
            "threshold": params["model"]["auc_threshold"],
            "passed": metrics.get("roc_auc", 0) >= params["model"]["auc_threshold"],
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(card, fh, indent=2)

    logger.info("Model card written to %s", output_path)


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict,
    run_id: str,
    reports_dir: str = "reports",
) -> Dict[str, float]:
    """
    Full evaluation: metrics + SHAP + model card.  Logs to MLflow.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X_test: Test feature dataframe.
        y_test: Test target series.
        params: Full pipeline params dict.
        run_id: Active MLflow run ID.
        reports_dir: Directory for saving artefacts.

    Returns:
        Metrics dict.
    """
    y_score = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.values, y_score)

    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)
        mlflow.log_metric(k, v)

    # ---- SHAP ----
    logger.info("Computing SHAP values ...")
    try:
        shap_values, feature_names = compute_shap_values(pipeline, X_test, max_samples=2000)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": mean_abs_shap}
        ).sort_values("mean_abs_shap", ascending=False)

        Path(reports_dir).mkdir(parents=True, exist_ok=True)
        shap_path = f"{reports_dir}/shap_importance.csv"
        importance_df.to_csv(shap_path, index=False)
        mlflow.log_artifact(shap_path)
        logger.info("SHAP importance saved to %s", shap_path)
    except Exception as exc:
        logger.warning("SHAP computation failed (non-fatal): %s", exc)

    # ---- Model card ----
    card_path = f"{reports_dir}/model_card.json"
    build_model_card(metrics, params, run_id, card_path)
    mlflow.log_artifact(card_path)

    return metrics


def main() -> None:
    """CLI entry-point for evaluate stage (called from scripts/evaluate_model.py)."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--model-uri", default=None, help="MLflow model URI, overrides params.yaml")
    args = parser.parse_args()

    params = _load_params(args.params)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = args.model_uri or params["api"]["model_uri"]
    logger.info("Loading model from %s", model_uri)
    pipeline = mlflow.sklearn.load_model(model_uri)

    test_df = pd.read_parquet(params["data"]["test_path"])

    from src.features import prepare_features

    X_test, y_test = prepare_features(test_df, args.params)

    with mlflow.start_run(run_name="evaluate") as run:
        metrics = evaluate_model(pipeline, X_test, y_test, params, run.info.run_id)

    auc = metrics["roc_auc"]
    threshold = params["model"]["auc_threshold"]
    if auc < threshold:
        raise RuntimeError(f"AUC {auc:.4f} below gate {threshold}")
    logger.info("Evaluation passed gate: AUC=%.4f", auc)


if __name__ == "__main__":
    main()
