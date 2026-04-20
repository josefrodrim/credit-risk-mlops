"""DVC evaluate stage: load production model + test set, compute metrics, write artefacts."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    """CLI entry-point for the DVC evaluate stage."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="DVC evaluate stage")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--model-uri", default=None, help="Override model URI")
    parser.add_argument("--reports-dir", default="reports", help="Output directory for artefacts")
    args = parser.parse_args()

    params = _load_params(args.params)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    model_uri = args.model_uri or os.getenv("MODEL_URI") or params["api"]["model_uri"]
    logger.info("Loading model from %s", model_uri)

    try:
        pipeline = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        sys.exit(1)

    test_path = params["data"]["test_path"]
    logger.info("Loading test data from %s", test_path)
    test_df = pd.read_parquet(test_path, engine="pyarrow")

    from src.features import prepare_features
    from src.evaluate import evaluate_model

    X_test, y_test = prepare_features(test_df, args.params)

    with mlflow.start_run(run_name="evaluate-stage") as run:
        metrics = evaluate_model(
            pipeline=pipeline,
            X_test=X_test,
            y_test=y_test,
            params=params,
            run_id=run.info.run_id,
            reports_dir=args.reports_dir,
        )

    auc = metrics["roc_auc"]
    threshold = params["model"]["auc_threshold"]

    # Write DVC-compatible metrics file
    metrics_path = Path(args.reports_dir) / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as fh:
        json.dump({k: round(v, 5) for k, v in metrics.items()}, fh, indent=2)
    logger.info("Metrics written to %s", metrics_path)

    if auc < threshold:
        logger.error("AUC gate FAILED: %.4f < %.4f", auc, threshold)
        sys.exit(1)

    logger.info("Evaluate stage PASSED: AUC=%.4f >= %.4f", auc, threshold)


if __name__ == "__main__":
    main()
