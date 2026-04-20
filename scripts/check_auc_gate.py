"""Jenkins AUC gate check: read metrics.json and exit non-zero if AUC < threshold."""

from __future__ import annotations

import argparse
import json
import logging
import sys

import yaml

logger = logging.getLogger(__name__)


def _load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as fh:
        return yaml.safe_load(fh)


def check_auc_gate(
    metrics_path: str,
    auc_threshold: float,
    metric_key: str = "roc_auc",
) -> float:
    """
    Read a metrics JSON file and assert AUC meets the threshold.

    Args:
        metrics_path: Path to the JSON file containing model metrics.
        auc_threshold: Minimum acceptable AUC (e.g. 0.87).
        metric_key: Key in the metrics dict holding the AUC value.

    Returns:
        Observed AUC value.

    Raises:
        FileNotFoundError: If metrics_path does not exist.
        KeyError: If metric_key is absent from the JSON.
        ValueError: If AUC is below the threshold.
    """
    with open(metrics_path, "r") as fh:
        metrics: dict = json.load(fh)

    if metric_key not in metrics:
        raise KeyError(
            f"Metric '{metric_key}' not found in {metrics_path}. Available: {list(metrics.keys())}"
        )

    auc = float(metrics[metric_key])
    logger.info("AUC gate check: %s=%.5f | threshold=%.5f", metric_key, auc, auc_threshold)

    if auc < auc_threshold:
        raise ValueError(
            f"AUC gate FAILED: {metric_key}={auc:.5f} < threshold={auc_threshold:.5f}. "
            "Model is not production-ready."
        )

    logger.info("AUC gate PASSED: %.5f >= %.5f", auc, auc_threshold)
    return auc


def main() -> None:
    """CLI entry-point called from Jenkins post-evaluate step."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="AUC gate check for Jenkins CI")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument(
        "--metrics",
        default="reports/metrics.json",
        help="Path to metrics JSON written by evaluate_model.py",
    )
    parser.add_argument(
        "--metric-key",
        default="roc_auc",
        help="Key in metrics JSON to compare against threshold",
    )
    args = parser.parse_args()

    params = _load_params(args.params)
    threshold: float = params["model"]["auc_threshold"]

    try:
        auc = check_auc_gate(
            metrics_path=args.metrics,
            auc_threshold=threshold,
            metric_key=args.metric_key,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    print(f"::set-output name=auc::{auc:.5f}")


if __name__ == "__main__":
    main()
