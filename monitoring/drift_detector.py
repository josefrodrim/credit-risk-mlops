"""Evidently AI data drift detection for credit-risk-mlops.

Compares a reference dataset (training) against a current dataset (test/production)
and generates an HTML report plus a structured JSON summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as fh:
        return yaml.safe_load(fh)


def load_dataset(path: str, n_sample: int | None = None, seed: int = 42) -> pd.DataFrame:
    """
    Load a parquet dataset, optionally sampling for speed.

    Args:
        path: Path to parquet file.
        n_sample: If set, randomly sample this many rows.
        seed: Random seed for reproducible sampling.

    Returns:
        DataFrame.
    """
    df = pd.read_parquet(path, engine="pyarrow")
    if n_sample and len(df) > n_sample:
        df = df.sample(n=n_sample, random_state=seed)
    return df


def run_drift_detection(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    drift_threshold: float = 0.05,
    output_path: str = "reports/drift_report.html",
    summary_path: str = "reports/drift_summary.json",
) -> dict:
    """
    Run Evidently data drift + target drift analysis.

    Args:
        reference_df: Reference (training) dataset.
        current_df: Current (test/production) dataset.
        feature_cols: Feature columns to monitor.
        target_col: Target column name.
        drift_threshold: p-value / share-of-drifted-features threshold.
        output_path: Path to write the HTML report.
        summary_path: Path to write the JSON summary.

    Returns:
        Dict with drift summary statistics.
    """
    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
        from evidently.report import Report
    except ImportError as exc:
        raise ImportError("evidently is required: pip install evidently") from exc

    available_cols = [c for c in feature_cols if c in reference_df.columns and c in current_df.columns]
    if not available_cols:
        raise ValueError("No feature columns found in both reference and current datasets.")

    target_available = target_col in reference_df.columns and target_col in current_df.columns
    column_mapping = ColumnMapping(
        target=target_col if target_available else None,
        numerical_features=[c for c in available_cols if reference_df[c].dtype != object],
        categorical_features=[c for c in available_cols if reference_df[c].dtype == object],
    )

    presets = [DataDriftPreset()]
    if target_available:
        presets.append(TargetDriftPreset())

    report = Report(metrics=presets)
    report.run(
        reference_data=reference_df[available_cols + ([target_col] if target_available else [])],
        current_data=current_df[available_cols + ([target_col] if target_available else [])],
        column_mapping=column_mapping,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(output_path)
    logger.info("Drift report saved to %s", output_path)

    # Extract summary from report dict
    report_dict = report.as_dict()
    metrics_list = report_dict.get("metrics", [])

    drift_summary: dict = {
        "reference_rows": len(reference_df),
        "current_rows": len(current_df),
        "n_features_checked": len(available_cols),
        "drift_detected": False,
        "drifted_features": [],
        "share_drifted": 0.0,
    }

    for metric in metrics_list:
        result = metric.get("result", {})
        # DatasetDriftMetric result
        if "share_of_drifted_columns" in result:
            drift_summary["share_drifted"] = result["share_of_drifted_columns"]
            drift_summary["drift_detected"] = result["share_of_drifted_columns"] >= drift_threshold
        if "drift_by_columns" in result:
            drift_summary["drifted_features"] = [
                col
                for col, stats in result["drift_by_columns"].items()
                if stats.get("drift_detected", False)
            ]

    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump(drift_summary, fh, indent=2)
    logger.info("Drift summary saved to %s", summary_path)

    return drift_summary


def main() -> None:
    """CLI entry-point for standalone drift detection."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Evidently drift detection")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--reference", default=None, help="Override reference parquet path")
    parser.add_argument("--current", default=None, help="Override current parquet path")
    parser.add_argument("--output", default=None, help="Override HTML report output path")
    parser.add_argument("--summary", default="reports/drift_summary.json")
    parser.add_argument("--n-sample", type=int, default=50_000, help="Max rows to sample per dataset")
    parser.add_argument("--fail-on-drift", action="store_true", help="Exit 1 if drift detected")
    args = parser.parse_args()

    params = _load_params(args.params)
    mon = params["monitoring"]
    feat = params["features"]

    reference_path = args.reference or mon["reference_data_path"]
    current_path = args.current or params["data"]["test_path"]
    output_path = args.output or mon["report_path"]
    threshold = float(mon["drift_threshold"])

    logger.info("Loading reference: %s", reference_path)
    ref_df = load_dataset(reference_path, n_sample=args.n_sample)

    logger.info("Loading current: %s", current_path)
    cur_df = load_dataset(current_path, n_sample=args.n_sample)

    feature_cols = feat["numeric"] + feat["categorical"]
    target_col = feat["target"]

    summary = run_drift_detection(
        reference_df=ref_df,
        current_df=cur_df,
        feature_cols=feature_cols,
        target_col=target_col,
        drift_threshold=threshold,
        output_path=output_path,
        summary_path=args.summary,
    )

    if summary["drift_detected"]:
        logger.warning(
            "DRIFT DETECTED — %.1f%% of features drifted | drifted: %s",
            summary["share_drifted"] * 100,
            summary["drifted_features"],
        )
        if args.fail_on_drift:
            sys.exit(1)
    else:
        logger.info("No significant drift detected (share=%.3f < threshold=%.3f)", summary["share_drifted"], threshold)


if __name__ == "__main__":
    main()
