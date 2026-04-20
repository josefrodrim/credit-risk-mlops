"""Great Expectations data validation runner for the credit-risk pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as fh:
        return yaml.safe_load(fh)


def _load_expectations(suite_path: str) -> Dict[str, Any]:
    with open(suite_path, "r") as fh:
        return json.load(fh)


def _run_expectation(df: pd.DataFrame, expectation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch a single expectation dict against a dataframe.

    Supported expectation types mirror the Great Expectations API subset
    used in credit_risk_suite.json.

    Args:
        df: DataFrame to validate.
        expectation: Expectation dict with 'expectation_type' and 'kwargs'.

    Returns:
        Result dict with keys: expectation_type, success, observed_value, kwargs.
    """
    etype: str = expectation["expectation_type"]
    kwargs: dict = expectation.get("kwargs", {})
    success = False
    observed: Any = None

    try:
        if etype == "expect_column_to_exist":
            col = kwargs["column"]
            success = col in df.columns
            observed = list(df.columns)

        elif etype == "expect_column_values_to_not_be_null":
            col = kwargs["column"]
            mostly = kwargs.get("mostly", 1.0)
            null_rate = df[col].isna().mean()
            success = (1 - null_rate) >= mostly
            observed = round(1 - null_rate, 4)

        elif etype == "expect_column_values_to_be_between":
            col = kwargs["column"]
            min_val = kwargs.get("min_value")
            max_val = kwargs.get("max_value")
            mostly = kwargs.get("mostly", 1.0)
            mask = pd.Series([True] * len(df))
            if min_val is not None:
                mask &= df[col] >= min_val
            if max_val is not None:
                mask &= df[col] <= max_val
            success = mask.mean() >= mostly
            observed = {"min": float(df[col].min()), "max": float(df[col].max())}

        elif etype == "expect_column_values_to_be_in_set":
            col = kwargs["column"]
            value_set = set(kwargs["value_set"])
            actual = set(df[col].dropna().unique())
            unexpected = actual - value_set
            success = len(unexpected) == 0
            observed = list(unexpected)

        elif etype == "expect_column_proportion_of_unique_values_to_be_between":
            col = kwargs["column"]
            min_val = kwargs.get("min_value", 0.0)
            max_val = kwargs.get("max_value", 1.0)
            prop = df[col].nunique() / len(df)
            success = min_val <= prop <= max_val
            observed = round(prop, 6)

        elif etype == "expect_table_row_count_to_be_between":
            min_val = kwargs.get("min_value", 0)
            max_val = kwargs.get("max_value", float("inf"))
            success = min_val <= len(df) <= max_val
            observed = len(df)

        elif etype == "expect_column_mean_to_be_between":
            col = kwargs["column"]
            min_val = kwargs.get("min_value")
            max_val = kwargs.get("max_value")
            mean = float(df[col].mean())
            lower_ok = min_val is None or mean >= min_val
            upper_ok = max_val is None or mean <= max_val
            success = lower_ok and upper_ok
            observed = round(mean, 4)

        else:
            logger.warning("Unsupported expectation type: %s — skipping", etype)
            success = True
            observed = "skipped"

    except (KeyError, TypeError) as exc:
        logger.error("Error evaluating %s: %s", etype, exc)
        success = False
        observed = str(exc)

    return {
        "expectation_type": etype,
        "kwargs": kwargs,
        "success": success,
        "observed_value": observed,
    }


def validate_dataframe(
    df: pd.DataFrame,
    suite: Dict[str, Any],
    fail_fast: bool = False,
) -> Dict[str, Any]:
    """
    Run all expectations in a suite against df.

    Args:
        df: DataFrame to validate.
        suite: Expectation suite dict (loaded from JSON).
        fail_fast: Raise immediately on first failure if True.

    Returns:
        Validation result dict with keys: success, statistics, results.

    Raises:
        RuntimeError: If fail_fast=True and any expectation fails.
    """
    expectations = suite.get("expectations", [])
    results = []
    n_failed = 0

    for exp in expectations:
        result = _run_expectation(df, exp)
        results.append(result)
        if not result["success"]:
            n_failed += 1
            logger.error(
                "FAILED  %s | kwargs=%s | observed=%s",
                result["expectation_type"],
                result["kwargs"],
                result["observed_value"],
            )
            if fail_fast:
                raise RuntimeError(f"Expectation failed: {result['expectation_type']}")
        else:
            logger.info(
                "passed  %s | observed=%s",
                result["expectation_type"],
                result["observed_value"],
            )

    overall_success = n_failed == 0
    stats = {
        "evaluated_expectations": len(expectations),
        "successful_expectations": len(expectations) - n_failed,
        "unsuccessful_expectations": n_failed,
        "success_percent": round(100 * (len(expectations) - n_failed) / max(len(expectations), 1), 1),
    }

    return {"success": overall_success, "statistics": stats, "results": results}


def main() -> None:
    """CLI entry-point for the DVC validate stage."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Great Expectations data validation")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument(
        "--suite",
        default="great_expectations/expectations/credit_risk_suite.json",
        help="Path to expectation suite JSON",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to parquet file to validate (default: raw_path from params)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on first failed expectation",
    )
    args = parser.parse_args()

    params = _load_params(args.params)
    data_path = args.data or params["data"]["raw_path"]

    logger.info("Loading data from %s", data_path)
    df = pd.read_parquet(data_path, engine="pyarrow")

    logger.info("Loading expectation suite from %s", args.suite)
    suite = _load_expectations(args.suite)

    validation_result = validate_dataframe(df, suite, fail_fast=args.fail_fast)

    stats = validation_result["statistics"]
    logger.info(
        "Validation complete — %d/%d expectations passed (%.1f%%)",
        stats["successful_expectations"],
        stats["evaluated_expectations"],
        stats["success_percent"],
    )

    if not validation_result["success"]:
        logger.error("Data validation FAILED")
        sys.exit(1)

    logger.info("Data validation PASSED")


if __name__ == "__main__":
    main()
