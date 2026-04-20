"""Generate synthetic credit risk dataset mirroring the Give Me Some Credit schema."""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


CREDIT_SCORE_BANDS = ["Poor", "Fair", "Good", "Very Good", "Exceptional"]


def _load_params(params_path: str = "params.yaml") -> dict:
    """Load pipeline parameters from YAML file."""
    with open(params_path, "r") as fh:
        return yaml.safe_load(fh)


def _generate_periodo_series(n_samples: int, rng: np.random.Generator) -> pd.Series:
    """Generate YYYYMM period column spanning 202001–202604."""
    years = list(range(2020, 2027))
    months = list(range(1, 13))
    periods = [y * 100 + m for y in years for m in months if y * 100 + m <= 202604]
    return pd.Series(rng.choice(periods, size=n_samples), name="periodo")


def generate_dataset(params: dict) -> pd.DataFrame:
    """
    Generate a synthetic credit-risk dataset with ~7% default rate.

    Args:
        params: Pipeline params dict (from params.yaml).

    Returns:
        DataFrame with 1 000 000 rows and Give Me Some Credit schema columns.
    """
    n_samples: int = params["data"]["n_samples"]
    default_rate: float = params["data"]["default_rate"]
    seed: int = params["data"]["random_seed"]

    rng = np.random.default_rng(seed)

    logger.info("Generating %d rows with %.1f%% default rate", n_samples, default_rate * 100)

    # ---- target ----
    target = rng.binomial(1, default_rate, size=n_samples)

    # ---- numeric features ---- (distributions calibrated to GiveMe Some Credit)
    revolving_util = np.clip(rng.beta(0.5, 2, size=n_samples), 0, 1)
    age = rng.integers(18, 90, size=n_samples)
    past_due_30_59 = rng.negative_binomial(1, 0.8, size=n_samples)
    debt_ratio = np.clip(rng.lognormal(mean=-0.5, sigma=1.2, size=n_samples), 0, 50)
    monthly_income = rng.lognormal(mean=8.5, sigma=0.8, size=n_samples)
    monthly_income[rng.random(n_samples) < 0.02] = np.nan  # ~2% missing
    open_credit_lines = rng.negative_binomial(5, 0.4, size=n_samples)
    times_90_late = rng.negative_binomial(1, 0.9, size=n_samples)
    real_estate_loans = rng.integers(0, 5, size=n_samples)
    past_due_60_89 = rng.negative_binomial(1, 0.9, size=n_samples)
    n_dependents = rng.integers(0, 8, size=n_samples).astype(float)
    n_dependents[rng.random(n_samples) < 0.03] = np.nan  # ~3% missing
    loan_amount = np.clip(rng.lognormal(mean=10.5, sigma=1.0, size=n_samples), 500, 500_000)
    employment_years = np.clip(rng.exponential(scale=7, size=n_samples), 0, 45)

    # ---- categorical feature ----
    credit_score_band = rng.choice(CREDIT_SCORE_BANDS, size=n_samples, p=[0.15, 0.25, 0.35, 0.20, 0.05])

    # ---- inject signal: defaulters skew toward risky profiles ----
    default_mask = target == 1
    revolving_util[default_mask] = np.clip(revolving_util[default_mask] * 1.5 + 0.2, 0, 1)
    past_due_30_59[default_mask] += rng.integers(0, 5, size=default_mask.sum())
    times_90_late[default_mask] += rng.integers(0, 3, size=default_mask.sum())
    debt_ratio[default_mask] *= 1.8
    credit_score_band[default_mask] = rng.choice(
        CREDIT_SCORE_BANDS[:2], size=default_mask.sum(), p=[0.6, 0.4]
    )

    df = pd.DataFrame(
        {
            "periodo": _generate_periodo_series(n_samples, rng),
            "SeriousDlqin2yrs": target.astype(np.int8),
            "RevolvingUtilizationOfUnsecuredLines": revolving_util.astype(np.float32),
            "age": age.astype(np.int16),
            "NumberOfTime30-59DaysPastDueNotWorse": past_due_30_59.astype(np.int16),
            "DebtRatio": debt_ratio.astype(np.float32),
            "MonthlyIncome": monthly_income.astype(np.float32),
            "NumberOfOpenCreditLinesAndLoans": open_credit_lines.astype(np.int16),
            "NumberOfTimes90DaysLate": times_90_late.astype(np.int16),
            "NumberRealEstateLoansOrLines": real_estate_loans.astype(np.int16),
            "NumberOfTime60-89DaysPastDueNotWorse": past_due_60_89.astype(np.int16),
            "NumberOfDependents": n_dependents.astype(np.float32),
            "loan_amount": loan_amount.astype(np.float32),
            "employment_years": employment_years.astype(np.float32),
            "credit_score_band": credit_score_band,
        }
    )

    actual_rate = df["SeriousDlqin2yrs"].mean()
    logger.info("Dataset generated. Shape: %s | Default rate: %.3f", df.shape, actual_rate)
    return df


def main() -> None:
    """CLI entry-point for the generate stage."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate synthetic credit risk dataset")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--output", default=None, help="Override output path")
    args = parser.parse_args()

    params = _load_params(args.params)
    output_path = args.output or params["data"]["raw_path"]

    df = generate_dataset(params)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info("Saved raw dataset to %s", output_path)


if __name__ == "__main__":
    main()
