"""Shared pytest fixtures for credit-risk-mlops test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_full_pipeline,
    build_preprocessor,
)

# ---- constants -------------------------------------------------------------

N_ROWS = 200
RANDOM_SEED = 42
CREDIT_SCORE_BANDS = ["Poor", "Fair", "Good", "Very Good", "Exceptional"]


# ---- raw dataframe fixtures ------------------------------------------------

@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Minimal synthetic dataframe matching the full feature schema."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = N_ROWS
    return pd.DataFrame(
        {
            "periodo": rng.choice(
                [y * 100 + m for y in range(2020, 2027) for m in range(1, 13) if y * 100 + m <= 202604],
                size=n,
            ),
            "SeriousDlqin2yrs": rng.binomial(1, 0.07, size=n).astype(np.int8),
            "RevolvingUtilizationOfUnsecuredLines": rng.uniform(0, 1, size=n).astype(np.float32),
            "age": rng.integers(18, 90, size=n).astype(np.int16),
            "NumberOfTime30-59DaysPastDueNotWorse": rng.integers(0, 5, size=n).astype(np.int16),
            "DebtRatio": rng.uniform(0, 5, size=n).astype(np.float32),
            "MonthlyIncome": np.where(
                rng.random(n) < 0.05, np.nan, rng.lognormal(8.5, 0.8, size=n)
            ).astype(np.float32),
            "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 15, size=n).astype(np.int16),
            "NumberOfTimes90DaysLate": rng.integers(0, 3, size=n).astype(np.int16),
            "NumberRealEstateLoansOrLines": rng.integers(0, 4, size=n).astype(np.int16),
            "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 3, size=n).astype(np.int16),
            "NumberOfDependents": np.where(
                rng.random(n) < 0.04, np.nan, rng.integers(0, 6, size=n)
            ).astype(np.float32),
            "loan_amount": rng.lognormal(10.5, 1.0, size=n).astype(np.float32),
            "employment_years": rng.exponential(7, size=n).astype(np.float32),
            "credit_score_band": rng.choice(CREDIT_SCORE_BANDS, size=n),
        }
    )


@pytest.fixture(scope="session")
def X_y(raw_df: pd.DataFrame):
    """Return (X, y) split from raw_df."""
    from src.features import prepare_features

    return prepare_features(raw_df)


@pytest.fixture(scope="session")
def fitted_preprocessor(X_y):
    """Fitted ColumnTransformer (no model)."""
    X, _ = X_y
    prep = build_preprocessor()
    prep.fit(X)
    return prep


@pytest.fixture(scope="session")
def fitted_pipeline(X_y):
    """End-to-end pipeline with a lightweight LogisticRegression for speed."""
    X, y = X_y
    model = LogisticRegression(max_iter=200, random_state=RANDOM_SEED, class_weight="balanced")
    pipeline = build_full_pipeline(model)
    pipeline.fit(X, y)
    return pipeline


# ---- API fixtures ----------------------------------------------------------

@pytest.fixture(scope="session")
def sample_payload() -> dict:
    """Single valid prediction payload matching the API schema."""
    return {
        "RevolvingUtilizationOfUnsecuredLines": 0.35,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.25,
        "MonthlyIncome": 5000.0,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2.0,
        "loan_amount": 15000.0,
        "employment_years": 10.0,
        "credit_score_band": "Good",
    }
