"""Feature engineering pipeline for credit-risk-mlops.

Builds a reusable sklearn Pipeline that:
  - Imputes missing values in numeric/categorical columns
  - Scales numeric features
  - Ordinal-encodes credit_score_band (risk-ordered)
  - Adds derived interaction features
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)

CREDIT_SCORE_ORDER = [["Poor", "Fair", "Good", "Very Good", "Exceptional"]]

NUMERIC_FEATURES: List[str] = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "loan_amount",
    "employment_years",
    "delinquency_ratio",  # derived by DelinquencyRatioTransformer inside the pipeline
]

CATEGORICAL_FEATURES: List[str] = ["credit_score_band"]

TARGET_COL = "SeriousDlqin2yrs"
PERIOD_COL = "periodo"


def load_feature_config(params_path: str = "params.yaml") -> dict:
    """Return the features section of params.yaml."""
    with open(params_path, "r") as fh:
        return yaml.safe_load(fh)["features"]


class DelinquencyRatioTransformer(BaseEstimator, TransformerMixin):
    """Adds delinquency_ratio = total past-due events / (open_lines + 1)."""

    def fit(self, X: pd.DataFrame, y=None) -> "DelinquencyRatioTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        past_due_cols = [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
            "NumberOfTime60-89DaysPastDueNotWorse",
        ]
        existing = [c for c in past_due_cols if c in df.columns]
        if existing:
            df["delinquency_ratio"] = df[existing].sum(axis=1) / (
                df.get("NumberOfOpenCreditLinesAndLoans", pd.Series(0, index=df.index)) + 1
            )
        return df


class LogTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p to right-skewed monetary/ratio columns."""

    COLS_TO_LOG = ["MonthlyIncome", "loan_amount", "DebtRatio"]

    def fit(self, X: pd.DataFrame, y=None) -> "LogTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in self.COLS_TO_LOG:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))
        return df


def build_numeric_pipeline() -> Pipeline:
    """Return impute → scale pipeline for numeric features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_categorical_pipeline() -> Pipeline:
    """Return impute → ordinal-encode pipeline for categorical features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=CREDIT_SCORE_ORDER,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )


def build_preprocessor(
    numeric_features: List[str] | None = None,
    categorical_features: List[str] | None = None,
) -> ColumnTransformer:
    """
    Build the full ColumnTransformer preprocessor.

    Args:
        numeric_features: Override default numeric column list.
        categorical_features: Override default categorical column list.

    Returns:
        Fitted-ready ColumnTransformer.
    """
    num_cols = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    cat_cols = categorical_features if categorical_features is not None else CATEGORICAL_FEATURES

    return ColumnTransformer(
        transformers=[
            ("num", build_numeric_pipeline(), num_cols),
            ("cat", build_categorical_pipeline(), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_full_pipeline(
    model,
    numeric_features: List[str] | None = None,
    categorical_features: List[str] | None = None,
) -> Pipeline:
    """
    Wrap preprocessor + model into a single sklearn Pipeline.

    Args:
        model: Unfitted sklearn-compatible estimator.
        numeric_features: Numeric column names.
        categorical_features: Categorical column names.

    Returns:
        End-to-end Pipeline (preprocess → model).
    """
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    return Pipeline(
        steps=[
            ("log_transform", LogTransformer()),
            ("delinquency", DelinquencyRatioTransformer()),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def get_feature_names_out(preprocessor: ColumnTransformer) -> List[str]:
    """Extract output feature names from a fitted ColumnTransformer."""
    return list(preprocessor.get_feature_names_out())


def prepare_features(
    df: pd.DataFrame,
    params_path: str = "params.yaml",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a raw dataframe into feature matrix X and target y.

    Args:
        df: Raw dataframe including target and period columns.
        params_path: Path to params.yaml.

    Returns:
        Tuple of (X, y).
    """
    cfg = load_feature_config(params_path)
    target_col: str = cfg["target"]
    feature_cols = cfg["numeric"] + cfg["categorical"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in dataframe: {missing}")

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    logger.debug("prepare_features: X=%s y=%s positives=%.3f", X.shape, y.shape, y.mean())
    return X, y
