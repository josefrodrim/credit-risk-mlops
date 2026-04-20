"""Unit tests for src/features.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    DelinquencyRatioTransformer,
    LogTransformer,
    build_full_pipeline,
    build_preprocessor,
    get_feature_names_out,
    prepare_features,
)


class TestLogTransformer:
    def test_transform_reduces_skew(self, raw_df: pd.DataFrame) -> None:
        X, _ = prepare_features(raw_df)
        t = LogTransformer()
        X_out = t.fit_transform(X)
        for col in LogTransformer.COLS_TO_LOG:
            if col in X.columns:
                assert X_out[col].skew() < X[col].skew(), f"{col} skew did not decrease"

    def test_no_nan_introduced(self, raw_df: pd.DataFrame) -> None:
        """Transform must not add new NaN values beyond what was already present."""
        X, _ = prepare_features(raw_df)
        t = LogTransformer()
        X_out = t.fit_transform(X)
        for col in LogTransformer.COLS_TO_LOG:
            if col in X.columns:
                assert X_out[col].isna().sum() <= X[col].isna().sum(), (
                    f"LogTransformer introduced new NaN values in {col}"
                )

    def test_fit_returns_self(self, raw_df: pd.DataFrame) -> None:
        X, _ = prepare_features(raw_df)
        t = LogTransformer()
        assert t.fit(X) is t


class TestDelinquencyRatioTransformer:
    def test_adds_column(self, raw_df: pd.DataFrame) -> None:
        X, _ = prepare_features(raw_df)
        t = DelinquencyRatioTransformer()
        X_out = t.fit_transform(X)
        assert "delinquency_ratio" in X_out.columns

    def test_ratio_non_negative(self, raw_df: pd.DataFrame) -> None:
        X, _ = prepare_features(raw_df)
        t = DelinquencyRatioTransformer()
        X_out = t.fit_transform(X)
        assert (X_out["delinquency_ratio"] >= 0).all()

    def test_fit_returns_self(self, raw_df: pd.DataFrame) -> None:
        X, _ = prepare_features(raw_df)
        t = DelinquencyRatioTransformer()
        assert t.fit(X) is t


class TestBuildPreprocessor:
    def test_output_is_array(self, X_y) -> None:
        X, _ = X_y
        prep = build_preprocessor()
        out = prep.fit_transform(X)
        assert hasattr(out, "shape"), "Expected array-like output"

    def test_no_nan_in_output(self, X_y) -> None:
        X, _ = X_y
        prep = build_preprocessor()
        out = prep.fit_transform(X)
        assert not np.isnan(out).any(), "NaN values found in preprocessor output"

    def test_feature_names_out_length(self, fitted_preprocessor) -> None:
        names = get_feature_names_out(fitted_preprocessor)
        # Expect numeric + categorical features (delinquency_ratio added after preprocessor)
        assert len(names) == len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)

    def test_custom_columns(self, X_y) -> None:
        X, _ = X_y
        subset_num = NUMERIC_FEATURES[:3]
        prep = build_preprocessor(numeric_features=subset_num, categorical_features=[])
        out = prep.fit_transform(X)
        assert out.shape[1] == len(subset_num)


class TestBuildFullPipeline:
    def test_pipeline_has_required_steps(self, fitted_pipeline: Pipeline) -> None:
        step_names = [name for name, _ in fitted_pipeline.steps]
        assert "log_transform" in step_names
        assert "delinquency" in step_names
        assert "preprocessor" in step_names
        assert "model" in step_names

    def test_predict_proba_shape(self, fitted_pipeline: Pipeline, X_y) -> None:
        X, _ = X_y
        probas = fitted_pipeline.predict_proba(X)
        assert probas.shape == (len(X), 2)

    def test_probas_in_range(self, fitted_pipeline: Pipeline, X_y) -> None:
        X, _ = X_y
        probas = fitted_pipeline.predict_proba(X)[:, 1]
        assert (probas >= 0).all() and (probas <= 1).all()

    def test_predict_returns_binary(self, fitted_pipeline: Pipeline, X_y) -> None:
        X, _ = X_y
        preds = fitted_pipeline.predict(X)
        assert set(preds).issubset({0, 1})


class TestPrepareFeatures:
    def test_returns_correct_shape(self, raw_df: pd.DataFrame) -> None:
        X, y = prepare_features(raw_df)
        assert len(X) == len(raw_df)
        assert len(y) == len(raw_df)
        assert set(X.columns) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)

    def test_target_is_binary(self, raw_df: pd.DataFrame) -> None:
        _, y = prepare_features(raw_df)
        assert set(y.unique()).issubset({0, 1})

    def test_missing_column_raises(self, raw_df: pd.DataFrame) -> None:
        bad_df = raw_df.drop(columns=["age"])
        with pytest.raises(ValueError, match="Missing feature columns"):
            prepare_features(bad_df)
