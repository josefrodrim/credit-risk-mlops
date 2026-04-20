"""Inference module: load model from MLflow registry and score observations."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List, Union

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def load_model(model_uri: str) -> Pipeline:
    """
    Load and cache a fitted sklearn Pipeline from MLflow.

    Args:
        model_uri: MLflow model URI, e.g. 'models:/credit-risk-model/Production'.

    Returns:
        Fitted sklearn Pipeline.

    Raises:
        mlflow.exceptions.MlflowException: If the model URI cannot be resolved.
    """
    logger.info("Loading model from %s", model_uri)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    pipeline: Pipeline = mlflow.sklearn.load_model(model_uri)
    logger.info("Model loaded successfully")
    return pipeline


def predict_proba(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Return positive-class probabilities for each row in X.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X: Feature dataframe with the expected schema.

    Returns:
        1-D array of probabilities in [0, 1].

    Raises:
        ValueError: If X contains unexpected shape or all-null columns.
    """
    if X.empty:
        raise ValueError("Input dataframe is empty.")

    proba: np.ndarray = pipeline.predict_proba(X)[:, 1]
    logger.debug("Scored %d rows — mean_proba=%.4f", len(X), proba.mean())
    return proba


def predict_single(
    pipeline: Pipeline,
    features: dict,
) -> dict:
    """
    Score a single observation supplied as a plain dict.

    Args:
        pipeline: Fitted sklearn Pipeline.
        features: Dict mapping feature name → value.

    Returns:
        Dict with keys ``default_probability`` (float) and ``prediction`` (int).
    """
    X = pd.DataFrame([features])
    proba = predict_proba(pipeline, X)
    return {
        "default_probability": float(proba[0]),
        "prediction": int(proba[0] >= 0.5),
    }


def predict_batch(
    pipeline: Pipeline,
    records: List[dict],
    batch_limit: int = 1000,
) -> List[dict]:
    """
    Score a list of observation dicts.

    Args:
        pipeline: Fitted sklearn Pipeline.
        records: List of feature dicts.
        batch_limit: Maximum allowed batch size.

    Returns:
        List of result dicts with ``default_probability`` and ``prediction``.

    Raises:
        ValueError: If len(records) > batch_limit.
    """
    if len(records) > batch_limit:
        raise ValueError(
            f"Batch size {len(records)} exceeds limit {batch_limit}. "
            "Split the request into smaller chunks."
        )

    X = pd.DataFrame(records)
    probas = predict_proba(pipeline, X)

    return [
        {
            "default_probability": float(p),
            "prediction": int(p >= 0.5),
        }
        for p in probas
    ]


def get_model_uri(params_path: str = "params.yaml") -> str:
    """Resolve model URI from env var override or params.yaml."""
    env_uri = os.getenv("MODEL_URI")
    if env_uri:
        return env_uri
    return _load_params(params_path)["api"]["model_uri"]
