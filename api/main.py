"""FastAPI inference service for credit-risk-mlops.

Endpoints:
  POST /predict        — single observation
  POST /predict/batch  — up to 1000 observations
  GET  /health         — liveness probe
  GET  /metrics        — Prometheus metrics
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field, field_validator

from src.predict import get_model_uri, load_model, predict_batch, predict_single

logger = logging.getLogger(__name__)

# ---- Prometheus metrics ----------------------------------------------------

REQUEST_COUNT = Counter(
    "api_request_total",
    "Total API requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
PREDICTION_SCORE = Histogram(
    "prediction_default_probability",
    "Distribution of predicted default probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ---- Config ----------------------------------------------------------------

def _load_config() -> dict:
    params_path = os.getenv("PARAMS_PATH", "params.yaml")
    with open(params_path, "r") as fh:
        return yaml.safe_load(fh)


_config = _load_config()
BATCH_LIMIT: int = int(os.getenv("BATCH_LIMIT", _config["api"]["batch_limit"]))

# ---- Schemas ---------------------------------------------------------------

class CreditFeatures(BaseModel):
    """Single observation feature schema (mirrors Give Me Some Credit + extras)."""

    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0.0, description="Revolving utilization ratio")
    age: int = Field(..., ge=18, le=120)
    NumberOfTime30_59DaysPastDueNotWorse: float = Field(..., alias="NumberOfTime30-59DaysPastDueNotWorse", ge=0)
    DebtRatio: float = Field(..., ge=0.0)
    MonthlyIncome: Optional[float] = Field(None, ge=0.0)
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0)
    NumberOfTimes90DaysLate: int = Field(..., ge=0)
    NumberRealEstateLoansOrLines: int = Field(..., ge=0)
    NumberOfTime60_89DaysPastDueNotWorse: float = Field(..., alias="NumberOfTime60-89DaysPastDueNotWorse", ge=0)
    NumberOfDependents: Optional[float] = Field(None, ge=0)
    loan_amount: float = Field(..., gt=0)
    employment_years: float = Field(..., ge=0)
    credit_score_band: str = Field(..., description="One of: Poor, Fair, Good, Very Good, Exceptional")

    model_config = {"populate_by_name": True}

    @field_validator("credit_score_band")
    @classmethod
    def validate_credit_band(cls, v: str) -> str:
        valid = {"Poor", "Fair", "Good", "Very Good", "Exceptional"}
        if v not in valid:
            raise ValueError(f"credit_score_band must be one of {sorted(valid)}, got '{v}'")
        return v

    def to_feature_dict(self) -> dict:
        """Return dict with original hyphenated column names for the pipeline."""
        d = self.model_dump(by_alias=True)
        return d


class PredictionResponse(BaseModel):
    default_probability: float
    prediction: int


class BatchRequest(BaseModel):
    records: List[CreditFeatures] = Field(..., min_length=1)


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---- App lifecycle ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup; release at shutdown."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    model_uri = get_model_uri()
    try:
        load_model(model_uri)
        logger.info("Model loaded at startup from %s", model_uri)
    except Exception as exc:
        logger.warning("Could not pre-load model at startup: %s", exc)
    yield


app = FastAPI(
    title="Credit Risk MLOps API",
    description="Binary classification for credit default probability.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---- Middleware ------------------------------------------------------------

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    endpoint = request.url.path
    start = time.perf_counter()
    try:
        response = await call_next(request)
        REQUEST_COUNT.labels(endpoint=endpoint, status=response.status_code).inc()
        return response
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint=endpoint, status=500).inc()
        raise exc
    finally:
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)


# ---- Endpoints -------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe."""
    model_loaded = False
    try:
        model_uri = get_model_uri()
        load_model(model_uri)
        model_loaded = True
    except Exception:
        pass
    return HealthResponse(status="ok", model_loaded=model_loaded)


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
def metrics():
    """Prometheus metrics scrape endpoint."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(features: CreditFeatures) -> PredictionResponse:
    """
    Score a single credit application.

    Returns the predicted default probability and binary prediction.
    """
    try:
        model_uri = get_model_uri()
        pipeline = load_model(model_uri)
    except Exception as exc:
        logger.error("Model unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Model not available") from exc

    try:
        result = predict_single(pipeline, features.to_feature_dict())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail="Internal prediction error") from exc

    PREDICTION_SCORE.observe(result["default_probability"])
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchResponse, tags=["inference"])
def predict_batch_endpoint(request: BatchRequest) -> BatchResponse:
    """
    Score a batch of credit applications (max 1 000 per request).

    Returns a list of predictions in the same order as the input records.
    """
    if len(request.records) > BATCH_LIMIT:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {len(request.records)} exceeds limit {BATCH_LIMIT}.",
        )

    try:
        model_uri = get_model_uri()
        pipeline = load_model(model_uri)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Model not available") from exc

    records_dicts = [r.to_feature_dict() for r in request.records]

    try:
        results = predict_batch(pipeline, records_dicts, batch_limit=BATCH_LIMIT)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail="Internal prediction error") from exc

    for r in results:
        PREDICTION_SCORE.observe(r["default_probability"])

    return BatchResponse(
        predictions=[PredictionResponse(**r) for r in results],
        count=len(results),
    )
