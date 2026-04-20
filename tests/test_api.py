"""Integration tests for api/main.py using httpx + TestClient."""

from __future__ import annotations

from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure modules are in sys.modules before unittest.mock.patch resolves them
import src.predict  # noqa: F401
import api.main  # noqa: F401


# ---- helpers ---------------------------------------------------------------

def _make_mock_pipeline():
    """Return a mock sklearn Pipeline that returns plausible predict_proba output."""
    mock = MagicMock()
    mock.predict_proba.return_value = np.array([[0.72, 0.28]])
    return mock


def _make_mock_pipeline_batch(n: int):
    mock = MagicMock()
    mock.predict_proba.return_value = np.column_stack(
        [np.full(n, 0.72), np.full(n, 0.28)]
    )
    return mock


# ---- fixtures --------------------------------------------------------------

@pytest.fixture(scope="module")
def client(sample_payload: dict) -> Generator[TestClient, None, None]:
    """TestClient with model loading patched out."""
    mock_pipeline = _make_mock_pipeline()

    with patch("src.predict.load_model", return_value=mock_pipeline), \
         patch("src.predict.get_model_uri", return_value="models:/mock/1"), \
         patch("api.main.load_model", return_value=mock_pipeline), \
         patch("api.main.get_model_uri", return_value="models:/mock/1"):
        from api.main import app
        with TestClient(app) as c:
            yield c


# ---- /health ---------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_schema(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert "status" in data
        assert data["status"] == "ok"


# ---- /predict --------------------------------------------------------------

class TestPredictEndpoint:
    def test_predict_returns_200(self, client: TestClient, sample_payload: dict) -> None:
        resp = client.post("/predict", json=sample_payload)
        assert resp.status_code == 200

    def test_predict_response_schema(self, client: TestClient, sample_payload: dict) -> None:
        data = client.post("/predict", json=sample_payload).json()
        assert "default_probability" in data
        assert "prediction" in data

    def test_predict_probability_in_range(self, client: TestClient, sample_payload: dict) -> None:
        data = client.post("/predict", json=sample_payload).json()
        assert 0.0 <= data["default_probability"] <= 1.0

    def test_predict_prediction_is_binary(self, client: TestClient, sample_payload: dict) -> None:
        data = client.post("/predict", json=sample_payload).json()
        assert data["prediction"] in (0, 1)

    def test_predict_invalid_credit_band(self, client: TestClient, sample_payload: dict) -> None:
        bad = {**sample_payload, "credit_score_band": "Invalid"}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_negative_age_rejected(self, client: TestClient, sample_payload: dict) -> None:
        bad = {**sample_payload, "age": -1}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_missing_required_field(self, client: TestClient, sample_payload: dict) -> None:
        bad = {k: v for k, v in sample_payload.items() if k != "loan_amount"}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_null_monthly_income_accepted(
        self, client: TestClient, sample_payload: dict
    ) -> None:
        """MonthlyIncome is Optional in the schema."""
        payload = {**sample_payload, "MonthlyIncome": None}
        mock_pipeline = _make_mock_pipeline()
        with patch("api.main.load_model", return_value=mock_pipeline), \
             patch("api.main.get_model_uri", return_value="models:/mock/1"):
            resp = client.post("/predict", json=payload)
        assert resp.status_code == 200


# ---- /predict/batch --------------------------------------------------------

class TestPredictBatchEndpoint:
    def test_batch_returns_200(self, client: TestClient, sample_payload: dict) -> None:
        mock = _make_mock_pipeline_batch(3)
        with patch("api.main.load_model", return_value=mock), \
             patch("api.main.get_model_uri", return_value="models:/mock/1"):
            resp = client.post("/predict/batch", json={"records": [sample_payload] * 3})
        assert resp.status_code == 200

    def test_batch_count_matches_input(self, client: TestClient, sample_payload: dict) -> None:
        n = 5
        mock = _make_mock_pipeline_batch(n)
        with patch("api.main.load_model", return_value=mock), \
             patch("api.main.get_model_uri", return_value="models:/mock/1"):
            data = client.post("/predict/batch", json={"records": [sample_payload] * n}).json()
        assert data["count"] == n
        assert len(data["predictions"]) == n

    def test_batch_empty_records_rejected(self, client: TestClient) -> None:
        resp = client.post("/predict/batch", json={"records": []})
        assert resp.status_code == 422

    def test_batch_over_limit_rejected(self, client: TestClient, sample_payload: dict) -> None:
        """1001 records should return 422 (exceeds BATCH_LIMIT=1000)."""
        import api.main as main_mod

        original = main_mod.BATCH_LIMIT
        main_mod.BATCH_LIMIT = 2
        try:
            resp = client.post("/predict/batch", json={"records": [sample_payload] * 3})
            assert resp.status_code == 422
        finally:
            main_mod.BATCH_LIMIT = original


# ---- /metrics --------------------------------------------------------------

class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_content_type(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert "text/plain" in resp.headers["content-type"]
