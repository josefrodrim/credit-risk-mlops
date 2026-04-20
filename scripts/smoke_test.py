"""Post-deployment smoke test: verify the API is alive and returns valid predictions."""

from __future__ import annotations

import argparse
import logging
import sys
import time

import requests
import yaml

logger = logging.getLogger(__name__)

SAMPLE_PAYLOAD = {
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


def _load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as fh:
        return yaml.safe_load(fh)


def wait_for_api(base_url: str, max_retries: int = 10, delay: float = 5.0) -> None:
    """
    Poll /health until the API reports ready or retries are exhausted.

    Args:
        base_url: Base URL of the API (e.g. http://localhost:8000).
        max_retries: Maximum number of poll attempts.
        delay: Seconds to sleep between retries.

    Raises:
        RuntimeError: If the API does not become healthy within max_retries.
    """
    url = f"{base_url}/health"
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                logger.info("API is healthy (attempt %d)", attempt)
                return
            logger.warning("Health check attempt %d — status=%d", attempt, resp.status_code)
        except requests.exceptions.ConnectionError:
            logger.warning("Connection refused on attempt %d — retrying in %.0fs", attempt, delay)
        time.sleep(delay)

    raise RuntimeError(f"API not healthy after {max_retries} attempts at {url}")


def check_single_predict(base_url: str) -> None:
    """
    POST a single prediction and assert the response shape.

    Raises:
        AssertionError: If the response is malformed.
        requests.HTTPError: If the API returns a non-2xx status.
    """
    url = f"{base_url}/predict"
    logger.info("Testing POST %s", url)
    resp = requests.post(url, json=SAMPLE_PAYLOAD, timeout=10)
    resp.raise_for_status()

    data = resp.json()
    assert "default_probability" in data, f"Missing 'default_probability' in response: {data}"
    assert "prediction" in data, f"Missing 'prediction' in response: {data}"
    assert 0.0 <= data["default_probability"] <= 1.0, (
        f"default_probability out of range: {data['default_probability']}"
    )
    assert data["prediction"] in (0, 1), f"prediction must be 0 or 1, got: {data['prediction']}"
    logger.info("Single predict OK — probability=%.4f prediction=%d", data["default_probability"], data["prediction"])


def check_batch_predict(base_url: str, batch_size: int = 5) -> None:
    """
    POST a small batch and assert the response shape.

    Raises:
        AssertionError: If the response is malformed.
        requests.HTTPError: If the API returns a non-2xx status.
    """
    url = f"{base_url}/predict/batch"
    logger.info("Testing POST %s with batch_size=%d", url, batch_size)
    payload = {"records": [SAMPLE_PAYLOAD] * batch_size}
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()

    data = resp.json()
    assert "predictions" in data, f"Missing 'predictions' in batch response: {data}"
    assert data["count"] == batch_size, f"Expected count={batch_size}, got {data['count']}"
    assert len(data["predictions"]) == batch_size

    logger.info("Batch predict OK — %d predictions returned", batch_size)


def main() -> None:
    """CLI entry-point for smoke test (called post-deploy in Jenkins)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Post-deployment smoke test")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--base-url", default=None, help="API base URL (overrides params.yaml)")
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument("--delay", type=float, default=5.0)
    args = parser.parse_args()

    params = _load_params(args.params)
    host = params["api"]["host"]
    port = params["api"]["port"]
    base_url = args.base_url or f"http://{host}:{port}"

    logger.info("Smoke test target: %s", base_url)

    try:
        wait_for_api(base_url, max_retries=args.retries, delay=args.delay)
        check_single_predict(base_url)
        check_batch_predict(base_url)
        logger.info("All smoke tests PASSED")
    except (AssertionError, RuntimeError, requests.HTTPError) as exc:
        logger.error("Smoke test FAILED: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
