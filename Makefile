.PHONY: help install install-dev lint format typecheck test coverage \
        generate validate featurize train evaluate pipeline \
        docker-build docker-up docker-down docker-logs \
        drift-report clean

PYTHON := python
PYTEST := pytest
PIP    := pip
DC     := docker compose

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---- Setup -----------------------------------------------------------------

install:  ## Install production dependencies
	$(PIP) install -r requirements.txt

install-dev:  ## Install all dependencies (including dev)
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

# ---- Code quality ----------------------------------------------------------

lint:  ## Run ruff linter
	ruff check src/ scripts/ tests/ api/ app/

format:  ## Auto-format with black
	black src/ scripts/ tests/ api/ app/

typecheck:  ## Run mypy type checks
	mypy src/ api/ --ignore-missing-imports

# ---- Tests -----------------------------------------------------------------

test:  ## Run pytest
	$(PYTEST) tests/ -v

coverage:  ## Run pytest with coverage (>= 80% required)
	$(PYTEST) tests/ -v \
	  --cov=src --cov=api \
	  --cov-report=term-missing \
	  --cov-report=html:reports/htmlcov \
	  --cov-fail-under=80

# ---- DVC pipeline ----------------------------------------------------------

generate:  ## Generate synthetic dataset
	$(PYTHON) scripts/generate_dataset.py --params params.yaml

validate:  ## Run Great Expectations data validation
	$(PYTHON) scripts/validate_data.py --params params.yaml

featurize:  ## Run temporal split / featurize stage
	$(PYTHON) scripts/featurize.py --params params.yaml

train:  ## Train XGBoost + LightGBM with Optuna + MLflow
	$(PYTHON) src/train.py --params params.yaml

evaluate:  ## Evaluate model and write reports/metrics.json
	$(PYTHON) scripts/evaluate_model.py --params params.yaml

pipeline:  ## Run full DVC pipeline
	dvc repro

# ---- Docker ----------------------------------------------------------------

docker-build:  ## Build all Docker images
	$(DC) build

docker-up:  ## Start all services
	$(DC) up -d

docker-down:  ## Stop and remove containers
	$(DC) down

docker-logs:  ## Tail all container logs
	$(DC) logs -f

smoke-test:  ## Run post-deploy smoke test against running API
	$(PYTHON) scripts/smoke_test.py --base-url http://localhost:8000

# ---- Monitoring ------------------------------------------------------------

drift-report:  ## Generate Evidently drift report
	$(PYTHON) monitoring/drift_detector.py \
	  --reference data/processed/train.parquet \
	  --current   data/processed/test.parquet \
	  --output    reports/drift_report.html

# ---- Cleanup ---------------------------------------------------------------

clean:  ## Remove generated data, reports, caches
	rm -rf data/raw data/processed reports/__pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
