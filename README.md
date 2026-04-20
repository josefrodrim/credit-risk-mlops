# Credit Risk MLOps

> End-to-end MLOps platform for consumer credit default prediction.  
> Demonstrates production-grade practices on a 1 M-row synthetic dataset: versioned data pipelines, Bayesian HPO, model registry, REST API, real-time observability, and automated CI/CD.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green)
![MLflow](https://img.shields.io/badge/MLflow-3.11.1-blue?logo=mlflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136.0-teal?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![Jenkins](https://img.shields.io/badge/CI%2FCD-Jenkins-red?logo=jenkins)
![Tests](https://img.shields.io/badge/tests-33%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-%E2%89%A580%25-brightgreen)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [MLOps Architecture](#mlops-architecture)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Model Training & HPO](#model-training--hpo)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [API Reference](#api-reference)
- [Monitoring & Observability](#monitoring--observability)
- [CI/CD Pipeline](#cicd-pipeline)
- [Services](#services)
- [MLOps Practices](#mlops-practices)

---

## Problem Statement

Credit default prediction is a core use case in consumer lending. This project builds a **binary classifier** that estimates the probability that a borrower will become 90+ days delinquent within two years, using the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) schema extended with synthetic features.

The goal is not just to train an accurate model — it is to build the **full production system** around it: reproducible pipelines, API serving, drift detection, and automated deployment.

---

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                  │
│                                                                     │
│  generate_dataset.py ──► Great Expectations ──► featurize.py       │
│  (1M synthetic rows)      (schema validation)    (temporal split)  │
│           │                                            │            │
│           └──────────────── DVC ───────────────────────┘           │
│                      (pipeline versioning)                          │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       TRAINING LAYER                                │
│                                                                     │
│   src/train.py                                                      │
│   ├── LogTransformer + DelinquencyRatioTransformer                  │
│   ├── Optuna HPO  (XGBoost vs LightGBM, 5 trials × 3-fold CV)      │
│   ├── MLflow experiment tracking (params, metrics, artifacts)       │
│   └── Auto-promote best model → Production (Model Registry)        │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       SERVING LAYER                                 │
│                                                                     │
│   FastAPI (/predict, /predict/batch, /health, /metrics)            │
│       │                                                             │
│     Nginx (reverse proxy)                                           │
│       │                                                             │
│   Streamlit Dashboard (scoring + monitoring UI)                     │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY LAYER                              │
│                                                                     │
│   Prometheus ──► Grafana  (request rate, latency p50/p95/p99)      │
│   Evidently AI            (PSI, target drift, score distribution)  │
│   MLflow Model Registry   (version lineage, stage transitions)     │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       CI/CD LAYER                                   │
│                                                                     │
│   Jenkins ──► ruff/black ──► pytest ──► dvc repro ──► AUC gate    │
│           ──► docker build ──► smoke test ──► drift report         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Machine Learning

| Library | Version | Role |
|---------|---------|------|
| XGBoost | 3.2.0 | Gradient boosting model (candidate 1) |
| LightGBM | 4.6.0 | Gradient boosting model (candidate 2) |
| Scikit-learn | 1.8.0 | Preprocessing, pipelines, cross-validation |
| Optuna | 4.8.0 | Bayesian hyperparameter optimization |
| NumPy | 2.4.4 | Numerical computing |
| Pandas | 2.3.3 | Data manipulation |
| SciPy | 1.17.1 | Statistical functions (PSI calculation) |
| SHAP | — | Model explainability (feature importance) |

### MLOps & Experiment Tracking

| Tool | Version | Role |
|------|---------|------|
| MLflow | 3.11.1 | Experiment tracking, model registry, artifact store |
| DVC | — | Data & pipeline versioning, reproducibility |
| Great Expectations | — | Data schema validation & quality gates |
| Evidently AI | 0.7.21 | Data drift detection, PSI, HTML reports |

### Serving & API

| Tool | Version | Role |
|------|---------|------|
| FastAPI | 0.136.0 | REST API framework |
| Uvicorn | — | ASGI server |
| Nginx | — | Reverse proxy, load balancing |
| Streamlit | 1.56.0 | Interactive monitoring dashboard |
| Plotly | 6.1.2 | Interactive charts in dashboard |

### Observability

| Tool | Role |
|------|------|
| Prometheus | Metrics scraping & time-series storage |
| Grafana | Dashboards: request rate, latency p50/p95/p99, score distribution |
| prometheus-fastapi-instrumentator | Auto-instrument FastAPI with Prometheus metrics |

### Infrastructure & CI/CD

| Tool | Role |
|------|------|
| Docker + Docker Compose | Containerisation of all services |
| Jenkins | CI/CD pipeline (11 stages) |
| Ruff | Fast Python linter |
| Black | Code formatter |
| pytest + pytest-cov | Unit & integration tests, coverage gate ≥ 80% |
| Mypy | Static type checking |

---

## Dataset

| Property | Value |
|----------|-------|
| Rows | 1,000,000 synthetic |
| Features | 15 (12 numeric + 1 categorical + period + target) |
| Target | `SeriousDlqin2yrs` (binary, ~7% positive rate) |
| Schema | Give Me Some Credit (Kaggle) + synthetic extensions |
| Period range | 202001 – 202604 (monthly `periodo` column) |
| Generation | Fully reproducible via `scripts/generate_dataset.py` + `params.yaml` seed |

**Three-way temporal split** — no data leakage between train, test, and production monitoring:

| Split | Period | Rows | Default Rate | Purpose |
|-------|--------|------|-------------|---------|
| Train | 202001 – 202312 | 631,361 | 6.97% | HPO + model fitting |
| Test | 202401 – 202512 | 316,067 | 6.99% | Held-out evaluation + AUC gate |
| Monitor | 202601 – 202604 | 52,572 | 7.28% | Simulated production drift detection |

> Temporal splitting is critical in credit risk — random splits leak future information into training.

---

## Feature Engineering

All transformations are implemented as **scikit-learn compatible transformers** in `src/features.py` and composed into a single `Pipeline` object that is serialised and versioned with the model in MLflow.

### Custom Transformers

**`LogTransformer`**  
Applies `log1p` to heavily right-skewed columns (`RevolvingUtilizationOfUnsecuredLines`, `DebtRatio`, `MonthlyIncome`, `loan_amount`). Reduces skew without losing the zero-floor property — important for tree models that use split-point thresholds.

**`DelinquencyRatioTransformer`**  
Creates `delinquency_ratio = (30–59d late + 60–89d late + 90d+ late) / (open_credit_lines + 1)`.  
A single feature that captures relative delinquency burden — higher signal than the raw counts alone.

### Preprocessing Pipeline

```
Raw DataFrame
    │
    ▼
LogTransformer                 ← skew reduction
    │
    ▼
DelinquencyRatioTransformer    ← feature creation
    │
    ▼
ColumnTransformer
    ├── Numeric (13 cols): SimpleImputer(median) → StandardScaler
    └── Categorical (1 col): OrdinalEncoder (credit_score_band: Poor < Fair < Good < Very Good < Exceptional)
    │
    ▼
XGBoostClassifier / LGBMClassifier
```

> The full pipeline is applied **inside the Optuna cross-validation loop** to prevent preprocessing leakage across CV folds.

---

## Model Training & HPO

`src/train.py` trains two candidates independently and promotes the best to Production.

### Optuna Hyperparameter Search

Both XGBoost and LightGBM are tuned with **Bayesian optimization (TPE sampler)**:

| Hyperparameter | XGBoost range | LightGBM range |
|---------------|---------------|----------------|
| `n_estimators` | 100 – 500 | 100 – 500 |
| `max_depth` | 3 – 8 | 3 – 8 |
| `learning_rate` | 0.01 – 0.3 (log) | 0.01 – 0.3 (log) |
| `subsample` | 0.6 – 1.0 | 0.6 – 1.0 |
| `colsample_bytree` | 0.6 – 1.0 | 0.6 – 1.0 |
| `reg_alpha` | 1e-8 – 10 (log) | 1e-8 – 10 (log) |
| `reg_lambda` | 1e-8 – 10 (log) | 1e-8 – 10 (log) |

> Default: `n_trials=5, cv_folds=3` (fast demo, ~3 min). For production-quality HPO: `n_trials=50, cv_folds=5` (~15–20 min). Configure in `params.yaml`.

### MLflow Experiment Tracking

Every trial is logged to MLflow with:
- All hyperparameters
- Per-fold and mean CV AUC
- Test ROC-AUC, PR-AUC, Brier score
- Full serialised pipeline artifact
- Feature importance (SHAP values)

### Auto-Promotion

At the end of training, the best model is automatically transitioned to **Production** in the MLflow Model Registry, archiving all previous Production versions. The API loads the model at startup using `models:/credit-risk-model/Production`.

---

## Key Results

| Metric | Value |
|--------|-------|
| Test ROC-AUC | **0.9944** |
| Test PR-AUC | **0.9467** |
| LightGBM v2 AUC | **0.9945** |
| AUC gate threshold | 0.87 → **PASS** |
| Production monitoring (2026) | All 4 months pass AUC gate |
| Low-risk sample | **0.012%** default probability |
| High-risk sample | **99.96%** default probability |
| Test coverage | ≥ 80% |
| API tests | 14 passing |
| Feature tests | 19 passing |

---

## Project Structure

```
credit-risk-mlops/
├── params.yaml                       # single source of truth for all config
├── dvc.yaml                          # pipeline DAG: generate→validate→featurize→train→evaluate
│
├── src/
│   ├── features.py                   # LogTransformer, DelinquencyRatioTransformer, build_full_pipeline
│   ├── train.py                      # Optuna HPO + MLflow logging + auto-promote to Production
│   ├── evaluate.py                   # AUC gate, metrics output
│   └── predict.py                    # batch prediction helper (used by API)
│
├── api/
│   └── main.py                       # FastAPI: /predict, /predict/batch, /health, /metrics
│
├── app/
│   └── streamlit_app.py              # 3-page dashboard: scoring, batch, monitoring
│
├── scripts/
│   ├── generate_dataset.py           # reproducible 1M-row synthetic dataset
│   ├── featurize.py                  # temporal split → train/test/monitor parquets
│   ├── validate_data.py              # Great Expectations data quality gate
│   ├── evaluate_model.py             # loads Production model, writes reports/metrics.json
│   ├── check_auc_gate.py             # CI gate: fails if AUC < threshold
│   └── smoke_test.py                 # post-deploy HTTP health + prediction check
│
├── monitoring/
│   ├── drift_detector.py             # Evidently AI: PSI, target drift, score distribution
│   ├── prometheus.yml                # scrape config
│   └── grafana/                      # provisioned datasource + dashboard JSON
│
├── notebooks/
│   ├── 01_eda.ipynb                  # shape, nulls, temporal split, correlations
│   ├── 02_feature_engineering.ipynb  # log transforms, PSI, ordinal encoding
│   ├── 03_model_training.ipynb       # Optuna HPO, XGB vs LGBM comparison
│   ├── 04_model_evaluation.ipynb     # AUC, PR curve, SHAP, calibration, monthly AUC
│   └── 05_monitoring.ipynb           # target drift, PSI per feature, score drift 2026
│
├── tests/
│   ├── conftest.py                   # shared fixtures (synthetic DataFrames, fitted pipelines)
│   ├── test_features.py              # 19 unit tests for transformers and pipeline
│   └── test_api.py                   # 14 integration tests for all API endpoints
│
├── docker/
│   ├── Dockerfile.api                # python:3.11-slim + libgomp1
│   ├── Dockerfile.train
│   ├── Dockerfile.streamlit
│   ├── Dockerfile.prometheus         # embeds prometheus.yml (no bind mount)
│   ├── Dockerfile.nginx              # embeds nginx.conf (no bind mount)
│   └── Dockerfile.grafana            # embeds provisioning + dashboard JSON
│
├── docker-compose.yml
├── Jenkinsfile                       # 11-stage CI/CD pipeline
├── Makefile                          # developer shortcuts
├── setup.py
├── requirements.txt                  # runtime (versions pinned to match training env)
├── requirements-dev.txt              # testing + linting + evidently
└── requirements-notebooks.txt        # Jupyter exploration
```

---

## Quickstart

**Prerequisites:** Docker Desktop, Python 3.11+, ~6 GB disk, ~4 GB RAM.

### 1. Clone and build

```bash
git clone https://github.com/josefrodrim/credit-risk-mlops.git
cd credit-risk-mlops

# Build all Docker images
docker compose build
```

### 2. Start infrastructure

```bash
docker compose up -d mlflow prometheus grafana
```

Wait ~20 seconds for MLflow to become healthy (`docker compose ps`).

### 3. Generate data and train

```bash
# Create venv outside the repo (path with colon breaks python -m venv .venv)
python3 -m venv ~/.venvs/credit-risk-mlops
source ~/.venvs/credit-risk-mlops/bin/activate
pip install -r requirements.txt

# Generate 1M-row dataset and featurize
python scripts/generate_dataset.py --params params.yaml
python scripts/validate_data.py   --params params.yaml
python scripts/featurize.py       --params params.yaml

# Train (logs to MLflow, auto-promotes best model to Production)
PYTHONPATH=$(pwd) \
MLFLOW_TRACKING_URI=http://localhost:5001 \
python src/train.py --params params.yaml
```

### 4. Start API

```bash
docker compose up -d api nginx
```

The API loads the Production model from MLflow on startup.  
Test it: `curl http://localhost:8000/health`

### 5. (Optional) Streamlit dashboard

```bash
docker compose --profile demo up -d streamlit
# → http://localhost:8501
```

### Run everything with DVC

```bash
dvc repro     # generate → validate → featurize → train → evaluate
dvc dag       # view the pipeline DAG
```

---

## API Reference

The inference service runs on **FastAPI** with auto-generated OpenAPI docs at `http://localhost:8000/docs`.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Score a single application |
| `POST` | `/predict/batch` | Score up to 1,000 applications |
| `GET` | `/health` | Liveness check |
| `GET` | `/metrics` | Prometheus exposition format |

### Request Schema (`/predict`)

```json
{
  "RevolvingUtilizationOfUnsecuredLines": 0.05,
  "age": 52,
  "NumberOfTime30-59DaysPastDueNotWorse": 0,
  "DebtRatio": 0.15,
  "MonthlyIncome": 7500,
  "NumberOfOpenCreditLinesAndLoans": 9,
  "NumberOfTimes90DaysLate": 0,
  "NumberRealEstateLoansOrLines": 1,
  "NumberOfTime60-89DaysPastDueNotWorse": 0,
  "NumberOfDependents": 1,
  "loan_amount": 12000,
  "employment_years": 15,
  "credit_score_band": "Good"
}
```

`MonthlyIncome` and `NumberOfDependents` accept `null` (imputed at inference time).  
`credit_score_band` must be one of: `Poor`, `Fair`, `Good`, `Very Good`, `Exceptional`.

### Response

```json
{
  "default_probability": 0.00012,
  "prediction": 0,
  "model_version": "2"
}
```

### Single-application example

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "RevolvingUtilizationOfUnsecuredLines": 0.05,
    "age": 52,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.15,
    "MonthlyIncome": 7500,
    "NumberOfOpenCreditLinesAndLoans": 9,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1,
    "loan_amount": 12000,
    "employment_years": 15,
    "credit_score_band": "Good"
  }' | python -m json.tool
```

### Batch example

```bash
curl -s -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"records": [<record_1>, <record_2>]}' | python -m json.tool
```

---

## Monitoring & Observability

### Streamlit Dashboard (`http://localhost:8501`)

Three pages:

**Single Application** — interactive scoring form with real-time probability output.

**Batch Scoring** — CSV upload, returns predictions with download.

**Monitoring Dashboard:**
- Model metadata (version, source experiment, row counts, default-rate delta)
- Monthly AUC & Gini bar chart for 2026 production period (PASS/FAIL per month)
- Target drift — default rate per period with ±2 pp tolerance band
- PSI per feature — horizontal bar chart, color-coded (green < 0.10, orange 0.10–0.20, red > 0.20)
- Score distribution — density overlay: test reference vs monitor 2026

### Grafana (`http://localhost:3000`)

Pre-provisioned dashboard displays:
- Inference request rate (req/s)
- Latency percentiles (p50, p95, p99)
- Prediction score histogram
- HTTP error rate

### Evidently Drift Report

Generates a standalone HTML report comparing training vs production distributions:

```bash
python monitoring/drift_detector.py \
  --reference data/processed/train.parquet \
  --current   data/processed/test.parquet \
  --output    reports/drift_report.html
```

Reports on: dataset drift (per-feature), target drift, prediction drift, data quality.

### Prometheus (`http://localhost:9090`)

FastAPI is auto-instrumented via `prometheus-fastapi-instrumentator`. Metrics scraped every 15s.

---

## CI/CD Pipeline

Jenkins pipeline defined in `Jenkinsfile` — 11 stages, 90-minute timeout.

| Stage | Tool | Gate |
|-------|------|------|
| **Checkout** | git | — |
| **Setup** | pip | — |
| **Lint & Format Check** | ruff, black | Fails on any lint error |
| **Unit Tests** | pytest, pytest-cov | Coverage ≥ 80%, all tests pass |
| **Data Validation** | Great Expectations | Schema + quality checks |
| **DVC Reproduce** | dvc repro | Full pipeline: generate → featurize → train → evaluate |
| **AUC Gate** | custom script | Fails if Test AUC < 0.87 |
| **Build API Image** | docker build | — |
| **Deploy (Staging)** | docker compose | Starts api, mlflow, prometheus, grafana |
| **Smoke Test** | custom script | `/health` + `/predict` must return 200 |
| **Drift Detection** | Evidently AI | HTML report archived as build artifact |

### Developer shortcuts (Makefile)

```bash
make lint          # ruff check
make format        # black auto-format
make typecheck     # mypy
make test          # pytest
make coverage      # pytest + coverage report
make train         # python src/train.py
make pipeline      # dvc repro
make docker-build  # docker compose build
make docker-up     # docker compose up -d
make drift-report  # generate Evidently report
```

---

## Services

| Service | URL | Credentials |
|---------|-----|-------------|
| MLflow UI | http://localhost:5001 | — |
| FastAPI | http://localhost:8000 | — |
| API docs (Swagger) | http://localhost:8000/docs | — |
| Nginx | http://localhost:80 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | Ver `GRAFANA_ADMIN_PASSWORD` en `.env` |
| Streamlit | http://localhost:8501 | `--profile demo` required |

---

## MLOps Practices

This project demonstrates the following production MLOps practices:

| Practice | Implementation |
|----------|---------------|
| **Reproducibility** | DVC pipeline + `params.yaml` as single config source + pinned dependency versions |
| **Data versioning** | DVC tracks raw and processed parquets as pipeline outputs |
| **Data quality gates** | Great Expectations schema suite runs before featurization |
| **Temporal validation** | Three-way time-based split prevents data leakage |
| **Experiment tracking** | MLflow logs every HPO trial: params, metrics, artifacts |
| **Model registry** | MLflow stages: None → Staging → Production with version lineage |
| **Auto-promotion** | Best model promoted to Production at end of training run |
| **Serialized pipeline** | Full preprocessing + model saved as one artifact — no training/serving skew |
| **AUC gate** | Hard CI gate: pipeline fails if model doesn't meet threshold |
| **Containerisation** | All services in Docker; config embedded at build time (no bind-mount fragility) |
| **REST API** | FastAPI with Pydantic validation, batch endpoint, Prometheus metrics |
| **Observability** | Prometheus + Grafana for real-time latency and throughput |
| **Drift detection** | Evidently AI computes PSI, target drift, and score distribution shift |
| **CI/CD** | Jenkins 11-stage pipeline: lint → test → train → gate → build → deploy → smoke |
| **Test coverage** | 33 tests, ≥ 80% coverage gate enforced in CI |

---

## Environment Notes

- The repo path contains a trailing colon (`credit-risk-mlops:`) which prevents `python -m venv .venv` inside the directory. Always create the venv outside: `python3 -m venv ~/.venvs/credit-risk-mlops`.
- MLflow runs on port **5001** — macOS AirPlay/ControlCenter occupies 5000.
- Docker config files (Prometheus, Nginx, Grafana) are embedded via `COPY` in their Dockerfiles rather than bind-mounted, to avoid Docker Desktop's path-parsing bug with colons.
- Pickle serialisation is version-specific: the API container's `requirements.txt` pins the exact same numpy/scipy/sklearn versions used during training to guarantee model compatibility.

---

## License

MIT
