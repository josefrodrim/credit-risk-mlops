# credit-risk-mlops — Project Context

## What this project is

End-to-end MLOps portfolio project for credit default prediction.
Dataset: 1 M synthetic rows (202001–202604), target `SeriousDlqin2yrs` (~7% default rate).

## Stack

- **Python 3.14.3** (system) — venv at `~/.venvs/credit-risk-mlops` uses this
- XGBoost 3.2.0, LightGBM 4.6.0, Scikit-learn 1.8.0, numpy 2.4.4, scipy 1.17.1
- MLflow 3.11.1 (tracking server in Docker on port **5001** — AirPlay owns 5000)
- FastAPI 0.136.0 + Uvicorn, Streamlit 1.56.0, Optuna 4.8.0
- Docker Compose (all services containerised), Jenkins (CI/CD)
- DVC pipeline for data and model versioning

## Directory layout

```text
credit-risk-mlops/
├── params.yaml                       # single source of truth for all config
├── dvc.yaml                          # pipeline: generate→validate→featurize→train→evaluate
├── src/
│   ├── features.py                   # LogTransformer, DelinquencyRatioTransformer, build_full_pipeline
│   ├── train.py                      # Optuna HPO + MLflow logging + auto-promote to Production
│   ├── evaluate.py                   # AUC gate, saves metrics to metrics/
│   └── predict.py                    # batch prediction helper
├── api/main.py                       # FastAPI: /predict, /health, /predict/batch, /metrics
├── app/streamlit_app.py              # Streamlit UI (← PENDING: Gini/monitoring dashboard)
├── scripts/
│   ├── generate_dataset.py           # synthetic 1M-row dataset
│   └── featurize.py                  # three-way temporal split → parquets
├── notebooks/
│   ├── 01_eda.ipynb                  # shape, nulls, temporal split, correlations
│   ├── 02_feature_engineering.ipynb  # log transforms, PSI, ordinal encoding check
│   ├── 03_model_training.ipynb       # Optuna HPO (10 trials), XGB vs LGBM comparison
│   ├── 04_model_evaluation.ipynb     # AUC, PR, SHAP, calibration, monthly AUC
│   └── 05_monitoring.ipynb           # target drift, PSI per feature, score drift, monthly AUC 2026
├── docker/
│   ├── Dockerfile.api                # python:3.11-slim + libgomp1 (required by LightGBM/XGBoost)
│   ├── Dockerfile.train
│   ├── Dockerfile.streamlit
│   ├── Dockerfile.prometheus         # embeds monitoring/prometheus.yml (no bind mount)
│   ├── Dockerfile.nginx              # embeds docker/nginx.conf (no bind mount)
│   └── Dockerfile.grafana            # embeds monitoring/grafana/provisioning + dashboards
├── docker-compose.yml
├── Jenkinsfile
├── requirements.txt                  # runtime — versions MUST match local venv exactly
├── requirements-dev.txt              # testing
└── requirements-notebooks.txt       # pinned for Jupyter exploration
```

## Data splits (params.yaml)

| Split   | Period        | Rows    | Default rate |
|---------|---------------|---------|--------------|
| Train   | 202001–202312 | 631,361 | 6.97%        |
| Test    | 202401–202512 | 316,067 | 6.99%        |
| Monitor | 202601–202604 |  52,572 | 7.28%        |

## Bugs fixed in this session

1. **`use_label_encoder=False`** removed from `src/train.py` — XGBoost 3.x dropped this param.
2. **`delinquency_ratio` added to `NUMERIC_FEATURES`** in `src/features.py` — was created by
   `DelinquencyRatioTransformer` but silently dropped by `ColumnTransformer`.
3. **HPO CV now uses full pre-pipeline** in `src/train.py` — both `train_xgboost` and
   `train_lightgbm` now apply `LogTransformer → DelinquencyRatioTransformer → build_preprocessor()`
   before running Optuna, matching what `build_full_pipeline` does.
4. **Auto-promotion to Production** added at end of `src/train.py` `main()` — uses
   `client.transition_model_version_stage(... stage="Production", archive_existing_versions=True)`.

## Docker fixes applied in this session

All of these are already in the repo:

- **Colon in path breaks Docker bind mounts** — `docker-compose.yml` now uses custom-built images
  (`Dockerfile.prometheus`, `Dockerfile.nginx`, `Dockerfile.grafana`) that COPY configs at build
  time instead of bind-mounting from the host. This avoids the path-parsing bug in Docker Desktop.
- **MLflow port** changed from 5000 → **5001** (macOS AirPlay/ControlCenter owns 5000).
- **MLflow image** upgraded from `v2.13.0` → `latest` (3.11.1). Client and server must match.
- **MLflow server flags**: `--serve-artifacts` + `--artifacts-destination /mlflow/artifacts` +
  `--allowed-hosts "*"` (Docker containers send `Host: mlflow:5000` which triggers DNS-rebinding
  protection in MLflow 3.x; `--allowed-hosts "*"` disables it).
- **MLflow DB path**: `sqlite:////mlflow/mlflow.db` (4 slashes = absolute path inside the
  named volume). `sqlite:///mlflow.db` (3 slashes = relative to CWD) is lost on container restart.
- **API healthcheck**: changed from `curl -f` to Python `urllib.request` — `python:3.11-slim`
  doesn't include `curl`.
- **`libgomp1`** added to `Dockerfile.api` runtime stage — required by LightGBM and XGBoost;
  missing from `python:3.11-slim`.
- **requirements.txt versions** aligned exactly with local venv (numpy 2.4.4, sklearn 1.8.0,
  scipy 1.17.1, pandas 2.3.3, etc.). Pickle serialization is version-specific — a model trained
  with numpy 2.x cannot be loaded by numpy 1.x (`numpy._core` vs `numpy.core`).

## How to run the full system

```bash
# 1. Build all images (first time or after code/deps changes)
cd /Users/josefrodriguez/repos_publicar/credit-risk-mlops:
docker compose build

# 2. Start infrastructure
docker compose up -d mlflow prometheus grafana

# 3. Train model (runs locally, logs to Docker MLflow at localhost:5001)
source ~/.venvs/credit-risk-mlops/bin/activate
PYTHONPATH=$(pwd) MLFLOW_TRACKING_URI=http://localhost:5001 python src/train.py --params params.yaml

# 4. Start API + nginx (model must be in Production stage first)
docker compose up -d api nginx

# 5. (Optional) Streamlit demo
docker compose --profile demo up -d streamlit
```

**Note on n_trials**: `params.yaml` has `n_trials: 5` and `cv_folds: 3` for fast demo runs
(~3 min). For production-quality HPO change to `n_trials: 50, cv_folds: 5` (takes ~15–20 min).

## Service URLs (all running)

| Service    | URL                             | Credentials                  |
|------------|---------------------------------|------------------------------|
| MLflow UI  | <http://localhost:5001>         | —                            |
| FastAPI    | <http://localhost:8000>         | —                            |
| API docs   | <http://localhost:8000/docs>    | —                            |
| Nginx      | <http://localhost:80>           | —                            |
| Prometheus | <http://localhost:9090>         | —                            |
| Grafana    | <http://localhost:3000>         | admin/admin                  |
| Streamlit  | <http://localhost:8501>         | start with `--profile demo`  |

## Key results

- Test ROC AUC: **0.9944** | Test PR AUC: **0.9467** | AUC gate: 0.87 → PASS
- LightGBM v2 (Production) AUC: **0.9945**
- Low-risk sample: `0.012%` default probability | High-risk sample: `99.96%`
- Monthly AUC in production 2026 — all 4 months pass gate (chart in 05_monitoring.ipynb)

## Environment notes

- **Colon in repo path** (`/Users/.../credit-risk-mlops:`) breaks `python3 -m venv .venv`.
  Always create the venv outside: `python3 -m venv ~/.venvs/credit-risk-mlops`
- Activate: `source ~/.venvs/credit-risk-mlops/bin/activate`
- Notebooks deps: `pip install -r requirements-notebooks.txt`
- `docker compose` must be run from the repo directory with the colon in the path — it works
  for build/up/down. Only bind mounts break (solved with custom Dockerfiles above).

## Notebooks — MLflow fallback pattern

Both `04_model_evaluation.ipynb` and `05_monitoring.ipynb`:

```python
try:
    pipeline = mlflow.sklearn.load_model(MODEL_URI)
except Exception:
    pipeline = build_full_pipeline(LGBMClassifier(...))
    pipeline.fit(X_ref, y_ref)
```

## Completed tasks (2026-04-20)

### 1. Streamlit dashboard ✅

`app/streamlit_app.py` rewritten with three pages:
- **Single Application** — scoring form (unchanged)
- **Batch Scoring** — CSV upload (unchanged)
- **Monitoring Dashboard** — model metadata, monthly AUC/Gini bar charts, target drift with
  tolerance band, PSI per feature (color-coded), score distribution overlay (plotly charts).
  Reads parquets directly, MLflow fallback pattern.

`docker/Dockerfile.streamlit` updated: added `libgomp1`, `streamlit==1.56.0`, `plotly==6.1.2`.
`docker-compose.yml` streamlit service: added `./data:/app/data` volume + `DATA_DIR` env var.

### 2. README.md ✅

Created at repo root: overview, architecture ASCII diagram, dataset table, quickstart (3 steps),
training instructions, API endpoints + curl example, monitoring section, CI/CD table, key results.

### 3. Rebuild Streamlit Docker image ⚠️ (still needed)

Run before using Streamlit in Docker:
```bash
docker compose build streamlit
# then start:
docker compose --profile demo up -d streamlit
```

## Reglas de git para este proyecto

- **Nunca** incluir `Co-Authored-By: Claude` en los mensajes de commit.
  GitHub registra esa línea como contribuidor y aparece públicamente en el repo,
  lo cual no es apropiado para un proyecto de portfolio personal.
