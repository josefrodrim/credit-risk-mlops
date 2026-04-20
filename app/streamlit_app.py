"""Streamlit app for credit-risk-mlops.

Pages:
  - Single Application: scoring form with risk label
  - Batch Scoring: CSV upload with downloadable results
  - Monitoring Dashboard: AUC/Gini, target drift, PSI, score distribution
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ensure repo root is on the path so `src` is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ─── constants ───────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
BATCH_LIMIT = 1000
CREDIT_SCORE_BANDS = ["Poor", "Fair", "Good", "Very Good", "Exceptional"]

DATA_DIR     = Path(os.getenv("DATA_DIR", "data/processed"))
TRAIN_PATH   = DATA_DIR / "train.parquet"
TEST_PATH    = DATA_DIR / "test.parquet"
MONITOR_PATH = DATA_DIR / "monitor.parquet"

TARGET     = "SeriousDlqin2yrs"
PERIOD_COL = "periodo"
AUC_GATE   = 0.87
PSI_WARN   = 0.10
PSI_ALERT  = 0.20

RAW_NUMERIC = [
    "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents", "loan_amount", "employment_years",
]
RAW_CATEGORICAL = ["credit_score_band"]
FEATURE_COLS    = RAW_NUMERIC + RAW_CATEGORICAL

# ─── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk MLOps",
    page_icon="💳",
    layout="wide",
)


# ─── helper functions ────────────────────────────────────────────────────────
def _check_api_health() -> bool:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.status_code == 200 and resp.json().get("status") == "ok"
    except requests.exceptions.RequestException:
        return False


def _predict_single(payload: dict) -> dict:
    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _predict_batch(records: list[dict]) -> dict:
    resp = requests.post(
        f"{API_URL}/predict/batch", json={"records": records}, timeout=60
    )
    resp.raise_for_status()
    return resp.json()


def _color_risk(prob: float) -> str:
    if prob >= 0.5:
        return "🔴 HIGH RISK"
    if prob >= 0.3:
        return "🟡 MEDIUM RISK"
    return "🟢 LOW RISK"


# ─── data & model loading (cached) ───────────────────────────────────────────
@st.cache_data(show_spinner="Loading datasets…")
def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train   = pd.read_parquet(TRAIN_PATH)
    test    = pd.read_parquet(TEST_PATH)
    monitor = pd.read_parquet(MONITOR_PATH)
    return train, test, monitor


@st.cache_resource(show_spinner="Loading model…")
def _load_model() -> tuple:
    """Returns (pipeline, source_label). Falls back to a quick LightGBM fit."""
    try:
        import yaml
        import mlflow

        with open("params.yaml") as fh:
            cfg = yaml.safe_load(fh)

        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        model_uri = f"models:/{cfg['mlflow']['model_name']}/Production"
        pipeline = mlflow.sklearn.load_model(model_uri)
        return pipeline, "MLflow Production"
    except Exception:
        from lightgbm import LGBMClassifier
        from src.features import build_full_pipeline

        train = pd.read_parquet(TRAIN_PATH)
        sample = train.sample(min(50_000, len(train)), random_state=42)
        X = sample[FEATURE_COLS]
        y = sample[TARGET]
        pipeline = build_full_pipeline(
            LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1)
        )
        pipeline.fit(X, y)
        return pipeline, "LightGBM fallback"


# ─── monitoring computations ─────────────────────────────────────────────────
def _psi(ref: pd.Series, prod: pd.Series, n_bins: int = 10) -> float:
    eps = 1e-4
    if ref.dtype == object or str(ref.dtype) == "category":
        cats = sorted(set(ref.dropna().unique()) | set(prod.dropna().unique()))
        r = ref.value_counts(normalize=True).reindex(cats, fill_value=eps).values
        p = prod.value_counts(normalize=True).reindex(cats, fill_value=eps).values
    else:
        bp = np.unique(np.nanpercentile(ref.dropna(), np.linspace(0, 100, n_bins + 1)))
        if len(bp) < 2:
            return 0.0
        r = np.histogram(ref.dropna(), bins=bp)[0].astype(float)
        p = np.histogram(prod.dropna(), bins=bp)[0].astype(float)
        r = np.where(r == 0, eps, r / r.sum())
        p = np.where(p == 0, eps, p / p.sum())
    return float(np.sum((p - r) * np.log(p / r)))


@st.cache_data(show_spinner=False)
def _monthly_auc(_pipeline, cache_key: str) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score

    _, _, monitor = _load_data()
    rows = []
    for period, grp in monitor.groupby(PERIOD_COL):
        X = grp[FEATURE_COLS]
        y = grp[TARGET]
        if y.nunique() < 2:
            continue
        proba = _pipeline.predict_proba(X)[:, 1]
        auc   = roc_auc_score(y, proba)
        rows.append({
            "periodo":      str(period),
            "auc":          auc,
            "gini":         2 * auc - 1,
            "pass":         auc >= AUC_GATE,
            "n":            len(grp),
            "default_rate": float(y.mean()),
        })
    return pd.DataFrame(rows).sort_values("periodo")


@st.cache_data(show_spinner=False)
def _psi_table(cache_key: str) -> pd.DataFrame:
    train, _, monitor = _load_data()
    rows = [
        {"feature": col, "psi": _psi(train[col], monitor[col])}
        for col in FEATURE_COLS
        if col in monitor.columns
    ]
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


@st.cache_data(show_spinner=False)
def _score_distributions(_pipeline, cache_key: str):
    _, test, monitor = _load_data()
    ref_scores = _pipeline.predict_proba(test[FEATURE_COLS])[:, 1]
    mon_scores = _pipeline.predict_proba(monitor[FEATURE_COLS])[:, 1]
    return ref_scores, mon_scores


# ─── sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("Credit Risk MLOps")
st.sidebar.caption("LightGBM / XGBoost + MLflow")

api_healthy = _check_api_health()
if api_healthy:
    st.sidebar.success(f"API ✓  {API_URL}")
else:
    st.sidebar.warning(f"API offline — {API_URL}")

page = st.sidebar.radio(
    "Navigate",
    ["Single Application", "Batch Scoring", "Monitoring Dashboard"],
)


# ─── page: single application ─────────────────────────────────────────────────
if page == "Single Application":
    st.title("Single Credit Application Scoring")

    with st.form("credit_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Personal Info")
            age              = st.number_input("Age", min_value=18, max_value=100, value=45)
            n_dependents     = st.number_input("Number of Dependents", min_value=0, max_value=20, value=2)
            employment_years = st.number_input("Employment Years", min_value=0.0, max_value=50.0, value=10.0)

        with col2:
            st.subheader("Financial Profile")
            monthly_income    = st.number_input("Monthly Income ($)", min_value=0.0, value=5000.0)
            loan_amount       = st.number_input("Loan Amount ($)", min_value=100.0, value=15000.0)
            debt_ratio        = st.number_input("Debt Ratio", min_value=0.0, max_value=100.0, value=0.25)
            revolving_util    = st.slider("Revolving Credit Utilization", 0.0, 1.0, 0.35)
            credit_score_band = st.selectbox("Credit Score Band", CREDIT_SCORE_BANDS, index=2)

        with col3:
            st.subheader("Credit History")
            open_lines    = st.number_input("Open Credit Lines", min_value=0, value=8)
            real_estate   = st.number_input("Real Estate Loans", min_value=0, value=1)
            past_due_30   = st.number_input("30-59 Days Past Due", min_value=0, value=0)
            past_due_60   = st.number_input("60-89 Days Past Due", min_value=0, value=0)
            past_due_90   = st.number_input("90+ Days Late", min_value=0, value=0)

        submitted = st.form_submit_button("Score Application", type="primary")

    if submitted:
        if not api_healthy:
            st.error("API is not reachable. Please check the connection.")
        else:
            payload = {
                "RevolvingUtilizationOfUnsecuredLines": revolving_util,
                "age": int(age),
                "NumberOfTime30-59DaysPastDueNotWorse": int(past_due_30),
                "DebtRatio": debt_ratio,
                "MonthlyIncome": monthly_income,
                "NumberOfOpenCreditLinesAndLoans": int(open_lines),
                "NumberOfTimes90DaysLate": int(past_due_90),
                "NumberRealEstateLoansOrLines": int(real_estate),
                "NumberOfTime60-89DaysPastDueNotWorse": int(past_due_60),
                "NumberOfDependents": float(n_dependents),
                "loan_amount": loan_amount,
                "employment_years": employment_years,
                "credit_score_band": credit_score_band,
            }
            with st.spinner("Scoring…"):
                try:
                    result = _predict_single(payload)
                except requests.exceptions.HTTPError as exc:
                    st.error(f"API error: {exc.response.text}")
                    st.stop()

            prob  = result["default_probability"]
            label = _color_risk(prob)
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Default Probability", f"{prob:.1%}")
            with c2:
                st.markdown(f"### {label}")
            st.progress(prob)
            with st.expander("Raw API Response"):
                st.json(result)


# ─── page: batch scoring ──────────────────────────────────────────────────────
elif page == "Batch Scoring":
    st.title("Batch Credit Scoring")
    st.markdown(f"Upload a CSV with the required columns. Maximum **{BATCH_LIMIT} rows**.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(df)} rows × {len(df.columns)} columns")
        st.dataframe(df.head())

        missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            if len(df) > BATCH_LIMIT:
                st.warning(f"File has {len(df)} rows — only first {BATCH_LIMIT} will be scored.")
                df = df.head(BATCH_LIMIT)

            if st.button("Score Batch", type="primary"):
                if not api_healthy:
                    st.error("API is not reachable.")
                else:
                    records = df[FEATURE_COLS].to_dict(orient="records")
                    with st.spinner(f"Scoring {len(records)} applications…"):
                        try:
                            response = _predict_batch(records)
                        except requests.exceptions.HTTPError as exc:
                            st.error(f"API error: {exc.response.text}")
                            st.stop()

                    preds_df = pd.DataFrame(response["predictions"])
                    result_df = df.copy()
                    result_df["default_probability"] = preds_df["default_probability"].values
                    result_df["prediction"]          = preds_df["prediction"].values

                    st.success(f"Scored {response['count']} applications")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Applications", response["count"])
                    c2.metric("High Risk (≥50%)", int((result_df["default_probability"] >= 0.5).sum()))
                    c3.metric("Avg Default Prob", f"{result_df['default_probability'].mean():.1%}")

                    st.dataframe(
                        result_df[["default_probability", "prediction"] + FEATURE_COLS[:5]].head(20)
                    )
                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Results CSV", data=csv,
                        file_name="credit_risk_scores.csv", mime="text/csv",
                    )


# ─── page: monitoring dashboard ───────────────────────────────────────────────
elif page == "Monitoring Dashboard":
    import plotly.graph_objects as go

    st.title("Model Monitoring Dashboard")

    data_ok = TRAIN_PATH.exists() and TEST_PATH.exists() and MONITOR_PATH.exists()
    if not data_ok:
        st.warning(
            "Parquet files not found. "
            "Run `python scripts/generate_dataset.py && python scripts/featurize.py` "
            "then restart the app.\n\n"
            f"Expected: `{DATA_DIR.resolve()}`"
        )
        st.stop()

    train, test, monitor = _load_data()
    pipeline, model_source = _load_model()

    cache_key = f"{len(train)}-{len(test)}-{len(monitor)}"  # invalidate if data changes

    # ── Model metadata ──────────────────────────────────────────────────────
    train_dr   = float(train[TARGET].mean())
    monitor_dr = float(monitor[TARGET].mean())
    drift_pct  = (monitor_dr - train_dr) / train_dr * 100

    st.subheader("Model Metadata")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model source",            model_source)
    m2.metric("Train rows",              f"{len(train):,}")
    m3.metric("Monitor rows (2026)",     f"{len(monitor):,}")
    m4.metric("Train default rate",      f"{train_dr:.2%}",
              delta=f"{drift_pct:+.1f}% vs monitor")

    st.markdown("---")

    # ── Monthly AUC & Gini ──────────────────────────────────────────────────
    st.subheader("Monthly AUC & Gini — Monitor Period 2026")

    with st.spinner("Computing monthly AUC…"):
        monthly = _monthly_auc(pipeline, cache_key)

    if monthly.empty:
        st.info("No monthly AUC data available.")
    else:
        colors = ["green" if p else "red" for p in monthly["pass"]]

        left, right = st.columns(2)

        with left:
            fig_auc = go.Figure(go.Bar(
                x=monthly["periodo"], y=monthly["auc"],
                marker_color=colors,
                text=[f"{v:.4f}" for v in monthly["auc"]],
                textposition="outside",
            ))
            fig_auc.add_hline(y=AUC_GATE, line_dash="dash", line_color="orange",
                               annotation_text=f"Gate {AUC_GATE}")
            fig_auc.update_layout(title="ROC-AUC per month",
                                   yaxis_range=[0.80, 1.01], height=360,
                                   margin=dict(t=40, b=10))
            st.plotly_chart(fig_auc, use_container_width=True)

        with right:
            fig_gini = go.Figure(go.Bar(
                x=monthly["periodo"], y=monthly["gini"],
                marker_color=colors,
                text=[f"{v:.4f}" for v in monthly["gini"]],
                textposition="outside",
            ))
            gate_gini = 2 * AUC_GATE - 1
            fig_gini.add_hline(y=gate_gini, line_dash="dash", line_color="orange",
                                annotation_text=f"Gate {gate_gini:.2f}")
            fig_gini.update_layout(title="Gini = 2·AUC − 1",
                                    yaxis_range=[0.60, 1.01], height=360,
                                    margin=dict(t=40, b=10))
            st.plotly_chart(fig_gini, use_container_width=True)

        tbl = monthly[["periodo", "auc", "gini", "pass", "n", "default_rate"]].copy()
        tbl["auc"]          = tbl["auc"].round(4)
        tbl["gini"]         = tbl["gini"].round(4)
        tbl["default_rate"] = tbl["default_rate"].map("{:.2%}".format)
        tbl["pass"]         = tbl["pass"].map({True: "✅ PASS", False: "❌ FAIL"})
        st.dataframe(
            tbl.rename(columns={"periodo": "Period", "auc": "AUC", "gini": "Gini",
                                  "pass": "Gate", "n": "Rows", "default_rate": "Default Rate"}),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")

    # ── Target drift ────────────────────────────────────────────────────────
    st.subheader("Target Drift — Default Rate by Period")

    all_data = (
        pd.concat([train, test, monitor], ignore_index=True)
        .groupby(PERIOD_COL)[TARGET]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values(PERIOD_COL)
        .rename(columns={"mean": "default_rate", "count": "n"})
    )
    periods  = all_data[PERIOD_COL].astype(str).tolist()
    rates    = all_data["default_rate"].tolist()
    baseline  = train_dr
    tolerance = 0.02

    fig_drift = go.Figure()
    fig_drift.add_trace(go.Scatter(
        x=periods + periods[::-1],
        y=[baseline + tolerance] * len(periods) + [baseline - tolerance] * len(periods),
        fill="toself", fillcolor="rgba(0,100,80,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"±{tolerance:.0%} band",
        hoverinfo="skip",
    ))
    fig_drift.add_hline(y=baseline, line_dash="dash", line_color="gray",
                         annotation_text=f"Train baseline {baseline:.2%}")
    fig_drift.add_trace(go.Scatter(
        x=periods, y=rates,
        mode="lines+markers",
        line=dict(color="steelblue", width=2),
        name="Default rate",
        hovertemplate="%{x}: %{y:.2%}<extra></extra>",
    ))
    tick_every = max(1, len(periods) // 12)
    fig_drift.update_layout(
        height=380, margin=dict(t=20, b=10),
        xaxis=dict(title="Period (YYYYMM)",
                   tickmode="array",
                   tickvals=periods[::tick_every],
                   ticktext=periods[::tick_every]),
        yaxis=dict(title="Default rate", tickformat=".1%"),
    )
    st.plotly_chart(fig_drift, use_container_width=True)

    st.markdown("---")

    # ── PSI per feature ─────────────────────────────────────────────────────
    st.subheader("Population Stability Index — Train vs Monitor 2026")
    st.caption("Red > 0.20 (major shift)  |  Orange > 0.10 (moderate)  |  Blue = stable")

    with st.spinner("Computing PSI…"):
        psi_df = _psi_table(cache_key)

    psi_colors = [
        "crimson" if v > PSI_ALERT else "darkorange" if v > PSI_WARN else "steelblue"
        for v in psi_df["psi"]
    ]
    fig_psi = go.Figure(go.Bar(
        x=psi_df["psi"], y=psi_df["feature"],
        orientation="h",
        marker_color=psi_colors,
        text=[f"{v:.3f}" for v in psi_df["psi"]],
        textposition="outside",
    ))
    fig_psi.add_vline(x=PSI_WARN,  line_dash="dot", line_color="darkorange",
                       annotation_text="0.10", annotation_position="top")
    fig_psi.add_vline(x=PSI_ALERT, line_dash="dot", line_color="crimson",
                       annotation_text="0.20", annotation_position="top")
    fig_psi.update_layout(
        height=max(420, len(psi_df) * 36),
        margin=dict(t=20, b=10, l=260),
        xaxis_title="PSI",
    )
    st.plotly_chart(fig_psi, use_container_width=True)

    st.markdown("---")

    # ── Score distribution ──────────────────────────────────────────────────
    st.subheader("Score Distribution — Test Reference vs Monitor 2026")

    with st.spinner("Generating score distributions…"):
        ref_scores, mon_scores = _score_distributions(pipeline, cache_key)

    n_bins    = 50
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ref_hist, _ = np.histogram(ref_scores, bins=bin_edges, density=True)
    mon_hist, _ = np.histogram(mon_scores, bins=bin_edges, density=True)
    bin_centers  = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_w        = bin_edges[1] - bin_edges[0]

    fig_scores = go.Figure()
    fig_scores.add_trace(go.Bar(
        x=bin_centers, y=ref_hist, width=bar_w,
        name="Test reference (202401–202512)",
        marker_color="rgba(70,130,180,0.65)",
    ))
    fig_scores.add_trace(go.Bar(
        x=bin_centers, y=mon_hist, width=bar_w,
        name="Monitor 2026 (202601–202604)",
        marker_color="rgba(220,80,60,0.65)",
    ))
    fig_scores.update_layout(
        barmode="overlay",
        height=380, margin=dict(t=20, b=10),
        xaxis_title="Predicted default probability",
        yaxis_title="Density",
        legend=dict(x=0.55, y=0.95),
    )
    st.plotly_chart(fig_scores, use_container_width=True)
