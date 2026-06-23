"""
Section 3 — Advanced ML Diagnostics
=======================================
Pre-trained forecasting, anomaly detection, and SHAP results are loaded
from dashboard/artifacts/ (built offline via scripts/build_dashboard_artifacts.py).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.artifacts import (
    artifacts_ready,
    available_contaminations,
    available_forecast_horizons,
    load_anomaly_results,
    load_forecast,
    load_shap_bundle,
)
from dashboard.utils.data_loader import available_numeric_cols


def _render_forecast(target_col: str, target_label: str) -> None:
    horizons = available_forecast_horizons(target_col)
    if not horizons:
        st.warning(f"No pre-trained forecast artifacts found for `{target_col}`.")
        return

    default = 14 if 14 in horizons else horizons[len(horizons) // 2]
    horizon = st.selectbox(
        "Forecast horizon (days)",
        horizons,
        index=horizons.index(default),
        help="Pre-computed held-out validation windows — switching horizons is instant.",
    )

    result = load_forecast(target_col, horizon)
    if result is None:
        st.warning(f"No forecast artifact for horizon={horizon}.")
        return

    if not result.get("prophet_available", True):
        st.info("Prophet was unavailable during artifact build — ensemble used XGBoost-only fallback.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result["dates"], y=result["y_true"], name="Actual",
                              line=dict(color="#222222", width=2)))
    fig.add_trace(go.Scatter(x=result["dates"], y=result["xgb_pred"], name="XGBoost",
                              line=dict(color="#2E86AB", dash="dot")))
    fig.add_trace(go.Scatter(x=result["dates"], y=result["prophet_pred"], name="Prophet",
                              line=dict(color="#F4A261", dash="dot")))
    fig.add_trace(go.Scatter(x=result["dates"], y=result["ensemble_pred"], name="Weighted Ensemble",
                              line=dict(color="#E94F37", width=3)))
    fig.update_layout(height=420, title=f"Held-Out Forecast — {target_label}", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"**Ensemble weights:** XGBoost = `{result['weights']['xgboost']:.2f}` · "
        f"Prophet = `{result['weights']['prophet']:.2f}` "
        f"(inverse-RMSE weighted — better-performing model gets more weight)."
    )

    metrics_df = pd.DataFrame({
        "XGBoost": result["xgb_metrics"],
        "Prophet": result["prophet_metrics"],
        "Ensemble": result["ensemble_metrics"],
    }).T.round(3)
    st.dataframe(metrics_df, use_container_width=True)


def _render_anomaly(numeric_cols: list[str]) -> None:
    contaminations = available_contaminations()
    default = 0.02 if 0.02 in contaminations else contaminations[0]

    c1, c2, c3 = st.columns(3)
    x_col = c1.selectbox(
        "X-axis feature", numeric_cols,
        index=numeric_cols.index("temperature_celsius") if "temperature_celsius" in numeric_cols else 0,
    )
    y_col = c2.selectbox(
        "Y-axis feature", numeric_cols,
        index=numeric_cols.index("humidity") if "humidity" in numeric_cols else min(1, len(numeric_cols) - 1),
    )
    contamination = c3.selectbox(
        "Expected anomaly rate",
        contaminations,
        index=contaminations.index(default),
        format_func=lambda x: f"{x:.0%}",
    )

    result = load_anomaly_results(contamination)
    if result is None:
        st.warning("Anomaly detection artifacts not found.")
        return

    n_anom = int(result["anomaly"].sum())
    fig = go.Figure()
    normal = result[~result["anomaly"]]
    anomalous = result[result["anomaly"]]
    fig.add_trace(go.Scattergl(
        x=normal[x_col], y=normal[y_col], mode="markers", name="Normal",
        marker=dict(color="#2E86AB", size=5, opacity=0.4),
    ))
    fig.add_trace(go.Scattergl(
        x=anomalous[x_col], y=anomalous[y_col], mode="markers", name="Anomaly",
        marker=dict(color="#E94F37", size=7, symbol="x"),
    ))
    fig.update_layout(
        height=450,
        title=f"IsolationForest Anomalies ({n_anom:,} flagged of {len(result):,})",
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_shap(target_col: str) -> None:
    bundle = load_shap_bundle(target_col)
    if bundle is None:
        st.warning(f"No SHAP artifact found for `{target_col}`.")
        return

    if bundle["mode"] == "importance":
        st.info("SHAP was unavailable during artifact build — showing pre-computed XGBoost feature importance.")
        st.bar_chart(bundle["importance"].head(15))
        return

    fig, _ax = plt.subplots(figsize=(8, 6))
    import shap

    shap.summary_plot(bundle["shap_values"], bundle["X"], show=False, plot_size=None)
    st.pyplot(fig, use_container_width=True)
    plt.close("all")
    st.caption(
        "Each point is one observation. Color = feature value (red = high, blue = low). "
        "Position on the x-axis = impact on the predicted temperature for that observation."
    )


def render(df) -> None:
    st.subheader("🤖 Advanced ML Diagnostics")
    st.caption("Forecasting, anomaly detection, and explainability — loaded from pre-trained artifacts.")

    if not artifacts_ready():
        st.error(
            "Pre-trained model artifacts are missing. Run "
            "`python scripts/build_dashboard_artifacts.py` from the project root, then reload."
        )
        return

    if df.empty:
        st.info("No data loaded — see the error above.")
        return

    numeric_cols = available_numeric_cols(df)
    if not numeric_cols:
        st.warning("No numeric weather/AQI columns available for modeling.")
        return

    tab1, tab2, tab3 = st.tabs(["📈 Ensemble Forecast", "🚨 Anomaly Detection", "🧠 SHAP Importance"])

    with tab1:
        target_label = st.selectbox(
            "Forecast target", ["Temperature (°C)", "Precipitation (mm)"], key="forecast_target",
        )
        target_col = "temperature_celsius" if "Temperature" in target_label else "precip_mm"
        if target_col in df.columns:
            _render_forecast(target_col, target_label)
        else:
            st.warning(f"Column `{target_col}` not found.")

    with tab2:
        _render_anomaly(numeric_cols)

    with tab3:
        target_col = "temperature_celsius" if "temperature_celsius" in numeric_cols else numeric_cols[0]
        st.caption(f"Explaining predictions of: **{target_col}**")
        _render_shap(target_col)
