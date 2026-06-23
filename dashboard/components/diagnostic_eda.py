"""
Section 2 — Diagnostic EDA
=============================
Trend decomposition, distribution profiling, and seasonality analysis for
Temperature and Precipitation — the two variables most directly tied to
the assessment's forecasting objective.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

from dashboard.utils.data_loader import daily_global_series

TARGET_OPTIONS = {
    "Temperature (°C)": "temperature_celsius",
    "Precipitation (mm)": "precip_mm",
}


def _decompose_chart(series: pd.Series, period: int, label: str) -> go.Figure | None:
    """Additive seasonal decomposition rendered as a 3-row Plotly subplot."""
    if len(series) < 2 * period:
        st.warning(
            f"Not enough history to decompose **{label}** with period={period} "
            f"(need at least {2 * period} days, have {len(series)}). Skipping."
        )
        return None
    decomposition = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                         subplot_titles=("Trend", "Seasonal", "Residual"))
    fig.add_trace(go.Scatter(x=series.index, y=decomposition.trend, name="Trend",
                              line=dict(color="#1F4E79")), row=1, col=1)
    fig.add_trace(go.Scatter(x=series.index, y=decomposition.seasonal, name="Seasonal",
                              line=dict(color="#2A9D8F")), row=2, col=1)
    fig.add_trace(go.Scatter(x=series.index, y=decomposition.resid, name="Residual",
                              line=dict(color="#E94F37")), row=3, col=1)
    fig.update_layout(height=600, showlegend=False, title=f"Trend Decomposition — {label}")
    return fig


def render(df) -> None:
    st.subheader("🔍 Diagnostic EDA")
    st.caption("Decompose, distribute, and profile seasonality for the core forecasting targets.")

    if df.empty:
        st.info("No data loaded — see the error above.")
        return

    target_label = st.selectbox(
        "Select variable", list(TARGET_OPTIONS.keys()),
        help="Choose which variable drives the decomposition, distribution, and seasonality views below.",
    )
    target_col = TARGET_OPTIONS[target_label]

    if target_col not in df.columns:
        st.warning(f"Column `{target_col}` not present in this dataset.")
        return

    # ---------------------------------------------------------------- Trend decomposition
    st.markdown("**Trend Decomposition**")
    period = st.slider(
        "Seasonal period (days)", min_value=7, max_value=90, value=30, step=1,
        help="7 = weekly seasonality, 30 ≈ monthly, 90 ≈ quarterly. Choose based on your date range.",
    )
    try:
        series = daily_global_series(df, target_col)
        if series.empty:
            st.warning("Could not build a daily series for decomposition (insufficient data).")
        else:
            fig = _decompose_chart(series, period, target_label)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:  # noqa: BLE001
        st.error(f"Decomposition failed: {e}")

    st.divider()

    # ---------------------------------------------------------------- Distributions
    st.markdown("**Distribution Profile**")
    d1, d2 = st.columns(2)
    try:
        with d1:
            fig_hist = px.histogram(
                df, x=target_col, nbins=60, marginal="box",
                color_discrete_sequence=["#2E86AB"],
                labels={target_col: target_label},
            )
            fig_hist.update_traces(hovertemplate=f"{target_label}: %{{x}}<br>Count: %{{y}}<extra></extra>")
            fig_hist.update_layout(height=380, title=f"Distribution — {target_label}")
            st.plotly_chart(fig_hist, use_container_width=True)
        with d2:
            group_col = "country" if "country" in df.columns else None
            if group_col:
                top_countries = df[group_col].value_counts().head(8).index
                fig_box = px.box(
                    df[df[group_col].isin(top_countries)], x=group_col, y=target_col,
                    color=group_col, points=False,
                    labels={target_col: target_label, group_col: "Country"},
                )
                fig_box.update_layout(height=380, showlegend=False,
                                       title=f"{target_label} by Country (Top 8 by Volume)")
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("`country` column unavailable — skipping group comparison boxplot.")
    except Exception as e:  # noqa: BLE001
        st.error(f"Distribution charts failed: {e}")

    st.divider()

    # ---------------------------------------------------------------- Seasonality profiling
    st.markdown("**Seasonality Profiling**")
    s1, s2 = st.columns(2)
    try:
        with s1:
            if "month_name" in df.columns:
                month_order = ["January", "February", "March", "April", "May", "June",
                                "July", "August", "September", "October", "November", "December"]
                fig_month = px.box(
                    df, x="month_name", y=target_col, category_orders={"month_name": month_order},
                    labels={target_col: target_label, "month_name": "Month"},
                    color_discrete_sequence=["#1F4E79"],
                )
                fig_month.update_layout(height=380, title=f"Monthly Seasonality — {target_label}")
                st.plotly_chart(fig_month, use_container_width=True)
        with s2:
            if "day_of_week" in df.columns:
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                dow_agg = df.groupby("day_of_week")[target_col].mean().reindex(day_order)
                fig_dow = px.line(
                    x=dow_agg.index, y=dow_agg.values, markers=True,
                    labels={"x": "Day of Week", "y": f"Mean {target_label}"},
                )
                fig_dow.update_traces(line_color="#2A9D8F",
                                       hovertemplate="%{x}<br>Mean: %{y:.2f}<extra></extra>")
                fig_dow.update_layout(height=380, title=f"Day-of-Week Pattern — {target_label}")
                st.plotly_chart(fig_dow, use_container_width=True)
    except Exception as e:  # noqa: BLE001
        st.error(f"Seasonality charts failed: {e}")
