"""
Section 5 — Environmental Matrix
====================================
Multi-variable cross-correlation deep-dive: weather parameters vs.
PM2.5, PM10, CO, and O3.
"""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

WEATHER_CANDIDATES = [
    "temperature_celsius", "humidity", "pressure_mb", "wind_kph",
    "cloud", "visibility_km", "uv_index", "precip_mm", "gust_kph",
]
AQI_CANDIDATES = {
    "PM2.5": "air_quality_PM2.5",
    "PM10": "air_quality_PM10",
    "Carbon Monoxide (CO)": "air_quality_Carbon_Monoxide",
    "Ozone (O3)": "air_quality_Ozone",
}


def render(df) -> None:
    st.subheader("🧪 Environmental Matrix")
    st.caption("Cross-correlation between weather conditions and key air-quality pollutants.")

    if df.empty:
        st.info("No data loaded — see the error above.")
        return

    weather_cols = [c for c in WEATHER_CANDIDATES if c in df.columns]
    aqi_cols = {label: col for label, col in AQI_CANDIDATES.items() if col in df.columns}

    if not weather_cols or not aqi_cols:
        st.warning("Required weather or AQI columns not found in this dataset.")
        return

    method = st.radio(
        "Correlation method", ["pearson", "spearman"], horizontal=True,
        help="Pearson assumes a linear relationship; Spearman captures monotonic, non-linear relationships "
             "and is more robust to outliers — compare both if results disagree.",
    )

    try:
        sub = df[weather_cols + list(aqi_cols.values())].dropna()
        corr = sub.corr(method=method)
        matrix = corr.loc[list(aqi_cols.values()), weather_cols]
        matrix.index = list(aqi_cols.keys())

        fig = go.Figure(data=go.Heatmap(
            z=matrix.values, x=matrix.columns, y=matrix.index,
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=matrix.round(2).values, texttemplate="%{text}",
            hovertemplate="Pollutant: %{y}<br>Weather var: %{x}<br>Correlation: %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(
            height=420, title=f"{method.title()} Correlation — Pollutants vs. Weather Variables",
            xaxis_title="Weather Variable", yaxis_title="Pollutant",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:  # noqa: BLE001
        st.error(f"Correlation matrix failed: {e}")
        return

    st.divider()

    # ---------------------------------------------------------------- Drill-down scatter
    st.markdown("**Pairwise Drill-Down**")
    c1, c2 = st.columns(2)
    pollutant_label = c1.selectbox("Pollutant", list(aqi_cols.keys()),
                                    help="Pollutant plotted on the y-axis.")
    weather_var = c2.selectbox("Weather variable", weather_cols,
                                help="Weather variable plotted on the x-axis.")
    try:
        pollutant_col = aqi_cols[pollutant_label]
        plot_df = df[[weather_var, pollutant_col]].dropna()
        fig_scatter = px.scatter(
            plot_df, x=weather_var, y=pollutant_col, trendline="ols",
            opacity=0.35, color_discrete_sequence=["#2E86AB"],
            labels={weather_var: weather_var, pollutant_col: pollutant_label},
        )
        fig_scatter.update_layout(height=420, title=f"{pollutant_label} vs. {weather_var}")
        st.plotly_chart(fig_scatter, use_container_width=True)
    except Exception as e:  # noqa: BLE001
        st.error(f"Drill-down scatter failed: {e}")
