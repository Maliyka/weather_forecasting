"""
Section 4 — Spatial Insights
================================
Interactive geospatial mapping plus pre-computed regional climate clustering.
"""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboard.utils.artifacts import available_cluster_k, cluster_feature_names, load_cluster_results
from dashboard.utils.data_loader import available_numeric_cols


def render(df) -> None:
    st.subheader("🌍 Spatial Insights")
    st.caption("Geographical distribution of weather/AQI conditions and regional climate clustering.")

    if df.empty:
        st.info("No data loaded — see the error above.")
        return

    if not {"latitude", "longitude"}.issubset(df.columns):
        st.warning("`latitude`/`longitude` columns not found — spatial analysis unavailable.")
        return

    numeric_cols = available_numeric_cols(df)
    if not numeric_cols:
        st.warning("No numeric weather/AQI columns available to color the map.")
        return

    st.markdown("**Geographical Distribution**")
    color_col = st.selectbox(
        "Color map by", numeric_cols,
        index=numeric_cols.index("temperature_celsius") if "temperature_celsius" in numeric_cols else 0,
    )
    try:
        if "location_name" in df.columns and "last_updated" in df.columns:
            latest = df.sort_values("last_updated").groupby("location_name").tail(1)
        else:
            latest = df.drop_duplicates(subset=["latitude", "longitude"])

        hover_cols = [c for c in ["location_name", "country", color_col] if c in latest.columns]
        fig = px.scatter_geo(
            latest, lat="latitude", lon="longitude", color=color_col,
            hover_name="location_name" if "location_name" in latest.columns else None,
            hover_data=hover_cols,
            color_continuous_scale="RdYlBu_r" if "temp" in color_col.lower() else "Viridis",
            projection="natural earth",
        )
        fig.update_layout(height=520, title=f"Latest {color_col} by Location",
                           margin=dict(l=0, r=0, t=40, b=0))
        fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="white")))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:  # noqa: BLE001
        st.error(f"Map rendering failed: {e}")

    st.divider()

    st.markdown("**Regional Climate Clustering (KMeans)**")
    features = cluster_feature_names()
    st.caption(
        f"Pre-trained on: {', '.join(features)}. "
        "Clusters group locations by geography and climate signature."
    )

    k_options = available_cluster_k()
    default_k = 6 if 6 in k_options else k_options[len(k_options) // 2]
    k = st.selectbox("Number of clusters (k)", k_options, index=k_options.index(default_k))

    try:
        clustered, profile = load_cluster_results(k)
        if clustered is None:
            st.warning(f"No clustering artifact found for k={k}.")
            return

        fig = px.scatter_geo(
            clustered, lat="latitude", lon="longitude", color="climate_cluster",
            projection="natural earth",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(marker=dict(size=6, line=dict(width=0.3, color="white")))
        fig.update_layout(height=480, title=f"Climate Clusters (k={k})", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        if profile is not None:
            st.markdown("**Cluster Profiles (mean values)**")
            st.dataframe(profile, use_container_width=True)
    except Exception as e:  # noqa: BLE001
        st.error(f"Clustering display failed: {e}")
