"""
Section 1 — Executive KPIs
============================
Global statistical summaries, variance metrics, and missing-data profiling.
Audience: executives scanning for data health and headline numbers, not
modeling detail — so this section stays at the KPI-card / single-chart level.
"""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboard.utils.data_loader import available_numeric_cols, missingness_profile, variance_profile


def render(df) -> None:
    st.subheader("📊 Executive KPIs")
    st.caption(
        "Headline statistics, variability, and data-quality posture across the full dataset. "
        "Use this section to sanity-check coverage before trusting downstream models."
    )

    if df.empty:
        st.info("No data loaded — see the error above.")
        return

    numeric_cols = available_numeric_cols(df)
    if not numeric_cols:
        st.warning("No recognized numeric weather/AQI columns found in this file.")
        return

    # ---------------------------------------------------------------- KPI cards
    try:
        n_rows = len(df)
        n_countries = df["country"].nunique() if "country" in df.columns else None
        n_locations = df["location_name"].nunique() if "location_name" in df.columns else None
        date_min, date_max = df["last_updated"].min(), df["last_updated"].max()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{n_rows:,}", help="Total rows after timestamp parsing & dedup.")
        c2.metric("Countries Covered", f"{n_countries:,}" if n_countries else "N/A",
                  help="Distinct values in the `country` column.")
        c3.metric("Monitored Locations", f"{n_locations:,}" if n_locations else "N/A",
                  help="Distinct values in `location_name` — i.e. weather stations/cities.")
        c4.metric("Date Range", f"{(date_max - date_min).days} days",
                  help=f"From {date_min.date()} to {date_max.date()} (UTC).")
    except Exception as e:  # noqa: BLE001
        st.warning(f"Could not compute headline KPI cards: {e}")

    st.divider()

    # ---------------------------------------------------------------- Summary stats
    left, right = st.columns([1.3, 1])

    with left:
        st.markdown("**Global Statistical Summary**")
        try:
            summary = df[numeric_cols].describe().T.round(2)
            st.dataframe(
                summary.style.background_gradient(cmap="Blues", subset=["mean", "std"]),
                use_container_width=True,
                height=380,
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not compute summary statistics: {e}")

    with right:
        st.markdown("**Variance / Volatility (Coefficient of Variation)**")
        st.caption("Higher CV = more relative spread — useful for spotting which variables are most volatile.")
        try:
            var_df = variance_profile(df, numeric_cols).reset_index().rename(columns={"index": "feature"})
            fig = px.bar(
                var_df.head(12), x="cv_pct", y="feature", orientation="h",
                labels={"cv_pct": "Coefficient of Variation (%)", "feature": ""},
                color="cv_pct", color_continuous_scale="Blues",
            )
            fig.update_traces(hovertemplate="<b>%{y}</b><br>CV: %{x:.1f}%<extra></extra>")
            fig.update_layout(height=380, coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not compute variance profile: {e}")

    st.divider()

    # ---------------------------------------------------------------- Missing-data profiling
    st.markdown("**Missing-Data Profiling**")
    st.caption("Columns with non-trivial missingness should be treated cautiously in downstream modeling.")
    try:
        miss_df = missingness_profile(df)
        miss_df = miss_df[miss_df["missing_pct"] > 0].reset_index().rename(columns={"index": "column"})

        if miss_df.empty:
            st.success("No missing values detected in this dataset. ✅")
        else:
            fig = px.bar(
                miss_df.head(20), x="missing_pct", y="column", orientation="h",
                labels={"missing_pct": "% Missing", "column": ""},
                color="missing_pct", color_continuous_scale="Reds",
            )
            fig.update_traces(
                hovertemplate="<b>%{y}</b><br>Missing: %{x:.2f}%<extra></extra>"
            )
            fig.update_layout(height=max(300, 22 * len(miss_df.head(20))),
                               coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not build missing-data profile: {e}")
