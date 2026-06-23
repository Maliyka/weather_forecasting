"""
Data loading & lightweight preparation layer for the dashboard.

All functions are wrapped in try/except with Streamlit-surfaced error
messages (rather than raw tracebacks) and cached via st.cache_data so the
30MB+ CSV is parsed once per session, not once per widget interaction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root on sys.path
from src.config import AQI_COLS, GEO_COLS, NUMERIC_WEATHER_COLS, RAW_CSV_PATH, TIME_COL  # noqa: E402

REQUIRED_COLS = [TIME_COL] + GEO_COLS


@st.cache_data(show_spinner="Loading Global Weather Repository data...")
def load_data() -> pd.DataFrame:
    """Load + lightly clean the bundled CSV. Returns an empty DataFrame on failure
    (callers must check `.empty` and surface a user-facing message)."""
    path = RAW_CSV_PATH
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(
            f"Dataset not found at `{path}`. Download `GlobalWeatherRepository.csv` "
            f"from Kaggle and place it in the `data/` folder."
        )
        return pd.DataFrame()
    except pd.errors.ParserError as e:
        st.error(f"Could not parse the CSV — file may be corrupted or malformed: {e}")
        return pd.DataFrame()
    except Exception as e:  # noqa: BLE001 — surface unexpected errors to the user, not a stack trace
        st.error(f"Unexpected error loading data: {e}")
        return pd.DataFrame()

    missing_required = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_required:
        st.error(f"Dataset is missing required columns: {missing_required}. "
                 f"Confirm you downloaded the correct Kaggle dataset.")
        return pd.DataFrame()

    try:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
        df = df.dropna(subset=[TIME_COL])
        df = df.sort_values(TIME_COL)
        # Vectorized calendar features — reused across multiple dashboard tabs
        df["date"] = df[TIME_COL].dt.date
        df["year"] = df[TIME_COL].dt.year
        df["month"] = df[TIME_COL].dt.month
        df["month_name"] = df[TIME_COL].dt.month_name()
        df["day_of_week"] = df[TIME_COL].dt.day_name()
        df["week"] = df[TIME_COL].dt.isocalendar().week.astype(int)
    except Exception as e:  # noqa: BLE001
        st.error(f"Error while parsing timestamps / building calendar features: {e}")
        return pd.DataFrame()

    return df


def available_numeric_cols(df: pd.DataFrame) -> list[str]:
    """Numeric weather + AQI columns actually present in the loaded file."""
    return [c for c in NUMERIC_WEATHER_COLS + AQI_COLS if c in df.columns]


@st.cache_data(show_spinner=False)
def daily_global_series(df: pd.DataFrame, value_col: str) -> pd.Series:
    """Resample a numeric column to a daily global mean time series."""
    try:
        s = df.set_index(TIME_COL)[value_col].resample("D").mean()
        return s.interpolate(method="time").dropna()
    except Exception as e:  # noqa: BLE001
        st.warning(f"Could not build daily series for `{value_col}`: {e}")
        return pd.Series(dtype=float)


@st.cache_data(show_spinner=False)
def missingness_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing count/percentage, sorted descending."""
    miss = df.isna().sum()
    pct = (miss / len(df) * 100) if len(df) else miss * 0.0
    profile = pd.DataFrame({"missing_count": miss, "missing_pct": pct})
    return profile.sort_values("missing_pct", ascending=False)


@st.cache_data(show_spinner=False)
def variance_profile(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Mean / std / coefficient-of-variation per numeric column (vectorized)."""
    valid_cols = [c for c in cols if c in df.columns]
    means = df[valid_cols].mean()
    stds = df[valid_cols].std()
    cv = (stds / means.replace(0, np.nan)).abs() * 100
    return pd.DataFrame({"mean": means, "std": stds, "cv_pct": cv}).sort_values("cv_pct", ascending=False)
