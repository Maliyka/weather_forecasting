"""Load pre-trained dashboard artifacts (built offline via scripts/build_dashboard_artifacts.py)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


@lru_cache(maxsize=1)
def _manifest() -> dict:
    path = ARTIFACTS_DIR / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def artifacts_ready() -> bool:
    return (ARTIFACTS_DIR / "manifest.json").exists()


@st.cache_data(show_spinner=False)
def load_forecast(target_col: str, horizon: int) -> dict | None:
    path = ARTIFACTS_DIR / "forecast" / f"{target_col}_h{horizon}.joblib"
    if not path.exists():
        return None
    result = joblib.load(path)
    result["dates"] = pd.to_datetime(result["dates"])
    return result


@st.cache_data(show_spinner=False)
def load_anomaly_results(contamination: float) -> pd.DataFrame | None:
    tag = f"{int(contamination * 100):02d}"
    path = ARTIFACTS_DIR / "anomaly" / f"isolation_forest_{tag}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_cluster_results(k: int) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    clustered_path = ARTIFACTS_DIR / "clustering" / f"kmeans_k{k}.parquet"
    profile_path = ARTIFACTS_DIR / "clustering" / f"kmeans_k{k}_profile.parquet"
    if not clustered_path.exists():
        return None, None
    clustered = pd.read_parquet(clustered_path)
    profile = pd.read_parquet(profile_path) if profile_path.exists() else None
    return clustered, profile


@st.cache_data(show_spinner=False)
def load_shap_bundle(target_col: str) -> dict | None:
    shap_dir = ARTIFACTS_DIR / "shap"
    npz_path = shap_dir / f"{target_col}.npz"
    features_path = shap_dir / f"{target_col}_features.parquet"
    importance_path = shap_dir / f"{target_col}_importance.parquet"

    if npz_path.exists() and features_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        return {
            "mode": "shap",
            "shap_values": data["shap_values"],
            "feature_names": list(data["feature_names"]),
            "X": pd.read_parquet(features_path),
        }
    if importance_path.exists():
        return {"mode": "importance", "importance": pd.read_parquet(importance_path)}
    return None


def available_forecast_horizons(target_col: str) -> list[int]:
    manifest = _manifest()
    horizons = manifest.get("forecast_horizons", [7, 14, 21, 30, 60])
    return [h for h in horizons if (ARTIFACTS_DIR / "forecast" / f"{target_col}_h{h}.joblib").exists()]


def available_contaminations() -> list[float]:
    manifest = _manifest()
    return manifest.get("anomaly_contaminations", [0.01, 0.02, 0.05, 0.10])


def available_cluster_k() -> list[int]:
    manifest = _manifest()
    return manifest.get("cluster_k_range", list(range(2, 11)))


def cluster_feature_names() -> list[str]:
    manifest = _manifest()
    return manifest.get("cluster_features", [])
