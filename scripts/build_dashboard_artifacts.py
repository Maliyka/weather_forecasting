#!/usr/bin/env python3
"""Pre-train ML artifacts for the Streamlit dashboard (run once offline)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RAW_CSV_PATH, TIME_COL  # noqa: E402
from dashboard.utils.data_loader import available_numeric_cols  # noqa: E402

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

ARTIFACTS_DIR = ROOT / "dashboard" / "artifacts"
RANDOM_STATE = 42
FORECAST_HORIZONS = [7, 14, 21, 30, 60]
FORECAST_TARGETS = ["temperature_celsius", "precip_mm"]
ANOMALY_CONTAMINATIONS = [0.01, 0.02, 0.05, 0.10]
CLUSTER_K_RANGE = range(2, 11)
CLUSTER_FEATURES = ["latitude", "longitude", "temperature_celsius", "humidity", "precip_mm"]
SHAP_TARGET = "temperature_celsius"
SHAP_SAMPLE_SIZE = 3000


def load_prepared_df() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV_PATH)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL)
    return df


def daily_global_series(df: pd.DataFrame, value_col: str) -> pd.Series:
    s = df.set_index(TIME_COL)[value_col].resample("D").mean()
    return s.interpolate(method="time").dropna()


def make_lag_features(series: pd.Series, n_lags: int = 7) -> pd.DataFrame:
    frame = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        frame[f"lag_{lag}"] = frame["y"].shift(lag)
    frame["dayofyear"] = frame.index.dayofyear
    frame["month"] = frame.index.month
    frame["rolling_mean_7"] = frame["y"].shift(1).rolling(7).mean()
    return frame.dropna()


def run_ensemble_forecast(series: pd.Series, horizon: int) -> dict:
    sup = make_lag_features(series)
    train, test = sup.iloc[:-horizon], sup.iloc[-horizon:]
    y_true = test["y"].values
    feature_cols = [c for c in sup.columns if c != "y"]

    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb.fit(train[feature_cols], train["y"])
    xgb_pred = xgb.predict(test[feature_cols])

    if PROPHET_AVAILABLE:
        try:
            train_series = series.iloc[:-horizon]
            p_df = train_series.reset_index()
            p_df.columns = ["ds", "y"]
            p_df["ds"] = p_df["ds"].dt.tz_localize(None)
            prophet = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            prophet.fit(p_df)
            future = prophet.make_future_dataframe(periods=horizon, include_history=False)
            prophet_pred = prophet.predict(future)["yhat"].values
            prophet_pred = np.resize(prophet_pred, horizon)
        except Exception:
            prophet_pred = np.full(horizon, train["y"].mean())
    else:
        prophet_pred = np.full(horizon, train["y"].mean())

    def metrics(y_pred: np.ndarray) -> dict[str, float]:
        return {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAPE": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        }

    xgb_metrics = metrics(xgb_pred)
    prophet_metrics = metrics(prophet_pred)
    inv = {
        "xgboost": 1 / max(xgb_metrics["RMSE"], 1e-6),
        "prophet": 1 / max(prophet_metrics["RMSE"], 1e-6),
    }
    total = sum(inv.values())
    weights = {k: v / total for k, v in inv.items()}
    ensemble_pred = weights["xgboost"] * xgb_pred + weights["prophet"] * prophet_pred

    return {
        "dates": [d.isoformat() for d in test.index.to_pydatetime()],
        "y_true": y_true.tolist(),
        "xgb_pred": xgb_pred.tolist(),
        "prophet_pred": prophet_pred.tolist(),
        "ensemble_pred": ensemble_pred.tolist(),
        "xgb_metrics": xgb_metrics,
        "prophet_metrics": prophet_metrics,
        "ensemble_metrics": metrics(ensemble_pred),
        "weights": weights,
        "prophet_available": PROPHET_AVAILABLE,
    }


def run_isolation_forest(df: pd.DataFrame, feature_cols: list[str], contamination: float) -> pd.DataFrame:
    sub = df[feature_cols].dropna().copy()
    X = StandardScaler().fit_transform(sub)
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    preds = model.fit_predict(X)
    sub["anomaly"] = preds == -1
    sub["anomaly_score"] = model.decision_function(X)
    return sub


def run_kmeans(df: pd.DataFrame, feature_cols: list[str], k: int) -> pd.DataFrame:
    sub = df[feature_cols].dropna().copy()
    X = StandardScaler().fit_transform(sub)
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    sub["climate_cluster"] = model.fit_predict(X).astype(str)
    return sub


def build_shap_artifact(df: pd.DataFrame, numeric_cols: list[str], target_col: str) -> None:
    shap_dir = ARTIFACTS_DIR / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    feature_cols = [c for c in numeric_cols if c != target_col]
    model_df = df[feature_cols + [target_col]].dropna()
    if len(model_df) > SHAP_SAMPLE_SIZE:
        model_df = model_df.sample(SHAP_SAMPLE_SIZE, random_state=RANDOM_STATE)

    X = model_df[feature_cols]
    y = model_df[target_col]
    model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=RANDOM_STATE)
    model.fit(X, y)

    if SHAP_AVAILABLE:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        np.savez_compressed(
            shap_dir / f"{target_col}.npz",
            shap_values=shap_values,
            feature_names=np.array(feature_cols),
        )
        X.to_parquet(shap_dir / f"{target_col}_features.parquet", index=False)
    else:
        importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        importance.to_frame("importance").to_parquet(shap_dir / f"{target_col}_importance.parquet")


def main() -> None:
    print(f"Loading dataset from {RAW_CSV_PATH} ...")
    df = load_prepared_df()
    numeric_cols = available_numeric_cols(df)

    forecast_dir = ARTIFACTS_DIR / "forecast"
    anomaly_dir = ARTIFACTS_DIR / "anomaly"
    cluster_dir = ARTIFACTS_DIR / "clustering"
    for d in (forecast_dir, anomaly_dir, cluster_dir):
        d.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "forecast_targets": FORECAST_TARGETS,
        "forecast_horizons": FORECAST_HORIZONS,
        "anomaly_contaminations": ANOMALY_CONTAMINATIONS,
        "cluster_k_range": list(CLUSTER_K_RANGE),
        "cluster_features": CLUSTER_FEATURES,
        "shap_target": SHAP_TARGET,
        "prophet_available": PROPHET_AVAILABLE,
        "shap_available": SHAP_AVAILABLE,
    }

    print("Building forecast artifacts ...")
    for target in FORECAST_TARGETS:
        if target not in df.columns:
            continue
        series = daily_global_series(df, target)
        for horizon in FORECAST_HORIZONS:
            if len(series) < horizon + 30:
                continue
            result = run_ensemble_forecast(series, horizon)
            out = forecast_dir / f"{target}_h{horizon}.joblib"
            joblib.dump(result, out)
            print(f"  saved {out.name}")

    print("Building anomaly detection artifacts ...")
    feature_cols = list(dict.fromkeys(numeric_cols))
    for contamination in ANOMALY_CONTAMINATIONS:
        result = run_isolation_forest(df, feature_cols, contamination)
        tag = f"{int(contamination * 100):02d}"
        out = anomaly_dir / f"isolation_forest_{tag}.parquet"
        result.to_parquet(out, index=False)
        print(f"  saved {out.name} ({int(result['anomaly'].sum())} anomalies)")

    print("Building clustering artifacts ...")
    cluster_features = [c for c in CLUSTER_FEATURES if c in df.columns]
    for k in CLUSTER_K_RANGE:
        clustered = run_kmeans(df, cluster_features, k)
        out = cluster_dir / f"kmeans_k{k}.parquet"
        clustered.to_parquet(out, index=False)
        profile = clustered.groupby("climate_cluster")[cluster_features].mean().round(2)
        profile["n_locations"] = clustered["climate_cluster"].value_counts()
        profile.to_parquet(cluster_dir / f"kmeans_k{k}_profile.parquet")
        print(f"  saved k={k}")

    print("Building SHAP artifact ...")
    if SHAP_TARGET in df.columns:
        build_shap_artifact(df, numeric_cols, SHAP_TARGET)

    manifest_path = ARTIFACTS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Done. Artifacts written to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
