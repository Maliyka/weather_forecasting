"""Shared configuration for the notebook and Streamlit dashboard."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TIME_COL = "last_updated"

GEO_COLS = ["country", "location_name", "latitude", "longitude"]

NUMERIC_WEATHER_COLS = [
    "temperature_celsius",
    "temperature_fahrenheit",
    "wind_mph",
    "wind_kph",
    "wind_degree",
    "pressure_mb",
    "pressure_in",
    "precip_mm",
    "precip_in",
    "humidity",
    "cloud",
    "feels_like_celsius",
    "feels_like_fahrenheit",
    "visibility_km",
    "visibility_miles",
    "uv_index",
    "gust_mph",
    "gust_kph",
]

AQI_COLS = [
    "air_quality_Carbon_Monoxide",
    "air_quality_Ozone",
    "air_quality_Nitrogen_dioxide",
    "air_quality_Sulphur_dioxide",
    "air_quality_PM2.5",
    "air_quality_PM10",
    "air_quality_us-epa-index",
    "air_quality_gb-defra-index",
]

_CSV_CANDIDATES = [
    ROOT / "data" / "GlobalWeatherRepository.csv",
    ROOT / "GlobalWeatherRepository.csv",
]
RAW_CSV_PATH = next((p for p in _CSV_CANDIDATES if p.exists()), _CSV_CANDIDATES[0])
