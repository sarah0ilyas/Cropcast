"""
CropCast — Feature Engineering
Builds ML-ready features from the analytical base table.
Includes lag features, rolling averages, YoY changes,
weather anomalies, and trend features.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cropcast.config.settings import DATA_PROCESSED

logging.basicConfig(
    level="INFO",
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("features")


def load_base() -> pd.DataFrame:
    path = DATA_PROCESSED / "analytical_base.parquet"
    if not path.exists():
        raise FileNotFoundError("Run transform.py first.")
    df = pd.read_parquet(path)
    log.info("Loaded analytical base: %d rows", len(df))
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["country", "crop", "year"]).copy()
    for lag in [1, 2, 3]:
        df[f"yield_lag_{lag}"]      = df.groupby(["country", "crop"])["yield_mt_ha"].shift(lag)
        df[f"production_lag_{lag}"] = df.groupby(["country", "crop"])["production_mt"].shift(lag)
        df[f"area_lag_{lag}"]       = df.groupby(["country", "crop"])["area_ha"].shift(lag)
    log.info("Added lag features (1, 2, 3 years)")
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["country", "crop", "year"]).copy()
    for window in [3, 5]:
        df[f"yield_rolling_{window}y"] = (
            df.groupby(["country", "crop"])["yield_mt_ha"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=2).mean())
            .round(4)
        )
        df[f"production_rolling_{window}y"] = (
            df.groupby(["country", "crop"])["production_mt"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=2).mean())
            .round(2)
        )
    for col in ["avg_temp_max_c", "total_precip_mm",
                "growing_season_temp_max_c", "growing_season_precip_mm"]:
        if col in df.columns:
            df[f"{col}_rolling_3y"] = (
                df.groupby("country")[col]
                .transform(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
                .round(2)
            )
    log.info("Added rolling features (3y, 5y)")
    return df


def add_yoy_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["country", "crop", "year"]).copy()
    for col in ["yield_mt_ha", "production_mt", "area_ha"]:
        prev = df.groupby(["country", "crop"])[col].shift(1)
        df[f"{col}_yoy_pct"] = ((df[col] - prev) / prev * 100).round(2)
    for col in ["avg_temp_max_c", "total_precip_mm"]:
        if col in df.columns:
            prev = df.groupby("country")[col].shift(1)
            df[f"{col}_yoy_pct"] = ((df[col] - prev) / prev * 100).round(2)
    log.info("Added YoY change features")
    return df


def add_weather_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    weather_cols = [
        "avg_temp_max_c", "total_precip_mm",
        "growing_season_temp_max_c", "growing_season_precip_mm",
    ]
    for col in weather_cols:
        if col not in df.columns:
            continue
        mean = df.groupby("country")[col].transform("mean")
        std  = df.groupby("country")[col].transform("std")
        df[f"{col}_anomaly"] = ((df[col] - mean) / std).round(3)
    log.info("Added weather anomaly features")
    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["years_since_2000"] = df["year"] - 2000
    df["crop_year_rank"]   = (
        df.groupby(["country", "crop"])["year"].rank(method="first") - 1
    )
    # Latest year available in the dataset (used for nowcast/forecast)
    latest_year = df["year"].max()
    df["years_to_latest"] = latest_year - df["year"]
    log.info("Added trend features (latest year: %d)", latest_year)
    return df


def add_country_encoding(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.preprocessing import LabelEncoder
    df = df.copy()
    le_country = LabelEncoder()
    le_crop    = LabelEncoder()
    df["country_encoded"] = le_country.fit_transform(df["country"])
    df["crop_encoded"]    = le_crop.fit_transform(df["crop"])
    log.info("Added country and crop encodings")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_yoy_features(df)
    df = add_weather_anomalies(df)
    df = add_trend_features(df)
    df = add_country_encoding(df)
    return df


def save_features(df: pd.DataFrame) -> Path:
    out = DATA_PROCESSED / "features.parquet"
    df.to_parquet(out, index=False, compression="snappy")
    log.info("Saved features → %s", out)
    return out


def run_feature_engineering() -> pd.DataFrame:
    log.info("=" * 60)
    log.info("Starting feature engineering")
    log.info("=" * 60)

    df = load_base()
    df = build_features(df)
    save_features(df)

    log.info("=" * 60)
    log.info("Feature engineering complete — %d rows, %d columns",
             len(df), len(df.columns))
    log.info("=" * 60)
    return df


if __name__ == "__main__":
    df = run_feature_engineering()
    print("\nShape:", df.shape)
    print("\nAll columns:")
    for col in df.columns:
        nulls = df[col].isnull().sum()
        pct = round(nulls / len(df) * 100, 1)
        print(f"  {col:<45} nulls: {nulls} ({pct}%)")

    print("\nSample — Argentina Grapes:")
    cols = ["country", "crop", "year", "yield_mt_ha", "yield_lag_1",
            "yield_rolling_3y", "yield_mt_ha_yoy_pct", "avg_temp_max_c_anomaly"]
    print(df[(df["country"] == "Argentina") & (df["crop"] == "Grapes")][cols].head(10).to_string())
