"""
CropCast — Forecast Engine
Rolling 5-year forecast with:
- Auto-detects latest FAO data year
- Nowcasts current unreported year (FAO lags 12-18 months)
- Forecasts 4 years beyond nowcast
- Prediction intervals that widen over time
- Prophet trend extrapolation for long-horizon stability
- Window auto-advances when new FAO data is ingested

Usage:
    python3 cropcast/forecast/engine.py
    python3 cropcast/forecast/engine.py --crop Grapes
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

from cropcast.config.settings import DATA_PROCESSED, MODELS_DIR, config

logging.basicConfig(
    level="INFO",
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("forecast")

FEATURE_COLS = [
    "yield_lag_1", "yield_lag_2", "yield_lag_3",
    "yield_rolling_3y", "yield_rolling_5y",
    "yield_mt_ha_yoy_pct",
    "area_ha", "area_lag_1", "area_ha_yoy_pct",
    "avg_temp_max_c", "avg_temp_min_c", "total_precip_mm", "avg_et0_mm",
    "growing_season_temp_max_c", "growing_season_temp_min_c",
    "growing_season_precip_mm",
    "avg_temp_max_c_anomaly", "total_precip_mm_anomaly",
    "growing_season_temp_max_c_anomaly", "growing_season_precip_mm_anomaly",
    "years_since_2000", "crop_year_rank", "country_encoded",
]


def load_model(crop: str) -> XGBRegressor:
    slug = crop.lower().replace(" ", "_").replace(",", "")
    path = MODELS_DIR / f"xgb_{slug}.json"
    if not path.exists():
        raise FileNotFoundError(f"No model found for {crop}. Run train.py first.")
    model = XGBRegressor()
    model.load_model(str(path))
    return model


def load_features() -> pd.DataFrame:
    path = DATA_PROCESSED / "features.parquet"
    return pd.read_parquet(path)


def get_latest_year(df: pd.DataFrame) -> int:
    """Auto-detect the latest year with actual FAO data."""
    return int(df["year"].max())


def build_forecast_row(df: pd.DataFrame, country: str, crop: str,
                        target_year: int) -> pd.Series:
    """
    Build a feature vector for a future year using:
    - Lag features from the most recent actual data
    - Rolling averages from historical data
    - Weather features: use country average (best estimate for future)
    - Trend features extrapolated forward
    """
    history = df[(df["country"] == country) & (df["crop"] == crop)].copy()
    history = history.sort_values("year")

    if len(history) == 0:
        return None

    latest = history.iloc[-1]
    latest_actual_year = history["year"].max()

    row = {}
    years_ahead = target_year - latest_actual_year

    # Lag features — shift by years_ahead
    for lag in [1, 2, 3]:
        effective_lag = lag + years_ahead - 1
        if effective_lag <= len(history):
            row[f"yield_lag_{lag}"] = history.iloc[-(effective_lag)]["yield_mt_ha"]
        else:
            row[f"yield_lag_{lag}"] = history["yield_mt_ha"].mean()

        if effective_lag <= len(history):
            row[f"production_lag_{lag}"] = history.iloc[-(effective_lag)]["production_mt"] if "production_mt" in history.columns else np.nan
            row[f"area_lag_{lag}"]       = history.iloc[-(effective_lag)]["area_ha"] if "area_ha" in history.columns else np.nan
        else:
            row[f"production_lag_{lag}"] = history["production_mt"].mean() if "production_mt" in history.columns else np.nan
            row[f"area_lag_{lag}"]       = history["area_ha"].mean() if "area_ha" in history.columns else np.nan

    # Rolling averages from recent history
    recent = history.tail(5)
    row["yield_rolling_3y"]      = history.tail(3)["yield_mt_ha"].mean()
    row["yield_rolling_5y"]      = history.tail(5)["yield_mt_ha"].mean()
    row["production_rolling_3y"] = history.tail(3)["production_mt"].mean() if "production_mt" in history.columns else np.nan
    row["production_rolling_5y"] = history.tail(5)["production_mt"].mean() if "production_mt" in history.columns else np.nan

    # YoY features — use recent average trend
    if "yield_mt_ha_yoy_pct" in history.columns:
        row["yield_mt_ha_yoy_pct"] = history.tail(3)["yield_mt_ha_yoy_pct"].mean()
    if "production_mt_yoy_pct" in history.columns:
        row["production_mt_yoy_pct"] = history.tail(3)["production_mt_yoy_pct"].mean()
    if "area_ha_yoy_pct" in history.columns:
        row["area_ha_yoy_pct"] = history.tail(3)["area_ha_yoy_pct"].mean()

    # Area — extrapolate recent trend
    if "area_ha" in history.columns:
        recent_area_growth = history.tail(3)["area_ha"].pct_change().mean()
        row["area_ha"] = latest["area_ha"] * (1 + recent_area_growth) ** years_ahead if pd.notna(latest.get("area_ha")) else np.nan

    # Weather — use country long-run average (best estimate for future)
    weather_cols = [
        "avg_temp_max_c", "avg_temp_min_c", "total_precip_mm", "avg_et0_mm",
        "growing_season_temp_max_c", "growing_season_temp_min_c",
        "growing_season_precip_mm",
    ]
    for col in weather_cols:
        if col in history.columns:
            row[col] = history[col].mean()

    # Weather anomalies — zero for future (unknown deviation from mean)
    anomaly_cols = [
        "avg_temp_max_c_anomaly", "total_precip_mm_anomaly",
        "growing_season_temp_max_c_anomaly", "growing_season_precip_mm_anomaly",
    ]
    for col in anomaly_cols:
        row[col] = 0.0

    # Rolling weather averages
    rolling_weather = [
        "avg_temp_max_c_rolling_3y", "total_precip_mm_rolling_3y",
        "growing_season_temp_max_c_rolling_3y", "growing_season_precip_mm_rolling_3y",
    ]
    for col in rolling_weather:
        base = col.replace("_rolling_3y", "")
        if base in history.columns:
            row[col] = history.tail(3)[base].mean()

    # Trend features
    row["years_since_2000"] = target_year - 2000
    row["crop_year_rank"]   = len(history) + years_ahead
    row["years_to_latest"]  = 0
    row["country_encoded"]  = latest.get("country_encoded", 0)
    row["crop_encoded"]     = latest.get("crop_encoded", 0)

    return pd.Series(row)


def prophet_trend_forecast(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Use Prophet to capture long-term trend for ensemble blending."""
    series = history.groupby("year")["yield_mt_ha"].mean().reset_index()
    if len(series) < 5:
        return None

    prophet_df = pd.DataFrame({
        "ds": pd.to_datetime(series["year"].astype(str) + "-06-01"),
        "y":  series["yield_mt_ha"]
    })

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.90,
        changepoint_prior_scale=0.05,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prophet_df)

    future = model.make_future_dataframe(periods=horizon, freq="YE")
    forecast = model.predict(future)
    forecast["year"] = forecast["ds"].dt.year
    return forecast[["year", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon)


def forecast_crop(crop: str, horizon: int = 5) -> pd.DataFrame:
    """
    Generate rolling 5-year forecast for all countries for a given crop.
    Returns a DataFrame with point forecasts and prediction intervals.
    """
    log.info("Forecasting: %s (%d years ahead)", crop, horizon)

    df      = load_features()
    df_crop = df[df["crop"] == crop].copy()
    model   = load_model(crop)

    latest_year    = get_latest_year(df_crop)
    nowcast_year   = latest_year + 1
    forecast_years = list(range(nowcast_year, nowcast_year + horizon))

    log.info("Latest FAO data: %d | Nowcast: %d | Forecast: %d-%d",
             latest_year, nowcast_year,
             forecast_years[1], forecast_years[-1])

    results = []
    countries = df_crop["country"].unique()

    for country in countries:
        country_history = df_crop[df_crop["country"] == country]
        if len(country_history) < 5:
            continue

        # Prophet trend for this country
        prophet_fc = prophet_trend_forecast(country_history, horizon + 2)

        for i, target_year in enumerate(forecast_years):
            row = build_forecast_row(df_crop, country, crop, target_year)
            if row is None:
                continue

            available = [c for c in FEATURE_COLS if c in row.index]
            X = pd.DataFrame([row[available].fillna(row[available].median())])
            X = X.fillna(0)

            xgb_pred = float(model.predict(X)[0])

            # Blend with Prophet trend (weighted ensemble)
            prophet_pred = xgb_pred
            if prophet_fc is not None:
                prophet_row = prophet_fc[prophet_fc["year"] == target_year]
                if len(prophet_row) > 0:
                    prophet_pred = float(prophet_row["yhat"].iloc[0])

            # Ensemble: XGBoost weight decreases for longer horizons
            xgb_weight     = max(0.4, config.model.xgb_weight - i * 0.05)
            prophet_weight = 1 - xgb_weight
            ensemble_pred  = xgb_weight * xgb_pred + prophet_weight * prophet_pred

            # Prediction intervals — widen as horizon increases
            base_std      = country_history["yield_mt_ha"].std()
            horizon_factor = 1 + i * 0.3
            pi_width      = base_std * horizon_factor * 1.645

            is_nowcast = (target_year == nowcast_year)

            results.append({
                "crop":           crop,
                "country":        country,
                "year":           target_year,
                "forecast_type":  "nowcast" if is_nowcast else "forecast",
                "years_ahead":    i + 1,
                "xgb_forecast":   round(xgb_pred, 2),
                "prophet_forecast": round(prophet_pred, 2),
                "ensemble_forecast": round(ensemble_pred, 2),
                "pi_lower":       round(max(0, ensemble_pred - pi_width), 2),
                "pi_upper":       round(ensemble_pred + pi_width, 2),
                "pi_width":       round(pi_width * 2, 2),
                "confidence_pct": round(max(50, 90 - i * 8), 1),
            })

    forecast_df = pd.DataFrame(results)
    log.info("Generated %d forecasts for %s across %d countries",
             len(forecast_df), crop, len(countries))
    return forecast_df


def run_all_forecasts(horizon: int = 5) -> pd.DataFrame:
    """Generate forecasts for all crops and save to disk."""
    log.info("=" * 60)
    log.info("Running CropCast rolling forecast engine")
    log.info("=" * 60)

    df       = load_features()
    crops    = df["crop"].unique().tolist()
    all_fc   = []

    for crop in crops:
        try:
            fc = forecast_crop(crop, horizon=horizon)
            all_fc.append(fc)
        except Exception as e:
            log.error("Failed to forecast %s: %s", crop, e)

    if not all_fc:
        log.error("No forecasts generated")
        return pd.DataFrame()

    combined = pd.concat(all_fc, ignore_index=True)

    out_path = DATA_PROCESSED / "forecasts.parquet"
    combined.to_parquet(out_path, index=False, compression="snappy")
    log.info("Saved %d forecasts -> %s", len(combined), out_path)

    log.info("=" * 60)
    log.info("Forecast complete")
    log.info("  Crops:     %d", combined["crop"].nunique())
    log.info("  Countries: %d", combined["country"].nunique())
    log.info("  Years:     %s - %s", combined["year"].min(), combined["year"].max())
    log.info("=" * 60)

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop",    default=None)
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()

    if args.crop:
        fc = forecast_crop(args.crop, horizon=args.horizon)
        print(f"\nForecast for {args.crop}:")
        print(fc[["country", "year", "forecast_type", "ensemble_forecast",
                   "pi_lower", "pi_upper", "confidence_pct"]].to_string())
    else:
        fc = run_all_forecasts(horizon=args.horizon)
        print(f"\nTotal forecasts: {len(fc)}")
        print("\nSample — Grapes:")
        sample = fc[fc["crop"] == "Grapes"][
            ["country", "year", "forecast_type",
             "ensemble_forecast", "pi_lower", "pi_upper", "confidence_pct"]
        ].head(15)
        print(sample.to_string())
