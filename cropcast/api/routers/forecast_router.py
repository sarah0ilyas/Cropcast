"""
CropCast — Forecast Router
Serves rolling 5-year forecasts and historical data.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


@router.get("/")
def get_forecasts(
    crop:    str           = Query(..., description="Crop name e.g. Grapes"),
    country: Optional[str] = Query(None, description="Filter by country"),
    year:    Optional[int] = Query(None, description="Filter by year"),
):
    """Rolling 5-year forecast for a crop, optionally filtered."""
    from cropcast.api.main import state
    df = state.forecasts_df.copy()
    df = df[df["crop"] == crop]
    if df.empty:
        raise HTTPException(404, f"No forecasts for crop: {crop}")
    if country:
        df = df[df["country"] == country]
        if df.empty:
            raise HTTPException(404, f"No forecasts for {crop} in {country}")
    if year:
        df = df[df["year"] == year]

    return {
        "crop":     crop,
        "country":  country or "all",
        "records":  df.to_dict(orient="records"),
        "n_records": len(df),
        "latest_actual_year": int(state.features_df["year"].max()),
        "nowcast_year": int(state.forecasts_df["year"].min()),
    }


@router.get("/nowcast")
def get_nowcast(
    crop:    str           = Query(..., description="Crop name"),
    country: Optional[str] = Query(None, description="Filter by country"),
):
    """Current year nowcast (latest unreported FAO year + 1)."""
    from cropcast.api.main import state
    df = state.forecasts_df.copy()
    df = df[(df["crop"] == crop) & (df["forecast_type"] == "nowcast")]
    if country:
        df = df[df["country"] == country]
    if df.empty:
        raise HTTPException(404, f"No nowcast for {crop}")
    return {
        "crop":      crop,
        "year":      int(df["year"].iloc[0]),
        "type":      "nowcast",
        "records":   df.to_dict(orient="records"),
    }


@router.get("/history")
def get_history(
    crop:    str           = Query(..., description="Crop name"),
    country: Optional[str] = Query(None, description="Filter by country"),
):
    """Historical actual yield data from FAO."""
    from cropcast.api.main import state
    df = state.features_df.copy()
    df = df[df["crop"] == crop].dropna(subset=["yield_mt_ha"])
    if country:
        df = df[df["country"] == country]
    if df.empty:
        raise HTTPException(404, f"No history for {crop}")

    cols = ["country", "crop", "year", "yield_mt_ha",
            "production_mt", "area_ha",
            "avg_temp_max_c", "total_precip_mm",
            "yield_mt_ha_yoy_pct"]
    available = [c for c in cols if c in df.columns]
    return {
        "crop":      crop,
        "country":   country or "all",
        "records":   df[available].sort_values(["country","year"]).to_dict(orient="records"),
        "n_records": len(df),
    }


@router.get("/combined")
def get_combined(
    crop:    str           = Query(..., description="Crop name"),
    country: Optional[str] = Query(None, description="Filter by country"),
):
    """
    Combined historical + forecast in one response.
    Ideal for charting actual vs forecast with confidence intervals.
    """
    from cropcast.api.main import state
    import pandas as pd

    hist = state.features_df[state.features_df["crop"] == crop].copy()
    fc   = state.forecasts_df[state.forecasts_df["crop"] == crop].copy()

    if country:
        hist = hist[hist["country"] == country]
        fc   = fc[fc["country"] == country]

    hist_records = []
    for _, row in hist.dropna(subset=["yield_mt_ha"]).iterrows():
        hist_records.append({
            "country":    row["country"],
            "year":       int(row["year"]),
            "type":       "actual",
            "yield":      round(float(row["yield_mt_ha"]), 2),
            "pi_lower":   None,
            "pi_upper":   None,
            "confidence": 100,
        })

    fc_records = []
    for _, row in fc.iterrows():
        fc_records.append({
            "country":    row["country"],
            "year":       int(row["year"]),
            "type":       row["forecast_type"],
            "yield":      round(float(row["ensemble_forecast"]), 2),
            "pi_lower":   round(float(row["pi_lower"]), 2),
            "pi_upper":   round(float(row["pi_upper"]), 2),
            "confidence": int(row["confidence_pct"]),
        })

    return {
        "crop":    crop,
        "country": country or "all",
        "history": hist_records,
        "forecast": fc_records,
        "latest_actual_year": int(state.features_df["year"].max()),
        "nowcast_year": int(state.forecasts_df["year"].min()),
        "forecast_horizon": int(state.forecasts_df["year"].max()),
    }


@router.get("/risk")
def get_risk(
    crop:    str           = Query(..., description="Crop name"),
    year:    Optional[int] = Query(None, description="Year (default: nowcast year)"),
):
    """
    Supply chain risk scores derived from weather anomalies and
    forecast uncertainty. Higher score = higher risk.
    """
    from cropcast.api.main import state
    import numpy as np

    fc = state.forecasts_df.copy()
    fc = fc[fc["crop"] == crop]

    if year:
        fc = fc[fc["year"] == year]
    else:
        fc = fc[fc["forecast_type"] == "nowcast"]

    if fc.empty:
        raise HTTPException(404, f"No risk data for {crop}")

    hist = state.features_df[state.features_df["crop"] == crop].copy()

    records = []
    for _, row in fc.iterrows():
        country = row["country"]
        country_hist = hist[hist["country"] == country]

        pi_width_pct = (row["pi_upper"] - row["pi_lower"]) / row["ensemble_forecast"] * 100 if row["ensemble_forecast"] > 0 else 50

        temp_anomaly = 0
        if "avg_temp_max_c_anomaly" in country_hist.columns:
            recent = country_hist.tail(3)["avg_temp_max_c_anomaly"].mean()
            temp_anomaly = abs(recent) if pd.notna(recent) else 0

        risk_score = round(min(100, pi_width_pct * 0.4 + temp_anomaly * 15), 1)

        if risk_score >= 60:
            risk_level = "High"
        elif risk_score >= 35:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        records.append({
            "country":       country,
            "year":          int(row["year"]),
            "risk_score":    risk_score,
            "risk_level":    risk_level,
            "pi_width_pct":  round(pi_width_pct, 1),
            "temp_anomaly":  round(temp_anomaly, 2),
            "forecast":      round(float(row["ensemble_forecast"]), 2),
            "confidence":    int(row["confidence_pct"]),
        })

    records.sort(key=lambda x: x["risk_score"], reverse=True)
    return {"crop": crop, "records": records, "n_records": len(records)}
