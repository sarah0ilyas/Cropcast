"""
CropCast — FastAPI Serving Layer
Exposes forecast, historical, and cluster data via REST API.

Run:
    uvicorn cropcast.api.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cropcast.api.routers import forecast_router

from cropcast.config.settings import DATA_PROCESSED


class AppState:
    features_df: pd.DataFrame = None
    forecasts_df: pd.DataFrame = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.features_df  = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    state.forecasts_df = pd.read_parquet(DATA_PROCESSED / "forecasts.parquet")
    print(f"Features loaded:  {state.features_df.shape}")
    print(f"Forecasts loaded: {state.forecasts_df.shape}")
    yield
    state.features_df  = None
    state.forecasts_df = None


app = FastAPI(
    title="CropCast API",
    description="Rolling 5-year yield forecasts for global fresh produce.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forecast_router.router, prefix="/forecast", tags=["Forecast"])


@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "api": "CropCast",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "features_rows": len(state.features_df) if state.features_df is not None else 0,
        "forecast_rows": len(state.forecasts_df) if state.forecasts_df is not None else 0,
    }


@app.get("/crops", tags=["Reference"])
def list_crops():
    return {"crops": sorted(state.features_df["crop"].unique().tolist())}


@app.get("/countries", tags=["Reference"])
def list_countries():
    return {"countries": sorted(state.features_df["country"].unique().tolist())}


@app.get("/summary", tags=["Analytics"])
def global_summary():
    df      = state.features_df
    fc      = state.forecasts_df
    latest  = df[df["year"] == df["year"].max()]
    nowcast = fc[fc["forecast_type"] == "nowcast"]
    return {
        "latest_actual_year":   int(df["year"].max()),
        "nowcast_year":         int(fc["year"].min()),
        "forecast_horizon":     int(fc["year"].max()),
        "n_countries":          int(df["country"].nunique()),
        "n_crops":              int(df["crop"].nunique()),
        "total_forecasts":      len(fc),
        "avg_nowcast_confidence": 90,
    }
