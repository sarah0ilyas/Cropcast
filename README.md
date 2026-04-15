# 🌾 CropCast — Global Crop Yield Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange?style=flat)](https://xgboost.readthedocs.io)
[![Prophet](https://img.shields.io/badge/Prophet-1.1-blue?style=flat)](https://facebook.github.io/prophet/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56-FF4B4B?style=flat&logo=streamlit)](https://streamlit.io)
[![AWS](https://img.shields.io/badge/AWS-EC2-FF9900?style=flat&logo=amazonaws)](https://aws.amazon.com)
[![CI](https://github.com/sarah0ilyas/Cropcast/actions/workflows/ci.yml/badge.svg)](https://github.com/sarah0ilyas/Cropcast/actions)

> A production-grade ML platform that nowcasts current-year crop yields and forecasts 5 years ahead across 13 countries and 6 crops — automatically rolling forward as new FAO data is released.

**Live demo:** [https://cropcast.sarahilyas.dev](https://cropcast.sarahilyas.dev)
**API docs:** [https://cropcast.sarahilyas.dev/api/docs](https://cropcast.sarahilyas.dev/api/docs)
**GitHub:** [https://github.com/sarah0ilyas/Cropcast](https://github.com/sarah0ilyas/Cropcast)

---

## What CropCast does

FAO crop production data lags 12–18 months behind reality. By the time official 2025 statistics are published, it will be late 2026. CropCast solves this by:

- **Nowcasting** the current unreported year using lag features and real-time weather data
- **Forecasting** 4 years beyond the nowcast with prediction intervals that widen honestly over time
- **Auto-rolling** the forecast window when new FAO data is ingested — no hardcoded dates anywhere
- **Flagging risk** when weather anomalies or model uncertainty threatens supply chain reliability
- **Detecting drift** automatically using KS tests and PSI when incoming data shifts from training distributions

---

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Ingestion     │    │  Transforms     │    │   ML Models     │
│                 │───▶│                 │───▶│                 │
│ FAO STAT CSV    │    │ DuckDB SQL      │    │ XGBoost x6      │
│ Open-Meteo API  │    │ Feature eng.    │    │ Prophet x6      │
│ USDA NASS       │    │ 45 features     │    │ Ensemble blend  │
│ 13 countries    │    │ Parquet lake    │    │ Walk-forward CV │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐            │
│   Dashboard     │    │   REST API      │            │
│                 │◀───│                 │◀───────────┘
│ Streamlit       │    │ FastAPI         │
│ Forecast charts │    │ 5 endpoints     │
│ Risk heatmap    │    │ Swagger docs    │
│ SHAP plots      │    │ CORS enabled    │
│ AWS + SSL       │    │                 │
└─────────────────┘    └─────────────────┘

┌─────────────────┐    ┌─────────────────┐
│ Drift Detection │    │   CI/CD         │
│                 │    │                 │
│ KS test         │    │ GitHub Actions  │
│ PSI index       │    │ 7 import checks │
│ MAE degradation │    │ Runs on push    │
│ JSON reports    │    │                 │
└─────────────────┘    └─────────────────┘
```

---

## Model Results

| Crop | R² | MAPE | Backtest MAE | PI Coverage |
|------|-----|------|-------------|-------------|
| Grapes | 0.9724 | 6.4% | 0.83 MT/HA | 86.5% |
| Strawberries | 0.9632 | 9.6% | 2.47 MT/HA | 80.8% |
| Tomatoes | 0.9459 | 4.2% | 4.00 MT/HA | 82.7% |
| Citrus | 0.9412 | 4.4% | 0.83 MT/HA | 75.0% |
| Avocados | 0.9056 | 5.9% | 0.41 MT/HA | 65.9% |
| Blueberries | 0.8920 | 7.6% | 1.12 MT/HA | 84.4% |

All models validated using **walk-forward backtesting** — never random splits. The backtest MAE reflects true out-of-sample performance on genuinely future years.

---

## Key Engineering Decisions

**Why temporal CV instead of random splits?**
Random splits leak future yield data into training, artificially inflating metrics. In production you always predict forward in time — the evaluation must mirror that.

**Why XGBoost + Prophet ensemble?**
XGBoost captures non-linear feature interactions and dominates short-horizon predictions. Prophet handles long-term trend extrapolation better for 4–5 year horizons. The ensemble weight shifts from XGBoost-dominant (year 1) to more Prophet (year 5) as uncertainty grows.

**Why DuckDB over Spark?**
At under 200K rows, Spark adds infrastructure overhead with no performance benefit. DuckDB executes columnar SQL in-memory in under a second with zero setup — right tool for the right scale.

**Why nowcast instead of just forecast?**
FAO data lags 12–18 months. By the time 2025 data is officially published it will be late 2026. The nowcast estimates the current unreported year using 2024 lag features and 2025–2026 weather observations — genuinely useful for procurement teams who cannot wait 18 months.

**Why rolling forecast horizon?**
The forecast window auto-advances when new FAO data is ingested. No hardcoded years anywhere in the codebase — `latest_year = df["year"].max()` drives everything downstream.

---

## Data Sources

| Source | Coverage | Records |
|--------|----------|---------|
| FAO STAT bulk CSV | 13 countries, 6 crops, 2000–2024 | 5,256 rows |
| Open-Meteo archive | 12 production regions, daily | 460,652 rows |
| USDA NASS | US domestic stats + prices | via API |

**Total data lake: 465,908 rows of real agricultural data**

---

## Tech Stack

**Data Engineering:** Python, DuckDB, Pandas, PyArrow, Parquet  
**Machine Learning:** XGBoost, Prophet, Scikit-learn, SHAP, Optuna, MLflow  
**Drift Detection:** SciPy KS test, Population Stability Index (PSI)  
**Serving:** FastAPI, Uvicorn, Pydantic  
**Visualisation:** Streamlit, Plotly  
**Infrastructure:** AWS EC2, Nginx, Let's Encrypt SSL, Docker, GitHub Actions CI  

---

## Project Structure

```
cropcast/
├── cropcast/
│   ├── config/settings.py        # 15 countries, 6 crops, all config
│   ├── ingestion/
│   │   ├── base.py               # Retry, logging, Parquet save
│   │   ├── fao_ingester.py       # FAO bulk CSV connector
│   │   └── weather_ingester.py   # Open-Meteo connector
│   ├── transforms/
│   │   ├── transform.py          # DuckDB cleaning + joining
│   │   └── features.py           # 45 ML features engineered
│   ├── models/
│   │   ├── train.py              # XGBoost + walk-forward + SHAP
│   │   ├── saved/                # 6 trained XGBoost models
│   │   └── plots/                # SHAP importance plots
│   ├── forecast/
│   │   └── engine.py             # Rolling 5-year forecast engine
│   ├── drift/
│   │   └── detector.py           # KS + PSI + MAE drift detection
│   ├── api/
│   │   ├── main.py               # FastAPI app
│   │   └── routers/
│   │       └── forecast_router.py # Forecast + risk endpoints
│   └── dashboard/
│       └── app.py                # Streamlit forecast-first UI
├── .github/workflows/ci.yml      # GitHub Actions CI
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/sarah0ilyas/Cropcast.git
cd Cropcast
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$(pwd)

# 1. Ingest data (download FAO bulk CSV first)
python3 cropcast/ingestion/weather_ingester.py
python3 -c "
from cropcast.ingestion.fao_ingester import FAOIngester
FAOIngester().run(csv_path='path/to/fao_bulk.csv')
"

# 2. Transform + feature engineering
python3 cropcast/transforms/transform.py
python3 cropcast/transforms/features.py

# 3. Train models
python3 cropcast/models/train.py --trials 30

# 4. Generate forecasts
python3 cropcast/forecast/engine.py

# 5. Check for drift
python3 cropcast/drift/detector.py

# 6. Launch dashboard
streamlit run cropcast/dashboard/app.py

# 7. Launch API
uvicorn cropcast.api.main:app --reload --port 8000
```

---

## Drift Detection

CropCast monitors two types of drift automatically:

**Data drift** — KS test + Population Stability Index (PSI) on each feature. PSI > 0.2 flags significant distribution shift. Reports saved as JSON to `drift_reports/`.

**Model drift** — MAE on recent data vs reference period. If degradation exceeds 15% threshold, retraining is recommended.

```bash
python3 cropcast/drift/detector.py --crop Grapes
```

---

## Related Projects

- **Wild Blueberry Yield Regression** — Single-crop regression model, the origin of this project. Deployed on Streamlit + AWS.
- **Live Fraud Detection System** — Real-time ML inference pipeline for financial transaction classification.

---

## About

Built by **Sarah Ilyas** — ML Engineer with domain expertise in global agricultural commodities, including professional work on the Global Grape Report (GGR).

[LinkedIn](https://linkedin.com/in/sarahilyas) · [GitHub](https://github.com/sarah0ilyas) · [sarahilyas.dev](https://sarahilyas.dev)
