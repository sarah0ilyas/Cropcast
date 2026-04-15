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

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         STAGE 1 — INGESTION                          │
│                                                                      │
│  FAO STAT Bulk CSV          Open-Meteo Archive API                   │
│  ├── 13 countries           ├── 12 production regions                │
│  ├── 6 crops                ├── Daily weather 2000–2026              │
│  ├── 2000–2024              ├── Temp, precip, ET0                    │
│  └── 5,256 rows             └── 460,652 rows                         │
│                                                                      │
│  base.py — retry logic, schema validation, Parquet save              │
└─────────────────────────────┬────────────────────────────────────────┘
                              │ raw Parquet files
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        STAGE 2 — TRANSFORMS                          │
│                                                                      │
│  transform.py (DuckDB in-memory SQL)                                 │
│  ├── Pivot FAO long → wide                                           │
│  ├── Aggregate daily weather → annual growing-season features        │
│  ├── Left join FAO + weather on (country, year)                      │
│  └── analytical_base.parquet (1,723 rows, 14 cols)                   │
│                                                                      │
│  features.py                                                         │
│  ├── Lag features: yield lag 1, 2, 3                                 │
│  ├── Rolling averages: 3-year, 5-year                                │
│  ├── YoY change: yield, production, area, weather                    │
│  ├── Weather anomalies: z-scores vs country mean                     │
│  ├── Trend features: years since 2000, crop rank                     │
│  └── features.parquet (1,723 rows, 45 cols)                          │
└─────────────────────────────┬────────────────────────────────────────┘
                              │ features.parquet
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        STAGE 3 — ML TRAINING                         │
│                                                                      │
│  train.py                                                            │
│  ├── Walk-forward backtesting (never random splits)                  │
│  ├── Optuna hyperparameter tuning (30 trials per crop)               │
│  ├── XGBoost trained on 23 features                                  │
│  ├── Prophet trained per country time series                         │
│  ├── Bootstrap prediction intervals (90% confidence)                 │
│  ├── SHAP explainability plots                                       │
│  ├── MLflow experiment tracking                                      │
│  └── 6 models saved → models/saved/xgb_*.json                       │
│                                                                      │
│  Results: R² 0.89–0.97 | Backtest MAPE 4.2%–9.7%                    │
└─────────────────────────────┬────────────────────────────────────────┘
                              │ trained models
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      STAGE 4 — FORECAST ENGINE                       │
│                                                                      │
│  engine.py                                                           │
│  ├── Auto-detects latest FAO year (currently 2024)                   │
│  ├── Nowcasts 2025 — FAO hasn't released yet                         │
│  ├── Forecasts 2026–2029 (5-year horizon)                            │
│  ├── XGBoost + Prophet ensemble                                      │
│  │   └── XGBoost weight decreases for longer horizons                │
│  ├── Prediction intervals widen: 90% → 58% confidence by 2029       │
│  ├── Rolling window auto-advances when new FAO data ingested         │
│  └── forecasts.parquet (355 forecasts, 13 countries × 6 crops)      │
│                                                                      │
│  drift/detector.py                                                   │
│  ├── KS test — detects feature distribution shifts                   │
│  ├── PSI — quantifies magnitude (>0.2 = significant)                 │
│  ├── MAE degradation — flags model performance decay                 │
│  └── JSON drift reports saved to drift_reports/                      │
└──────────────┬──────────────────────────┬───────────────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────┐    ┌─────────────────────────────────────────┐
│   STAGE 5 — API      │    │         STAGE 5 — DASHBOARD             │
│                      │    │                                         │
│  FastAPI             │    │  Streamlit                              │
│  ├── /forecast       │    │  ├── Forecast view                      │
│  ├── /nowcast        │    │  │   ├── 5-year rolling chart           │
│  ├── /history        │    │  │   ├── Prediction interval bands      │
│  ├── /combined       │    │  │   ├── Nowcast by country             │
│  └── /risk           │    │  │   └── 5-year summary table           │
│                      │    │  ├── Historical view                    │
│  Swagger docs /docs  │    │  │   ├── Yield trends 2000–2024         │
│                      │    │  │   ├── Production bar chart           │
│                      │    │  │   └── YoY change boxplot             │
│                      │    │  └── Risk view                          │
│                      │    │      ├── Risk score heatmap             │
│                      │    │      └── SHAP feature importance        │
└──────────────────────┘    └─────────────────────────────────────────┘
               │                          │
               └──────────────┬───────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       DEPLOYMENT                                     │
│                                                                      │
│  AWS EC2 t2.micro (eu-west-1)                                        │
│  ├── Nginx reverse proxy                                             │
│  ├── Let's Encrypt SSL (auto-renews)                                 │
│  ├── Crontab @reboot auto-start                                      │
│  └── https://cropcast.sarahilyas.dev                                 │
│                                                                      │
│  GitHub Actions CI                                                   │
│  └── Import checks on every push to main                            │
│                                                                      │
│  Docker + docker-compose                                             │
│  └── Containerised for reproducible deployment                       │
└──────────────────────────────────────────────────────────────────────┘
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

All models validated using **walk-forward backtesting** — never random splits. Backtest MAPE reflects true out-of-sample performance on genuinely future years.

---

## Key Engineering Decisions

**Why temporal CV instead of random splits?**
Random splits leak future yield data into training, artificially inflating metrics. In production you always predict forward in time — the evaluation must mirror that.

**Why XGBoost + Prophet ensemble?**
XGBoost captures non-linear feature interactions and dominates short-horizon predictions. Prophet handles long-term trend extrapolation better for 4–5 year horizons. The ensemble weight shifts from XGBoost-dominant (year 1) to more Prophet (year 5) as uncertainty grows.

**Why DuckDB over Spark?**
At under 200K rows, Spark adds infrastructure overhead with no performance benefit. DuckDB executes columnar SQL in-memory in under a second with zero setup — right tool for the right scale.

**Why nowcast instead of just forecast?**
FAO data lags 12–18 months. The nowcast estimates the current unreported year using 2024 lag features and 2025–2026 weather observations — genuinely useful for procurement teams who cannot wait 18 months for official statistics.

**Why rolling forecast horizon?**
The forecast window auto-advances when new FAO data is ingested. No hardcoded years anywhere — `latest_year = df["year"].max()` drives everything downstream.

---

## Data Sources

| Source | Coverage | Records |
|--------|----------|---------|
| FAO STAT bulk CSV | 13 countries, 6 crops, 2000–2024 | 5,256 rows |
| Open-Meteo archive | 12 production regions, daily 2000–2026 | 460,652 rows |

**Total: 465,908 rows of real agricultural data**

---

## Tech Stack

**Data Engineering:** Python, DuckDB, Pandas, PyArrow, Parquet  
**Machine Learning:** XGBoost, Prophet, Scikit-learn, SHAP, Optuna, MLflow  
**Drift Detection:** SciPy KS test, Population Stability Index (PSI)  
**Serving:** FastAPI, Uvicorn, Pydantic  
**Visualisation:** Streamlit, Plotly  
**Infrastructure:** AWS EC2, Nginx, Let's Encrypt SSL, Docker, GitHub Actions CI  

---

## Quickstart

```bash
git clone https://github.com/sarah0ilyas/Cropcast.git
cd Cropcast
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$(pwd)

# 1. Ingest data (download FAO bulk CSV first from FAO STAT)
python3 cropcast/ingestion/weather_ingester.py
python3 -c "
from cropcast.ingestion.fao_ingester import FAOIngester
FAOIngester().run(csv_path='path/to/fao_bulk.csv')
"

# 2. Transform + feature engineering
python3 cropcast/transforms/transform.py
python3 cropcast/transforms/features.py

# 3. Train models (all 6 crops)
python3 cropcast/models/train.py --trials 30

# 4. Generate rolling 5-year forecasts
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
python3 cropcast/drift/detector.py           # all crops
python3 cropcast/drift/detector.py --crop Grapes  # single crop
```

---

## Related Projects

- **Wild Blueberry Yield Regression** — Single-crop regression model. Deployed on Streamlit + AWS.
- **Live Fraud Detection System** — Real-time ML inference pipeline for financial transaction classification.

---

## About

Built by **Sarah Ilyas** — ML Engineer with domain expertise in global agricultural commodities, including professional work on the Global Grape Report (GGR).

[LinkedIn](https://linkedin.com/in/sarahilyas) · [GitHub](https://github.com/sarah0ilyas) · [cropcast.sarahilyas.dev](https://cropcast.sarahilyas.dev)
