"""
CropCast — Global configuration
All secrets loaded from environment variables.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_RAW       = BASE_DIR / "cropcast" / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "cropcast" / "data" / "processed"
MODELS_DIR     = BASE_DIR / "cropcast" / "models" / "saved"
PLOTS_DIR      = BASE_DIR / "cropcast" / "models" / "plots"
LOGS_DIR       = BASE_DIR / "logs"
MLRUNS_DIR     = BASE_DIR / "mlruns"

for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, PLOTS_DIR, LOGS_DIR, MLRUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class FAOConfig:
    base_url: str = "https://fenixservices.fao.org/faostat/api/v1"
    crops: List[str] = field(default_factory=lambda: [
        "Grapes",
        "Blueberries",
        "Avocados",
        "Tomatoes",
        "Strawberries",
        "Citrus Fruit, Total",
    ])
    elements: List[str] = field(default_factory=lambda: [
        "Area harvested",
        "Production",
        "Yield",
    ])
    # Expanded to 15 countries
    area_codes: List[str] = field(default_factory=lambda: [
        "Peru", "Chile", "South Africa", "Spain", "Italy",
        "United States of America", "China", "Argentina",
        "Australia", "France", "Turkey", "India", "Iran",
        "Portugal", "Greece",
    ])
    start_year: int = 2000
    timeout: int = 60


@dataclass
class WeatherConfig:
    base_url: str = "https://archive-api.open-meteo.com/v1/archive"
    locations: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Ica_Peru":           (-14.07, -75.73),
        "Maule_Chile":        (-35.43, -71.67),
        "Western_Cape_SA":    (-33.93,  18.86),
        "Murcia_Spain":       ( 37.98,  -1.13),
        "California_US":      ( 36.78,-119.42),
        "Mendoza_Argentina":  (-32.89, -68.84),
        "Southeast_Australia":(-34.93, 138.60),
        "Bordeaux_France":    ( 44.84,  -0.58),
        "Aegean_Turkey":      ( 38.42,  27.14),
        "Maharashtra_India":  ( 19.75,  75.71),
        "Alentejo_Portugal":  ( 38.57,  -8.00),
        "Crete_Greece":       ( 35.24,  25.11),
    })
    variables: List[str] = field(default_factory=lambda: [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "et0_fao_evapotranspiration",
    ])
    start_date: str = "2000-01-01"
    timeout: int = 60


@dataclass
class ModelConfig:
    # Temporal split — last N years as test set
    test_years: int = 4
    # Optuna trials for XGBoost tuning
    optuna_trials: int = 30
    # Forecast horizon
    forecast_horizon: int = 5
    # Ensemble weights (XGBoost, Prophet) — auto-tuned but start here
    xgb_weight: float = 0.65
    prophet_weight: float = 0.35
    # Prediction interval confidence level
    prediction_interval: float = 0.90
    random_state: int = 42


@dataclass
class DriftConfig:
    # Evidently drift detection thresholds
    psi_threshold: float = 0.2       # Population Stability Index
    js_threshold: float = 0.1        # Jensen-Shannon divergence
    mae_degradation_pct: float = 15.0 # % MAE increase triggers retraining
    reference_window_years: int = 5   # years used as reference distribution


@dataclass
class StorageConfig:
    compression: str = "snappy"
    use_s3: bool = field(default_factory=lambda: os.getenv("USE_S3", "false").lower() == "true")
    s3_bucket: str = field(default_factory=lambda: os.getenv("S3_BUCKET", "cropcast-data"))
    aws_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "eu-west-1"))


@dataclass
class SchedulerConfig:
    # Monthly refresh on 1st at 03:00 UTC
    schedule_interval: str = "0 3 1 * *"
    retries: int = 3
    retry_delay_minutes: int = 10


@dataclass
class CropCastConfig:
    fao: FAOConfig           = field(default_factory=FAOConfig)
    weather: WeatherConfig   = field(default_factory=WeatherConfig)
    model: ModelConfig       = field(default_factory=ModelConfig)
    drift: DriftConfig       = field(default_factory=DriftConfig)
    storage: StorageConfig   = field(default_factory=StorageConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    log_level: str           = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


config = CropCastConfig()
