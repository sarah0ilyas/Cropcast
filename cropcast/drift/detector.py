"""
CropCast — Drift Detection
Statistical drift detection using:
1. KS test — detects feature distribution shifts
2. PSI (Population Stability Index) — industry standard for model monitoring
3. MAE degradation — model performance drift

Usage:
    python3 cropcast/drift/detector.py
    python3 cropcast/drift/detector.py --crop Grapes
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

from cropcast.config.settings import DATA_PROCESSED, MODELS_DIR, config

logging.basicConfig(
    level="INFO",
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("drift")

DRIFT_REPORTS_DIR = Path(__file__).resolve().parents[2] / "drift_reports"
DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

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
TARGET = "yield_mt_ha"


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_model(crop: str) -> XGBRegressor:
    slug  = crop.lower().replace(" ", "_").replace(",", "")
    path  = MODELS_DIR / f"xgb_{slug}.json"
    model = XGBRegressor()
    model.load_model(str(path))
    return model


def get_reference_current(df: pd.DataFrame, crop: str):
    crop_df   = df[df["crop"] == crop].dropna(subset=[TARGET]).copy()
    reference = crop_df[crop_df["year"] <= 2018].copy()
    current   = crop_df[crop_df["year"] > 2018].copy()
    return reference, current


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in FEATURE_COLS if c in df.columns]
    return df[available].fillna(df[available].median())


def compute_psi(reference: np.ndarray, current: np.ndarray,
                n_bins: int = 10) -> float:
    """
    Population Stability Index (PSI).
    PSI < 0.1: stable | 0.1-0.2: moderate shift | > 0.2: significant drift
    """
    ref_min = min(reference.min(), current.min())
    ref_max = max(reference.max(), current.max())
    bins    = np.linspace(ref_min, ref_max, n_bins + 1)

    ref_counts = np.histogram(reference, bins=bins)[0]
    cur_counts = np.histogram(current,   bins=bins)[0]

    ref_pct = (ref_counts + 1e-6) / (len(reference) + 1e-6 * n_bins)
    cur_pct = (cur_counts + 1e-6) / (len(current)   + 1e-6 * n_bins)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def detect_data_drift(reference: pd.DataFrame,
                       current: pd.DataFrame,
                       crop: str) -> dict:
    log.info("Running data drift detection for %s...", crop)

    ref_X = get_features(reference)
    cur_X = get_features(current)

    if len(ref_X) < 10 or len(cur_X) < 10:
        return {"drift_detected": False, "reason": "insufficient_data"}

    feature_results = {}
    drifted_cols    = []

    for col in ref_X.columns:
        ref_vals = ref_X[col].dropna().values
        cur_vals = cur_X[col].dropna().values

        if len(ref_vals) < 5 or len(cur_vals) < 5:
            continue

        ks_stat, ks_pval = stats.ks_2samp(ref_vals, cur_vals)
        psi = compute_psi(ref_vals, cur_vals)

        col_drift = bool((ks_pval < 0.05) or (psi > config.drift.psi_threshold))
        if col_drift:
            drifted_cols.append(col)

        feature_results[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue":    round(float(ks_pval), 4),
            "psi":          round(float(psi), 4),
            "drifted":      col_drift,
        }

    drift_share    = len(drifted_cols) / len(ref_X.columns)
    drift_detected = bool(drift_share > 0.3)

    log.info("Data drift — detected: %s | drifted cols: %d/%d | share: %.1f%%",
             drift_detected, len(drifted_cols), len(ref_X.columns), drift_share * 100)

    result = {
        "crop":             crop,
        "drift_detected":   drift_detected,
        "drift_share":      round(float(drift_share), 3),
        "n_drifted_cols":   int(len(drifted_cols)),
        "drifted_features": drifted_cols,
        "feature_results":  feature_results,
        "reference_period": "2000-2018",
        "current_period":   f"2019-{int(current['year'].max())}",
        "timestamp":        datetime.utcnow().isoformat(),
    }

    report_path = DRIFT_REPORTS_DIR / f"data_drift_{crop.lower().replace(' ', '_')}.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    log.info("Drift report saved -> %s", report_path)

    return result


def detect_model_drift(reference: pd.DataFrame,
                        current: pd.DataFrame,
                        crop: str) -> dict:
    log.info("Running model performance drift for %s...", crop)

    try:
        model = load_model(crop)
    except FileNotFoundError:
        log.warning("No model found for %s", crop)
        return {"drift_detected": False, "reason": "no_model"}

    ref_X = get_features(reference)
    cur_X = get_features(current)

    if len(ref_X) < 5 or len(cur_X) < 5:
        return {"drift_detected": False, "reason": "insufficient_data"}

    ref_preds = model.predict(ref_X)
    cur_preds = model.predict(cur_X)

    ref_mae = float(np.mean(np.abs(reference[TARGET].values - ref_preds)))
    cur_mae = float(np.mean(np.abs(current[TARGET].values  - cur_preds)))

    degradation_pct = float((cur_mae - ref_mae) / ref_mae * 100)
    threshold       = config.drift.mae_degradation_pct
    drift_detected  = bool(degradation_pct > threshold)

    log.info("Model drift — ref MAE: %.3f | cur MAE: %.3f | degradation: %.1f%% | threshold: %.1f%%",
             ref_mae, cur_mae, degradation_pct, threshold)

    if drift_detected:
        log.warning("MODEL DRIFT DETECTED for %s — retraining recommended", crop)

    return {
        "crop":            crop,
        "drift_detected":  drift_detected,
        "reference_mae":   round(ref_mae, 4),
        "current_mae":     round(cur_mae, 4),
        "degradation_pct": round(degradation_pct, 2),
        "threshold_pct":   float(threshold),
        "action_required": drift_detected,
        "timestamp":       datetime.utcnow().isoformat(),
    }


def run_drift_check(crop: str) -> dict:
    log.info("=" * 60)
    log.info("Drift check: %s", crop)
    log.info("=" * 60)

    df = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    reference, current = get_reference_current(df, crop)

    data_drift  = detect_data_drift(reference, current, crop)
    model_drift = detect_model_drift(reference, current, crop)

    result = {
        "crop":            crop,
        "data_drift":      data_drift,
        "model_drift":     model_drift,
        "action_required": bool(data_drift.get("drift_detected") or model_drift.get("drift_detected")),
        "timestamp":       datetime.utcnow().isoformat(),
    }

    summary_path = DRIFT_REPORTS_DIR / f"drift_summary_{crop.lower().replace(' ', '_')}.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    if result["action_required"]:
        log.warning("ACTION REQUIRED: Retraining recommended for %s", crop)
    else:
        log.info("No drift detected for %s — models are healthy", crop)

    return result


def run_all_drift_checks() -> list:
    log.info("=" * 60)
    log.info("Running full drift detection suite")
    log.info("=" * 60)

    df    = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    crops = df["crop"].unique().tolist()

    results = []
    for crop in crops:
        try:
            r = run_drift_check(crop)
            results.append(r)
        except Exception as e:
            log.error("Drift check failed for %s: %s", crop, e)

    print("\n" + "=" * 70)
    print("CROPCAST DRIFT DETECTION SUMMARY")
    print("=" * 70)
    print(f"{'Crop':<25} {'Data Drift':>12} {'Model Drift':>12} {'Action':>10}")
    print("-" * 70)
    for r in results:
        dd = "YES" if r["data_drift"].get("drift_detected") else "no"
        md = "YES" if r["model_drift"].get("drift_detected") else "no"
        ac = "RETRAIN" if r["action_required"] else "ok"
        print(f"{r['crop']:<25} {dd:>12} {md:>12} {ac:>10}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", default=None)
    args = parser.parse_args()

    if args.crop:
        run_drift_check(args.crop)
    else:
        run_all_drift_checks()
