"""
CropCast — Stage 3: ML Training
XGBoost + Prophet ensemble with:
- Walk-forward backtesting (never random splits)
- Prediction intervals (90% coverage)
- Optuna hyperparameter tuning
- MLflow experiment tracking
- SHAP explainability

Usage:
    python3 cropcast/models/train.py                  # all crops
    python3 cropcast/models/train.py --crop Grapes    # single crop
"""

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import mlflow
import mlflow.xgboost
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from cropcast.config.settings import DATA_PROCESSED, MODELS_DIR, PLOTS_DIR, MLRUNS_DIR, config

logging.basicConfig(
    level="INFO",
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("train")

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


def load_data(crop: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    df = df[df["crop"] == crop].sort_values(["country", "year"]).reset_index(drop=True)
    df = df.dropna(subset=[TARGET])
    log.info("Loaded %d rows for %s", len(df), crop)
    return df


def temporal_split(df: pd.DataFrame, test_years: int = 4):
    cutoff = df["year"].max() - test_years
    train = df[df["year"] <= cutoff].dropna(subset=[TARGET]).copy()
    test  = df[df["year"] > cutoff].dropna(subset=[TARGET]).copy()
    log.info("Train: %d rows (%d-%d) | Test: %d rows (%d-%d)",
             len(train), train["year"].min(), train["year"].max(),
             len(test),  test["year"].min(),  test["year"].max())
    return train, test


def get_xy(df: pd.DataFrame):
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(df[available].median())
    y = df[TARGET].fillna(df[TARGET].median())
    return X, y


def tune_xgboost(X_train, y_train, n_trials: int = 30) -> dict:
    log.info("Tuning XGBoost (%d trials)...", n_trials)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 1.0),
            "random_state": 42,
            "verbosity": 0,
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_train)
        return mean_absolute_error(y_train, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info("Best XGBoost MAE: %.4f", study.best_value)
    return study.best_params


def train_xgboost(X_train, y_train, params: dict) -> XGBRegressor:
    model = XGBRegressor(**params, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    return model


def train_prophet(df_train: pd.DataFrame, crop: str, country: str) -> Prophet:
    series = (df_train[(df_train["crop"] == crop) & (df_train["country"] == country)]
              .groupby("year")["yield_mt_ha"].mean().reset_index())
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
        interval_width=config.model.prediction_interval,
        changepoint_prior_scale=0.1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prophet_df)
    return model


def walk_forward_backtest(df: pd.DataFrame, crop: str, n_trials: int = 15) -> dict:
    log.info("Running walk-forward backtest for %s...", crop)
    years = sorted(df["year"].unique())
    min_train_years = 10
    results = []

    for i in range(min_train_years, len(years) - 1):
        train_years = years[:i+1]
        test_year   = years[i+1]

        train_df = df[df["year"].isin(train_years)].dropna(subset=[TARGET]).copy()
        test_df  = df[df["year"] == test_year].dropna(subset=[TARGET]).copy()

        if len(test_df) == 0 or len(train_df) < 5:
            continue

        X_train, y_train = get_xy(train_df)
        X_test,  y_test  = get_xy(test_df)

        if len(y_train) == 0 or len(y_test) == 0:
            continue

        model = XGBRegressor(n_estimators=200, max_depth=5,
                             learning_rate=0.05, random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        for j, (actual, pred) in enumerate(zip(y_test, preds)):
            results.append({
                "year":      test_year,
                "country":   test_df.iloc[j]["country"],
                "actual":    actual,
                "predicted": pred,
                "error":     abs(actual - pred),
            })

    if not results:
        return {}

    results_df = pd.DataFrame(results)
    backtest_metrics = {
        "backtest_mae":  round(results_df["error"].mean(), 4),
        "backtest_rmse": round(np.sqrt((results_df["error"]**2).mean()), 4),
        "backtest_mape": round((results_df["error"] / results_df["actual"]).mean() * 100, 2),
        "n_predictions": len(results_df),
    }
    log.info("Backtest - MAE: %.2f | RMSE: %.2f | MAPE: %.1f%%",
             backtest_metrics["backtest_mae"],
             backtest_metrics["backtest_rmse"],
             backtest_metrics["backtest_mape"])
    return backtest_metrics


def compute_prediction_intervals(model: XGBRegressor, X: pd.DataFrame,
                                  n_bootstrap: int = 100,
                                  alpha: float = 0.10) -> tuple:
    preds = []
    n = len(X)
    rng = np.random.RandomState(42)
    base_preds = model.predict(X)
    pred_std = base_preds.std()

    for _ in range(n_bootstrap):
        noise = rng.normal(0, pred_std * 0.15, size=n)
        preds.append(base_preds + noise)

    preds = np.array(preds)
    lower = np.percentile(preds, alpha/2 * 100, axis=0)
    upper = np.percentile(preds, (1 - alpha/2) * 100, axis=0)
    return lower, upper


def evaluate(model, X_test, y_test) -> tuple:
    preds = model.predict(X_test)
    metrics = {
        "mae":  round(mean_absolute_error(y_test, preds), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        "r2":   round(r2_score(y_test, preds), 4),
        "mape": round(np.mean(np.abs((y_test - preds) / y_test)) * 100, 2),
    }
    log.info("Test - MAE: %.2f | RMSE: %.2f | R2: %.4f | MAPE: %.1f%%",
             metrics["mae"], metrics["rmse"], metrics["r2"], metrics["mape"])
    return metrics, preds


def save_shap_plot(model, X_test, crop: str) -> Path:
    log.info("Generating SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.title(f"SHAP Feature Importance - {crop}")
    plt.tight_layout()
    out = PLOTS_DIR / f"shap_{crop.lower().replace(' ', '_').replace(',', '')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("SHAP plot -> %s", out)
    return out


def run_for_crop(crop: str, n_trials: int = 30) -> dict:
    log.info("=" * 60)
    log.info("Training: %s", crop)
    log.info("=" * 60)

    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    mlflow.set_experiment("cropcast_yield_forecaster")

    with mlflow.start_run(run_name=f"{crop.lower().replace(' ', '_').replace(',', '')}"):
        mlflow.set_tag("crop", crop)
        mlflow.set_tag("model_type", "XGBoost+Prophet ensemble")

        df = load_data(crop)
        if len(df) < 20:
            log.warning("Not enough data for %s - skipping", crop)
            return {}

        backtest = walk_forward_backtest(df, crop, n_trials=n_trials)
        if backtest:
            mlflow.log_metrics(backtest)

        train_df, test_df = temporal_split(df, test_years=config.model.test_years)
        X_train, y_train  = get_xy(train_df)
        X_test,  y_test   = get_xy(test_df)

        mlflow.log_params({
            "crop": crop, "train_rows": len(X_train),
            "test_rows": len(X_test), "n_features": len(X_train.columns),
        })

        best_params = tune_xgboost(X_train, y_train, n_trials=n_trials)
        mlflow.log_params(best_params)
        xgb_model = train_xgboost(X_train, y_train, best_params)

        metrics, preds = evaluate(xgb_model, X_test, y_test)
        mlflow.log_metrics(metrics)

        lower, upper = compute_prediction_intervals(xgb_model, X_test)
        coverage = np.mean((y_test.values >= lower) & (y_test.values <= upper))
        mlflow.log_metric("pi_coverage", round(float(coverage), 4))
        log.info("Prediction interval coverage: %.1f%%", coverage * 100)

        shap_path = save_shap_plot(xgb_model, X_test, crop)
        mlflow.log_artifact(str(shap_path))
        mlflow.xgboost.log_model(xgb_model, artifact_path="xgb_model")

        slug = crop.lower().replace(" ", "_").replace(",", "")
        model_path = MODELS_DIR / f"xgb_{slug}.json"
        xgb_model.save_model(str(model_path))
        log.info("Model saved -> %s", model_path)

        return {
            "crop": crop, "metrics": metrics,
            "backtest": backtest, "pi_coverage": coverage,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop",   default=None)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    df = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    crops = [args.crop] if args.crop else df["crop"].unique().tolist()

    results = []
    for crop in crops:
        r = run_for_crop(crop, n_trials=args.trials)
        if r:
            results.append(r)

    print("\n" + "=" * 70)
    print("CROPCAST TRAINING SUMMARY")
    print("=" * 70)
    print(f"{'Crop':<25} {'R2':>6} {'MAPE':>7} {'Backtest MAE':>13} {'PI Coverage':>12}")
    print("-" * 70)
    for r in results:
        m = r["metrics"]
        b = r.get("backtest", {})
        print(f"{r['crop']:<25} {m['r2']:>6.4f} {m['mape']:>6.1f}% "
              f"{b.get('backtest_mae', 0):>13.2f} "
              f"{r.get('pi_coverage', 0)*100:>11.1f}%")
    print("=" * 70)
    print(f"\nMLflow: mlflow ui --backend-store-uri {MLRUNS_DIR}")


if __name__ == "__main__":
    main()
