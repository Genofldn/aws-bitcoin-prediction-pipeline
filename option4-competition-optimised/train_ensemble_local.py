#!/usr/bin/env python3
"""
Local ensemble training: XGBoost + Random Forest + Gradient Boosting.
Reads training data from S3, trains, packages model.tar.gz, uploads back to S3.
Runs on CPU — expected time ~45 minutes.

Usage:
    python3 train_ensemble_local.py
"""

import os
import sys
import json
import pickle
import tarfile
import tempfile
import warnings
from datetime import datetime, timezone

import boto3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

REGION  = "eu-west-2"
BUCKET  = os.environ.get("DATA_BUCKET", "your-bucket-name")
S3_KEY  = "training-data/btc_hourly.parquet"
OUT_KEY = "models/ensemble/model.tar.gz"


def load_data():
    print("Loading training data from S3...")
    s3 = boto3.client("s3", region_name=REGION)
    local = "/tmp/btc_training.parquet"
    s3.download_file(BUCKET, S3_KEY, local)
    df = pd.read_parquet(local)
    print(f"  Loaded {len(df):,} rows")
    return df


def engineer_features(df):
    print("Engineering features...")
    df = df.sort_values("timestamp").copy()

    # Price features
    for w in [6, 12, 24, 48, 168]:
        df[f"sma_{w}"]       = df["price"].rolling(w).mean()
        df[f"ema_{w}"]       = df["price"].ewm(span=w, adjust=False).mean()
        df[f"price_vs_sma_{w}"] = df["price"] / df[f"sma_{w}"]

    # Returns
    for p in [1, 3, 6, 12, 24, 48, 168]:
        df[f"return_{p}h"]   = df["price"].pct_change(p)
        df[f"price_lag_{p}"] = df["price"].shift(p)

    # Volatility
    for w in [6, 12, 24, 48]:
        df[f"vol_{w}"] = df["price"].pct_change().rolling(w).std()

    # RSI
    delta = df["price"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # MACD
    ema12 = df["price"].ewm(span=12, adjust=False).mean()
    ema26 = df["price"].ewm(span=26, adjust=False).mean()
    df["macd"]      = ema12 - ema26
    df["macd_sig"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    # Bollinger
    rm  = df["price"].rolling(20).mean()
    rs  = df["price"].rolling(20).std()
    df["bb_upper"]  = rm + 2 * rs
    df["bb_lower"]  = rm - 2 * rs
    df["bb_pos"]    = (df["price"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / rm

    # Volume
    for w in [6, 12, 24]:
        df[f"vol_sma_{w}"]   = df["volume"].rolling(w).mean()
        df[f"vol_ratio_{w}"] = df["volume"] / df[f"vol_sma_{w}"]

    # Sentiment
    for w in [3, 6, 12, 24]:
        df[f"sent_sma_{w}"]  = df["sentiment_score"].rolling(w).mean()
    df["sent_trend"]         = df["sentiment_score"] - df["sent_sma_24"]
    for lag in [1, 3, 6, 12]:
        df[f"sent_lag_{lag}"] = df["sentiment_score"].shift(lag)
    df["news_24h_sum"]       = df["news_count"].rolling(24).sum()

    # Time cyclical
    df["hour"]     = df["timestamp"].dt.hour
    df["dow"]      = df["timestamp"].dt.dayofweek
    df["month"]    = df["timestamp"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["dow"]  / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["dow"]  / 7)
    df["is_wknd"]  = (df["dow"] >= 5).astype(int)

    # Target: price 24 hours ahead
    df["target_24h"] = df["price"].shift(-24)

    df = df.dropna().reset_index(drop=True)

    EXCLUDE = {"timestamp", "target_24h", "price"}
    feature_cols = [c for c in df.columns if c not in EXCLUDE]
    print(f"  {len(feature_cols)} features, {len(df):,} samples after dropna")
    return df, feature_cols


def time_split(df):
    """80% train, 20% test — chronological."""
    n = len(df)
    cut = int(n * 0.80)
    return df.iloc[:cut], df.iloc[cut:]


def train_models(X_train, y_train, feature_cols):
    print("\nTraining models...")

    # ── XGBoost ───────────────────────────────────────────
    print("  [1/3] XGBoost — tuning with Optuna (20 trials)...")
    def xgb_objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("lr", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child", 1, 10),
            "reg_alpha":        trial.suggest_float("alpha", 1e-4, 10, log=True),
            "reg_lambda":       trial.suggest_float("lambda", 1e-4, 10, log=True),
        }
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for tr_idx, va_idx in tscv.split(X_train):
            Xt, Xv = X_train[tr_idx], X_train[va_idx]
            yt, yv = y_train[tr_idx], y_train[va_idx]
            m = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
            m.fit(Xt, yt)
            scores.append(mean_squared_error(yv, m.predict(Xv)))
        return np.mean(scores)

    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(xgb_objective, n_trials=20, show_progress_bar=False)
    # Map Optuna suggest-names back to XGBoost constructor param names
    p = study_xgb.best_params
    xgb_params = {
        "n_estimators":     p["n_estimators"],
        "max_depth":        p["max_depth"],
        "learning_rate":    p["lr"],
        "subsample":        p["subsample"],
        "colsample_bytree": p["colsample"],
        "min_child_weight": p["min_child"],
        "reg_alpha":        p["alpha"],
        "reg_lambda":       p["lambda"],
    }
    best_xgb = xgb.XGBRegressor(**xgb_params, random_state=42, n_jobs=-1, verbosity=0)
    best_xgb.fit(X_train, y_train)
    print(f"     Best MSE: {study_xgb.best_value:,.0f}")

    # ── Random Forest ──────────────────────────────────────
    print("  [2/3] Random Forest — tuning with Optuna (15 trials)...")
    def rf_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_est", 100, 500),
            "max_depth":    trial.suggest_int("depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_split", 2, 20),
            "max_features": trial.suggest_categorical("max_feat", ["sqrt", "log2", 0.5, 0.7]),
        }
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for tr_idx, va_idx in tscv.split(X_train):
            m = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            m.fit(X_train[tr_idx], y_train[tr_idx])
            scores.append(mean_squared_error(y_train[va_idx], m.predict(X_train[va_idx])))
        return np.mean(scores)

    study_rf = optuna.create_study(direction="minimize")
    study_rf.optimize(rf_objective, n_trials=15, show_progress_bar=False)
    # Map Optuna suggest-names back to RandomForest constructor param names
    p = study_rf.best_params
    rf_params = {
        "n_estimators":      p["n_est"],
        "max_depth":         p["depth"],
        "min_samples_split": p["min_split"],
        "max_features":      p["max_feat"],
    }
    best_rf = RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1)
    best_rf.fit(X_train, y_train)
    print(f"     Best MSE: {study_rf.best_value:,.0f}")

    # ── Gradient Boosting ──────────────────────────────────
    print("  [3/3] Gradient Boosting — tuning with Optuna (15 trials)...")
    def gb_objective(trial):
        params = {
            "n_estimators":  trial.suggest_int("n_est", 100, 400),
            "max_depth":     trial.suggest_int("depth", 2, 6),
            "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
            "subsample":     trial.suggest_float("sub", 0.6, 1.0),
            "min_samples_split": trial.suggest_int("min_split", 2, 20),
        }
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for tr_idx, va_idx in tscv.split(X_train):
            m = GradientBoostingRegressor(**params, random_state=42)
            m.fit(X_train[tr_idx], y_train[tr_idx])
            scores.append(mean_squared_error(y_train[va_idx], m.predict(X_train[va_idx])))
        return np.mean(scores)

    study_gb = optuna.create_study(direction="minimize")
    study_gb.optimize(gb_objective, n_trials=15, show_progress_bar=False)
    # Map Optuna suggest-names back to GradientBoosting constructor param names
    p = study_gb.best_params
    gb_params = {
        "n_estimators":      p["n_est"],
        "max_depth":         p["depth"],
        "learning_rate":     p["lr"],
        "subsample":         p["sub"],
        "min_samples_split": p["min_split"],
    }
    best_gb = GradientBoostingRegressor(**gb_params, random_state=42)
    best_gb.fit(X_train, y_train)
    print(f"     Best MSE: {study_gb.best_value:,.0f}")

    return best_xgb, best_rf, best_gb


def evaluate(models, X_test, y_test, scaler):
    xgb_m, rf_m, gb_m = models
    preds = {
        "XGBoost":          scaler.inverse_transform(xgb_m.predict(X_test).reshape(-1,1)).flatten(),
        "RandomForest":     scaler.inverse_transform(rf_m.predict(X_test).reshape(-1,1)).flatten(),
        "GradientBoosting": scaler.inverse_transform(gb_m.predict(X_test).reshape(-1,1)).flatten(),
    }
    y_true = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

    print("\nTest set results (24h price prediction):")
    for name, pred in preds.items():
        mae  = mean_absolute_error(y_true, pred)
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        r2   = r2_score(y_true, pred)
        print(f"  {name:<20} MAE: ${mae:,.0f}   RMSE: ${rmse:,.0f}   R²: {r2:.4f}")

    # Simple equal-weight ensemble
    ensemble_pred = np.mean([p for p in preds.values()], axis=0)
    mae  = mean_absolute_error(y_true, ensemble_pred)
    rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))
    r2   = r2_score(y_true, ensemble_pred)
    print(f"  {'Ensemble (equal wt)':<20} MAE: ${mae:,.0f}   RMSE: ${rmse:,.0f}   R²: {r2:.4f}")
    return mae, rmse, r2


def build_inference_script():
    return '''
import json
import pickle
import numpy as np
import os

def model_fn(model_dir):
    with open(os.path.join(model_dir, "ensemble.pkl"), "rb") as f:
        return pickle.load(f)

def input_fn(request_body, content_type="application/json"):
    data = json.loads(request_body)
    return np.array(data["instances"])

def predict_fn(input_data, model_artifacts):
    xgb_m   = model_artifacts["xgboost"]
    rf_m    = model_artifacts["random_forest"]
    gb_m    = model_artifacts["gradient_boosting"]
    scaler  = model_artifacts["scaler"]

    features = input_data
    xgb_pred = xgb_m.predict(features)
    rf_pred  = rf_m.predict(features)
    gb_pred  = gb_m.predict(features)

    ensemble = (xgb_pred + rf_pred + gb_pred) / 3
    # Inverse-transform from scaled space
    prediction = scaler.inverse_transform(ensemble.reshape(-1, 1)).flatten()[0]
    return {"prediction": float(prediction), "price_24h": float(prediction)}

def output_fn(prediction, content_type="application/json"):
    return json.dumps(prediction)
'''


def package_and_upload(models, scaler, feature_cols):
    xgb_m, rf_m, gb_m = models
    model_dir = tempfile.mkdtemp()

    # Save artifact bundle
    artifacts = {
        "xgboost":          xgb_m,
        "random_forest":    rf_m,
        "gradient_boosting":gb_m,
        "scaler":           scaler,
        "feature_columns":  feature_cols,
        "model_version":    "1.0",
        "trained_at":       datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(model_dir, "ensemble.pkl"), "wb") as f:
        pickle.dump(artifacts, f)

    # Save inference script
    with open(os.path.join(model_dir, "inference.py"), "w") as f:
        f.write(build_inference_script())

    # Save config
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump({
            "feature_columns": feature_cols,
            "model_type": "ensemble_xgb_rf_gb",
            "target": "price_24h",
        }, f, indent=2)

    # Tar it up
    tar_path = "/tmp/ensemble_model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_dir, arcname=".")
    print(f"\nModel archive: {tar_path}  ({os.path.getsize(tar_path)/1e6:.1f} MB)")

    # Upload to S3
    s3 = boto3.client("s3", region_name=REGION)
    s3.upload_file(tar_path, BUCKET, OUT_KEY)
    print(f"Uploaded → s3://{BUCKET}/{OUT_KEY}")


def main():
    print("=" * 60)
    print("Option 4 — Ensemble Model Training (local)")
    print("=" * 60)
    start = datetime.now()

    df, feature_cols = engineer_features(load_data())
    train_df, test_df = time_split(df)

    # Scale target
    scaler = RobustScaler()
    y_train = scaler.fit_transform(train_df["target_24h"].values.reshape(-1, 1)).flatten()
    y_test  = test_df["target_24h"].values

    X_train = train_df[feature_cols].values
    X_test  = test_df[feature_cols].values

    # Scale features too (same scaler for X, separate for y is fine)
    feat_scaler = RobustScaler()
    X_train = feat_scaler.fit_transform(X_train)
    X_test  = feat_scaler.transform(X_test)

    models = train_models(X_train, y_train, feature_cols)
    evaluate(models, X_test, y_test, scaler)

    print("\nPackaging and uploading to S3...")
    package_and_upload(models, scaler, feature_cols)

    elapsed = (datetime.now() - start).seconds // 60
    print(f"\n{'='*60}")
    print(f"Ensemble training complete in {elapsed} minutes ✅")
    print(f"Model at: s3://{BUCKET}/{OUT_KEY}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
