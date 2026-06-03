#!/usr/bin/env python3
"""
Trains Prophet + ARIMA time series model and packages FinBERT sentiment model.
Both are fast — expected total runtime ~20 minutes.
Uploads both model.tar.gz files to S3.

Usage:
    python3 train_timeseries_and_package_finbert.py
"""

import os
import json
import pickle
import tarfile
import tempfile
import warnings
from datetime import datetime, timezone
warnings.filterwarnings("ignore")

import boto3
import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

REGION   = "eu-west-2"
BUCKET   = os.environ.get("DATA_BUCKET", "your-bucket-name")
DATA_KEY = "training-data/btc_hourly.parquet"

s3 = boto3.client("s3", region_name=REGION)


# ═══════════════════════════════════════════════════════════════
# PART 1: TIME SERIES MODEL (Prophet + ARIMA ensemble)
# ═══════════════════════════════════════════════════════════════

def load_data():
    print("Loading training data from S3...")
    local = "/tmp/btc_ts_training.parquet"
    s3.download_file(BUCKET, DATA_KEY, local)
    df = pd.read_parquet(local)
    print(f"  {len(df):,} rows loaded")
    return df


def train_timeseries(df):
    print("\n" + "="*50)
    print("PART 1: Training Time Series model (Prophet + ARIMA)")
    print("="*50)

    # Use daily aggregated data for Prophet (smoother signal)
    df_daily = (
        df.set_index("timestamp")["price"]
        .resample("D").last()
        .reset_index()
        .rename(columns={"timestamp": "ds", "price": "y"})
        .dropna()
    )
    # Prophet requires timezone-naive datetimes
    df_daily["ds"] = df_daily["ds"].dt.tz_localize(None)
    print(f"  Daily data: {len(df_daily)} days")

    # Train/test split — last 60 days as test
    train_daily = df_daily.iloc[:-60]
    test_daily  = df_daily.iloc[-60:]

    # ── Prophet ────────────────────────────────────────────────
    print("\n[1/2] Fitting Prophet...")
    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        interval_width=0.80,
    )
    # Add crypto-specific weekly pattern
    prophet_model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    prophet_model.fit(train_daily)

    # Forecast 1 day ahead iteratively for test evaluation
    prophet_mae_vals = []
    for i in range(len(test_daily)):
        future = prophet_model.make_future_dataframe(periods=i+1, freq="D")
        fc     = prophet_model.predict(future)
        pred   = fc.iloc[-1]["yhat"]
        true   = test_daily.iloc[i]["y"]
        prophet_mae_vals.append(abs(pred - true))

    prophet_mae = np.mean(prophet_mae_vals)
    print(f"  Prophet test MAE: ${prophet_mae:,.0f}")

    # ── ARIMA ──────────────────────────────────────────────────
    print("\n[2/2] Fitting ARIMA on hourly data...")
    # Use last 2,000 hourly points for ARIMA (enough history, fast to fit)
    hourly_prices = df.sort_values("timestamp")["price"].values[-2000:]

    # ARIMA(2,1,2) — standard for financial time series
    arima_model = ARIMA(hourly_prices, order=(2, 1, 2))
    arima_fit   = arima_model.fit()

    # Evaluate on last 168 hours (1 week)
    arima_test  = hourly_prices[-168:]
    arima_train = hourly_prices[:-168]
    arima_eval  = ARIMA(arima_train, order=(2, 1, 2)).fit()

    # 24-step ahead forecast
    arima_forecast = arima_eval.forecast(steps=24)
    arima_pred_24h = arima_forecast[-1]  # 24h price prediction
    arima_true_24h = arima_test[23]
    arima_mae      = abs(arima_pred_24h - arima_true_24h)
    print(f"  ARIMA 24h test MAE: ${arima_mae:,.0f}")

    # Evaluate metrics
    print(f"\n  Summary:")
    print(f"    Prophet MAE (daily):  ${prophet_mae:,.0f}")
    print(f"    ARIMA MAE (24h):      ${arima_mae:,.0f}")

    return prophet_model, arima_fit, df_daily, hourly_prices


def build_timeseries_inference():
    return '''
import json
import pickle
import numpy as np

def model_fn(model_dir):
    with open(f"{model_dir}/timeseries.pkl", "rb") as f:
        return pickle.load(f)

def input_fn(request_body, content_type="application/json"):
    return json.loads(request_body)

def predict_fn(input_data, artifacts):
    from prophet import Prophet
    from statsmodels.tsa.arima.model import ARIMA
    import pandas as pd
    from datetime import datetime, timezone, timedelta

    prophet_model  = artifacts["prophet"]
    arima_fit      = artifacts["arima"]
    current_price  = input_data.get("current_price", 0)

    # Prophet: predict tomorrow
    future = prophet_model.make_future_dataframe(periods=1, freq="D")
    fc     = prophet_model.predict(future)
    prophet_pred = float(fc.iloc[-1]["yhat"])

    # ARIMA: 24-step ahead from last known prices
    try:
        prices = input_data.get("recent_prices", [current_price] * 24)
        arima_model = ARIMA(prices[-200:], order=(2, 1, 2))
        arima_result = arima_model.fit()
        arima_pred = float(arima_result.forecast(steps=24)[-1])
    except Exception:
        arima_pred = prophet_pred  # fallback

    # Equal-weight ensemble
    prediction = (prophet_pred + arima_pred) / 2
    return {"prediction": prediction, "price_24h": prediction,
            "prophet": prophet_pred, "arima": arima_pred}

def output_fn(prediction, content_type="application/json"):
    return json.dumps(prediction)
'''


def package_timeseries(prophet_model, arima_fit):
    print("\nPackaging time series model...")
    model_dir = tempfile.mkdtemp()

    artifacts = {
        "prophet":       prophet_model,
        "arima":         arima_fit,
        "model_type":    "prophet_arima_ensemble",
        "trained_at":    datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(model_dir, "timeseries.pkl"), "wb") as f:
        pickle.dump(artifacts, f)

    with open(os.path.join(model_dir, "inference.py"), "w") as f:
        f.write(build_timeseries_inference())

    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump({"model_type": "prophet_arima", "target": "price_24h"}, f)

    tar_path = "/tmp/timeseries_model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_dir, arcname=".")
    print(f"  Archive: {tar_path}  ({os.path.getsize(tar_path)/1e6:.1f} MB)")

    s3_key = "models/timeseries/model.tar.gz"
    s3.upload_file(tar_path, BUCKET, s3_key)
    print(f"  Uploaded → s3://{BUCKET}/{s3_key} ✅")


# ═══════════════════════════════════════════════════════════════
# PART 2: FINBERT — Pre-trained, no training needed
# ═══════════════════════════════════════════════════════════════

def package_finbert():
    print("\n" + "="*50)
    print("PART 2: Packaging FinBERT sentiment model")
    print("="*50)
    print("Downloading ProsusAI/finbert from HuggingFace...")
    print("(Pre-trained on financial news — no fine-tuning needed)")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    model_name = "ProsusAI/finbert"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Quick sanity check
    test_texts = [
        "Bitcoin surges to all-time high as institutional adoption grows",
        "Crypto market crashes amid regulatory crackdown",
        "Bitcoin trading sideways ahead of Fed decision",
    ]
    print("\nSanity check on 3 headlines:")
    for text in test_texts:
        inputs  = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1).numpy()[0]
        labels  = ["negative", "neutral", "positive"]
        pred    = labels[probs.argmax()]
        score   = float(probs[2] - probs[0])  # positive - negative
        print(f"  [{pred:8s} | score={score:+.2f}] {text[:60]}")

    # Save model and tokenizer locally
    model_dir = tempfile.mkdtemp()
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)

    # Write inference script
    with open(os.path.join(model_dir, "inference.py"), "w") as f:
        f.write('''
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def model_fn(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return {"tokenizer": tokenizer, "model": model}

def input_fn(request_body, content_type="application/json"):
    return json.loads(request_body)

def predict_fn(input_data, artifacts):
    tokenizer = artifacts["tokenizer"]
    model     = artifacts["model"]
    headlines = input_data.get("headlines", [])
    current_price = input_data.get("current_price", 50000)

    if not headlines:
        return {"prediction": current_price, "price_24h": current_price, "sentiment": 0.0}

    scores = []
    for text in headlines[:20]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
        # positive - negative → range [-1, 1]
        scores.append(float(probs[2] - probs[0]))

    avg_sentiment = float(sum(scores) / len(scores))
    # Translate sentiment to price adjustment: ±2% max swing from sentiment
    price_adjustment = current_price * avg_sentiment * 0.02
    prediction = current_price + price_adjustment

    return {
        "prediction": prediction,
        "price_24h":  prediction,
        "sentiment":  avg_sentiment,
    }

def output_fn(prediction, content_type="application/json"):
    return json.dumps(prediction)
''')

    with open(os.path.join(model_dir, "config_meta.json"), "w") as f:
        json.dump({
            "base_model": "ProsusAI/finbert",
            "model_type": "finbert_sentiment",
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }, f)

    tar_path = "/tmp/sentiment_model.tar.gz"
    print(f"\nPackaging FinBERT model (this may be ~400MB)...")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_dir, arcname=".")
    size_mb = os.path.getsize(tar_path) / 1e6
    print(f"  Archive: {tar_path}  ({size_mb:.0f} MB)")

    s3_key = "models/sentiment/model.tar.gz"
    print(f"  Uploading to S3 ({size_mb:.0f} MB — may take a minute)...")
    s3.upload_file(tar_path, BUCKET, s3_key)
    print(f"  Uploaded → s3://{BUCKET}/{s3_key} ✅")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    start = datetime.now()
    print("="*50)
    print("Time Series + FinBERT Training/Packaging")
    print("="*50)

    # Time series
    df = load_data()
    prophet_model, arima_fit, _, _ = train_timeseries(df)
    package_timeseries(prophet_model, arima_fit)

    # FinBERT
    package_finbert()

    elapsed = (datetime.now() - start).seconds // 60
    print(f"\n{'='*50}")
    print(f"Done in {elapsed} minutes ✅")
    print(f"  s3://{BUCKET}/models/timeseries/model.tar.gz")
    print(f"  s3://{BUCKET}/models/sentiment/model.tar.gz")
    print(f"{'='*50}")
