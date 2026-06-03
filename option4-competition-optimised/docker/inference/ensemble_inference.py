"""
Ensemble inference.py — XGBoost + RandomForest + GradientBoosting
Reconstructs all 74 training features from raw hourly price/volume data
sent by the FeatureEngineer Lambda.

Input JSON (sent by Lambda via invoke_endpoint):
{
  "hourly_prices":  [96000, 95800, ...],   # last 200+ hourly prices (most recent last)
  "hourly_volumes": [1e8, 1.1e8, ...],     # matching hourly volumes
  "daily_prices":   [90000, 91000, ...],   # last 60 daily prices (fallback)
  "sentiment_score": 0.3,                  # keyword-based sentiment [-1, 1]
  "news_count":      25,
  "current_price":   96000,
  "date":            "2026-06-03",
  "run_type":        "morning"
}

Output JSON:
{
  "prediction": 97500.0,
  "price_24h":  97500.0
}
"""

import json
import os
import pickle
import numpy as np
from datetime import datetime, timezone


# ── EMA helper ────────────────────────────────────────────────────────────────
def _ema_series(prices, span):
    """Compute EMA of a price array; returns final scalar."""
    k = 2.0 / (span + 1)
    ema = float(prices[0])
    for p in prices[1:]:
        ema = float(p) * k + ema * (1 - k)
    return ema


def _macd_signal(prices, fast=12, slow=26, signal=9):
    """Return (macd_line, signal_line) scalars from a price array."""
    k_f, k_s, k_sig = 2/(fast+1), 2/(slow+1), 2/(signal+1)
    ema_f = ema_s = float(prices[0])
    macd_hist = []
    for p in prices:
        ema_f = float(p) * k_f + ema_f * (1 - k_f)
        ema_s = float(p) * k_s + ema_s * (1 - k_s)
        macd_hist.append(ema_f - ema_s)
    sig = macd_hist[0]
    for m in macd_hist[1:]:
        sig = m * k_sig + sig * (1 - k_sig)
    return macd_hist[-1], sig


# ── Feature engineering (mirrors train_ensemble_local.py) ─────────────────────
def _engineer_features(prices, volumes, sentiment, news_count, now):
    """
    Build the 74-feature dict from a price/volume series.
    prices, volumes — numpy arrays, most-recent element last.
    """
    p = prices
    v = volumes
    N = len(p)

    feat = {}

    # SMAs, EMAs, price_vs_sma
    for w in [6, 12, 24, 48, 168]:
        win = min(w, N)
        sma = float(np.mean(p[-win:]))
        ema = _ema_series(p[-win:], win)
        feat[f"sma_{w}"]           = sma
        feat[f"ema_{w}"]           = ema
        feat[f"price_vs_sma_{w}"]  = float(p[-1] / sma) if sma != 0 else 1.0

    # Returns and lags
    for lag in [1, 3, 6, 12, 24, 48, 168]:
        idx = min(lag, N - 1)
        feat[f"return_{lag}h"]  = float((p[-1] - p[-1-idx]) / p[-1-idx]) if p[-1-idx] != 0 else 0.0
        feat[f"price_lag_{lag}"] = float(p[-1-idx])

    # Volatility (std of returns)
    for w in [6, 12, 24, 48]:
        win = min(w + 1, N)
        rets = np.diff(p[-win:]) / np.where(p[-win:-1] != 0, p[-win:-1], 1)
        feat[f"vol_{w}"] = float(np.std(rets)) if len(rets) > 0 else 0.0

    # RSI-14
    win14 = min(15, N)
    diffs = np.diff(p[-win14:])
    gains = diffs.clip(min=0)
    losses = (-diffs).clip(min=0)
    avg_gain = float(np.mean(gains[-14:])) if len(gains) >= 14 else float(np.mean(gains))
    avg_loss = float(np.mean(losses[-14:])) if len(losses) >= 14 else float(np.mean(losses))
    rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
    feat["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD
    macd_val, macd_sig = _macd_signal(p[-max(26, N):] if N > 26 else p)
    feat["macd"]      = macd_val
    feat["macd_sig"]  = macd_sig
    feat["macd_hist"] = macd_val - macd_sig

    # Bollinger Bands
    win20 = min(20, N)
    bb_window = p[-win20:]
    bb_mid    = float(np.mean(bb_window))
    bb_std    = float(np.std(bb_window))
    bb_upper  = bb_mid + 2 * bb_std
    bb_lower  = bb_mid - 2 * bb_std
    feat["bb_upper"] = bb_upper
    feat["bb_lower"] = bb_lower
    feat["bb_pos"]   = float((p[-1] - bb_lower) / (bb_upper - bb_lower)) if (bb_upper - bb_lower) != 0 else 0.5
    feat["bb_width"] = float((bb_upper - bb_lower) / bb_mid) if bb_mid != 0 else 0.0

    # Volume features
    feat["volume"] = float(v[-1])
    for w in [6, 12, 24]:
        win = min(w, len(v))
        vsma = float(np.mean(v[-win:]))
        feat[f"vol_sma_{w}"]   = vsma
        feat[f"vol_ratio_{w}"] = float(v[-1] / vsma) if vsma != 0 else 1.0

    # Sentiment (no multi-step history — use current value for all lags/SMAs)
    s = float(sentiment)
    feat["sentiment_score"] = s
    feat["news_count"]      = float(news_count)
    for w in [3, 6, 12, 24]:
        feat[f"sent_sma_{w}"] = s
    feat["sent_trend"] = 0.0
    for lag in [1, 3, 6, 12]:
        feat[f"sent_lag_{lag}"] = s
    feat["news_24h_sum"] = float(news_count) * 24

    # Time cyclical
    feat["hour"]     = float(now.hour)
    feat["dow"]      = float(now.weekday())
    feat["month"]    = float(now.month)
    feat["hour_sin"] = float(np.sin(2 * np.pi * now.hour / 24))
    feat["hour_cos"] = float(np.cos(2 * np.pi * now.hour / 24))
    feat["dow_sin"]  = float(np.sin(2 * np.pi * now.weekday() / 7))
    feat["dow_cos"]  = float(np.cos(2 * np.pi * now.weekday() / 7))
    feat["is_wknd"]  = float(1 if now.weekday() >= 5 else 0)

    return feat


# ── SageMaker contract ────────────────────────────────────────────────────────

def model_fn(model_dir):
    pkl_path = os.path.join(model_dir, "ensemble.pkl")
    with open(pkl_path, "rb") as f:
        artifacts = pickle.load(f)
    print(f"[ensemble] Loaded model — features: {len(artifacts['feature_columns'])}")
    return artifacts


def input_fn(request_body, content_type="application/json"):
    return json.loads(request_body)


def predict_fn(input_data, artifacts):
    xgb_m   = artifacts["xgboost"]
    rf_m    = artifacts["random_forest"]
    gb_m    = artifacts["gradient_boosting"]
    scaler  = artifacts["scaler"]
    feature_cols = artifacts["feature_columns"]

    current_price   = float(input_data.get("current_price", 50000))
    hourly_prices   = input_data.get("hourly_prices") or input_data.get("daily_prices") or [current_price] * 200
    hourly_volumes  = input_data.get("hourly_volumes") or input_data.get("daily_volumes") or [1e8] * len(hourly_prices)
    sentiment_score = float(input_data.get("sentiment_score", 0.0))
    news_count      = int(input_data.get("news_count", 10))
    now             = datetime.now(timezone.utc)

    prices  = np.array(hourly_prices, dtype=float)
    volumes = np.array(hourly_volumes, dtype=float)

    # Pad if too short (need at least 170 for sma_168)
    MIN_LEN = 170
    if len(prices) < MIN_LEN:
        prices  = np.pad(prices,  (MIN_LEN - len(prices),  0), mode="edge")
        volumes = np.pad(volumes, (MIN_LEN - len(volumes), 0), mode="edge")

    feat_dict = _engineer_features(prices, volumes, sentiment_score, news_count, now)

    # Build feature row in exact training order
    row = [feat_dict.get(c, 0.0) for c in feature_cols]
    X   = np.array(row, dtype=float).reshape(1, -1)

    # Predict (models output scaled price)
    xgb_pred = xgb_m.predict(X)
    rf_pred  = rf_m.predict(X)
    gb_pred  = gb_m.predict(X)

    ensemble_scaled = (xgb_pred + rf_pred + gb_pred) / 3.0
    prediction = float(
        scaler.inverse_transform(ensemble_scaled.reshape(-1, 1)).flatten()[0]
    )

    print(f"[ensemble] XGB={scaler.inverse_transform(xgb_pred.reshape(-1,1))[0,0]:,.0f} "
          f"RF={scaler.inverse_transform(rf_pred.reshape(-1,1))[0,0]:,.0f} "
          f"GB={scaler.inverse_transform(gb_pred.reshape(-1,1))[0,0]:,.0f} "
          f"→ ensemble=${prediction:,.0f}")

    return {"prediction": prediction, "price_24h": prediction}


def output_fn(prediction, content_type="application/json"):
    return json.dumps(prediction)
