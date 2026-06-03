"""
LSTM/GRU inference.py — Multi-output deep learning model (85-feature version).
Replicates the pandas feature engineering from train_lstm_local.py exactly so
the feature matrix matches what the model was trained on.

Input JSON (from FeatureEngineer Lambda):
{
  "hourly_prices":    [96000, 95800, ...],  # at least 168 values, most recent last
  "hourly_volumes":   [1e8, ...],           # matching hourly volumes
  "current_price":    96000,
  "sentiment_score":  0.3,                  # current scalar (fallback)
  "sentiment_history": [0.1, -0.2, ...],   # 168-point series, oldest first (preferred)
  "news_count": 15,
  "date": "2026-06-03",
  "run_type": "morning"
}

Output JSON:
{
  "prediction": 97500.0,
  "price_24h":  97500.0,
  "price_1h":   96100.0,
  "price_6h":   96800.0,
  "price_7d":   99000.0
}
"""

import json
import os
import pickle
import numpy as np


# ── SageMaker contract ─────────────────────────────────────────────────────────

def model_fn(model_dir):
    """Load Keras model, scalers, and config from model directory."""
    import tensorflow as tf

    h5_path         = os.path.join(model_dir, "lstm_model.h5")
    keras_path      = os.path.join(model_dir, "lstm_model.keras")
    saved_model_dir = os.path.join(model_dir, "lstm_model")

    # Priority: .keras (portable) → SavedModel dir → .h5
    if os.path.exists(keras_path):
        model = tf.keras.models.load_model(keras_path, compile=False)
        print(f"[lstm] Loaded from {keras_path}")
    elif os.path.isdir(saved_model_dir):
        model = tf.saved_model.load(saved_model_dir)
        print(f"[lstm] Loaded SavedModel from {saved_model_dir}")
    elif os.path.exists(h5_path):
        model = tf.keras.models.load_model(h5_path, compile=False)
        print(f"[lstm] Loaded from {h5_path}")
    else:
        raise FileNotFoundError(f"No LSTM model file found in {model_dir}")

    scalers = {}
    scalers_path = os.path.join(model_dir, "scalers.pkl")
    if os.path.exists(scalers_path):
        with open(scalers_path, "rb") as f:
            scalers = pickle.load(f)
        print(f"[lstm] Scalers loaded: {list(scalers.keys())}")

    config = {}
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        print(f"[lstm] Config: seq_len={config.get('sequence_length', 168)}, "
              f"features={len(config.get('feature_columns', []))}")

    return {"model": model, "scalers": scalers, "config": config}


def input_fn(request_body, content_type="application/json"):
    return json.loads(request_body)


def _build_feature_matrix(prices_arr, volumes_arr, sentiment, news_count, now,
                           sentiment_history=None):
    """
    Build a [N, 85] feature matrix by replicating the pandas feature engineering
    from train_lstm_local.py exactly. Uses approximations for OHLCV columns that
    are not available at inference (open ≈ prev close, high/low ≈ candle extremes).

    prices_arr       : numpy array of hourly close prices, oldest first
    volumes_arr      : numpy array of hourly volumes, same length
    sentiment        : float, current sentiment score (scalar fallback)
    news_count       : int, news count for this hour
    now              : datetime-like with .hour, .weekday(), .month attributes
    sentiment_history: list/array of length N with per-hour sentiment values (optional).
                       When provided, gives the model realistic time-varying sentiment
                       rather than a constant fill. Oldest value first.
    """
    import pandas as pd

    N = len(prices_arr)
    timestamps = pd.date_range(end=now, periods=N, freq="1h")

    # Build per-hour sentiment series
    if sentiment_history is not None and len(sentiment_history) >= N:
        # Use the provided history aligned to our N timestamps
        sent_series = np.array(sentiment_history[-N:], dtype=float)
    elif sentiment_history is not None and len(sentiment_history) > 0:
        # Shorter than N: pad older end with first value
        sh = np.array(sentiment_history, dtype=float)
        pad_len = N - len(sh)
        sent_series = np.concatenate([np.full(pad_len, sh[0]), sh])
    else:
        # Fallback: constant fill with current score
        sent_series = np.full(N, float(sentiment))

    df = pd.DataFrame({
        "timestamp":       timestamps,
        "price":           prices_arr.astype(float),
        "volume":          volumes_arr.astype(float),
        # Approximate OHLCV columns from close-price series
        "open":            np.concatenate([[prices_arr[0]], prices_arr[:-1]]),
        "high":            np.maximum(np.concatenate([[prices_arr[0]], prices_arr[:-1]]), prices_arr),
        "low":             np.minimum(np.concatenate([[prices_arr[0]], prices_arr[:-1]]), prices_arr),
        "trades":          np.full(N, 10000.0),   # typical BTC hourly trades
        "sentiment_score": sent_series,
        "news_count":      np.full(N, float(news_count)),
    })

    # ── Returns ────────────────────────────────────────────────────────────────
    df["returns"]     = df["price"].pct_change()
    df["log_returns"] = np.log(df["price"] / df["price"].shift(1))

    # returns_1h / returns_24h — original parquet columns (same as computed returns)
    df["returns_1h"]  = df["returns"]
    df["returns_24h"] = df["price"].pct_change(24)

    # ── Moving averages ────────────────────────────────────────────────────────
    for w in [6, 12, 24, 48, 168]:
        df[f"sma_{w}"]             = df["price"].rolling(w, min_periods=1).mean()
        df[f"ema_{w}"]             = df["price"].ewm(span=w, min_periods=1).mean()
        df[f"price_ratio_sma_{w}"] = df["price"] / df[f"sma_{w}"]

    # ── Volatility ─────────────────────────────────────────────────────────────
    for w in [6, 12, 24, 48]:
        df[f"volatility_{w}"]     = df["returns"].rolling(w, min_periods=1).std()
        df[f"volatility_ema_{w}"] = df["returns"].ewm(span=w, min_periods=1).std()

    # ── RSI ────────────────────────────────────────────────────────────────────
    delta = df["price"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - 100 / (1 + rs)).fillna(50)

    # ── MACD ───────────────────────────────────────────────────────────────────
    ema12          = df["price"].ewm(span=12, min_periods=1).mean()
    ema26          = df["price"].ewm(span=26, min_periods=1).mean()
    df["macd"]          = ema12 - ema26
    df["macd_signal"]   = df["macd"].ewm(span=9, min_periods=1).mean()
    df["macd_histogram"]= df["macd"] - df["macd_signal"]

    # ── Bollinger Bands ────────────────────────────────────────────────────────
    rm  = df["price"].rolling(20, min_periods=1).mean()
    rs2 = df["price"].rolling(20, min_periods=1).std().fillna(0)
    df["bb_upper"]    = rm + 2 * rs2
    df["bb_lower"]    = rm - 2 * rs2
    bb_range          = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_width"]    = bb_range / rm
    df["bb_position"] = ((df["price"] - df["bb_lower"]) / bb_range).fillna(0.5)

    # ── Momentum ───────────────────────────────────────────────────────────────
    for period in [1, 3, 6, 12, 24]:
        df[f"momentum_{period}"] = (df["price"] / df["price"].shift(period)).fillna(1.0)

    # ── Volume indicators ──────────────────────────────────────────────────────
    for w in [6, 12, 24, 48]:
        vsma = df["volume"].rolling(w, min_periods=1).mean()
        df[f"volume_sma_{w}"]   = vsma
        df[f"volume_ratio_{w}"] = (df["volume"] / vsma.replace(0, np.nan)).fillna(1.0)

    df["vpt"]      = (df["volume"] * df["returns"].fillna(0)).cumsum()
    df["vpt_sma_24"] = df["vpt"].rolling(24, min_periods=1).mean()
    df["obv"]      = (np.sign(df["returns"].fillna(0)) * df["volume"]).cumsum()

    # ── Sentiment & news ───────────────────────────────────────────────────────
    for w in [3, 6, 12, 24]:
        df[f"sentiment_sma_{w}"] = df["sentiment_score"].rolling(w, min_periods=1).mean()
        df[f"sentiment_ema_{w}"] = df["sentiment_score"].ewm(span=w, min_periods=1).mean()

    sent_sma24 = df["sentiment_score"].rolling(24, min_periods=1).mean()
    df["sentiment_momentum"]  = df["sentiment_score"] - sent_sma24
    df["sentiment_volatility"]= df["sentiment_score"].rolling(24, min_periods=1).std().fillna(0)

    for w in [6, 12, 24]:
        nsum = df["news_count"].rolling(w, min_periods=1).sum()
        nmean = nsum.rolling(w, min_periods=1).mean()
        df[f"news_count_sum_{w}"]  = nsum
        df[f"news_intensity_{w}"]  = (df["news_count"] / nmean.replace(0, np.nan)).fillna(1.0)

    # ── Time features ──────────────────────────────────────────────────────────
    df["hour"]        = df["timestamp"].dt.hour.astype(float)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(float)
    df["month"]       = df["timestamp"].dt.month.astype(float)
    df["quarter"]     = df["timestamp"].dt.quarter.astype(float)
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(float)
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)

    # Fill any remaining NaN with 0
    df = df.fillna(0)
    return df


def predict_fn(input_data, artifacts):
    import tensorflow as tf_lib
    from datetime import datetime, timezone

    model   = artifacts["model"]
    scalers = artifacts["scalers"]
    config  = artifacts["config"]

    current_price     = float(input_data.get("current_price", 50000))
    hourly_prices     = input_data.get("hourly_prices") or input_data.get("daily_prices") or []
    hourly_volumes    = input_data.get("hourly_volumes") or input_data.get("daily_volumes") or []
    sentiment_score   = float(input_data.get("sentiment_score", 0.0))
    sentiment_history = input_data.get("sentiment_history")   # 168-point list or None
    news_count        = int(input_data.get("news_count", 10))

    seq_len         = config.get("sequence_length", 168)
    feature_columns = config.get("feature_columns", [])
    target_columns  = config.get("target_columns", ["price_1h", "price_6h", "price_24h", "price_7d"])

    # ── Prepare price/volume arrays ────────────────────────────────────────────
    prices  = np.array(hourly_prices, dtype=float)  if hourly_prices  else np.full(seq_len + 50, current_price)
    volumes = np.array(hourly_volumes, dtype=float) if hourly_volumes else np.ones(len(prices)) * 1e8

    # Need at least seq_len + some buffer for rolling windows
    MIN_LEN = max(seq_len + 50, 220)
    if len(prices) < MIN_LEN:
        prices  = np.pad(prices,  (MIN_LEN - len(prices),  0), mode="edge")
    if len(volumes) < len(prices):
        volumes = np.pad(volumes, (len(prices) - len(volumes), 0), mode="edge")

    now = datetime.now(timezone.utc)

    # Log sentiment source for debugging
    if sentiment_history:
        print(f"[lstm] Using sentiment_history ({len(sentiment_history)} points), "
              f"current={sentiment_score:.3f}, "
              f"mean={np.mean(sentiment_history):.3f}")
    else:
        print(f"[lstm] No sentiment_history — constant fill with {sentiment_score:.3f}")

    # ── Build 85-feature DataFrame ─────────────────────────────────────────────
    df_feat = _build_feature_matrix(prices, volumes, sentiment_score, news_count, now,
                                     sentiment_history=sentiment_history)

    # ── Select and order features to match training ────────────────────────────
    if feature_columns:
        # Use exact training order; fill 0 for any column missing from our DataFrame
        available = set(df_feat.columns)
        missing   = [c for c in feature_columns if c not in available]
        if missing:
            print(f"[lstm] Warning: {len(missing)} features missing, defaulting to 0: {missing}")
            for c in missing:
                df_feat[c] = 0.0
        raw_matrix = df_feat[feature_columns].values.astype(float)
    else:
        # Fallback: use price/volume/sentiment (3-feature mode)
        raw_matrix = np.stack([prices[-seq_len:],
                                volumes[-seq_len:],
                                np.full(seq_len, sentiment_score)], axis=1).astype(float)

    # ── Scale features ─────────────────────────────────────────────────────────
    feature_scaler = scalers.get("features") or scalers.get("feature_scaler")
    if feature_scaler is not None:
        try:
            scaled_matrix = feature_scaler.transform(raw_matrix)
        except Exception as e:
            print(f"[lstm] Feature scaling failed: {e}; using raw values")
            scaled_matrix = raw_matrix
    else:
        scaled_matrix = raw_matrix

    # Clip to [0, 1] — MinMaxScaler may produce slightly out-of-range for unseen values
    scaled_matrix = np.clip(scaled_matrix, -3, 3)

    # ── Build sequence ─────────────────────────────────────────────────────────
    sequence = scaled_matrix[-seq_len:].reshape(1, seq_len, -1).astype("float32")
    X_tensor = tf_lib.constant(sequence)

    # ── Predict ────────────────────────────────────────────────────────────────
    try:
        raw_preds = model.predict(sequence, verbose=0)
    except AttributeError:
        # SavedModel: use serving signature
        try:
            infer = model.signatures.get("serving_default") or model.signatures.get("__call__")
            if infer is not None:
                output = infer(X_tensor)
                raw_preds = [v.numpy() for v in output.values()]
            else:
                raw_preds = model(X_tensor, training=False)
                raw_preds = [raw_preds.numpy()]
        except Exception as e2:
            print(f"[lstm] SavedModel call failed: {e2}")
            raw_preds = model(X_tensor, training=False).numpy()
    except Exception as e:
        print(f"[lstm] model.predict failed: {e}")
        raw_preds = model(X_tensor, training=False).numpy()

    # ── Inverse-transform targets ──────────────────────────────────────────────
    target_scaler = scalers.get("targets") or scalers.get("target_scaler")
    results = {}

    if isinstance(raw_preds, (list, tuple)):
        # 4 separate output arrays, one per target
        for i, col in enumerate(target_columns):
            if i < len(raw_preds):
                scaled_val = float(np.array(raw_preds[i]).flatten()[0])
                if target_scaler is not None:
                    try:
                        dummy = np.zeros((1, len(target_columns)))
                        dummy[0, i] = scaled_val
                        results[col] = float(target_scaler.inverse_transform(dummy)[0, i])
                    except Exception:
                        results[col] = scaled_val
                else:
                    results[col] = scaled_val
    else:
        # Single [1, n_targets] output
        preds_flat = np.array(raw_preds).flatten()
        if target_scaler is not None:
            try:
                inv = target_scaler.inverse_transform(preds_flat.reshape(1, -1)).flatten()
                for i, col in enumerate(target_columns):
                    if i < len(inv):
                        results[col] = float(inv[i])
            except Exception:
                for i, col in enumerate(target_columns):
                    if i < len(preds_flat):
                        results[col] = float(preds_flat[i])
        else:
            for i, col in enumerate(target_columns):
                if i < len(preds_flat):
                    results[col] = float(preds_flat[i])

    # ── Sanity cap: ±30% from current price ────────────────────────────────────
    price_24h = results.get("price_24h", current_price)
    if abs(price_24h - current_price) / max(current_price, 1) > 0.30:
        direction = 1 if price_24h > current_price else -1
        print(f"[lstm] Capping prediction ${price_24h:,.0f} to ±30% of ${current_price:,.0f}")
        price_24h = current_price * (1 + direction * 0.30)
        results["price_24h"] = price_24h

    print(f"[lstm] 1h=${results.get('price_1h', 0):,.0f} "
          f"6h=${results.get('price_6h', 0):,.0f} "
          f"24h=${price_24h:,.0f} "
          f"7d=${results.get('price_7d', 0):,.0f}")

    return {
        "prediction": price_24h,
        "price_24h":  price_24h,
        "price_1h":   results.get("price_1h", current_price),
        "price_6h":   results.get("price_6h", current_price),
        "price_7d":   results.get("price_7d", current_price),
    }


def output_fn(prediction, content_type="application/json"):
    return json.dumps(prediction)
