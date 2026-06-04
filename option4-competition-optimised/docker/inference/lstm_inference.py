"""
LSTM Inference — v3
Canonical reference — added by post-training Lambda (compile=False).

v3 change: model predicts 24h RETURN (not absolute price).
  prediction_usd = current_price * (1 + predicted_return)
  current_price comes from features['current_price'].
"""
import json, os, pickle, traceback
import numpy as np


def model_fn(model_dir):
    import tensorflow as tf
    model_path = os.path.join(model_dir, 'lstm_model.h5')
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        return data
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(data, model):
    import tensorflow as tf

    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')

    # Load scalers and config
    with open(os.path.join(model_dir, 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    feat_scaler  = scalers['features']
    tgt_scaler   = scalers['targets']
    feature_cols = config['feature_columns']
    seq_len      = config['sequence_length']          # 72 in v3
    target_type  = config.get('target_type', 'return')  # 'return' or 'price'

    # Current BTC price — needed to convert return → USD
    current_price = float(data.get('current_price', 0))

    # ── Build feature sequence ────────────────────────────────────────────────
    # Expected input: dict with list fields (hourly_prices, hourly_volumes, etc.)
    # or pre-computed feature vectors.

    hourly_prices   = data.get('hourly_prices', [])
    hourly_volumes  = data.get('hourly_volumes', [])
    sentiment_hist  = data.get('sentiment_history', [])
    news_count_hist = data.get('news_count_history', [])

    import pandas as pd
    from datetime import datetime, timezone, timedelta

    n = max(len(hourly_prices), seq_len + 200)
    now = datetime.now(timezone.utc)
    timestamps = [now - timedelta(hours=n - 1 - i) for i in range(n)]

    prices  = list(hourly_prices)[-n:]   if hourly_prices  else [current_price] * n
    volumes = list(hourly_volumes)[-n:]  if hourly_volumes else [1e9] * n

    # Pad if short
    while len(prices)  < n: prices.insert(0,  prices[0]  if prices  else current_price)
    while len(volumes) < n: volumes.insert(0, volumes[0] if volumes else 1e9)

    df = pd.DataFrame({
        'timestamp': timestamps[-n:],
        'price':     prices[-n:],
        'volume':    volumes[-n:],
    })

    # Add sentiment
    if sentiment_hist:
        sh = list(sentiment_hist)
        while len(sh) < n: sh.insert(0, sh[0] if sh else 0.0)
        df['sentiment_score'] = sh[-n:]
    else:
        df['sentiment_score'] = 0.0

    # Add news count
    if news_count_hist:
        nc = list(news_count_hist)
        while len(nc) < n: nc.insert(0, nc[0] if nc else 0)
        df['news_count'] = nc[-n:]
    else:
        df['news_count'] = 0

    # Feature engineering (must match training — see training script)
    df['returns']     = df['price'].pct_change()
    df['log_returns'] = np.log(df['price'] / df['price'].shift(1))

    for w in [6, 12, 24, 48, 168]:
        df[f'sma_{w}']             = df['price'].rolling(w).mean()
        df[f'ema_{w}']             = df['price'].ewm(span=w).mean()
        df[f'price_ratio_sma_{w}'] = df['price'] / df[f'sma_{w}']
    for w in [6, 12, 24, 48]:
        df[f'volatility_{w}']     = df['returns'].rolling(w).std()
        df[f'volatility_ema_{w}'] = df['returns'].ewm(span=w).std()

    delta = df['price'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))

    p = df['price']
    ema12 = p.ewm(span=12).mean()
    ema26 = p.ewm(span=26).mean()
    df['macd']           = (ema12 - ema26) / p
    df['macd_signal']    = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    rm = p.rolling(20).mean()
    rs = p.rolling(20).std()
    df['bb_upper_ratio'] = (rm + 2 * rs) / p
    df['bb_lower_ratio'] = (rm - 2 * rs) / p
    df['bb_width']       = (4 * rs) / rm
    df['bb_position']    = (p - (rm - 2*rs)) / (4 * rs)

    for lag in [1, 3, 6, 12, 24]:
        df[f'momentum_{lag}'] = p / p.shift(lag)

    vol = df['volume']
    for w in [6, 12, 24, 48]:
        df[f'volume_sma_{w}']   = vol.rolling(w).mean()
        df[f'volume_ratio_{w}'] = vol / df[f'volume_sma_{w}']
    raw_obv = (np.sign(df['returns']) * vol).cumsum()
    raw_vpt = (vol * df['returns']).cumsum()
    obv_std = raw_obv.rolling(168).std().replace(0, 1)
    vpt_std = raw_vpt.rolling(168).std().replace(0, 1)
    df['obv']        = raw_obv / obv_std
    df['vpt']        = raw_vpt / vpt_std
    df['vpt_sma_24'] = df['vpt'].rolling(24).mean()

    for w in [3, 6, 12, 24]:
        df[f'sentiment_sma_{w}'] = df['sentiment_score'].rolling(w).mean()
        df[f'sentiment_ema_{w}'] = df['sentiment_score'].ewm(span=w).mean()
    df['sentiment_momentum']   = df['sentiment_score'] - df['sentiment_sma_24']
    df['sentiment_volatility'] = df['sentiment_score'].rolling(24).std()

    for w in [6, 12, 24]:
        df[f'news_count_sum_{w}']  = df['news_count'].rolling(w).sum()
        df[f'news_intensity_{w}']  = (df['news_count'] /
                                       df[f'news_count_sum_{w}'].rolling(w).mean())

    df['hour']        = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month']       = df['timestamp'].dt.month
    df['quarter']     = df['timestamp'].dt.quarter
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(float)
    df['hour_sin']    = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']    = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin']     = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos']     = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin']   = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']   = np.cos(2 * np.pi * df['month'] / 12)

    df = df.dropna()

    # Select and order features to match training
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Missing features: {missing} — filling with 0")
        for c in missing:
            df[c] = 0.0

    feat_matrix = df[feature_cols].values
    if len(feat_matrix) < seq_len:
        raise ValueError(f"Not enough rows ({len(feat_matrix)}) for seq_len={seq_len}")

    # Take last seq_len rows → scale → reshape for model
    X = feat_scaler.transform(feat_matrix[-seq_len:])
    X = X.reshape(1, seq_len, len(feature_cols))

    # ── Predict ───────────────────────────────────────────────────────────────
    pred_scaled = model.predict(X, verbose=0)[0][0]
    pred_return = float(tgt_scaler.inverse_transform([[pred_scaled]])[0][0])

    # Convert return → USD
    if target_type == 'return' and current_price > 0:
        prediction_usd = current_price * (1.0 + pred_return)
    else:
        # Fallback for older models or if current_price missing
        prediction_usd = pred_return

    print(f"[LSTM v3] current=${current_price:,.0f}  return={pred_return:+.4f}  "
          f"prediction=${prediction_usd:,.0f}")
    return prediction_usd


def output_fn(prediction, accept):
    return json.dumps({'prediction': float(prediction)}), 'application/json'
