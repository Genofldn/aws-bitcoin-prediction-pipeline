#!/usr/bin/env python3
"""
Bitcoin LSTM Training Script — v3

Architecture fixes over v2:
  1. RETURN-BASED TARGET  — predicts 24h % return, not absolute price.
     Model is now price-level agnostic: no bias toward training-data mean.
     At inference: prediction_usd = current_price * (1 + predicted_return)

  2. SINGLE TARGET        — only return_24h. Multi-target (1h/6h/24h/7d) caused
     the shared representation to be polluted by longer-horizon signals,
     dragging 24h predictions toward 7-day bullish patterns.

  3. RATIO-NORMALISED FEATURES — MACD/signal/histogram now divided by price
     (→ dimensionless). bb_upper/bb_lower replaced by bb_upper_ratio/bb_lower_ratio.
     OBV/VPT normalised by rolling std. Prevents out-of-distribution inputs
     when price is far from training-data mean.

  4. 72h SEQUENCE (was 168h) — 3 days of hourly data is sufficient for 24h
     prediction. Shorter sequences: faster training, less attention diffusion,
     better focus on recent candles.
"""

import os, sys, json, io, tarfile, pickle, argparse, warnings
import numpy as np
import pandas as pd
import boto3
from datetime import datetime, timezone
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

REGION          = os.environ.get('AWS_DEFAULT_REGION', 'eu-west-2')
SEQUENCE_LENGTH = 72          # v3: 3 days (was 168h = 1 week)
TARGET_COL      = 'return_24h' # v3: single return target (was 4 absolute prices)


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df):
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    if 'price' in df.columns:
        p = df['price']

        # Returns (dimensionless ✅)
        df['returns']     = p.pct_change()
        df['log_returns'] = np.log(p / p.shift(1))

        for w in [6, 12, 24, 48, 168]:
            df[f'sma_{w}']             = p.rolling(w).mean()
            df[f'ema_{w}']             = p.ewm(span=w).mean()
            df[f'price_ratio_sma_{w}'] = p / df[f'sma_{w}']   # ratio ✅

        for w in [6, 12, 24, 48]:
            df[f'volatility_{w}']     = df['returns'].rolling(w).std()   # dimensionless ✅
            df[f'volatility_ema_{w}'] = df['returns'].ewm(span=w).std()

        # RSI (dimensionless ✅)
        delta = p.diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))

        # MACD — v3: normalised by price → dimensionless ✅
        ema12 = p.ewm(span=12).mean()
        ema26 = p.ewm(span=26).mean()
        df['macd']           = (ema12 - ema26) / p
        df['macd_signal']    = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Bollinger Bands — v3: upper/lower as price ratios ✅ (drop absolute values)
        rm = p.rolling(20).mean()
        rs = p.rolling(20).std()
        df['bb_upper_ratio'] = (rm + 2 * rs) / p  # e.g. 1.065
        df['bb_lower_ratio'] = (rm - 2 * rs) / p  # e.g. 0.975
        df['bb_width']       = (4 * rs) / rm       # dimensionless ✅
        df['bb_position']    = (p - (rm - 2*rs)) / (4 * rs)  # 0–1 ✅

        # Momentum (price ratios ✅)
        for lag in [1, 3, 6, 12, 24]:
            df[f'momentum_{lag}'] = p / p.shift(lag)

        # v3 TARGET: 24h percentage return (dimensionless, price-level agnostic ✅)
        df[TARGET_COL] = (p.shift(-24) - p) / p

    if 'volume' in df.columns:
        vol = df['volume']
        for w in [6, 12, 24, 48]:
            df[f'volume_sma_{w}']   = vol.rolling(w).mean()
            df[f'volume_ratio_{w}'] = vol / df[f'volume_sma_{w}']  # ratio ✅

        # OBV / VPT — normalise by rolling std to remove cumulative drift ✅
        raw_obv = (np.sign(df['returns']) * vol).cumsum()
        raw_vpt = (vol * df['returns']).cumsum()
        obv_std = raw_obv.rolling(168).std().replace(0, 1)
        vpt_std = raw_vpt.rolling(168).std().replace(0, 1)
        df['obv']        = raw_obv / obv_std
        df['vpt']        = raw_vpt / vpt_std
        df['vpt_sma_24'] = df['vpt'].rolling(24).mean()

    if 'sentiment_score' in df.columns:
        for w in [3, 6, 12, 24]:
            df[f'sentiment_sma_{w}'] = df['sentiment_score'].rolling(w).mean()
            df[f'sentiment_ema_{w}'] = df['sentiment_score'].ewm(span=w).mean()
        df['sentiment_momentum']   = df['sentiment_score'] - df['sentiment_sma_24']
        df['sentiment_volatility'] = df['sentiment_score'].rolling(24).std()

    if 'news_count' in df.columns:
        for w in [6, 12, 24]:
            df[f'news_count_sum_{w}']  = df['news_count'].rolling(w).sum()
            df[f'news_intensity_{w}']  = (df['news_count'] /
                                          df[f'news_count_sum_{w}'].rolling(w).mean())

    if 'timestamp' in df.columns:
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
    exclude      = {'timestamp', TARGET_COL, 'price'}
    feature_cols = [c for c in df.columns if c not in exclude]
    return df, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Model — v3: single return output
# ─────────────────────────────────────────────────────────────────────────────

def build_model(input_shape, params):
    inp = keras.Input(shape=input_shape)

    # Two LSTM layers
    x = layers.LSTM(params['lstm_units_1'], return_sequences=True)(inp)
    x = layers.Dropout(params['dropout_rate'])(x)
    x = layers.LSTM(params['lstm_units_2'], return_sequences=True)(x)
    x = layers.Dropout(params['dropout_rate'])(x)

    # Multi-head attention — focuses on most relevant timesteps
    attn = layers.MultiHeadAttention(
        num_heads=params['num_heads'],
        key_dim=params['attention_key_dim'],
        dropout=params['dropout_rate'] / 2
    )(x, x)
    x = layers.Add()([x, attn])                    # residual connection
    x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(params['dense_units'], activation='relu')(x)
    x = layers.Dropout(params['dropout_rate'])(x)
    x = layers.Dense(params['dense_units'] // 2, activation='relu')(x)

    # v3: SINGLE output — 24h return (dimensionless)
    output = layers.Dense(1, activation='linear', name='return_24h')(x)

    model = Model(inputs=inp, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=params['learning_rate']),
        loss='huber',         # more robust to outlier return spikes than MSE
        metrics=['mae']
    )
    return model


def make_sequences(feat, tgt, seq_len):
    X, y = [], []
    for i in range(seq_len, len(feat)):
        X.append(feat[i - seq_len:i])
        y.append(tgt[i])
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    s3 = boto3.client('s3', region_name=REGION)

    print(f"\n{'='*60}")
    print(f"Bitcoin LSTM v3 Training")
    print(f"  Sequence:  {SEQUENCE_LENGTH}h  (was 168h)")
    print(f"  Target:    {TARGET_COL}  (return, not absolute price)")
    print(f"  Optuna:    {args.optuna_trials} trials × {args.optuna_epochs} epochs")
    print(f"  Final:     {args.epochs} epochs, patience=15")
    print(f"{'='*60}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("STEP 1 — Loading training data")
    local_data = '/tmp/btc_hourly.parquet'
    s3.download_file(args.s3_bucket, args.s3_data_path, local_data)
    df = pd.read_parquet(local_data)
    print(f"  Full dataset: {len(df):,} rows × {len(df.columns)} cols")

    if len(df) > args.training_window_hours:
        df = df.tail(args.training_window_hours).reset_index(drop=True)
        print(f"  Trimmed to last {args.training_window_hours:,}h "
              f"({args.training_window_hours/24/30:.1f} months): {len(df):,} rows")

    # ── Feature engineering ───────────────────────────────────────────────────
    print("\nSTEP 2 — Feature engineering (v3: ratio-normalised)")
    df, feature_cols = build_features(df)
    n_feats = len(feature_cols)
    print(f"  {n_feats} features, {len(df):,} usable rows")
    print(f"  Target 24h return stats: "
          f"mean={df[TARGET_COL].mean():.4f}  "
          f"std={df[TARGET_COL].std():.4f}  "
          f"min={df[TARGET_COL].min():.4f}  "
          f"max={df[TARGET_COL].max():.4f}")

    # Scale features with MinMaxScaler; target with StandardScaler
    # (returns are approx Gaussian — StandardScaler more appropriate than MinMax)
    feat_scaler = MinMaxScaler()
    tgt_scaler  = StandardScaler()

    feat_scaled = feat_scaler.fit_transform(df[feature_cols].values)
    tgt_scaled  = tgt_scaler.fit_transform(df[[TARGET_COL]].values).flatten()

    # Time-series split: last 30d benchmark, prior 80/20 train/val
    BENCH_HOURS = 720
    X_all, y_all = make_sequences(feat_scaled, tgt_scaled, SEQUENCE_LENGTH)
    bench_start  = max(0, len(X_all) - BENCH_HOURS)
    X_bench, y_bench = X_all[bench_start:], y_all[bench_start:]
    X_main,  y_main  = X_all[:bench_start],  y_all[:bench_start]
    val_split = int(len(X_main) * 0.8)
    X_train, y_train = X_main[:val_split], y_main[:val_split]
    X_val,   y_val   = X_main[val_split:], y_main[val_split:]
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Benchmark: {len(X_bench):,}")

    # ── Optuna hyperparameter search ──────────────────────────────────────────
    print(f"\nSTEP 3 — Optuna search "
          f"({args.optuna_trials} trials × {args.optuna_epochs} epochs)")

    def objective(trial):
        p = {
            'lstm_units_1':      trial.suggest_int('lstm_units_1',      32,  256),
            'lstm_units_2':      trial.suggest_int('lstm_units_2',      16,  128),
            'dropout_rate':      trial.suggest_float('dropout_rate',    0.1,  0.5),
            'num_heads':         trial.suggest_int('num_heads',          2,    8),
            'attention_key_dim': trial.suggest_int('attention_key_dim', 16,   64),
            'dense_units':       trial.suggest_int('dense_units',       32,  128),
            'learning_rate':     trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        }
        m = build_model((SEQUENCE_LENGTH, n_feats), p)
        h = m.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.optuna_epochs,
            batch_size=args.batch_size,
            verbose=0,
            callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        )
        return min(h.history['val_loss'])

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.optuna_trials)
    best_params = study.best_params
    print(f"  Best val_loss: {study.best_value:.6f}")
    print(f"  Best params:   {best_params}")

    # ── Final model using best hyperparameters ────────────────────────────────
    print(f"\nSTEP 4 — Final training ({args.epochs} epochs, patience=15, Huber loss)")
    model = build_model((SEQUENCE_LENGTH, n_feats), best_params)
    cbs = [
        callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-7, monitor='val_loss'),
        callbacks.ModelCheckpoint('/tmp/best.weights.h5', save_best_only=True,
                                  save_weights_only=True, monitor='val_loss'),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cbs,
        verbose=1
    )
    model.load_weights('/tmp/best.weights.h5')
    best_val_loss = min(history.history['val_loss'])
    print(f"  Best val_loss: {best_val_loss:.6f}  "
          f"({len(history.history['val_loss'])} epochs)")

    # ── Walk-forward benchmark (last 30 days) ─────────────────────────────────
    print(f"\nSTEP 5 — Walk-forward benchmark (last 30 days)")

    bench_preds_scaled = model.predict(X_bench, verbose=0).flatten()
    bench_preds_return = tgt_scaler.inverse_transform(
        bench_preds_scaled.reshape(-1, 1)).flatten()
    bench_true_return  = tgt_scaler.inverse_transform(
        y_bench.reshape(-1, 1)).flatten()

    # Convert returns to USD MAE using actual prices in benchmark window
    bench_prices = df['price'].values[bench_start + SEQUENCE_LENGTH:]
    bench_prices = bench_prices[:len(bench_preds_return)]
    pred_prices  = bench_prices * (1 + bench_preds_return)
    true_prices  = bench_prices * (1 + bench_true_return)
    bench_mae    = mean_absolute_error(true_prices, pred_prices)
    bench_pct    = bench_mae / np.mean(np.abs(true_prices)) * 100

    # Return-level MAE
    return_mae   = mean_absolute_error(bench_true_return, bench_preds_return)
    print(f"  24h MAE (USD):    ${bench_mae:,.0f}  ({bench_pct:.2f}%)")
    print(f"  24h MAE (return): {return_mae:.4f}  ({return_mae*100:.2f}%)")

    # Compare to previous benchmark
    METRICS_KEY = f'{args.s3_prefix}/training_metrics.json'
    prev_mae = None
    try:
        obj      = s3.get_object(Bucket=args.s3_bucket, Key=METRICS_KEY)
        prev     = json.loads(obj['Body'].read())
        prev_mae = prev.get('bench_mae_24h')
        pct_chg  = (bench_mae - prev_mae) / prev_mae * 100
        status   = 'better ✅' if pct_chg < 0 else (
                   'WARNING: significantly worse ⚠️' if pct_chg > 20 else 'similar ➡️')
        print(f"  Previous MAE: ${prev_mae:,.0f}  | Change: {pct_chg:+.1f}%  [{status}]")
    except Exception:
        print("  No previous metrics — first v3 benchmark.")

    metrics = {
        'bench_mae_24h':      round(float(bench_mae), 2),
        'bench_mae_pct':      round(float(bench_pct), 4),
        'bench_return_mae':   round(float(return_mae), 6),
        'best_val_loss':      round(float(best_val_loss), 8),
        'optuna_trials':      args.optuna_trials,
        'training_window_h':  args.training_window_hours,
        'sequence_length':    SEQUENCE_LENGTH,
        'target':             TARGET_COL,
        'model_version':      'v3',
        'epochs_run':         len(history.history['val_loss']),
        'trained_at':         datetime.now(timezone.utc).isoformat(),
        'prev_bench_mae_24h': prev_mae,
    }

    # ── Save model artifacts ──────────────────────────────────────────────────
    print(f"\nSTEP 6 — Saving model artifacts")
    model_dir = '/tmp/model_out'
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'lstm_model.h5'))

    with open(os.path.join(model_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump({'features': feat_scaler, 'targets': tgt_scaler}, f)

    config = {
        'sequence_length':    SEQUENCE_LENGTH,
        'feature_columns':    feature_cols,
        'target_column':      TARGET_COL,       # singular — v3
        'target_type':        'return',         # v3: return not absolute price
        'model_version':      'v3',
        'model_type':         'lstm_attention',
        'framework':          'tensorflow_keras',
        'training_timestamp': datetime.now(timezone.utc).isoformat(),
        'bench_mae_24h':      round(float(bench_mae), 2),
        'best_params':        best_params,
    }
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # NOTE: inference.py NOT saved here — post-training Lambda adds canonical v3 version.

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for fname in os.listdir(model_dir):
            fpath = os.path.join(model_dir, fname)
            if not fname.startswith('._'):
                tar.add(fpath, arcname=fname)
    buf.seek(0)

    model_key = f'{args.s3_prefix}/model.tar.gz'
    s3.put_object(Bucket=args.s3_bucket, Key=model_key, Body=buf.getvalue())
    s3.put_object(Bucket=args.s3_bucket, Key=METRICS_KEY,
                  Body=json.dumps(metrics, indent=2).encode())
    print(f"  Model   → s3://{args.s3_bucket}/{model_key}")
    print(f"  Metrics → s3://{args.s3_bucket}/{METRICS_KEY}")

    print(f"\n{'='*60}")
    print(f"v3 Training complete!")
    print(f"  24h MAE: ${bench_mae:,.0f} ({bench_pct:.2f}%)  |  return MAE: {return_mae*100:.2f}%")
    print(f"  Sequence: {SEQUENCE_LENGTH}h  |  Target: {TARGET_COL}  |  Features: {n_feats}")
    print(f"{'='*60}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--s3-bucket',             required=True)
    p.add_argument('--s3-data-path',          default='training-data/btc_hourly.parquet')
    p.add_argument('--s3-prefix',             default='models/deeplearning')
    p.add_argument('--training-window-hours', type=int, default=4380)
    p.add_argument('--optuna-trials',         type=int, default=50)
    p.add_argument('--optuna-epochs',         type=int, default=30)
    p.add_argument('--epochs',                type=int, default=150)
    p.add_argument('--batch-size',            type=int, default=32)
    args = p.parse_args()
    print(f"Config: window={args.training_window_hours}h  "
          f"optuna={args.optuna_trials}×{args.optuna_epochs}ep  "
          f"final={args.epochs}ep  seq={SEQUENCE_LENGTH}h")
    train(args)


if __name__ == '__main__':
    main()
