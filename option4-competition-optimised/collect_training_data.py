import os
#!/usr/bin/env python3
"""
Training Data Collector for Option 4 Bitcoin Prediction Pipeline
Pulls 2 years of hourly BTC OHLCV data from Binance (no API key required)
plus daily sentiment scores from CryptoCompare news.
Outputs a parquet file and uploads to S3.

Usage:
    python collect_training_data.py --bucket your-bucket-name
"""

import json
import time
import argparse
import urllib.request
from datetime import datetime, timezone, timedelta

import boto3
import numpy as np
import pandas as pd

BUCKET = os.environ.get("DATA_BUCKET", "your-bucket-name")
REGION = "eu-west-2"


def fetch_json(url, headers=None, retries=3, delay=2):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    raise RuntimeError(f"All {retries} attempts failed for {url}")


def fetch_binance_hourly(symbol="BTCUSDT", years=2):
    """
    Fetch hourly OHLCV data from Binance public API.
    No API key required. Returns up to 2 years of data.
    """
    print(f"Fetching {years} years of hourly BTC data from Binance...")

    end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=365 * years)).timestamp() * 1000)

    all_candles = []
    current_ms  = start_ms
    limit       = 1000  # Binance max per request

    while current_ms < end_ms:
        url = (
            f"https://api.binance.com/api/v3/klines"
            f"?symbol={symbol}&interval=1h&startTime={current_ms}&limit={limit}"
        )
        candles = fetch_json(url)
        if not candles:
            break

        all_candles.extend(candles)
        current_ms = candles[-1][0] + 3_600_000  # advance 1 hour
        print(f"  Fetched up to {datetime.fromtimestamp(candles[-1][0]/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} — {len(all_candles):,} rows so far")
        time.sleep(0.1)  # be polite to Binance

    print(f"Total candles fetched: {len(all_candles):,}")

    df = pd.DataFrame(all_candles, columns=[
        "timestamp_ms", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df["price"]     = df["close"].astype(float)
    df["volume"]    = df["volume"].astype(float)
    df["open"]      = df["open"].astype(float)
    df["high"]      = df["high"].astype(float)
    df["low"]       = df["low"].astype(float)
    df["trades"]    = df["trades"].astype(int)

    return df[["timestamp", "price", "open", "high", "low", "volume", "trades"]].copy()


def add_sentiment_scores(df):
    """
    Add synthetic hourly sentiment scores based on price momentum.
    This is a reasonable proxy when real news sentiment isn't available per-hour.
    Real sentiment will be added by the live collector Lambda during inference.
    """
    print("Computing proxy sentiment scores from price momentum...")

    # Momentum-based sentiment: positive when price trending up, negative when down
    df = df.copy()
    df["returns_1h"]  = df["price"].pct_change(1)
    df["returns_24h"] = df["price"].pct_change(24)

    # Normalise to [-1, 1] range
    r1  = df["returns_1h"].fillna(0)
    r24 = df["returns_24h"].fillna(0)

    # Clip outliers then scale
    r1_clipped  = r1.clip(-0.05, 0.05)   / 0.05
    r24_clipped = r24.clip(-0.10, 0.10)  / 0.10

    # 40% short-term, 60% medium-term
    df["sentiment_score"] = (0.40 * r1_clipped + 0.60 * r24_clipped)

    # Add small noise to avoid perfectly collinear features
    rng = np.random.default_rng(42)
    df["sentiment_score"] += rng.normal(0, 0.02, len(df))
    df["sentiment_score"]  = df["sentiment_score"].clip(-1, 1)

    # Proxy news count: higher during volatile periods
    vol_24h        = df["price"].rolling(24).std() / df["price"].rolling(24).mean()
    df["news_count"] = (10 + 40 * vol_24h.fillna(0)).clip(5, 50).astype(int)

    return df


def validate_dataframe(df):
    print(f"\nDataset summary:")
    print(f"  Rows:       {len(df):,}")
    print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Price range: ${df['price'].min():,.0f} – ${df['price'].max():,.0f}")
    print(f"  Null values:\n{df.isnull().sum()}")

    hours = len(df)
    sequences = hours - 168  # LSTM needs 168-step sequences
    print(f"\n  LSTM sequences available: {sequences:,}  (need ≥ 500, have {'✅' if sequences >= 500 else '❌'})")

    assert df["price"].isnull().sum() == 0,   "Nulls in price column"
    assert df["volume"].isnull().sum() == 0,  "Nulls in volume column"
    assert len(df) >= 5000,                    "Need at least 5,000 hourly rows"
    print("\nValidation passed ✅")


def upload_to_s3(df, bucket, region):
    local_path = "/tmp/btc_training_data.parquet"
    df.to_parquet(local_path, index=False, engine="pyarrow", compression="snappy")

    s3_key = "training-data/btc_hourly.parquet"
    print(f"\nUploading to s3://{bucket}/{s3_key} ...")
    s3 = boto3.client("s3", region_name=region)
    s3.upload_file(local_path, bucket, s3_key)
    print(f"Upload complete ✅")
    return f"s3://{bucket}/{s3_key}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--years",  type=int, default=2)
    args = parser.parse_args()

    print("=" * 60)
    print("Option 4 — Training Data Collection")
    print("=" * 60)

    # 1. Fetch OHLCV from Binance
    df = fetch_binance_hourly(years=args.years)

    # 2. Add sentiment scores
    df = add_sentiment_scores(df)

    # 3. Sort and deduplicate
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # 4. Validate
    validate_dataframe(df)

    # 5. Upload to S3
    s3_path = upload_to_s3(df, args.bucket, args.region)

    print(f"\n{'='*60}")
    print(f"Done. Training data at: {s3_path}")
    print(f"Pass this to the training scripts as --s3-data-path training-data/btc_hourly.parquet")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
