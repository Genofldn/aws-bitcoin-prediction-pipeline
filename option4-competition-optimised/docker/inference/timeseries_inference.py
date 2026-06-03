"""
TimeSeries inference.py — ARIMA(2,1,2) + Prophet ensemble.
Uses daily price history to forecast 24h ahead.

Input JSON:
{
  "daily_prices":  [90000, 91000, ...],  # 30-60 daily BTC prices, most-recent last
  "daily_volumes": [5e9, ...],           # matching daily volumes
  "hourly_prices": [96000, ...],         # optional, for short-term ARIMA
  "current_price": 96000,
  "date":          "2026-06-03",
  "run_type":      "morning"
}

Output JSON:
{
  "prediction": 97200.0,
  "price_24h":  97200.0,
  "arima_pred": 97000.0,
  "prophet_pred": 97400.0
}
"""

import json
import os
import pickle
import numpy as np
from datetime import datetime, timezone, timedelta


def model_fn(model_dir):
    """Load Prophet model and ARIMA pickle from model directory."""
    artifacts = {}

    # ARIMA
    arima_path = os.path.join(model_dir, "arima_model.pkl")
    if os.path.exists(arima_path):
        with open(arima_path, "rb") as f:
            artifacts["arima"] = pickle.load(f)
        print("[timeseries] ARIMA model loaded")

    # Prophet
    prophet_path = os.path.join(model_dir, "prophet_model.pkl")
    if os.path.exists(prophet_path):
        with open(prophet_path, "rb") as f:
            artifacts["prophet"] = pickle.load(f)
        print("[timeseries] Prophet model loaded")

    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            artifacts["config"] = json.load(f)

    if not artifacts:
        raise RuntimeError(f"No model files found in {model_dir}")

    return artifacts


def input_fn(request_body, content_type="application/json"):
    return json.loads(request_body)


def predict_fn(input_data, artifacts):
    current_price = float(input_data.get("current_price", 50000))
    daily_prices  = input_data.get("daily_prices") or input_data.get("hourly_prices") or []
    date_str      = input_data.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    if not daily_prices:
        daily_prices = [current_price] * 30

    prices = np.array(daily_prices, dtype=float)

    arima_pred   = None
    prophet_pred = None

    # ── ARIMA Forecast ────────────────────────────────────────────────────────
    arima_model = artifacts.get("arima")
    if arima_model is not None:
        try:
            # Re-fit ARIMA on latest prices and forecast 1 day ahead
            from statsmodels.tsa.arima.model import ARIMA

            series = prices[-min(60, len(prices)):]
            model  = ARIMA(series, order=(2, 1, 2))
            fitted = model.fit()
            forecast = fitted.forecast(steps=1)
            arima_pred = float(forecast.iloc[0]) if hasattr(forecast, "iloc") else float(forecast[0])
            print(f"[timeseries] ARIMA 24h forecast: ${arima_pred:,.0f}")
        except Exception as e:
            print(f"[timeseries] ARIMA forecast error: {e}")
            arima_pred = None

    # ── Prophet Forecast ──────────────────────────────────────────────────────
    prophet_model = artifacts.get("prophet")
    if prophet_model is not None:
        try:
            import pandas as pd
            from prophet import Prophet

            # Build a daily DS dataframe ending "today"
            end_date = datetime.strptime(date_str, "%Y-%m-%d")
            n_days   = len(prices)
            dates    = [end_date - timedelta(days=n_days - 1 - i) for i in range(n_days)]

            df_hist = pd.DataFrame({
                "ds": pd.to_datetime(dates),
                "y":  prices,
            })
            # Prophet requires timezone-naive
            df_hist["ds"] = df_hist["ds"].dt.tz_localize(None)

            # Fit on latest history (Prophet is a generative model — always refit)
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_mode="multiplicative",
            )
            m.fit(df_hist)

            future = m.make_future_dataframe(periods=1, freq="D")
            forecast = m.predict(future)
            prophet_pred = float(forecast.iloc[-1]["yhat"])
            print(f"[timeseries] Prophet 24h forecast: ${prophet_pred:,.0f}")
        except Exception as e:
            print(f"[timeseries] Prophet forecast error: {e}")
            prophet_pred = None

    # ── Combine ───────────────────────────────────────────────────────────────
    predictions = [p for p in [arima_pred, prophet_pred] if p is not None]
    if predictions:
        # Equal weight between available models
        final_pred = float(np.mean(predictions))
    else:
        # Fallback: simple linear extrapolation
        if len(prices) >= 2:
            last_delta = float(prices[-1] - prices[-2])
            final_pred = float(prices[-1]) + last_delta
        else:
            final_pred = current_price

    # Sanity check: cap at ±20% from current
    if abs(final_pred - current_price) / current_price > 0.20:
        direction = 1 if final_pred > current_price else -1
        final_pred = current_price * (1 + direction * 0.20)
        print(f"[timeseries] Prediction capped at ${final_pred:,.0f}")

    print(f"[timeseries] Final: ARIMA=${arima_pred or 0:,.0f} "
          f"Prophet=${prophet_pred or 0:,.0f} → ${final_pred:,.0f}")

    return {
        "prediction":   final_pred,
        "price_24h":    final_pred,
        "arima_pred":   arima_pred or final_pred,
        "prophet_pred": prophet_pred or final_pred,
    }


def output_fn(prediction, content_type="application/json"):
    return json.dumps(prediction)
