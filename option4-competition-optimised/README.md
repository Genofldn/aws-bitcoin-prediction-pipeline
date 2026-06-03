# Option 4 — Competition-Optimised Bitcoin Prediction Pipeline

> **Goal:** Fully automated 24h BTC price prediction pipeline on AWS, optimised for a daily competition.  
> **Cost:** ~$15/month (vs $500–700/month for a provisioned equivalent)

---

## Architecture

```
07:00 UTC — EventBridge (morning, 30% weight)
20:00 UTC — EventBridge (evening, 70% weight)
        │
        ▼
Lambda: Pipeline Coordinator
        │
        ├─► Lambda: Data Collector
        │     → CoinGecko (live price/volume)
        │     → NewsAPI (headlines)
        │     → CryptoCompare (OHLCV)
        │     → S3: raw-data/{date}/{run_type}.json
        │
        ├─► Lambda: Feature Engineer
        │     → 85 features: RSI, MACD, Bollinger Bands, sentiment history, etc.
        │     → Validates data quality (stale/corrupt detection + S3 fallback)
        │     → S3: features/{date}/{run_type}.json
        │
        ├─► SageMaker Serverless Endpoints (4× invoked in parallel)
        │     → Ensemble:     XGBoost + Random Forest + Gradient Boosting
        │     → Deep Learning: LSTM with multi-head attention (85 features)
        │     → Sentiment:    FinBERT-based model
        │     → Time Series:  ARIMA + Prophet (refits on live data each run)
        │
        └─► Lambda: Ensemble Orchestrator
              → Hallucination guard: aborts if prediction outside [$15k–$600k]
              → Cross-model spread warning (>$25k gap)
              → 30/70 morning/evening blend → DynamoDB
              → SNS email with prediction + model breakdown

07:30 UTC daily — Accuracy Tracker Lambda
        → Fetches actual BTC price from CoinGecko
        → Computes error vs yesterday's prediction
        → Builds 7-day rolling MAE summary
        → Emails performance report

02:00 UTC Sunday — Weekly Retraining Lambda
        ├─ Data refresh: CoinGecko 7-day hourly → appends to parquet (2-year rolling window)
        ├─ Ensemble retrain: XGB+RF+GB in-Lambda (~7 min)
        └─ LSTM training job: ml.m5.4xlarge (~4–5h)
                │
                └─► EventBridge (on job completion)
                      → Lambda: Post-Training Auto-Deploy
                      → Repacks model with canonical inference.py
                      → Creates new SageMaker model + endpoint config
                      → Updates LSTM endpoint (zero-downtime)
                      → Emails "✅ LSTM Model Auto-Deployed"
```

---

## LSTM Training (v2)

| Setting | Value |
|---------|-------|
| Training window | Last 6 months (regime-aware, not 2-year) |
| Optuna trials | 50 × 30 epochs (hyperparameter search) |
| Final training | 150 epochs, patience=15, LR decay |
| Best params applied | ✅ (v1 bug: always used defaults) |
| Walk-forward benchmark | 24h MAE on last 30 days, saved to S3 |
| Comparison | Flags if new model is >20% worse than previous |

---

## Safety Features

| Feature | Detail |
|---------|--------|
| Input data validator | Checks: length ≥24, no stale/identical prices, no >50% single-candle moves. Falls back to previous S3 features on failure |
| Hallucination guard | Blended prediction must be $15k–$600k or pipeline aborts + alerts |
| Model spread warning | Logs warning if any two models diverge >$25k |
| Accuracy tracker | Daily MAE tracking, 7-day rolling email summary |
| Deduplication | S3 `IfNoneMatch` lock prevents double EventBridge fires |

---

## Cost Breakdown (~$15.20/month)

| Service | Monthly |
|---------|---------|
| SageMaker Training (Sunday, ml.m5.4xlarge, ~5h) | ~$14.75 |
| SageMaker Serverless (4 endpoints × 2 runs/day) | ~$0.42 |
| Lambda, S3, DynamoDB, SNS, EventBridge | ~$0.03 |
| **Total** | **~$15.20** |

---

## Deployment

### Prerequisites

1. **S3 bucket** — create before deploying, pass as `DataBucketName`
2. **ECR images** — build and push all 4 Docker images:
   ```bash
   ./build_and_push.sh
   ```
3. **Model artifacts** — upload initial models to S3:
   ```bash
   python repackage_models.py
   ```
4. **API keys** (free tiers sufficient):
   - [NewsAPI](https://newsapi.org) — 100 req/day free
   - [CryptoCompare](https://min-api.cryptocompare.com) — free tier

### Deploy

```bash
export DATA_BUCKET=your-bucket-name
export NEWS_API_KEY=your_newsapi_key
export CRYPTOCOMPARE_API_KEY=your_cryptocompare_key

aws cloudformation deploy \
  --template-file infrastructure/main-cloudformation.yaml \
  --stack-name bitcoin-prediction-option4 \
  --region eu-west-2 \
  --capabilities CAPABILITY_NAMED_IAM \
  --s3-bucket $DATA_BUCKET \
  --s3-prefix cloudformation \
  --parameter-overrides \
      Environment=production \
      DataBucketName=$DATA_BUCKET \
      NotificationEmail=your-email@example.com \
      NewsApiKey=$NEWS_API_KEY \
      CryptoCompareApiKey=$CRYPTOCOMPARE_API_KEY \
  --no-fail-on-empty-changeset
```

### Verify

```bash
# Check all 4 endpoints are InService
aws sagemaker list-endpoints --region eu-west-2 \
  --query 'Endpoints[?contains(EndpointName,`bitcoin`)].{Name:EndpointName,Status:EndpointStatus}' \
  --output table

# Trigger a manual run
aws lambda invoke \
  --function-name bitcoin-pipeline-coordinator-production \
  --region eu-west-2 \
  --payload '{"run_type":"evening"}' \
  --cli-binary-format raw-in-base64-out \
  /tmp/out.json && cat /tmp/out.json

# Check today's prediction in DynamoDB
aws dynamodb get-item \
  --table-name bitcoin-predictions-production \
  --region eu-west-2 \
  --key '{"date":{"S":"'$(date +%Y-%m-%d)'"},"run_type":{"S":"evening"}}'
```

---

## Key Lambda Functions

| Function | Purpose |
|----------|---------|
| `bitcoin-pipeline-coordinator-production` | Orchestrates each run: triggers collector → features → orchestrator |
| `bitcoin-feature-engineer-production` | Fetches live data, builds 85 features, validates quality |
| `bitcoin-ensemble-orchestrator-production` | Calls all 4 endpoints, blends, applies guards, writes to DynamoDB |
| `bitcoin-weekly-retrain-production` | Sunday retraining: data refresh + ensemble + LSTM job submission |
| `bitcoin-lstm-post-training-production` | Auto-deploys new LSTM model on training completion |
| `bitcoin-accuracy-tracker-production` | Daily MAE tracking and 7-day rolling email summary |

---

## S3 Structure

```
{bucket}/
├── raw-data/{date}/{morning|evening}.json      # Live fetched data per run
├── features/{date}/{morning|evening}.json      # Engineered features per run
├── sentiment-history/{date}/{run_type}.json   # FinBERT score history (168-point series)
├── accuracy/{date}.json                        # Daily prediction vs actual
├── models/
│   ├── ensemble/model.tar.gz                  # Live ensemble model
│   ├── ensemble/inference.py                  # Canonical reference inference.py
│   ├── deeplearning/model.tar.gz              # Live LSTM model
│   ├── deeplearning/inference.py              # Canonical reference inference.py
│   ├── deeplearning/training_metrics.json     # Latest training benchmark MAE
│   ├── sentiment/model.tar.gz                 # FinBERT model
│   └── timeseries/model.tar.gz                # ARIMA/Prophet model
├── training-data/btc_hourly.parquet           # 2-year rolling training data
├── training-scripts/lstm_source.tar.gz        # LSTM training script (v2)
└── locks/post-train/{job_name}.lock           # Deduplication locks
```
