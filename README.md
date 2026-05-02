# AWS Bitcoin Prediction Pipeline

A production-grade, serverless ML pipeline on AWS that collects real-time market data, performs sentiment analysis, engineers features, and generates multi-horizon Bitcoin price predictions.

## Architecture

```
NewsAPI / CryptoCompare / CoinAPI
            │
            ▼
    EventBridge (scheduled)
            │
            ▼
  Lambda Data Collectors ──► SQS Queue ──► Dead Letter Queue
            │
            ▼
  Lambda Sentiment Analyzer + Feature Engineer
            │
            ▼
       DynamoDB (raw data + feature vectors)
            │
            ▼
  Lambda Predictor (ML model from S3) ──► DynamoDB (predictions)
            │
            ▼
  QuickSight Dashboard + CloudWatch Monitoring
```

## Key Features

- **Serverless-first** — Lambda, SQS, DynamoDB, EventBridge; zero idle infrastructure cost
- **Resilient by design** — circuit breaker pattern, exponential backoff with jitter, dead letter queues for zero data loss
- **ML-powered predictions** — ensemble model (Random Forest, Gradient Boosting, Ridge Regression) with 1h, 6h, 24h, 7-day forecasts and confidence scoring
- **Full observability** — CloudWatch custom metrics, 15+ alarms, composite health checks, QuickSight dashboards
- **Security** — least-privilege IAM roles, encrypted Parameter Store for API keys, S3 encryption at rest, VPC endpoints
- **Infrastructure as Code** — complete CloudFormation templates, one-command deployment

## Project Structure

```
├── option1-robust-pipeline/          # Production-grade pipeline (recommended)
│   ├── infrastructure/
│   │   ├── cloudformation-template.yaml   # Full AWS infrastructure definition
│   │   ├── iam-roles.yaml                 # Least-privilege IAM roles
│   │   └── deploy.sh                      # One-command deployment script
│   ├── lambda-functions/
│   │   ├── data-ingestion/                # NewsAPI, CryptoCompare, CoinAPI collectors
│   │   ├── data-processor/                # Sentiment analyzer, feature engineer, predictor
│   │   └── shared/                        # Circuit breaker, retry handler, utilities
│   ├── model/
│   │   └── training.py                    # Multi-model training pipeline
│   ├── monitoring/
│   │   ├── cloudwatch-dashboards.json     # 12 dashboard visualisations
│   │   ├── alarms.yaml                    # 15+ CloudWatch alarms
│   │   └── health-check.py               # Automated system diagnostics
│   └── quicksight/
│       └── setup-quicksight.py            # Automated dashboard provisioning
│
├── option2-simplified-pipeline/      # Lower cost variant (~$10/month)
└── option3-realtime-architecture/    # High-frequency real-time variant (Kinesis + SageMaker)
```

## Infrastructure Components

| Component | Service | Purpose |
|---|---|---|
| Data collection | AWS Lambda + EventBridge | Scheduled every 5 minutes |
| Message queue | SQS + DLQ | Decoupled processing, no data loss |
| Storage | DynamoDB (on-demand) | Raw data + predictions with TTL |
| ML models | S3 + Lambda | Versioned model storage and inference |
| Monitoring | CloudWatch | Custom metrics, alarms, dashboards |
| Visualisation | QuickSight | Real-time prediction dashboards |
| Secrets | SSM Parameter Store | Encrypted API key management |
| IaC | CloudFormation | Repeatable, version-controlled infrastructure |

## Deployment

### Prerequisites
- AWS CLI configured (`aws configure`)
- Python 3.9+
- Appropriate IAM permissions

### Quick Start

```bash
# Set your API keys as environment variables (never hardcode secrets)
export NEWSAPI_KEY="your-newsapi-key"
export CRYPTOCOMPARE_KEY="your-cryptocompare-key"

# Deploy infrastructure
cd option1-robust-pipeline/infrastructure
./deploy.sh
```

### Post-Deployment
- Data collection begins immediately
- First predictions available within 1–2 hours
- CloudWatch dashboards auto-populated
- QuickSight dashboard URL printed on completion

## Resilience Patterns

### Circuit Breaker
Prevents cascade failures when upstream APIs are unavailable. Each data source has an independent circuit that opens after repeated failures and resets automatically.

### Dead Letter Queues
Every SQS queue has a DLQ configured with 14-day retention. No messages are lost during processing failures — they are reprocessed once the issue is resolved.

### Fallback Model
If the trained ML model cannot be loaded from S3, the predictor falls back to a linear model, ensuring predictions continue to be served even during model update failures.

## Monitoring

- **15+ CloudWatch alarms** covering Lambda error rates, DLQ message depth, prediction latency, and data collection failures
- **Custom metrics namespace** `Bitcoin/Prediction` for prediction confidence scores, processing duration, and storage success rates
- **Composite health check script** — run `python3 monitoring/health-check.py` for a full system diagnostic

## Cost

| Variant | Estimated Monthly Cost |
|---|---|
| Option 1 (Production) | $70–112/month |
| Option 2 (Simplified) | $8–12/month |
| Option 3 (Real-time) | $594–750/month |

## Tech Stack

`Python` `AWS Lambda` `Amazon SQS` `Amazon DynamoDB` `Amazon S3` `AWS CloudFormation` `Amazon CloudWatch` `Amazon QuickSight` `AWS EventBridge` `Amazon Kinesis` `Amazon SageMaker` `Docker` `Terraform`
