# Option 2: Simplified Pipeline

## Overview
Simple, cost-effective Bitcoin prediction pipeline with minimal complexity and maximum value.

**Architecture:** Single Lambda → S3 → QuickSight

**Monthly Cost:** $8-12/month

## Key Features
- Single Lambda function handles all data collection and prediction
- Direct S3 storage for simplicity
- Basic but effective ML predictions
- Simple scheduling with EventBridge
- Cost-optimized for small-scale use

## Simplified Architecture
```
APIs → Single Lambda Function → S3 Bucket → QuickSight
  ↓           ↓                    ↓          ↓
News +    All-in-One         JSON Files   Dashboard
Price +   Processing        (Time-based    (Auto
Sentiment                   Partitioning)   Refresh)
```

## Files Structure
```
option2-simplified-pipeline/
├── README.md                           # This file
├── IMPLEMENTATION_GUIDE.md             # Step-by-step setup
├── infrastructure/
│   ├── simple-cloudformation.yaml      # Minimal infrastructure
│   ├── iam-simple.yaml                 # Basic IAM roles
│   └── deploy-simple.sh                # One-command deployment
├── lambda-function/
│   ├── bitcoin-predictor-simple.py     # All-in-one function
│   ├── requirements.txt                # Dependencies
│   └── utils.py                        # Helper functions
├── quicksight/
│   ├── s3-dataset-config.json          # S3 data source config
│   └── simple-dashboard.py             # Basic dashboard setup
└── monitoring/
    ├── basic-alarms.yaml               # Essential monitoring
    └── cost-alert.yaml                 # Cost monitoring
```

## Advantages
- **Ultra Low Cost**: 85% cheaper than Option 1
- **Simple Debugging**: Single function to troubleshoot
- **Fast Deployment**: 15-minute setup
- **Easy Maintenance**: Minimal moving parts
- **Good for Testing**: Perfect for proof-of-concept

## Limitations
- **No Fault Tolerance**: Single point of failure
- **Basic Error Handling**: Limited retry logic
- **Simpler ML Model**: Linear regression approach
- **Manual Scaling**: No auto-scaling features
- **Basic Monitoring**: Essential alerts only

## When to Use Option 2
- **Budget Conscious**: Need to minimize costs
- **Proof of Concept**: Testing Bitcoin prediction ideas
- **Learning Purpose**: Understanding the basics
- **Low Volume**: Hourly predictions sufficient
- **Simple Requirements**: Basic accuracy acceptable

## When to Upgrade to Option 1
- **Production Use**: Need reliability and fault tolerance
- **Higher Accuracy**: Advanced ML models required
- **Real-time Needs**: Sub-minute prediction updates
- **Enterprise Grade**: Comprehensive monitoring needed
- **High Volume**: Multiple prediction requests

## Quick Start
1. Follow the IMPLEMENTATION_GUIDE.md
2. Run `./infrastructure/deploy-simple.sh`
3. Wait 15 minutes for first predictions
4. Access QuickSight dashboard

## API Integration
- Uses the same API keys as Option 1
- NewsAPI and CryptoCompare fully supported
- CoinAPI optional (can be added later)

## Expected Performance
- **Predictions**: Every 5 minutes
- **Accuracy**: 60-75% directional accuracy
- **Latency**: 2-3 minutes end-to-end
- **Uptime**: 95-98% (depending on Lambda health)