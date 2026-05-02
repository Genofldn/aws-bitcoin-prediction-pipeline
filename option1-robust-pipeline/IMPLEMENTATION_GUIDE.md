# Option 1: Robust Production Pipeline - Implementation Guide

## Prerequisites
- AWS CLI configured with administrative access
- Python 3.9+ installed
- AWS account with billing enabled

## Step-by-Step Implementation

### Phase 1: Infrastructure Setup (30 minutes)

#### 1.1 Deploy IAM Roles
```bash
cd infrastructure/
aws cloudformation deploy \
  --template-file iam-roles.yaml \
  --stack-name bitcoin-prediction-iam \
  --capabilities CAPABILITY_IAM
```

#### 1.2 Deploy Core Infrastructure
```bash
aws cloudformation deploy \
  --template-file cloudformation-template.yaml \
  --stack-name bitcoin-prediction-infrastructure \
  --parameter-overrides \
    NewsApiKey=56edb41cdf0740528d0b5e6331678ad4 \
    CryptoCompareApiKey=a6f12c11c4057ac8e7e631be960f741de19f5b6c543a70f39682242a540729c8
```

#### 1.3 Verify Infrastructure
```bash
# Check SQS queues
aws sqs list-queues --query 'QueueUrls[?contains(@, `bitcoin`)]'

# Check DynamoDB table
aws dynamodb describe-table --table-name bitcoin-predictions

# Check Lambda functions
aws lambda list-functions --query 'Functions[?contains(FunctionName, `bitcoin`)]'
```

### Phase 2: Deploy Lambda Functions (45 minutes)

#### 2.1 Package and Deploy Data Ingestion Functions
```bash
cd lambda-functions/data-ingestion/

# NewsAPI Collector
zip -r newsapi-collector.zip newsapi-collector.py requirements.txt ../shared/
aws lambda update-function-code \
  --function-name bitcoin-newsapi-collector \
  --zip-file fileb://newsapi-collector.zip

# CryptoCompare Collector
zip -r cryptocompare-collector.zip cryptocompare-collector.py requirements.txt ../shared/
aws lambda update-function-code \
  --function-name bitcoin-cryptocompare-collector \
  --zip-file fileb://cryptocompare-collector.zip

# CoinAPI Collector
zip -r coinapi-collector.zip coinapi-collector.py requirements.txt ../shared/
aws lambda update-function-code \
  --function-name bitcoin-coinapi-collector \
  --zip-file fileb://coinapi-collector.zip
```

#### 2.2 Deploy Data Processing Functions
```bash
cd ../data-processor/

# Sentiment Analyzer
zip -r sentiment-analyzer.zip sentiment-analyzer.py requirements.txt ../shared/
aws lambda update-function-code \
  --function-name bitcoin-sentiment-analyzer \
  --zip-file fileb://sentiment-analyzer.zip

# Feature Engineer
zip -r feature-engineer.zip feature-engineer.py requirements.txt ../shared/
aws lambda update-function-code \
  --function-name bitcoin-feature-engineer \
  --zip-file fileb://feature-engineer.zip

# Predictor
zip -r predictor.zip predictor.py requirements.txt ../shared/ ../../model/
aws lambda update-function-code \
  --function-name bitcoin-predictor \
  --zip-file fileb://predictor.zip
```

#### 2.3 Test Lambda Functions
```bash
# Test NewsAPI collector
aws lambda invoke \
  --function-name bitcoin-newsapi-collector \
  --payload '{}' \
  response.json && cat response.json

# Test end-to-end flow
aws lambda invoke \
  --function-name bitcoin-cryptocompare-collector \
  --payload '{}' \
  response.json && cat response.json
```

### Phase 3: Configure Monitoring (20 minutes)

#### 3.1 Set up CloudWatch Dashboards
```bash
cd ../../monitoring/
aws cloudwatch put-dashboard \
  --dashboard-name "Bitcoin-Prediction-Pipeline" \
  --dashboard-body file://cloudwatch-dashboards.json
```

#### 3.2 Configure Alarms
```bash
aws cloudformation deploy \
  --template-file alarms.yaml \
  --stack-name bitcoin-prediction-alarms
```

#### 3.3 Test Health Check
```bash
python3 health-check.py
```

### Phase 4: Set up QuickSight (25 minutes)

#### 4.1 Enable QuickSight
```bash
# Enable QuickSight in your AWS account (manual step in console)
# Navigate to QuickSight service and sign up for Standard edition
```

#### 4.2 Configure Data Source
```bash
cd ../../quicksight/
python3 setup-quicksight.py --configure-datasource
```

#### 4.3 Create Dashboard
```bash
python3 setup-quicksight.py --create-dashboard
```

### Phase 5: Production Validation (15 minutes)

#### 5.1 End-to-End Test
```bash
# Trigger data collection manually
aws events put-events \
  --entries Source=bitcoin.prediction,DetailType="Manual Test",Detail='{}'

# Wait 5 minutes, then check DynamoDB
aws dynamodb scan \
  --table-name bitcoin-predictions \
  --limit 5 \
  --query 'Items[?attribute_exists(prediction_time)]'
```

#### 5.2 Verify QuickSight Updates
```bash
# Check QuickSight refresh status
python3 quicksight/setup-quicksight.py --check-refresh-status
```

#### 5.3 Monitor System Health
```bash
python3 monitoring/health-check.py --full-check
```

## Configuration

### API Keys Storage
All API keys are stored in AWS Systems Manager Parameter Store:
- `/bitcoin/newsapi/key`: NewsAPI key
- `/bitcoin/cryptocompare/key`: CryptoCompare key
- `/bitcoin/coinapi/key`: CoinAPI key (you'll need to add this)

### Scheduling
- Data collection: Every 5 minutes
- Sentiment analysis: Every 15 minutes  
- Predictions: Every hour
- Model retraining: Daily at 2 AM UTC

## Troubleshooting

### Common Issues

#### 1. Lambda Function Timeouts
```bash
# Check function configuration
aws lambda get-function-configuration --function-name bitcoin-newsapi-collector

# Increase timeout if needed
aws lambda update-function-configuration \
  --function-name bitcoin-newsapi-collector \
  --timeout 300
```

#### 2. SQS Message Failures
```bash
# Check dead letter queue
aws sqs receive-message \
  --queue-url $(aws sqs get-queue-url --queue-name bitcoin-data-dlq --query 'QueueUrl' --output text)

# Redrive messages from DLQ
aws sqs redrive-messages \
  --source-queue-url $(aws sqs get-queue-url --queue-name bitcoin-data-dlq --query 'QueueUrl' --output text) \
  --destination-queue-url $(aws sqs get-queue-url --queue-name bitcoin-data-processing --query 'QueueUrl' --output text)
```

#### 3. DynamoDB Throttling
```bash
# Check CloudWatch metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/DynamoDB \
  --metric-name UserErrors \
  --dimensions Name=TableName,Value=bitcoin-predictions \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-01T23:59:59Z \
  --period 3600 \
  --statistics Sum
```

#### 4. API Rate Limiting
```bash
# Check circuit breaker status
python3 monitoring/health-check.py --check-circuit-breakers
```

## Maintenance

### Daily Tasks
- Review CloudWatch alarms
- Check system health dashboard
- Verify QuickSight data freshness

### Weekly Tasks  
- Review cost optimization opportunities
- Update model with new training data
- Performance analysis

### Monthly Tasks
- Security review and IAM policy updates
- Cost analysis and optimization
- Model performance evaluation

## Security Notes
- All API keys are encrypted in Parameter Store
- Lambda functions have minimal IAM permissions
- VPC endpoints used for internal communication
- CloudTrail enabled for audit logging

## Support
For issues or questions:
1. Check CloudWatch logs first
2. Run health-check.py for diagnostics  
3. Review alarm history in CloudWatch
4. Check this implementation guide for troubleshooting steps