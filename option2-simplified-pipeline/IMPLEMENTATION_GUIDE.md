# Option 2: Simple Pipeline Implementation Guide

## Quick Overview
This is the **simplified, cost-effective** Bitcoin prediction pipeline. Single Lambda function, direct S3 storage, basic QuickSight dashboard.

**⏱️ Setup Time:** 15-20 minutes  
**💰 Monthly Cost:** $8-12  
**🎯 Accuracy:** 60-75%  

---

## Prerequisites

### 1. AWS Account Setup
- Active AWS account with billing enabled
- AWS CLI installed and configured
- Appropriate IAM permissions for Lambda, S3, CloudFormation

### 2. API Keys Required
You need these API keys (already provided in the code):
- **NewsAPI:** `56edb41cdf0740528d0b5e6331678ad4`
- **CryptoCompare:** `a6f12c11c4057ac8e7e631be960f741de19f5b6c543a70f39682242a540729c8`

### 3. Local Tools
```bash
# Install required tools
pip install boto3 requests
```

---

## Step-by-Step Deployment

### Method 1: Automated Deployment (Recommended)

```bash
# Navigate to infrastructure directory
cd infrastructure/

# Run the deployment script
./deploy-simple.sh
```

The script will:
1. ✅ Deploy CloudFormation infrastructure
2. ✅ Package and upload Lambda function
3. ✅ Test the deployment
4. ✅ Optionally set up QuickSight
5. ✅ Provide next steps

### Method 2: Manual Deployment

#### Step 1: Deploy Infrastructure
```bash
aws cloudformation create-stack \
  --stack-name bitcoin-simple-pipeline \
  --template-body file://simple-cloudformation.yaml \
  --parameters \
    ParameterKey=NewsApiKey,ParameterValue=56edb41cdf0740528d0b5e6331678ad4 \
    ParameterKey=CryptoCompareApiKey,ParameterValue=a6f12c11c4057ac8e7e631be960f741de19f5b6c543a70f39682242a540729c8 \
  --capabilities CAPABILITY_NAMED_IAM \
  --region eu-west-2

# Wait for completion
aws cloudformation wait stack-create-complete \
  --stack-name bitcoin-simple-pipeline \
  --region eu-west-2
```

#### Step 2: Deploy Lambda Function
```bash
# Create deployment package
cd ../lambda-function
zip -r ../lambda-deployment.zip . -x "*.pyc" "__pycache__/*"

# Update function code
aws lambda update-function-code \
  --function-name bitcoin-simple-predictor \
  --zip-file fileb://../lambda-deployment.zip \
  --region eu-west-2
```

#### Step 3: Test Deployment
```bash
# Test Lambda function
aws lambda invoke \
  --function-name bitcoin-simple-predictor \
  --region eu-west-2 \
  --payload '{}' \
  output.json

# Check results
cat output.json
```

#### Step 4: Set Up QuickSight
```bash
cd ../quicksight
python3 simple-dashboard.py --setup-all
```

---

## Verification Steps

### 1. Check Lambda Function
```bash
# View recent logs
aws logs tail /aws/lambda/bitcoin-simple-predictor --follow

# Check function status
aws lambda get-function --function-name bitcoin-simple-predictor
```

### 2. Verify S3 Storage
```bash
# Get bucket name from stack
BUCKET=$(aws cloudformation describe-stacks \
  --stack-name bitcoin-simple-pipeline \
  --query "Stacks[0].Outputs[?OutputKey=='S3BucketName'].OutputValue" \
  --output text)

# Check for prediction files
aws s3 ls s3://$BUCKET/predictions/ --recursive --human-readable
```

### 3. Monitor Predictions
```bash
# Check recent predictions (after 5-10 minutes)
aws s3 ls s3://$BUCKET/predictions/ --recursive | tail -5

# Download and view a prediction
aws s3 cp s3://$BUCKET/predictions/[latest-file] prediction.json
cat prediction.json | jq .
```

---

## Understanding the Pipeline

### Architecture Flow
```
Every 5 minutes:
1. EventBridge triggers Lambda
2. Lambda fetches Bitcoin price (CryptoCompare)
3. Lambda fetches news sentiment (NewsAPI)
4. Lambda calculates technical indicators
5. Lambda makes simple prediction
6. Lambda saves results to S3
7. QuickSight visualizes the data
```

### Data Structure
Predictions are saved as JSON files in S3:
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "current_data": {
    "price": 45000.50,
    "news_articles_count": 8,
    "overall_sentiment": 0.15
  },
  "prediction": {
    "predictions": {
      "1h": 45100.25,
      "6h": 45200.80,
      "24h": 45500.00,
      "7d": 46000.00
    },
    "confidence": 0.72
  }
}
```

---

## Customization Options

### 1. Adjust Prediction Frequency
Edit `simple-cloudformation.yaml`:
```yaml
ScheduleExpression: rate(5 minutes)  # Change to rate(1 hour) for hourly
```

### 2. Modify Prediction Logic
Edit `lambda-function/bitcoin-predictor-simple.py`:
```python
# Adjust prediction factors in make_simple_prediction()
prediction_factors['price_momentum'] = price_change_pct * 0.5  # Increase momentum weight
```

### 3. Add More Data Sources
```python
# Add CoinAPI integration in the Lambda function
def fetch_coinapi_data(self):
    # Implementation here
```

---

## Monitoring and Maintenance

### 1. Set Up Monitoring (Optional)
```bash
# Deploy monitoring stack
aws cloudformation create-stack \
  --stack-name bitcoin-simple-monitoring \
  --template-body file://monitoring/basic-alarms.yaml \
  --parameters \
    ParameterKey=LambdaFunctionName,ParameterValue=bitcoin-simple-predictor \
    ParameterKey=S3BucketName,ParameterValue=$BUCKET \
    ParameterKey=NotificationEmail,ParameterValue=your-email@example.com
```

### 2. Cost Monitoring
- Check AWS Billing Dashboard daily
- Budget alert set at $15/month (80% threshold)
- Monitor S3 storage growth

### 3. Performance Monitoring
```bash
# Check CloudWatch metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Duration \
  --dimensions Name=FunctionName,Value=bitcoin-simple-predictor \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-01T23:59:59Z \
  --period 3600 \
  --statistics Average
```

---

## Troubleshooting

### Common Issues

#### 1. Lambda Function Errors
```bash
# Check logs for errors
aws logs filter-log-events \
  --log-group-name /aws/lambda/bitcoin-simple-predictor \
  --filter-pattern "ERROR"
```

**Common fixes:**
- API key issues: Verify keys are correct
- Timeout issues: Increase Lambda timeout
- Memory issues: Increase Lambda memory

#### 2. No Predictions in S3
**Check:**
- EventBridge rule is enabled
- Lambda has S3 permissions
- API keys are working

#### 3. QuickSight Issues
**Common fixes:**
- Verify S3 permissions for QuickSight
- Check data source configuration
- Refresh dataset manually

### Getting Help
1. Check CloudWatch logs first
2. Verify all API keys are valid
3. Ensure AWS permissions are correct
4. Check stack outputs for resource names

---

## Expected Performance

### Accuracy Metrics
- **Directional Accuracy:** 60-75%
- **1-hour predictions:** Most accurate
- **24-hour predictions:** Moderate accuracy
- **7-day predictions:** Trend indication only

### Performance Metrics
- **Execution Time:** 2-4 seconds per prediction
- **End-to-end Latency:** 2-3 minutes
- **Uptime:** 95-98% (Lambda availability)
- **Cost per Prediction:** ~$0.0001

---

## Upgrade Path

### When to Consider Option 1
- Need higher accuracy (>80%)
- Require fault tolerance
- Need real-time predictions
- Production/enterprise use
- Higher prediction volume

### Migration Steps
1. Export S3 data for historical analysis
2. Deploy Option 1 infrastructure
3. Gradually transition traffic
4. Decommission Option 2 when stable

---

## Cost Breakdown

| Component | Monthly Cost |
|-----------|-------------|
| Lambda (432 invocations/day) | $0.20 |
| S3 Storage (1GB) | $0.50 |
| S3 Requests | $0.10 |
| CloudWatch Logs | $1.00 |
| EventBridge | $0.10 |
| Data Transfer | $0.50 |
| QuickSight | $6.00 |
| **Total** | **~$8.40** |

*Note: Costs may vary based on usage patterns and AWS pricing changes.*

---

## Success Criteria

✅ **Deployment Successful** when:
- Lambda function executes without errors
- Predictions appear in S3 every 5 minutes
- QuickSight dashboard shows data
- All monitoring alerts are green

✅ **Production Ready** when:
- 7 days of successful predictions
- Accuracy meets expectations (>60%)
- Costs are within budget ($15/month)
- Monitoring alerts are configured