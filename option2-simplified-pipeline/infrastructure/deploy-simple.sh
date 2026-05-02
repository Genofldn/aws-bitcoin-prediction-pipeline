#!/bin/bash

# Simple Bitcoin Prediction Pipeline Deployment Script
# This script deploys the complete simplified pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
STACK_NAME="bitcoin-simple-pipeline"
REGION="eu-west-2"
LAMBDA_FUNCTION_NAME="bitcoin-simple-predictor"

# API Keys (set these before running)
NEWSAPI_KEY="${NEWSAPI_KEY:?Set NEWSAPI_KEY environment variable}"
CRYPTOCOMPARE_KEY="${CRYPTOCOMPARE_KEY:?Set CRYPTOCOMPARE_KEY environment variable}"

echo -e "${BLUE}=== Simple Bitcoin Pipeline Deployment ===${NC}"
echo "Stack Name: $STACK_NAME"
echo "Region: $REGION"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

# Check if AWS CLI is installed and configured
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    print_error "AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
print_status "AWS Account ID: $ACCOUNT_ID"

# Check if jq is available
if ! command -v jq &> /dev/null; then
    print_warning "jq not found. JSON parsing will be limited."
fi

# Validate API keys
if [[ -z "$NEWSAPI_KEY" || "$NEWSAPI_KEY" == "your-newsapi-key" ]]; then
    print_error "Please set NEWSAPI_KEY in this script"
    exit 1
fi

if [[ -z "$CRYPTOCOMPARE_KEY" || "$CRYPTOCOMPARE_KEY" == "your-cryptocompare-key" ]]; then
    print_error "Please set CRYPTOCOMPARE_KEY in this script"
    exit 1
fi

print_status "API keys validated"

# Step 1: Deploy CloudFormation stack
echo -e "\n${BLUE}Step 1: Deploying infrastructure...${NC}"

if aws cloudformation describe-stacks --stack-name $STACK_NAME --region $REGION &> /dev/null; then
    print_warning "Stack exists. Updating..."
    aws cloudformation update-stack \
        --stack-name $STACK_NAME \
        --template-body file://simple-cloudformation.yaml \
        --parameters \
            ParameterKey=NewsApiKey,ParameterValue=$NEWSAPI_KEY \
            ParameterKey=CryptoCompareApiKey,ParameterValue=$CRYPTOCOMPARE_KEY \
        --capabilities CAPABILITY_NAMED_IAM \
        --region $REGION
    
    aws cloudformation wait stack-update-complete --stack-name $STACK_NAME --region $REGION
else
    print_status "Creating new stack..."
    aws cloudformation create-stack \
        --stack-name $STACK_NAME \
        --template-body file://simple-cloudformation.yaml \
        --parameters \
            ParameterKey=NewsApiKey,ParameterValue=$NEWSAPI_KEY \
            ParameterKey=CryptoCompareApiKey,ParameterValue=$CRYPTOCOMPARE_KEY \
        --capabilities CAPABILITY_NAMED_IAM \
        --region $REGION
    
    aws cloudformation wait stack-create-complete --stack-name $STACK_NAME --region $REGION
fi

print_status "Infrastructure deployed"

# Step 2: Package and deploy Lambda function
echo -e "\n${BLUE}Step 2: Deploying Lambda function...${NC}"

# Create deployment package
TEMP_DIR=$(mktemp -d)
LAMBDA_DIR="../lambda-function"

print_status "Creating deployment package..."

# Copy Lambda files
cp $LAMBDA_DIR/*.py $TEMP_DIR/
cp $LAMBDA_DIR/requirements.txt $TEMP_DIR/

# Install dependencies
cd $TEMP_DIR
if [ -f requirements.txt ]; then
    pip install -r requirements.txt -t .
fi

# Create zip package
zip -r lambda-deployment.zip . -x "*.pyc" "__pycache__/*" "*.git*"

print_status "Lambda package created"

# Update Lambda function code
aws lambda update-function-code \
    --function-name $LAMBDA_FUNCTION_NAME \
    --zip-file fileb://lambda-deployment.zip \
    --region $REGION

print_status "Lambda function updated"

# Cleanup
rm -rf $TEMP_DIR

# Step 3: Get stack outputs
echo -e "\n${BLUE}Step 3: Getting deployment information...${NC}"

S3_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --region $REGION \
    --query "Stacks[0].Outputs[?OutputKey=='S3BucketName'].OutputValue" \
    --output text)

LAMBDA_ARN=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --region $REGION \
    --query "Stacks[0].Outputs[?OutputKey=='LambdaFunctionArn'].OutputValue" \
    --output text)

print_status "S3 Bucket: $S3_BUCKET"
print_status "Lambda ARN: $LAMBDA_ARN"

# Step 4: Test Lambda function
echo -e "\n${BLUE}Step 4: Testing Lambda function...${NC}"

TEST_RESULT=$(aws lambda invoke \
    --function-name $LAMBDA_FUNCTION_NAME \
    --region $REGION \
    --payload '{}' \
    /tmp/lambda-test-output.json)

if [ $? -eq 0 ]; then
    print_status "Lambda function test successful"
    if command -v jq &> /dev/null; then
        echo "Response preview:"
        cat /tmp/lambda-test-output.json | jq '.body | fromjson | {success, current_price, processing_time}'
    fi
else
    print_warning "Lambda test had issues - check CloudWatch logs"
fi

# Step 5: Set up QuickSight (optional)
echo -e "\n${BLUE}Step 5: QuickSight setup (optional)...${NC}"

read -p "Do you want to set up QuickSight dashboard now? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Setting up QuickSight..."
    cd ../quicksight
    python3 simple-dashboard.py --region $REGION --account-id $ACCOUNT_ID --setup-all
    print_status "QuickSight setup attempted"
else
    print_warning "QuickSight setup skipped. Run manually later with:"
    echo "cd quicksight && python3 simple-dashboard.py --setup-all"
fi

# Step 6: Display final information
echo -e "\n${GREEN}=== Deployment Complete ===${NC}"
echo ""
echo "📊 Stack Name: $STACK_NAME"
echo "🪣 S3 Bucket: $S3_BUCKET"
echo "⚡ Lambda Function: $LAMBDA_FUNCTION_NAME"
echo "🌍 Region: $REGION"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Wait 5-10 minutes for first predictions to appear in S3"
echo "2. Check CloudWatch logs: /aws/lambda/$LAMBDA_FUNCTION_NAME"
echo "3. Monitor S3 bucket for prediction files"
echo "4. Set up QuickSight if not done above"
echo ""
echo -e "${BLUE}Useful Commands:${NC}"
echo "# Check recent predictions:"
echo "aws s3 ls s3://$S3_BUCKET/predictions/ --recursive --human-readable | tail -10"
echo ""
echo "# View Lambda logs:"
echo "aws logs tail /aws/lambda/$LAMBDA_FUNCTION_NAME --follow"
echo ""
echo "# Invoke function manually:"
echo "aws lambda invoke --function-name $LAMBDA_FUNCTION_NAME output.json"
echo ""
echo -e "${BLUE}Cost Monitoring:${NC}"
echo "- Basic budget alert set at \$15/month"
echo "- Monitor AWS Billing Dashboard"
echo "- Check CloudWatch metrics for usage"
echo ""
echo -e "${GREEN}✅ Simple Bitcoin Pipeline is ready!${NC}"

# Return to original directory
cd - > /dev/null

exit 0