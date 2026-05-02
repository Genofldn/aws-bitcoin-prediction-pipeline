#!/bin/bash

# Bitcoin Prediction Pipeline Deployment Script
# This script deploys the complete Option 1 (Robust Production Pipeline)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REGION=${AWS_REGION:-eu-west-2}
STACK_NAME_IAM="bitcoin-prediction-iam"
STACK_NAME_INFRA="bitcoin-prediction-infrastructure"
STACK_NAME_ALARMS="bitcoin-prediction-alarms"
NEWSAPI_KEY="${NEWSAPI_KEY:?Set NEWSAPI_KEY environment variable}"
CRYPTOCOMPARE_KEY="${CRYPTOCOMPARE_KEY:?Set CRYPTOCOMPARE_KEY environment variable}"

echo -e "${BLUE}=== Bitcoin Prediction Pipeline Deployment ===${NC}"
echo "Region: $REGION"
echo "IAM Stack: $STACK_NAME_IAM"
echo "Infrastructure Stack: $STACK_NAME_INFRA"
echo ""

# Function to print colored output
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if AWS CLI is configured
check_aws_cli() {
    log_info "Checking AWS CLI configuration..."
    
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS CLI is not configured. Please run 'aws configure' first."
        exit 1
    fi
    
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    log_success "AWS CLI configured. Account ID: $ACCOUNT_ID"
}

# Function to validate CloudFormation template
validate_template() {
    local template_file=$1
    log_info "Validating CloudFormation template: $template_file"
    
    if aws cloudformation validate-template --template-body file://$template_file &> /dev/null; then
        log_success "Template validation passed: $template_file"
    else
        log_error "Template validation failed: $template_file"
        exit 1
    fi
}

# Function to deploy CloudFormation stack
deploy_stack() {
    local stack_name=$1
    local template_file=$2
    local parameters=$3
    
    log_info "Deploying stack: $stack_name"
    
    # Build the deploy command
    local deploy_cmd="aws cloudformation deploy --template-file $template_file --stack-name $stack_name --region $REGION"
    
    if [ ! -z "$parameters" ]; then
        deploy_cmd="$deploy_cmd --parameter-overrides $parameters"
    fi
    
    # Add capabilities for IAM resources
    if [[ $template_file == *"iam"* ]] || [[ $stack_name == *"iam"* ]]; then
        deploy_cmd="$deploy_cmd --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM"
    fi
    
    # Execute deployment
    if eval $deploy_cmd; then
        log_success "Stack deployed successfully: $stack_name"
        return 0
    else
        log_error "Stack deployment failed: $stack_name"
        return 1
    fi
}

# Function to wait for stack completion
wait_for_stack() {
    local stack_name=$1
    log_info "Waiting for stack to complete: $stack_name"
    
    aws cloudformation wait stack-deploy-complete --stack-name $stack_name --region $REGION
    
    if [ $? -eq 0 ]; then
        log_success "Stack completed successfully: $stack_name"
    else
        log_error "Stack deployment timed out or failed: $stack_name"
        exit 1
    fi
}

# Function to package and deploy Lambda functions
deploy_lambda_functions() {
    log_info "Packaging and deploying Lambda functions..."
    
    # Create temporary directory for packaging
    local temp_dir=$(mktemp -d)
    local script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    local lambda_dir="$script_dir/../lambda-functions"
    
    # Data ingestion functions
    local ingestion_functions=("newsapi-collector" "cryptocompare-collector" "coinapi-collector")
    
    for func in "${ingestion_functions[@]}"; do
        log_info "Packaging $func..."
        
        local func_dir="$temp_dir/$func"
        mkdir -p "$func_dir"
        
        # Copy function code
        cp "$lambda_dir/data-ingestion/$func.py" "$func_dir/"
        cp "$lambda_dir/data-ingestion/requirements.txt" "$func_dir/"
        
        # Copy shared utilities
        mkdir -p "$func_dir/shared"
        cp "$lambda_dir/shared/"*.py "$func_dir/shared/"
        
        # Create deployment package
        cd "$func_dir"
        pip install -r requirements.txt -t .
        zip -r "../$func.zip" . -x "*.pyc" "__pycache__/*"
        
        # Deploy function
        log_info "Updating Lambda function: bitcoin-$func"
        aws lambda update-function-code \
            --function-name "bitcoin-$func" \
            --zip-file "fileb://$temp_dir/$func.zip" \
            --region $REGION
        
        if [ $? -eq 0 ]; then
            log_success "Updated Lambda function: bitcoin-$func"
        else
            log_warning "Failed to update Lambda function: bitcoin-$func (may not exist yet)"
        fi
    done
    
    # Data processing functions
    local processing_functions=("sentiment-analyzer" "feature-engineer" "predictor")
    
    for func in "${processing_functions[@]}"; do
        log_info "Packaging $func..."
        
        local func_dir="$temp_dir/$func"
        mkdir -p "$func_dir"
        
        # Copy function code
        cp "$lambda_dir/data-processor/$func.py" "$func_dir/"
        cp "$lambda_dir/data-processor/requirements.txt" "$func_dir/"
        
        # Copy shared utilities
        mkdir -p "$func_dir/shared"
        cp "$lambda_dir/shared/"*.py "$func_dir/shared/"
        
        # Copy model files for predictor
        if [ "$func" == "predictor" ]; then
            mkdir -p "$func_dir/model"
            cp "$lambda_dir/../model/"*.py "$func_dir/model/" 2>/dev/null || true
        fi
        
        # Create deployment package
        cd "$func_dir"
        pip install -r requirements.txt -t .
        zip -r "../$func.zip" . -x "*.pyc" "__pycache__/*"
        
        # Deploy function
        log_info "Updating Lambda function: bitcoin-$func"
        aws lambda update-function-code \
            --function-name "bitcoin-$func" \
            --zip-file "fileb://$temp_dir/$func.zip" \
            --region $REGION
        
        if [ $? -eq 0 ]; then
            log_success "Updated Lambda function: bitcoin-$func"
        else
            log_warning "Failed to update Lambda function: bitcoin-$func (may not exist yet)"
        fi
    done
    
    # Cleanup
    rm -rf "$temp_dir"
    cd "$script_dir"
}

# Function to setup monitoring
setup_monitoring() {
    log_info "Setting up CloudWatch monitoring..."
    
    # Deploy CloudWatch dashboard
    local dashboard_file="../monitoring/cloudwatch-dashboards.json"
    if [ -f "$dashboard_file" ]; then
        aws cloudwatch put-dashboard \
            --dashboard-name "Bitcoin-Prediction-Pipeline" \
            --dashboard-body file://$dashboard_file \
            --region $REGION
        
        if [ $? -eq 0 ]; then
            log_success "CloudWatch dashboard created"
        else
            log_warning "Failed to create CloudWatch dashboard"
        fi
    fi
    
    # Deploy alarms
    log_info "Deploying CloudWatch alarms..."
    validate_template "../monitoring/alarms.yaml"
    
    local alerts_topic_arn="arn:aws:sns:$REGION:$ACCOUNT_ID:bitcoin-prediction-alerts"
    
    if deploy_stack "$STACK_NAME_ALARMS" "../monitoring/alarms.yaml" "AlertsTopicArn=$alerts_topic_arn"; then
        wait_for_stack "$STACK_NAME_ALARMS"
    else
        log_warning "Failed to deploy alarms stack"
    fi
}

# Function to setup QuickSight
setup_quicksight() {
    log_info "Setting up QuickSight integration..."
    
    local quicksight_script="../quicksight/setup-quicksight.py"
    if [ -f "$quicksight_script" ]; then
        python3 "$quicksight_script" --region $REGION --setup-all
        
        if [ $? -eq 0 ]; then
            log_success "QuickSight setup completed"
        else
            log_warning "QuickSight setup failed or requires manual configuration"
        fi
    else
        log_warning "QuickSight setup script not found"
    fi
}

# Function to run health check
run_health_check() {
    log_info "Running system health check..."
    
    local health_script="../monitoring/health-check.py"
    if [ -f "$health_script" ]; then
        python3 "$health_script" --region $REGION --full-check
        
        if [ $? -eq 0 ]; then
            log_success "Health check passed"
        elif [ $? -eq 2 ]; then
            log_warning "Health check completed with warnings"
        else
            log_error "Health check failed"
        fi
    else
        log_warning "Health check script not found"
    fi
}

# Function to create CoinAPI parameter if needed
setup_coinapi_parameter() {
    log_info "Setting up CoinAPI parameter..."
    
    # Check if parameter exists
    if aws ssm get-parameter --name "/bitcoin/coinapi/key" --region $REGION &>/dev/null; then
        log_info "CoinAPI parameter already exists"
    else
        log_warning "CoinAPI parameter not found. You can add it later with:"
        echo "aws ssm put-parameter --name '/bitcoin/coinapi/key' --value 'YOUR_COINAPI_KEY' --type 'SecureString' --region $REGION"
    fi
}

# Main deployment function
main() {
    local script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    cd "$script_dir"
    
    echo -e "${BLUE}Starting deployment process...${NC}"
    
    # Pre-deployment checks
    check_aws_cli
    
    # Validate all templates
    log_info "Validating CloudFormation templates..."
    validate_template "iam-roles.yaml"
    validate_template "cloudformation-template.yaml"
    
    # Deploy IAM roles first
    log_info "=== Phase 1: Deploying IAM Roles ==="
    if deploy_stack "$STACK_NAME_IAM" "iam-roles.yaml"; then
        wait_for_stack "$STACK_NAME_IAM"
    else
        log_error "Failed to deploy IAM stack. Exiting."
        exit 1
    fi
    
    # Deploy main infrastructure
    log_info "=== Phase 2: Deploying Infrastructure ==="
    local parameters="NewsApiKey=$NEWSAPI_KEY CryptoCompareApiKey=$CRYPTOCOMPARE_KEY"
    
    if deploy_stack "$STACK_NAME_INFRA" "cloudformation-template.yaml" "$parameters"; then
        wait_for_stack "$STACK_NAME_INFRA"
    else
        log_error "Failed to deploy infrastructure stack. Exiting."
        exit 1
    fi
    
    # Deploy Lambda functions
    log_info "=== Phase 3: Deploying Lambda Functions ==="
    deploy_lambda_functions
    
    # Setup monitoring
    log_info "=== Phase 4: Setting up Monitoring ==="
    setup_monitoring
    
    # Setup additional parameters
    log_info "=== Phase 5: Setting up Additional Parameters ==="
    setup_coinapi_parameter
    
    # Setup QuickSight
    log_info "=== Phase 6: Setting up QuickSight ==="
    setup_quicksight
    
    # Final health check
    log_info "=== Phase 7: Running Health Check ==="
    run_health_check
    
    # Success message
    echo ""
    log_success "=== DEPLOYMENT COMPLETED SUCCESSFULLY ==="
    echo ""
    echo -e "${GREEN}Your Bitcoin Prediction Pipeline is now deployed!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Check the CloudWatch dashboard: https://$REGION.console.aws.amazon.com/cloudwatch/home?region=$REGION#dashboards:name=Bitcoin-Prediction-Pipeline"
    echo "2. Monitor the system with: python3 ../monitoring/health-check.py --region $REGION"
    echo "3. Access QuickSight dashboard (if configured)"
    echo "4. Add CoinAPI key if needed: aws ssm put-parameter --name '/bitcoin/coinapi/key' --value 'YOUR_KEY' --type 'SecureString'"
    echo ""
    echo -e "${YELLOW}Note: The system will start collecting data automatically. First predictions will be available within 1-2 hours.${NC}"
}

# Handle script arguments
case "${1:-}" in
    --validate-only)
        log_info "Validation mode: checking templates only"
        check_aws_cli
        validate_template "iam-roles.yaml"
        validate_template "cloudformation-template.yaml"
        log_success "All templates are valid"
        ;;
    --lambda-only)
        log_info "Lambda deployment mode: updating functions only"
        check_aws_cli
        deploy_lambda_functions
        ;;
    --monitoring-only)
        log_info "Monitoring setup mode"
        check_aws_cli
        setup_monitoring
        ;;
    --health-check)
        log_info "Health check mode"
        run_health_check
        ;;
    --help)
        echo "Bitcoin Prediction Pipeline Deployment Script"
        echo ""
        echo "Usage: $0 [option]"
        echo ""
        echo "Options:"
        echo "  (no option)        Full deployment"
        echo "  --validate-only    Validate CloudFormation templates only"
        echo "  --lambda-only      Deploy Lambda functions only"
        echo "  --monitoring-only  Setup monitoring only"
        echo "  --health-check     Run health check only"
        echo "  --help            Show this help message"
        echo ""
        ;;
    *)
        main "$@"
        ;;
esac