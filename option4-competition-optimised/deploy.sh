#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy.sh — Deploy the Option 4 CloudFormation stack
# Run from the option4-competition-optimised/ directory.
#
# Prerequisites:
#   1. All 4 Docker images pushed to ECR  (run build_and_push.sh first)
#   2. All 4 model.tar.gz files in S3     (run repackage_models.py first)
#   3. NewsAPI and CryptoCompare API keys ready
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REGION="eu-west-2"
STACK_NAME="bitcoin-prediction-option4"
TEMPLATE="infrastructure/main-cloudformation.yaml"
BUCKET="${DATA_BUCKET:-your-bucket-name}"  # pre-existing bucket
EMAIL="your-email@example.com"

echo "============================================================"
echo "Deploying Option 4 — Bitcoin Prediction Pipeline"
echo "Stack:  ${STACK_NAME}"
echo "Region: ${REGION}"
echo "============================================================"

# ── Prompt for API keys if not set ───────────────────────────────────────────
if [[ -z "${NEWS_API_KEY:-}" ]]; then
  read -rp "NewsAPI key (newsapi.org): " NEWS_API_KEY
fi
if [[ -z "${CRYPTOCOMPARE_API_KEY:-}" ]]; then
  read -rp "CryptoCompare API key: " CRYPTOCOMPARE_API_KEY
fi

# ── Upload template to S3 (avoids 51KB inline limit) ─────────────────────────
TEMPLATE_S3_KEY="cloudformation/option4-main.yaml"
echo ""
echo "▶ Uploading CloudFormation template to S3..."
aws s3 cp "${TEMPLATE}" "s3://${BUCKET}/${TEMPLATE_S3_KEY}" --region "${REGION}"
TEMPLATE_URL="https://${BUCKET}.s3.${REGION}.amazonaws.com/${TEMPLATE_S3_KEY}"

# ── Deploy / Update stack ─────────────────────────────────────────────────────
echo "▶ Deploying stack ${STACK_NAME} ..."
aws cloudformation deploy \
  --template-url "${TEMPLATE_URL}" \
  --stack-name   "${STACK_NAME}" \
  --region       "${REGION}" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
      Environment=production \
      DataBucketName=bitcoin-prediction-option4 \
      NotificationEmail="${EMAIL}" \
      NewsApiKey="${NEWS_API_KEY}" \
      CryptoCompareApiKey="${CRYPTOCOMPARE_API_KEY}" \
  --no-fail-on-empty-changeset

echo ""
echo "▶ Stack status:"
aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --region     "${REGION}" \
  --query "Stacks[0].{Status:StackStatus,Reason:StackStatusReason}" \
  --output table

echo ""
echo "▶ Key outputs:"
aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --region     "${REGION}" \
  --query "Stacks[0].Outputs" \
  --output table

echo ""
echo "============================================================"
echo "Deployment complete ✅"
echo ""
echo "Next steps:"
echo "  1. Check your email (${EMAIL}) and confirm the SNS topic subscription"
echo "  2. Test manually:"
echo "       aws lambda invoke --function-name bitcoin-pipeline-coordinator-production \\"
echo "         --payload '{\"run_type\":\"morning\"}' /tmp/test_out.json \\"
echo "         --region ${REGION} && cat /tmp/test_out.json"
echo ""
echo "  3. Evening pipeline fires at 21:00 BST (20:00 UTC)"
echo "  4. Submission reminder fires at 23:30 BST (22:30 UTC)"
echo "============================================================"
