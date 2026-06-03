#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# build_and_push.sh — Build all 4 Docker images and push to ECR
# Run from the option4-competition-optimised/ directory.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REGION="eu-west-2"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
ECR_BASE="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

MODELS=(ensemble lstm sentiment timeseries)

echo "============================================================"
echo "Build & Push — Option 4 SageMaker containers"
echo "Account: ${ACCOUNT}  Region: ${REGION}"
echo "============================================================"

# ── ECR login ────────────────────────────────────────────────────────────────
echo ""
echo "▶ Logging into ECR..."
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ECR_BASE}"

# ── Create repos if needed ────────────────────────────────────────────────────
for model in "${MODELS[@]}"; do
  repo="bitcoin-${model}"
  echo "  Ensuring ECR repo: ${repo}"
  aws ecr describe-repositories --repository-names "${repo}" \
      --region "${REGION}" > /dev/null 2>&1 \
    || aws ecr create-repository \
        --repository-name "${repo}" \
        --region "${REGION}" \
        --image-scanning-configuration scanOnPush=false \
        --query 'repository.repositoryUri' --output text
done

# ── Build and push each image ─────────────────────────────────────────────────
for model in "${MODELS[@]}"; do
  echo ""
  echo "── Building bitcoin-${model} ──────────────────────────────"

  IMAGE="${ECR_BASE}/bitcoin-${model}:latest"
  DOCKERFILE="docker/Dockerfile.${model}"

  docker build \
    --platform linux/amd64 \
    -f "${DOCKERFILE}" \
    -t "bitcoin-${model}:latest" \
    -t "${IMAGE}" \
    docker/      # build context = docker/ folder (has serve.py)

  echo "  Pushing ${IMAGE} ..."
  docker push "${IMAGE}"
  echo "  ✅ bitcoin-${model} pushed"
done

echo ""
echo "============================================================"
echo "All 4 images pushed to ECR ✅"
echo ""
echo "  ${ECR_BASE}/bitcoin-ensemble:latest"
echo "  ${ECR_BASE}/bitcoin-lstm:latest"
echo "  ${ECR_BASE}/bitcoin-sentiment:latest"
echo "  ${ECR_BASE}/bitcoin-timeseries:latest"
echo "============================================================"
