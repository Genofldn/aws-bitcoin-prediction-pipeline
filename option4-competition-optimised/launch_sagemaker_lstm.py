#!/usr/bin/env python3
"""
Submits the LSTM training job to SageMaker using a GPU instance.
Runs overnight — takes ~3 hours on ml.p3.2xlarge.
Cost: ~$11 total.

Usage:
    python3 launch_sagemaker_lstm.py
"""

import boto3
import json
import tarfile
import os
from datetime import datetime, timezone

REGION       = "eu-west-2"
ACCOUNT_ID   = os.environ.get("AWS_ACCOUNT_ID", "your-account-id")
BUCKET       = os.environ.get("DATA_BUCKET", "your-bucket-name")
ROLE_NAME    = "BitcoinSageMakerTrainingRole"

# AWS Deep Learning Container — TensorFlow 2.12, GPU, Python 3.10
TF_IMAGE = (
    "763104351884.dkr.ecr.eu-west-2.amazonaws.com/"
    "tensorflow-training:2.12.0-gpu-py310-cu116-ubuntu20.04-ec2"
)

iam = boto3.client("iam",    region_name=REGION)
sm  = boto3.client("sagemaker", region_name=REGION)
s3  = boto3.client("s3",    region_name=REGION)


def ensure_sagemaker_role():
    """Create SageMaker training execution role if it doesn't already exist."""
    try:
        role = iam.get_role(RoleName=ROLE_NAME)
        arn  = role["Role"]["Arn"]
        print(f"Using existing role: {arn}")
        return arn
    except iam.exceptions.NoSuchEntityException:
        pass

    print("Creating SageMaker execution role...")
    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }
    role = iam.create_role(
        RoleName=ROLE_NAME,
        AssumeRolePolicyDocument=json.dumps(trust),
        Description="SageMaker training role for bitcoin prediction Option 4"
    )
    iam.attach_role_policy(
        RoleName=ROLE_NAME,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
    )
    iam.attach_role_policy(
        RoleName=ROLE_NAME,
        PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess"
    )
    arn = role["Role"]["Arn"]
    print(f"Role created: {arn}")
    # Small wait for IAM propagation
    import time; time.sleep(10)
    return arn


def package_and_upload_script():
    """Tar the training script and upload to S3 for SageMaker Script Mode."""
    script_src  = "/Users/terrysmac/project/option3-realtime-architecture/sagemaker/training/deep-learning-training.py"
    local_tar   = "/tmp/lstm_training_source.tar.gz"
    s3_key      = "training-scripts/lstm_source.tar.gz"

    print("Packaging training script...")
    with tarfile.open(local_tar, "w:gz") as tar:
        tar.add(script_src, arcname="deep-learning-training.py")

    print(f"Uploading script to s3://{BUCKET}/{s3_key} ...")
    s3.upload_file(local_tar, BUCKET, s3_key)
    return f"s3://{BUCKET}/{s3_key}"


def submit_training_job(role_arn, source_s3_uri):
    job_name = f"bitcoin-lstm-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    print(f"\nSubmitting SageMaker training job: {job_name}")
    print(f"  Instance:  ml.g4dn.2xlarge (1× T4 GPU)")
    print(f"  Container: TensorFlow 2.12 GPU")
    print(f"  Data:      s3://{BUCKET}/training-data/btc_hourly.parquet")
    print(f"  Est. time: ~3 hours | Est. cost: ~$11")

    response = sm.create_training_job(
        TrainingJobName=job_name,
        RoleArn=role_arn,

        AlgorithmSpecification={
            "TrainingImage":     TF_IMAGE,
            "TrainingInputMode": "File",
        },

        # Pass our training script as the entry point
        HyperParameters={
            "sagemaker_program":          "deep-learning-training.py",
            "sagemaker_submit_directory": source_s3_uri,
            "s3_bucket":                  BUCKET,
            "s3_data_path":               "training-data/btc_hourly.parquet",
            "s3_prefix":                  "models/deeplearning",
            "region":                     REGION,
            "sequence_length":            "168",
            "epochs":                     "100",
            "batch_size":                 "32",
        },

        InputDataConfig=[{
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri":      f"s3://{BUCKET}/training-data/",
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
            "ContentType": "application/octet-stream",
        }],

        OutputDataConfig={
            "S3OutputPath": f"s3://{BUCKET}/sagemaker-output/lstm/"
        },

        ResourceConfig={
            "InstanceType":   "ml.g4dn.2xlarge",  # NVIDIA T4 GPU — available eu-west-2
            "InstanceCount":  1,
            "VolumeSizeInGB": 50,
        },

        StoppingCondition={
            "MaxRuntimeInSeconds": 18000  # 5 hour hard cap
        },

        Tags=[
            {"Key": "Project",     "Value": "BitcoinPredictionOption4"},
            {"Key": "ModelType",   "Value": "LSTM"},
            {"Key": "Environment", "Value": "production"},
        ],
    )

    print(f"\nJob submitted ✅")
    print(f"Job name: {job_name}")
    print(f"\nMonitor with:")
    print(f"  aws sagemaker describe-training-job --training-job-name {job_name} --query 'TrainingJobStatus' --region eu-west-2")
    print(f"\nOr view in AWS Console:")
    print(f"  https://eu-west-2.console.aws.amazon.com/sagemaker/home?region=eu-west-2#/jobs/{job_name}")
    return job_name


if __name__ == "__main__":
    role_arn       = ensure_sagemaker_role()
    source_s3_uri  = package_and_upload_script()
    job_name       = submit_training_job(role_arn, source_s3_uri)

    # Save job name for later reference
    with open("/tmp/lstm_job_name.txt", "w") as f:
        f.write(job_name)
    print(f"\nJob name saved to /tmp/lstm_job_name.txt")
