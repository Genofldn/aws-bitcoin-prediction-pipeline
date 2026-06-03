#!/usr/bin/env python3
"""
repackage_models.py — inject the correct inference.py into each model tar.gz.

Downloads each model from S3, replaces inference.py with the version from
docker/inference/, and re-uploads. Run this AFTER training finishes.

Usage:
    python3 repackage_models.py [--models ensemble lstm timeseries]
    # sentiment already patched by fix_sentiment_inference.py
"""

import os
import sys
import tarfile
import tempfile
import shutil
import boto3
import argparse
from datetime import datetime, timezone

REGION = "eu-west-2"
BUCKET = os.environ.get("DATA_BUCKET", "your-bucket-name")

INFERENCE_DIR = os.path.join(os.path.dirname(__file__), "docker", "inference")

MODEL_CONFIG = {
    "ensemble": {
        "s3_key":    "models/ensemble/model.tar.gz",
        "local_src": os.path.join(INFERENCE_DIR, "ensemble_inference.py"),
    },
    "lstm": {
        "s3_key":    "models/deeplearning/model.tar.gz",
        "local_src": os.path.join(INFERENCE_DIR, "lstm_inference.py"),
    },
    "timeseries": {
        "s3_key":    "models/timeseries/model.tar.gz",
        "local_src": os.path.join(INFERENCE_DIR, "timeseries_inference.py"),
    },
}


def repackage(model_name: str, s3: boto3.client):
    cfg     = MODEL_CONFIG[model_name]
    s3_key  = cfg["s3_key"]
    src_inf = cfg["local_src"]

    if not os.path.exists(src_inf):
        print(f"  [SKIP] {src_inf} not found — skipping {model_name}")
        return

    local_tar = f"/tmp/{model_name}_orig.tar.gz"
    fixed_tar = f"/tmp/{model_name}_fixed.tar.gz"

    print(f"\n── {model_name} {'─'*40}")
    print(f"  Downloading s3://{BUCKET}/{s3_key} ...")
    s3.download_file(BUCKET, s3_key, local_tar)
    size_mb = os.path.getsize(local_tar) / 1e6
    print(f"  Downloaded ({size_mb:.0f} MB)")

    extract_dir = tempfile.mkdtemp()
    try:
        print("  Extracting...")
        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(extract_dir)

        print(f"  Injecting {os.path.basename(src_inf)} as inference.py ...")
        dest_inf = os.path.join(extract_dir, "inference.py")
        shutil.copy2(src_inf, dest_inf)

        # Update metadata
        import json
        meta_path = os.path.join(extract_dir, "config_meta.json")
        if not os.path.exists(meta_path):
            meta_path = os.path.join(extract_dir, "config.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                try:
                    meta = json.load(f)
                except Exception:
                    meta = {}
        meta["inference_updated_at"] = datetime.now(timezone.utc).isoformat()
        meta["inference_source"] = os.path.basename(src_inf)
        with open(os.path.join(extract_dir, "config_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print("  Re-packaging...")
        with tarfile.open(fixed_tar, "w:gz") as tar:
            tar.add(extract_dir, arcname=".")
        fixed_mb = os.path.getsize(fixed_tar) / 1e6
        print(f"  Fixed archive: {fixed_tar}  ({fixed_mb:.0f} MB)")

        print(f"  Uploading to s3://{BUCKET}/{s3_key} ...")
        s3.upload_file(fixed_tar, BUCKET, s3_key)
        print(f"  ✅ {model_name} repackaged and uploaded")

    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        default=list(MODEL_CONFIG.keys()),
        choices=list(MODEL_CONFIG.keys()),
    )
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)

    print("=" * 60)
    print("Model Inference Repackaging")
    print(f"Models: {args.models}")
    print("=" * 60)

    for model_name in args.models:
        repackage(model_name, s3)

    print("\n" + "=" * 60)
    print("Repackaging complete ✅")
    print(f"  s3://{BUCKET}/models/ensemble/model.tar.gz")
    print(f"  s3://{BUCKET}/models/deeplearning/model.tar.gz")
    print(f"  s3://{BUCKET}/models/timeseries/model.tar.gz")
    print("=" * 60)


if __name__ == "__main__":
    main()
