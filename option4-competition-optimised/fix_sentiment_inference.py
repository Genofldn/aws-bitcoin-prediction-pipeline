#!/usr/bin/env python3
"""
Fix FinBERT label mapping in the sentiment model.
ProsusAI/finbert label order: index 0=positive, 1=negative, 2=neutral
Original inference script had it reversed.
Re-packages and re-uploads the sentiment model.tar.gz.
"""

import os
import json
import tarfile
import tempfile
import shutil
import boto3
from datetime import datetime, timezone

REGION  = "eu-west-2"
BUCKET  = os.environ.get("DATA_BUCKET", "your-bucket-name")
S3_KEY  = "models/sentiment/model.tar.gz"

CORRECTED_INFERENCE = '''
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ProsusAI/finbert label mapping (DO NOT CHANGE):
#   index 0 = positive
#   index 1 = negative
#   index 2 = neutral
LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}

def model_fn(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return {"tokenizer": tokenizer, "model": model}

def input_fn(request_body, content_type="application/json"):
    return json.loads(request_body)

def predict_fn(input_data, artifacts):
    tokenizer     = artifacts["tokenizer"]
    model         = artifacts["model"]
    headlines     = input_data.get("headlines", [])
    current_price = input_data.get("current_price", 50000)

    if not headlines:
        return {"prediction": current_price, "price_24h": current_price, "sentiment": 0.0}

    scores = []
    for text in headlines[:20]:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=128, padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
        # Score = positive_prob - negative_prob → range approx [-1, 1]
        # Positive score → bullish → price expected higher
        # Negative score → bearish → price expected lower
        score = float(probs[0] - probs[1])
        scores.append(score)

    avg_sentiment = float(sum(scores) / len(scores))

    # Translate sentiment to price adjustment: ±2% max swing
    price_adjustment = current_price * avg_sentiment * 0.02
    prediction = current_price + price_adjustment

    return {
        "prediction":  prediction,
        "price_24h":   prediction,
        "sentiment":   avg_sentiment,
        "num_articles": len(scores),
    }

def output_fn(prediction, content_type="application/json"):
    return json.dumps(prediction)
'''

def repackage_sentiment_model():
    s3 = boto3.client("s3", region_name=REGION)

    print("Downloading existing sentiment model from S3...")
    local_tar = "/tmp/sentiment_model_orig.tar.gz"
    s3.download_file(BUCKET, S3_KEY, local_tar)
    print(f"  Downloaded ({os.path.getsize(local_tar)/1e6:.0f} MB)")

    print("Extracting...")
    extract_dir = tempfile.mkdtemp()
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(extract_dir)

    print("Replacing inference.py with corrected label mapping...")
    inference_path = os.path.join(extract_dir, "inference.py")
    with open(inference_path, "w") as f:
        f.write(CORRECTED_INFERENCE)

    # Update config with fix note
    config_path = os.path.join(extract_dir, "config_meta.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}
    config["label_fix_applied"] = True
    config["label_mapping"]     = {"0": "positive", "1": "negative", "2": "neutral"}
    config["score_formula"]     = "probs[0] - probs[1]  (positive - negative)"
    config["fixed_at"]          = datetime.now(timezone.utc).isoformat()
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("Re-packaging...")
    fixed_tar = "/tmp/sentiment_model_fixed.tar.gz"
    with tarfile.open(fixed_tar, "w:gz") as tar:
        tar.add(extract_dir, arcname=".")
    print(f"  Fixed archive: {fixed_tar}  ({os.path.getsize(fixed_tar)/1e6:.0f} MB)")

    print(f"Uploading fixed model to s3://{BUCKET}/{S3_KEY} ...")
    s3.upload_file(fixed_tar, BUCKET, S3_KEY)
    print("  Uploaded ✅")

    # Quick sanity check using the corrected logic
    print("\nSanity check with corrected labels:")
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        tokenizer = AutoTokenizer.from_pretrained(extract_dir)
        model     = AutoModelForSequenceClassification.from_pretrained(extract_dir)
        model.eval()
        test_cases = [
            ("Bitcoin surges to all-time high as institutional adoption grows",  "expect: POSITIVE score"),
            ("Crypto market crashes amid regulatory crackdown",                  "expect: NEGATIVE score"),
            ("Bitcoin trading sideways ahead of Fed decision",                   "expect: NEUTRAL score"),
        ]
        for text, expectation in test_cases:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
            label = {0: "positive", 1: "negative", 2: "neutral"}[probs.argmax()]
            score = float(probs[0] - probs[1])
            print(f"  [{label:8s} | score={score:+.2f}] {text[:55]}...")
            print(f"              → {expectation}")
    except Exception as e:
        print(f"  Sanity check skipped: {e}")

    shutil.rmtree(extract_dir)
    print(f"\nSentiment model fix complete ✅")


if __name__ == "__main__":
    repackage_sentiment_model()
