#!/usr/bin/env python3
"""
create_lstm_placeholder.py — Create a minimal LSTM placeholder model.

Builds a tiny LSTM with random weights and the correct architecture.
Uploads to S3 as models/deeplearning/model.tar.gz so CloudFormation
can deploy the endpoint immediately. The real trained model replaces
it automatically when training completes (via repackage_models.py).
"""

import os
import json
import pickle
import tarfile
import tempfile
import shutil
import boto3
import numpy as np
from datetime import datetime, timezone

REGION  = "eu-west-2"
BUCKET  = os.environ.get("DATA_BUCKET", "your-bucket-name")
S3_KEY  = "models/deeplearning/model.tar.gz"

INFERENCE_PY = os.path.join(
    os.path.dirname(__file__), "docker", "inference", "lstm_inference.py"
)

# Target: 4 predictions (1h, 6h, 24h, 7d)
TARGET_COLUMNS   = ["price_1h", "price_6h", "price_24h", "price_7d"]
FEATURE_COLUMNS  = ["price", "volume", "sentiment_score"]
SEQUENCE_LENGTH  = 168


def build_placeholder_model():
    """
    Build a SIMPLE Dense model (no LSTM) as placeholder.
    LSTM layers use Metal/CudnnRNN ops on M2 Mac that can't run in Linux CPU containers.
    Dense model avoids this issue and loads cleanly in any TF environment.
    The real trained LSTM replaces this when training completes.
    """
    import os
    import tensorflow as tf
    # Force CPU-only mode — no GPU-specific ops
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    tf.config.set_visible_devices([], "GPU")

    from tensorflow import keras
    from tensorflow.keras import layers

    seq_len   = SEQUENCE_LENGTH
    n_feat    = len(FEATURE_COLUMNS)
    n_targets = len(TARGET_COLUMNS)

    # Simple Dense network that takes the flattened sequence
    # Input: (batch, seq_len, n_feat) — we flatten for simplicity
    inputs  = keras.Input(shape=(seq_len, n_feat), name="sequence_input")
    flat    = layers.Flatten()(inputs)
    dense1  = layers.Dense(128, activation="relu")(flat)
    dense2  = layers.Dense(64, activation="relu")(dense1)
    outputs = layers.Dense(n_targets, name="price_outputs")(dense2)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    print(f"  Placeholder model built (Dense, CPU-compatible): {model.count_params():,} params")
    return model


def build_scalers():
    """Build identity-like scalers so inverse_transform is a no-op."""
    from sklearn.preprocessing import MinMaxScaler

    # Feature scaler — fit on a plausible BTC price range
    feature_scaler = MinMaxScaler()
    dummy_features = np.array([
        [40000, 1e8, -1],
        [130000, 1e10, 1],
    ])
    feature_scaler.fit(dummy_features)

    # Target scaler — same price range for all 4 targets
    target_scaler = MinMaxScaler()
    dummy_targets = np.array([
        [40000, 40000, 40000, 40000],
        [130000, 130000, 130000, 130000],
    ])
    target_scaler.fit(dummy_targets)

    return {"features": feature_scaler, "targets": target_scaler}


def main():
    import tensorflow as tf

    print("Building placeholder LSTM model...")
    model   = build_placeholder_model()
    scalers = build_scalers()

    model_dir = tempfile.mkdtemp()
    try:
        # Save model as SavedModel format (version-agnostic, avoids Keras H5 compat issues)
        # Also save H5 as fallback
        saved_model_path = os.path.join(model_dir, "lstm_model")
        model.export(saved_model_path)  # Keras 3.x: export() creates TF SavedModel
        print(f"  Saved SavedModel: {saved_model_path}/")

        # Also save a minimal H5 for older-TF compatibility
        h5_path = os.path.join(model_dir, "lstm_model.h5")
        try:
            # Try native keras format first
            keras_path = os.path.join(model_dir, "lstm_model.keras")
            model.save(keras_path)
            print(f"  Saved .keras: {os.path.getsize(keras_path)/1e6:.1f} MB")
        except Exception as e:
            print(f"  Skipping .keras save: {e}")

        # Save scalers
        with open(os.path.join(model_dir, "scalers.pkl"), "wb") as f:
            pickle.dump(scalers, f)

        # Save config
        config = {
            "sequence_length": SEQUENCE_LENGTH,
            "feature_columns":  FEATURE_COLUMNS,
            "target_columns":   TARGET_COLUMNS,
            "is_placeholder":   True,
            "created_at":       datetime.now(timezone.utc).isoformat(),
            "note": "Placeholder model — replace with real trained weights via repackage_models.py"
        }
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Copy inference.py
        shutil.copy2(INFERENCE_PY, os.path.join(model_dir, "inference.py"))
        print("  inference.py copied")

        # Package
        tar_path = "/tmp/lstm_placeholder.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(model_dir, arcname=".")
        print(f"  Archive: {tar_path}  ({os.path.getsize(tar_path)/1e6:.1f} MB)")

        # Upload
        s3 = boto3.client("s3", region_name=REGION)
        print(f"  Uploading to s3://{BUCKET}/{S3_KEY} ...")
        s3.upload_file(tar_path, BUCKET, S3_KEY)
        print("  ✅ Placeholder LSTM uploaded to S3")

    finally:
        shutil.rmtree(model_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
