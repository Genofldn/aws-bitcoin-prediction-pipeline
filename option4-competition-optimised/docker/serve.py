#!/usr/bin/env python3
"""
SageMaker-compatible serving script for BYO containers.
Loads inference.py from /opt/ml/model/inference.py and serves predictions
via a minimal Flask app on port 8080.

SageMaker contract:
  GET  /ping         → 200 OK (health check)
  POST /invocations  → prediction result (JSON)
"""

import importlib.util
import os
import sys
import json
import logging
from flask import Flask, request, Response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
PORT = int(os.environ.get("SAGEMAKER_BIND_TO_PORT", 8080))

# ── Load inference.py from model dir ─────────────────────────────────────────
inference_path = os.path.join(MODEL_DIR, "inference.py")
if not os.path.exists(inference_path):
    logger.error(f"inference.py not found at {inference_path}")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("inference", inference_path)
inference_module = importlib.util.module_from_spec(spec)
sys.modules["inference"] = inference_module
spec.loader.exec_module(inference_module)
logger.info("inference.py loaded")

# ── Load model ────────────────────────────────────────────────────────────────
logger.info("Calling model_fn ...")
model_artifacts = inference_module.model_fn(MODEL_DIR)
logger.info("Model ready ✓")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    return Response("", status=200)


@app.route("/invocations", methods=["POST"])
def invocations():
    content_type = request.content_type or "application/json"
    try:
        input_data = inference_module.input_fn(request.data, content_type)
        prediction = inference_module.predict_fn(input_data, model_artifacts)
        accept = request.accept_mimetypes.best or "application/json"
        result = inference_module.output_fn(prediction, accept)
        return Response(result, status=200, mimetype="application/json")
    except Exception as e:
        logger.exception("Prediction error")
        return Response(
            json.dumps({"error": str(e)}),
            status=500,
            mimetype="application/json",
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
