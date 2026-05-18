#!/usr/bin/env python3
"""
Unified Prediction API using FastAPI and MLflow.

Purpose:
- Serve predictions through one HTTP API.
- Load a model from MLflow registry when available.
- Fall back to a locally saved MLflow model folder when registry access fails.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio_password")

os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT
os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

MODEL_NAME = os.getenv("MODEL_NAME", "Videometer_Model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
MODEL_URI = os.getenv("MODEL_URI", f"models:/{MODEL_NAME}/{MODEL_STAGE}")
MODEL_LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH", "/opt/spark-app/data/models")

mlflow_model = None
mlflow_model_source = "unloaded"


def _load_local_model_fallback():
    base_path = Path(MODEL_LOCAL_PATH)
    if not base_path.exists():
        raise FileNotFoundError(f"Local model folder does not exist: {MODEL_LOCAL_PATH}")

    candidates = sorted(
        [p for p in base_path.iterdir() if p.is_dir() and (p / "MLmodel").exists()]
    )
    if not candidates:
        raise FileNotFoundError(f"No MLflow model folders found under: {MODEL_LOCAL_PATH}")

    local_model_path = candidates[0].resolve()
    return mlflow.pyfunc.load_model(local_model_path.as_uri()), str(local_model_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mlflow_model, mlflow_model_source

    print(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}")
    print(f"Attempting to load model from URI: {MODEL_URI}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        mlflow_model = mlflow.pyfunc.load_model(MODEL_URI)
        mlflow_model_source = MODEL_URI
        print(f"Loaded model from MLflow URI: {MODEL_URI}")
    except Exception as exc:
        print(f"Registry load failed for '{MODEL_URI}': {exc}")
        try:
            mlflow_model, local_path = _load_local_model_fallback()
            mlflow_model_source = f"file://{local_path}"
            print(f"Loaded model from local path: {mlflow_model_source}")
        except Exception as fallback_exc:
            mlflow_model = None
            mlflow_model_source = "unloaded"
            print(f"Local model fallback failed: {fallback_exc}")

    yield

    mlflow_model = None
    mlflow_model_source = "unloaded"


app = FastAPI(
    title="Videometer Lakehouse Prediction API",
    description="Unified API to serve MLflow model predictions for seed data.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    features: List[Dict[str, Any]]


class PredictionResponse(BaseModel):
    model_name: str
    model_version: str
    predictions: List[Any]


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Videometer Lakehouse Prediction API. Go to /docs to view the Swagger UI."
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "mlflow_uri": MLFLOW_TRACKING_URI,
        "model_loaded": mlflow_model is not None,
        "model_source": mlflow_model_source,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if mlflow_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or unavailable.")

    if not request.features:
        raise HTTPException(status_code=400, detail="features cannot be empty")

    try:
        df = pd.DataFrame(request.features)
        predictions = mlflow_model.predict(df)

        if hasattr(predictions, "tolist"):
            predictions = predictions.tolist()

        return PredictionResponse(
            model_name=MODEL_NAME,
            model_version=MODEL_STAGE,
            predictions=predictions,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(exc)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("05_prediction_api:app", host="0.0.0.0", port=8000, reload=True)