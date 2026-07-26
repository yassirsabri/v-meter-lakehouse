#!/usr/bin/env python3
"""
Unified Prediction API using FastAPI and MLflow.

Purpose:
- Serve predictions through one HTTP API.
- Load a model from MLflow registry when available.
- Fall back to a locally saved MLflow model folder when registry access fails.
- Support Zero-Downtime Hot Reloading via the /reload endpoint.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

import pandas as pd
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Query
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pyarrow.compute as pc


# ============================================================================
# Configuration
# ============================================================================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
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

# Global state
mlflow_model = None
bundle_model = None
mlflow_model_source = "unloaded"
last_error = "No error recorded"


def _patch_xgboost_model(model):
    if hasattr(model, "predict") and not hasattr(model, "use_label_encoder"):
        model.use_label_encoder = False
    return model


def _load_local_model_fallback():
    base_path = Path(MODEL_LOCAL_PATH)
    if not base_path.exists():
        raise FileNotFoundError(f"Local model folder does not exist: {MODEL_LOCAL_PATH}")

    candidates = sorted(
        [p for p in base_path.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for candidate in candidates:
        bundle_path = candidate / "model_bundle.joblib"
        if bundle_path.exists():
            bundle = joblib.load(bundle_path)
            if isinstance(bundle, dict) and "model" in bundle:
                bundle["model"] = _patch_xgboost_model(bundle["model"])
            return bundle, str(candidate)

        if (candidate / "MLmodel").exists():
            model = mlflow.pyfunc.load_model(candidate.resolve().as_uri())
            return model, str(candidate)

    raise FileNotFoundError(f"No supported model found under: {MODEL_LOCAL_PATH}")


def load_active_model():
    """Loads the model from MLflow registry or local fallback into memory."""
    global mlflow_model, bundle_model, mlflow_model_source, last_error

    print(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}", flush=True)
    print(f"Attempting to load model from URI: {MODEL_URI}", flush=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    temp_mlflow_model = None
    temp_bundle_model = None
    temp_source = "unloaded"
    temp_error = "No error recorded"

    try:
        temp_mlflow_model = mlflow.pyfunc.load_model(MODEL_URI)
        temp_source = MODEL_URI
        print(f"Loaded model from MLflow URI: {MODEL_URI}", flush=True)
    except Exception as exc:
        print(f"Registry load failed for '{MODEL_URI}': {exc}", flush=True)
        try:
            temp_bundle_model, local_path = _load_local_model_fallback()
            temp_source = f"file://{local_path}"
            print(f"Loaded local model from: {temp_source}", flush=True)
        except Exception as fallback_exc:
            temp_error = str(fallback_exc)
            print(f"Local model fallback failed: {fallback_exc}", flush=True)
            raise RuntimeError(f"Could not load model from registry or fallback: {fallback_exc}")

    # Atomic swap for zero-downtime
    mlflow_model = temp_mlflow_model
    bundle_model = temp_bundle_model
    mlflow_model_source = temp_source
    last_error = temp_error


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    try:
        load_active_model()
    except Exception as e:
        print(f"Startup model load failed, API will be unavailable until /reload succeeds. Error: {e}", flush=True)
    yield


app = FastAPI(
    title="Videometer Lakehouse Prediction API",
    description="Unified API to serve MLflow model predictions for seed data with zero-downtime reloads.",
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
        "model_loaded": mlflow_model is not None or bundle_model is not None,
        "model_source": mlflow_model_source,
        "last_error": last_error,
    }


@app.post("/reload")
def reload_model():
    """Hot-reloads the model from the MLflow registry or local disk without downtime."""
    try:
        load_active_model()
        return {
            "status": "success", 
            "message": "Model reloaded successfully", 
            "model_source": mlflow_model_source
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not request.features:
        raise HTTPException(status_code=400, detail="features cannot be empty")

    try:
        df = pd.DataFrame(request.features)
        predictions = []

        if bundle_model is not None:
            model = _patch_xgboost_model(bundle_model["model"])
            imputer = bundle_model["imputer"]
            feature_names = bundle_model["feature_names"]

            for feature in feature_names:
                if feature not in df.columns:
                    df[feature] = np.nan

            df = df[feature_names]
            df_imputed = pd.DataFrame(imputer.transform(df), columns=feature_names)
            predictions = model.predict(df_imputed)

        elif mlflow_model is not None:
            # FIX FOR XGBOOST gpu_id BUG
            import xgboost as xgb
            
            # Extract the underlying XGBoost booster/model from the MLflow PyFunc wrapper
            underlying_model = mlflow_model.unwrap_python_model()
            
            # Check if it's an XGBoost model
            if hasattr(underlying_model, "xgb_model"):
                xgb_model = underlying_model.xgb_model
                
                # If it's a Booster, it needs a DMatrix
                if isinstance(xgb_model, xgb.Booster):
                    dtrain = xgb.DMatrix(df)
                    preds = xgb_model.predict(dtrain)
                    # Convert probabilities to classes
                    predictions = (preds > 0.5).astype(int)
                # If it's a Scikit-Learn wrapper (XGBClassifier)
                else:
                    predictions = xgb_model.predict(df)
            else:
                # Fallback if it's not XGBoost
                predictions = mlflow_model.predict(df)

        else:
            raise HTTPException(status_code=503, detail="Model is not loaded or unavailable.")

        if hasattr(predictions, "tolist"):
            predictions = predictions.tolist()

        if bundle_model is not None and bundle_model.get("task") == "viability":
            predictions = ["Viable" if p == 0 else "Non Viable" for p in predictions]
        elif mlflow_model is not None:
            # Format outputs for viability classification
            predictions = ["Viable" if p == 0 else "Non Viable" for p in predictions]

        return PredictionResponse(
            model_name=MODEL_NAME,
            model_version=MODEL_STAGE,
            predictions=predictions,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(exc)}")

@app.get("/data/bronze")
def query_bronze_by_accession(
    accessions: str = Query(..., description="Comma-separated list of accessions (e.g., 8586, 16988)"),
    crop: str = Query(None, description="(Optional) Restrict search to a specific crop type (e.g., barley)"),
    limit: int = Query(50, ge=1, le=5000)
):
    """
    Query the Bronze data layer efficiently using Predicate Pushdown.
    Searches across all crops simultaneously by default.
    """
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.fs as pafs
    import pyarrow.compute as pc
    import pandas as pd
    
    raw_list = [acc.strip() for acc in accessions.split(",")]
    expanded_list = []
    
    # Variations intelligentes pour la recherche
    for acc in raw_list:
        expanded_list.append(acc)
        if not acc.lower().endswith(".png"):
            expanded_list.append(f"{acc}.png")
            expanded_list.append(f"{acc}.PNG")
            expanded_list.append(f"{acc}.jpg")
    
    endpoint_url = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    scheme = "https" if endpoint_url.startswith("https") else "http"
    endpoint_override = endpoint_url.replace(f"{scheme}://", "")
    
    try:
        s3_fs = pafs.S3FileSystem(
            endpoint_override=endpoint_override,
            access_key=os.environ.get("AWS_ACCESS_KEY_ID", "admin"),
            secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minio_password"),
            scheme=scheme
        )
        
        bronze_bucket = os.getenv("BRONZE_BUCKET", "bronze")
        base_path = f"{bronze_bucket}/warehouse/bronze"
        
        selector = pafs.FileSelector(base_path, recursive=True)
        try:
            file_infos = s3_fs.get_file_info(selector)
        except Exception:
            raise HTTPException(status_code=404, detail="No Bronze data warehouse found.")

        parquet_files = []
        for f in file_infos:
            if f.type == pafs.FileType.File and f.path.endswith('.parquet'):
                if crop:
                    if f"{crop}_" in f.path.lower() or f"/{crop}/" in f.path.lower():
                        parquet_files.append(f.path)
                else:
                    parquet_files.append(f.path)
        
        if not parquet_files:
            raise HTTPException(status_code=404, detail="No Parquet data files found in the datalake.")

        results = []
        for p_file in parquet_files:
            try:
                dataset = ds.dataset([p_file], format="parquet", filesystem=s3_fs)
                condition = pc.is_in(ds.field("Filename"), value_set=pa.array(expanded_list))
                scanner = dataset.scanner(filter=condition)
                table = scanner.head(limit) 
                
                if table.num_rows > 0:
                    results.append(table.to_pandas())
            except Exception as e:
                print(f"Skipping {p_file} due to error: {e}")
                continue
        
        if not results:
            raise HTTPException(
                status_code=404, 
                detail=f"The requested accessions were not found in the dataset. (Searched for: {', '.join(expanded_list)})"
            )
            

        # On fusionne avec Pandas
        df_result = pd.concat(results, ignore_index=True)
        
        # LA VRAIE CORRECTION EST ICI : On force la limite GLOBALE sur le résultat final
        df_result = df_result.head(limit)
        
        # Destruction totale des NaN, inf et -inf pour JSON
        df_result = df_result.astype(object).replace([np.inf, -np.inf, np.nan], None)
            
        return {
            "scanned_crops": "All" if not crop else crop,
            "requested_accessions": raw_list,
            "rows_returned": len(df_result),
            "data": df_result.to_dict(orient="records")
        }
        
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Server Error: {str(exc)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("05_prediction_api:app", host="0.0.0.0", port=8000, reload=True)