import os
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# Configuration des arguments par défaut du DAG
default_args = {
    "owner": "pfe",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# Répertoire contenant les datasets labellisés
ML_DATA_DIR = "/opt/spark-app/data_for_supervised_ml"
VALID_EXTENSIONS = {".csv"}

def discover_ml_datasets():
    """Découverte dynamique des fichiers datasets dans le dossier cible."""
    data_path = Path(ML_DATA_DIR)
    datasets = []
    if data_path.exists():
        for f in data_path.iterdir():
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS:
                datasets.append(f.name)
    return sorted(datasets)

DATASETS = discover_ml_datasets()

# Environnement réseau Docker requis
MLOPS_ENVIRONMENT = {
    "MLFLOW_TRACKING_URI": "http://mlflow:5000",
    "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
    "AWS_ACCESS_KEY_ID": "admin",
    "AWS_SECRET_ACCESS_KEY": "minio_password",
}

def reload_and_smoke_test_api():
    """
    Automated Smoke Test for the Prediction API.
    1. Calls /reload to fetch the latest model.
    2. Dynamically reads a real row from the dataset to test inference.
    """
    import time
    import requests
    import pandas as pd
    from pathlib import Path

    api_url = "http://prediction-api:8000"
    
    print("1. Triggering API Hot-Reload...")
    reload_resp = requests.post(f"{api_url}/reload", timeout=10)
    reload_resp.raise_for_status()
    print("API successfully reloaded the model.")
    
    # Give the API a moment to settle
    time.sleep(2)
    
    print("2. Preparing Smoke Test Data from actual dataset...")
    # Find the first available dataset to extract a real feature vector
    data_dir = Path("/opt/spark-app/data_for_supervised_ml")
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV found for smoke test. Skipping inference test.")
        return

    # Load just the first few rows to minimize memory usage
    df = pd.read_csv(csv_files[0], nrows=5)
    
    # Exclude metadata exactly like the training script does
    cols_to_exclude = [
        "IG", "id_measurement", "cluster_label", "is_anomaly", "is_geometric_outlier",
        "sourceimage_captureid", "sourceimageid", "dq_check_status", "TAX_NAME", "POP_TYPE",
        "ORI", "REGION", "MACRO_REGION", "ViabilityRate", "Viability_Class", "filename", "blobid"
    ]
    
    existing = [c for c in cols_to_exclude if c in df.columns]
    df_features = df.drop(columns=existing, errors="ignore").select_dtypes(include=["number"])
    
    if df_features.empty:
        print("No numeric features extracted. Skipping inference test.")
        return
        
    # Convert the first row to a dictionary, safely handling NaNs
    record = {k: (None if pd.isna(v) else float(v)) for k, v in df_features.iloc[0].items()}
    dummy_payload = {"features": [record]}
    
    print(f"3. Running Smoke Test Inference with {len(record)} features...")
    predict_resp = requests.post(f"{api_url}/predict", json=dummy_payload, timeout=10)
    
    if not predict_resp.ok:
        print(f"API Error Details: {predict_resp.text}")
        predict_resp.raise_for_status()
    
    result = predict_resp.json()
    print(f"Smoke Test Successful! API Predicted: {result['predictions'][0]}")


with DAG(
    dag_id="mlops_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=["lakehouse", "mlops", "training", "tracking"],
) as dag:

    # Tâche finale : Rechargement et Test (Zero-Downtime Smoke Test)
    api_integration_test = PythonOperator(
        task_id="api_reload_and_smoke_test",
        python_callable=reload_and_smoke_test_api,
    )

    if not DATASETS:
        no_data_task = BashOperator(
            task_id="no_datasets_found",
            bash_command="echo 'Aucun fichier CSV trouvé dans le dossier ML. Fin du pipeline.'",
        )
        no_data_task >> api_integration_test
    else:
        # Création dynamique des tâches d'entraînement
        for dataset_file in DATASETS:
            task_type = "viability" if "viabilit" in dataset_file.lower() else "origin"
            
            train_model = BashOperator(
                task_id=f"train_{task_type}_model",
                bash_command=f"""
                python /opt/spark-app/scripts/06_supervised_train_and_track.py \
                    --task {task_type} \
                    --data-path "/opt/spark-app/data_for_supervised_ml/{dataset_file}" \
                    --output-model-path "/opt/spark-app/data/models" \
                    --experiment-name "automated_mlops_{task_type}"
                """,
                env=MLOPS_ENVIRONMENT,
            )
            
            # Dépendance : L'entraînement déclenche le Smoke Test de l'API
            train_model >> api_integration_test