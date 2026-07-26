from datetime import datetime, timedelta
from pathlib import Path
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "pfe",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# Discover crops from the data directory
DATA_DIR = "/opt/spark-app/data"
VALID_EXTENSIONS = {".csv", ".xlsx", ".xls"}

def discover_crops():
    """Discover crop names from filenames in the data directory."""
    data_path = Path(DATA_DIR)
    crops = set()
    
    if data_path.exists():
        for f in data_path.iterdir():
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS:
                crop = f.stem.lower()  # filename without extension
                crops.add(crop)
    
    return sorted(list(crops))

# Get crops at DAG parse time
CROPS = discover_crops()

with DAG(
    dag_id="medallion_pipeline_v2",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=["lakehouse", "bronze", "silver", "gold"],
) as dag:

    # Bronze discovers and processes all crops
    bronze = BashOperator(
        task_id="bronze_ingestion",
        bash_command="python /opt/spark-app/scripts/01_bronze_ingestion.py",
    )

    for crop in CROPS:
        silver = BashOperator(
            task_id=f"silver_{crop}",
            bash_command=f"python /opt/spark-app/scripts/02_silver_transformation.py --crop {crop}",
        )

        # TÂCHE NON-SUPERVISÉE AJOUTÉE ICI
        unsupervised = BashOperator(
            task_id=f"unsupervised_ml_{crop}",
            bash_command=f"python /opt/spark-app/scripts/04_unsupervised_modeling.py --crop {crop}",
        )

        gold = BashOperator(
            task_id=f"gold_{crop}",
            bash_command=f"python /opt/spark-app/scripts/03_gold_kpi.py --crop {crop}",
        )

        # CHAINAGE : Bronze -> Silver, puis Silver déclenche Unsupervised et Gold en parallèle
        bronze >> silver
        silver >> unsupervised
        silver >> gold