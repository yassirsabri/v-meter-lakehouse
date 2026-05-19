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
    dag_id="medallion_pipeline",
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

    # Dynamically create Silver and Gold tasks per discovered crop
    silver_tasks = []
    gold_tasks = []
    
    for crop in CROPS:
        silver = BashOperator(
            task_id=f"silver_{crop}",
            bash_command=f"python /opt/spark-app/scripts/02_silver_transformation.py --crop {crop}",
        )
        silver_tasks.append(silver)

        gold = BashOperator(
            task_id=f"gold_{crop}",
            bash_command=f"python /opt/spark-app/scripts/03_gold_kpi.py --crop {crop}",
        )
        gold_tasks.append(gold)

        # Chain: silver -> gold for each crop
        silver >> gold

    # Chain: bronze -> all silvers -> all golds
    bronze >> silver_tasks