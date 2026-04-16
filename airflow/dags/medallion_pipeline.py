from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "pfe",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="medallion_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=["lakehouse", "bronze", "silver", "gold"],
) as dag:

    bronze = BashOperator(
        task_id="bronze_ingestion",
        bash_command="python /opt/spark-app/scripts/01_bronze_ingestion.py",
    )

    silver = BashOperator(
        task_id="silver_transformation",
        bash_command="python /opt/spark-app/scripts/02_silver_transformation.py",
    )

    gold = BashOperator(
        task_id="gold_kpi",
        bash_command="python /opt/spark-app/scripts/03_gold_kpi.py",
    )

    bronze >> silver >> gold