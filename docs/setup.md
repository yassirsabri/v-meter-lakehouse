# Setup

## Requirements

- Docker Desktop
- Docker Compose
- A terminal opened in the project folder

## Start the environment

```bash
docker compose up -d --build
```

## Check the services

Open the following interfaces in the browser:

- MinIO Console: `http://localhost:9001`
- Spark Master: `http://localhost:8080`
- Airflow: `http://localhost:8088`
- MLflow: `http://localhost:5000`
- Nessie: `http://localhost:19120`

## Run the pipeline

The pipeline can be executed either from Airflow or manually using the scripts in the `scripts` folder.

### Airflow

- Open the Airflow interface
- Trigger the `medallion_pipeline` workflow

### Manual execution

Run the three scripts in order:

1. `01_bronze_ingestion.py`
2. `02_silver_transformation.py`
3. `03_gold_kpi.py`

## Notes

- The source file is expected in the mounted `data` folder.
- MinIO credentials are taken from the environment or Docker Compose.
