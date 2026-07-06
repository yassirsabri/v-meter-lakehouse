
# Operations and Startup Runbook

## 1. System Initialization

To deploy the production-ready container architecture, run:


# Clean up any residual corrupted volumes
docker compose down -v

# Build and start all services
docker compose up -d --build

# Monitor the convergence of services until all are marked as healthy:
docker compose ps

## 2. Triggering the Workflows

The platform relies on two decoupled Apache Airflow DAGs to ensure data processing and machine learning tasks do not interfere with each other.

### The Medallion Data Pipeline (`medallion_pipeline`)
1. Log into Airflow at http://localhost:8088 (Credentials: `admin` / `admin`).
2. Unpause and trigger `medallion_pipeline`.
3. This scans the `data/` folder and processes all files into the Bronze, Silver, and Gold zones residing in MinIO.

### The MLOps Pipeline (`mlops_pipeline`)
1. In Airflow, trigger `mlops_pipeline`.
2. This DAG dynamically discovers labeled datasets inside `data_for_supervised_ml/ready for ml/`.
3. It triggers the supervised training scripts, logs metrics to MLflow, and automatically signals the FastAPI application to reload its active model into memory without dropping HTTP requests.

## 3. Real-Time Inference Testing

You can verify the active model is loaded in the API via the health check endpoint:

```bash
curl http://localhost:8000/health