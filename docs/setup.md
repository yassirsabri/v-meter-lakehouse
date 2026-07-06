# Setup Guide

## Requirements
- Docker Desktop (with Docker Compose v2 enabled)
- A terminal opened in the project folder

## Installation Steps

1. **Clean prior states**:
   ```bash
   docker compose down -v


  Boot the platform:


docker compose up -d --build

Verify the Initializers:
The services postgres-init and minio-init will run briefly to create isolated databases and buckets. Ensure they exit with status 0 (Success).

Access UI Consoles:

MinIO Console: http://localhost:9001

Spark Master: http://localhost:8080

Airflow: http://localhost:8088

MLflow: http://localhost:5000

Prediction API Docs: http://localhost:8000/docs

Run Workflows:
Log into Airflow (admin/admin) and trigger medallion_pipeline, followed by mlops_pipeline.