# System Architecture

The Videometer Lakehouse platform is designed with a decoupled, production-grade microservices architecture. 

## Component Overview

1. **Storage Tier**: MinIO provides an S3-compatible object storage layer containing isolated buckets (`bronze`, `silver`, `gold`, `mlflow`). 
2. **Compute Tier**: An Apache Spark Standalone cluster (Master/Worker) executes distributed data transformations.
3. **Database Tier**: A single PostgreSQL instance provides state management. Crucially, databases are strictly isolated at startup (`airflow`, `mlflow`, `nessie`) preventing schema corruption.
4. **Orchestration Tier**: Apache Airflow schedules and executes both data preparation and model training workflows.
5. **MLOps Tier**: MLflow manages the lifecycle, tracking metrics, and archiving artifacts for all trained models. Project Nessie versions the data catalog.
6. **Inference Tier**: FastAPI hosts the Prediction API.

## Workflow Integration

The system natively links Data Engineering and Data Science:

- **Decoupled DAGs**: The data transformation logic and model training logic run in separate DAGs. This guarantees that failures in model convergence do not halt data analytics pipelines.
- **Zero-Downtime Synchronization**: When the MLOps DAG successfully trains a new model, it executes an HTTP POST request to the API's `/reload` endpoint. The API seamlessly hot-swaps the model in memory.