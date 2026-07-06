# Videometer Lakehouse & MLOps Platform

This repository contains a fully automated, production-ready, containerized lakehouse and MLOps stack for the Videometer project.

## Architecture Highlights

The platform integrates data engineering and machine learning workflows seamlessly:

1. **Medallion Data Pipeline**: Orchestrated by Airflow (DAG 1), it processes raw crop image data through Bronze (ingestion), Silver (cleaning), and Gold (aggregation) layers.
2. **Automated MLOps Pipeline**: Orchestrated by Airflow (DAG 2), it dynamically discovers labeled datasets, trains XGBoost/RandomForest models, and tracks parameters, metrics, and artifacts via MLflow.
3. **Zero-Downtime Prediction API**: A FastAPI service that serves real-time inferences. It automatically hot-reloads the latest promoted model from the MLflow registry immediately after the MLOps pipeline finishes training.
4. **Isolated Metadata Stores**: PostgreSQL is initialized with strictly isolated databases (`airflow`, `mlflow`, `nessie`) to prevent schema collisions and guarantee transactional safety.

## Technology Stack

- **Data Processing**: Apache Spark
- **Object Storage**: MinIO (S3 compatible)
- **Metadata & Catalog**: PostgreSQL, Project Nessie
- **Orchestration**: Apache Airflow
- **Machine Learning**: MLflow, Scikit-Learn, XGBoost
- **Inference Serving**: FastAPI

## Repository Layout

- `airflow/`: Airflow DAGs (`medallion_pipeline.py`, `mlops_pipeline.py`), logs, and plugins
- `docker/`: Dockerfiles for Airflow, API, Spark, and MLflow
- `scripts/`: Bronze, Silver, Gold data scripts, Supervised training scripts, and API
- `data/`: Source data and generated local model bundles
- `docs/`: Setup, architecture, pipeline, and limitations documentation

## Prerequisites

Before starting the project, ensure you have installed:

- Docker Desktop (with Docker Compose v2 enabled)
- Minimum 8GB RAM allocated to Docker
- Git

## Quick Start

Build and start the fully automated infrastructure:

```bash
docker compose up -d --build