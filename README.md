# V-Meter Lakehouse (PFE)

This repository is part of the final year internship project:

Design and Implementation of a Containerized Lakehouse Architecture for Videometer Big Data and Predictive Modeling.

## Project Scope

- Build a containerized open source lakehouse platform.
- Implement a medallion data pipeline:
  - Bronze layer for raw ingestion.
  - Silver layer for cleaning, standardization, and data quality checks.
  - Gold layer for key performance indicator aggregation.
- Orchestrate pipeline execution with Apache Airflow.
- Prepare the platform for predictive modeling workflows.

## Technology Stack

- Docker Compose
- Apache Spark
- Apache Airflow
- MinIO (S3-compatible object storage)
- PostgreSQL
- MLflow
- Project Nessie

## Repository Structure

- [airflow](airflow)
- [docker](docker)
- [scripts](scripts)
- [docs](docs)
- [docker-compose.yml](docker-compose.yml)
- [STARTUP.MD](STARTUP.MD)

## Quick Start

1. Start all services:

docker compose up -d --build

2. Verify running services:

docker compose ps -a

3. Open local interfaces:

- MinIO Console: http://localhost:9001
- Spark Master UI: http://localhost:8080
- Airflow UI: http://localhost:8088
- MLflow UI: http://localhost:5000
- Nessie API: http://localhost:19120

## Pipeline Execution

Run through Airflow by triggering the medallion_pipeline workflow, or run scripts manually in this order:

1. scripts/01_bronze_ingestion.py
2. scripts/02_silver_transformation.py
3. scripts/03_gold_kpi.py

## Documentation

- [Project overview](docs/overview.md)
- [Architecture](docs/architecture.md)
- [Setup](docs/setup.md)
- [Pipeline](docs/pipeline.md)
- [Limitations](docs/limitations.md)

## License

See [LICENSE](LICENSE).