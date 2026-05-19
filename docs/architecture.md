
# Architecture

The project uses the following components:

- Docker Compose to start the local environment
- Apache Spark to process the data
- MinIO to store Bronze, Silver, and Gold outputs
- PostgreSQL to store metadata for supporting services
- Apache Airflow to orchestrate the pipeline
- MLflow for model tracking infrastructure
- Nessie for versioned data management infrastructure

## Data Flow

1. The raw CSV file is read from the mounted `data/` folder.
2. The Bronze script stores the raw data in MinIO and adds technical metadata.
3. The Silver script reads the Bronze data, cleans it, and removes duplicates.
4. The Gold script reads the Silver data and creates aggregated key performance indicators.
5. Airflow runs the three scripts in sequence.

## Storage Zones

- Bronze: raw data
- Silver: cleaned data

