# Data & MLOps Pipelines

The infrastructure operates via two interconnected pipelines orchestrated by Airflow.

## 1. Medallion Data Pipeline
- **Bronze**: Raw data ingestion. Auto-discovers files, attaches technical tracking metadata, and writes to MinIO.
- **Silver**: Data quality and standardization. Cleans headers, removes duplicates, filters outliers via Tukey fences, and drops empty rows.
- **Gold**: Business aggregation. Computes grouped Key Performance Indicators (KPIs), morphology statistics, and shape indices.

## 2. MLOps Pipeline
- **Auto-Discovery**: Scans designated supervised learning directories for ready-to-train datasets.
- **Supervised Training**: Applies class-weighting and trains XGBoost/Random Forest classifiers.
- **MLflow Tracking**: Logs confusion matrices, feature importance graphs, F1-Scores, and full model bundles directly into the MinIO backend via MLflow.
- **Hot-Reloading**: Automates model deployment by pinging the FastAPI inference engine to fetch the newest model artifacts without interrupting web traffic.