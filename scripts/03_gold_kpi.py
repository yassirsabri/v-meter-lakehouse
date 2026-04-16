#!/usr/bin/env python3
"""
Gold KPI aggregation pipeline.

Purpose
- Read curated Silver data from MinIO.
- Aggregate measurements by image and date bucket.
- Compute analytical metrics for reporting and downstream modeling.
- Write the final KPI dataset to the Gold zone in MinIO.

Why this stage matters
- Gold is the analytics-ready layer of the medallion architecture.
- It converts row-level measurements into business-level indicators.
"""

import os
import sys
from datetime import datetime, timezone

import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio_password")

SILVER_BUCKET = os.getenv("SILVER_BUCKET", "silver")
SILVER_KEY = os.getenv("SILVER_KEY", "videometer/cleaned_measurements")
GOLD_BUCKET = os.getenv("GOLD_BUCKET", "gold")
GOLD_KEY = os.getenv("GOLD_KEY", "videometer/kpi_viability")

S3_STORAGE_OPTIONS = {
    "key": MINIO_ACCESS_KEY,
    "secret": MINIO_SECRET_KEY,
    "client_kwargs": {"endpoint_url": MINIO_ENDPOINT},
    "anon": False,
}

# Grouping dimensions for KPI aggregation.
GROUP_KEYS = ["SourceImage_ID", "date_bucket"]

# Numeric columns included in KPI calculations.
NUMERIC_KPI_COLS = [
    "Area (mm2)",
    "Length (mm)",
    "Width (mm)",
    "RatioWidthLength",
    "AreaFraction",
    "Perimeter",
    "Compactness",
    "Eccentricity",
    "CIELab_L",
    "CIELab_A",
    "CIELab_B",
    "ReflectanceMean",
    "Volume",
    "data_quality_score",
]


# ============================================================================
# Helper functions
# ============================================================================

def fail(message: str) -> None:
    """Print an error message and stop execution."""
    print(f"[GOLD KPI] ERROR: {message}")
    sys.exit(1)


def get_s3_fs():
    """
    Initialize an authenticated S3 filesystem client for MinIO.

    This helper is kept for compatibility with environments that require
    explicit filesystem initialization.
    """
    try:
        import s3fs
        return s3fs.S3FileSystem(
            anon=False,
            use_ssl=False,
            key=MINIO_ACCESS_KEY,
            secret=MINIO_SECRET_KEY,
            client_kwargs={"endpoint_url": MINIO_ENDPOINT},
        )
    except ImportError:
        fail("s3fs not installed. Install: pip install s3fs")


def flatten_columns(columns) -> list[str]:
    """
    Flatten the MultiIndex columns created by pandas aggregation.

    Example:
    ('Area (mm2)', 'mean') -> 'Area (mm2)_mean'
    """
    flat = []
    for column in columns:
        if isinstance(column, tuple):
            parts = [str(part) for part in column if part and str(part) != "nan"]
            flat.append("_".join(parts))
        else:
            flat.append(column)
    return flat


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    """
    Execute the Gold KPI aggregation pipeline end to end.
    """
    print("[GOLD KPI] Starting...")
    print(f"[GOLD KPI] Reading Silver from MinIO s3://{SILVER_BUCKET}/{SILVER_KEY}")

    try:
        silver_path = f"s3://{SILVER_BUCKET}/{SILVER_KEY}"
        df = pd.read_parquet(silver_path, storage_options=S3_STORAGE_OPTIONS)
        print(f"[GOLD KPI] Loaded {len(df)} rows")
    except Exception as exc:
        fail(f"Failed to load Silver from MinIO: {exc}")

    # Required columns for aggregation and row counting.
    required = ["SourceImage_ID", "ingestion_timestamp", "id_measurement"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        fail(f"Missing required columns: {missing}")

    # Convert ingestion timestamps and derive a date-level bucket.
    df["ingestion_timestamp"] = pd.to_datetime(
        df["ingestion_timestamp"],
        errors="coerce",
        utc=True,
    )
    if df["ingestion_timestamp"].isna().any():
        fail("Invalid ingestion_timestamp values")

    df["date_bucket"] = df["ingestion_timestamp"].dt.date

    # Keep only numeric KPI columns that are present in the dataset.
    present_numeric = [column for column in NUMERIC_KPI_COLS if column in df.columns]
    if not present_numeric:
        fail("No KPI numeric columns found in Silver")

    # Aggregation plan:
    # - count records using id_measurement
    # - compute mean and standard deviation for numeric metrics
    # - compute mean only for data_quality_score
    agg_spec = {"id_measurement": "count"}
    for column in present_numeric:
        if column != "data_quality_score":
            agg_spec[column] = ["mean", "std"]
    if "data_quality_score" in present_numeric:
        agg_spec["data_quality_score"] = "mean"

    print(f"[GOLD KPI] Aggregating by {GROUP_KEYS}...")
    kpi = df.groupby(GROUP_KEYS).agg(agg_spec).reset_index()
    kpi.columns = flatten_columns(kpi.columns)

    # Rename selected output columns for readability.
    rename_map = {
        "id_measurement_count": "measurement_count",
        "data_quality_score_mean": "avg_quality_score",
    }
    kpi = kpi.rename(columns=rename_map)

    # Derived KPI: ratio between mean area and mean length.
    if "Area (mm2)_mean" in kpi.columns and "Length (mm)_mean" in kpi.columns:
        kpi["shape_index"] = (kpi["Area (mm2)_mean"] / kpi["Length (mm)_mean"]).round(4)

    # Derived KPI: categorical quality status.
    if "avg_quality_score" in kpi.columns:
        kpi["quality_status"] = kpi["avg_quality_score"].apply(
            lambda value: "all_pass"
            if value >= 95
            else ("with_warnings" if value >= 70 else "low_quality")
        )

    # Timestamp of the Gold snapshot.
    kpi["updated_timestamp"] = datetime.now(timezone.utc).isoformat()

    print(f"[GOLD KPI] Writing to MinIO s3://{GOLD_BUCKET}/{GOLD_KEY}")
    try:
        gold_path = f"s3://{GOLD_BUCKET}/{GOLD_KEY}"
        kpi.to_parquet(gold_path, index=False, storage_options=S3_STORAGE_OPTIONS)
    except Exception as exc:
        fail(f"Failed to save Gold to MinIO: {exc}")

    # Post-write verification ensures the output is readable and complete.
    try:
        verify = pd.read_parquet(gold_path, storage_options=S3_STORAGE_OPTIONS)
        print(f"[GOLD KPI] Saved to: {gold_path}")
        print(f"[GOLD KPI] Verification rows={len(verify)}, cols={len(verify.columns)}")
        print(verify.head(3).to_string())
        print("[GOLD KPI] COMPLETE")
    except Exception as exc:
        print(f"[GOLD KPI] Verification failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())