#!/usr/bin/env python3
"""
Silver transformation pipeline.

Purpose
- Read Bronze data from MinIO.
- Validate mandatory fields and ingestion timestamps.
- Normalize and validate the composite business key.
- Deduplicate records using the latest ingestion timestamp.
- Cast numeric measurement columns.
- Apply basic sanity rules and compute data quality scores.
- Write the curated dataset to the Silver zone in MinIO.

Why this stage matters
- Silver is the cleaned and trusted layer used for downstream analytics.
- It removes duplicates, enforces structure, and prepares data for aggregation.
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

BRONZE_BUCKET = os.getenv("BRONZE_BUCKET", "bronze")
BRONZE_KEY = os.getenv("BRONZE_KEY", "videometer/raw_measurements")
SILVER_BUCKET = os.getenv("SILVER_BUCKET", "silver")
SILVER_KEY = os.getenv("SILVER_KEY", "videometer/cleaned_measurements")

S3_STORAGE_OPTIONS = {
    "key": MINIO_ACCESS_KEY,
    "secret": MINIO_SECRET_KEY,
    "client_kwargs": {"endpoint_url": MINIO_ENDPOINT},
    "anon": False,
}

# Composite business key used to identify a unique measurement.
BUSINESS_KEY_COLUMNS = [
    "SourceImage_ID",
    "SourceImage_CaptureId",
    "Filename",
    "BlobId",
]

# Measurement columns that should be numeric in Silver.
TARGET_NUMERIC_COLUMNS = [
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
]

# Fields that must remain strictly positive.
POSITIVE_COLUMNS = [
    "Area (mm2)",
    "Length (mm)",
    "Width (mm)",
    "Perimeter",
    "Volume",
]


# ============================================================================
# Helper functions
# ============================================================================

def fail(message: str) -> None:
    """Print an error message and stop execution."""
    print(f"[SILVER] ERROR: {message}")
    sys.exit(1)


def get_s3_fs():
    """
    Initialize an authenticated S3 filesystem client for MinIO.

    This helper is kept for compatibility with environments that require
    an explicit filesystem object.
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


def ensure_required_columns(df: pd.DataFrame) -> None:
    """
    Verify that the minimum required columns exist before processing.
    """
    required = BUSINESS_KEY_COLUMNS + ["ingestion_date"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        fail(f"Missing required columns: {missing}")


def parse_ingestion_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse ingestion_date into a UTC timestamp.

    This is a blocking validation because deduplication depends on temporal order.
    """
    df["ingestion_timestamp"] = pd.to_datetime(
        df["ingestion_date"],
        errors="coerce",
        utc=True,
    )
    bad_count = int(df["ingestion_timestamp"].isna().sum())
    if bad_count > 0:
        fail(f"{bad_count} rows have invalid ingestion_date")
    return df


def normalize_business_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize, validate, and deduplicate the composite business key.

    Processing rules:
    - Trim whitespace from key fields.
    - Reject null or empty key values.
    - Deduplicate by keeping the latest record for each composite key.
    - Build a stable technical id_measurement identifier.
    """
    for column in BUSINESS_KEY_COLUMNS:
        df[column] = df[column].astype("string").str.strip()

    for column in BUSINESS_KEY_COLUMNS:
        bad_count = int(df[column].isna().sum() + (df[column] == "").sum())
        if bad_count > 0:
            fail(f"{bad_count} rows have null/empty {column} in business key")

    duplicate_count = int(df.duplicated(subset=BUSINESS_KEY_COLUMNS).sum())
    if duplicate_count > 0:
        print(f"[SILVER] WARNING: {duplicate_count} duplicates on composite key, keeping latest row")
        df = df.sort_values("ingestion_timestamp").drop_duplicates(
            subset=BUSINESS_KEY_COLUMNS,
            keep="last",
        )

    df["id_measurement"] = (
        df["SourceImage_ID"].astype("string")
        + "|"
        + df["SourceImage_CaptureId"].astype("string")
        + "|"
        + df["Filename"].astype("string")
        + "|"
        + df["BlobId"].astype("string")
    )

    return df


def cast_numeric_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Cast the target measurement columns to numeric dtype.

    Invalid values are coerced to NaN so they can be handled by quality checks.
    """
    present = []
    for column in TARGET_NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            present.append(column)
    return df, present


def apply_sanity_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-specific sanity checks.

    Values less than or equal to zero in strictly positive fields are replaced
    with NaN because they are considered invalid observations.
    """
    for column in POSITIVE_COLUMNS:
        if column in df.columns:
            invalid_mask = df[column] <= 0
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                print(f"[SILVER] WARNING: {invalid_count} invalid values in {column} (<=0), set to NaN")
                df.loc[invalid_mask, column] = pd.NA
    return df


def compute_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a row-level quality score and a categorical quality status.

    The score is based on missing values across a small set of core columns.
    """
    core_columns = [
        column
        for column in ["Area (mm2)", "Length (mm)", "Width (mm)", "ReflectanceMean", "Volume"]
        if column in df.columns
    ]

    if core_columns:
        null_ratio = df[core_columns].isnull().mean(axis=1)
        df["data_quality_score"] = (100.0 - (null_ratio * 100.0)).round(2)
    else:
        df["data_quality_score"] = 100.0

    def status(score: float) -> str:
        if score >= 95:
            return "all_pass"
        if score >= 70:
            return "with_warnings"
        return "low_quality"

    df["dq_check_status"] = df["data_quality_score"].apply(status)
    return df


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    """
    Execute the Silver transformation pipeline end to end.
    """
    print("[SILVER] Starting...")
    print(f"[SILVER] Reading Bronze from MinIO s3://{BRONZE_BUCKET}/{BRONZE_KEY}")

    try:
        bronze_path = f"s3://{BRONZE_BUCKET}/{BRONZE_KEY}"
        df = pd.read_parquet(bronze_path, storage_options=S3_STORAGE_OPTIONS)
    except Exception as exc:
        fail(f"Failed to read Bronze from MinIO: {exc}")

    if df.empty:
        fail("Bronze dataframe is empty")

    print(f"[SILVER] Loaded rows={len(df)}, cols={len(df.columns)}")

    ensure_required_columns(df)
    df = parse_ingestion_timestamp(df)
    df = normalize_business_key(df)

    df, numeric_columns_present = cast_numeric_columns(df)
    print(f"[SILVER] Numeric columns casted: {len(numeric_columns_present)}")

    df = apply_sanity_rules(df)

    # Processing timestamp indicates when Silver curation occurred.
    df["processed_date"] = datetime.now(timezone.utc).isoformat()
    df = compute_quality_score(df)

    print(f"[SILVER] Writing to MinIO s3://{SILVER_BUCKET}/{SILVER_KEY}")
    try:
        silver_path = f"s3://{SILVER_BUCKET}/{SILVER_KEY}"
        df.to_parquet(silver_path, index=False, storage_options=S3_STORAGE_OPTIONS)
    except Exception as exc:
        fail(f"Failed to save Silver to MinIO: {exc}")

    try:
        verify = pd.read_parquet(silver_path, storage_options=S3_STORAGE_OPTIONS)
        print(f"[SILVER] Saved to: {silver_path}")
        print(f"[SILVER] Verification rows={len(verify)}, cols={len(verify.columns)}")
        print(f"[SILVER] Total null cells={int(verify.isnull().sum().sum())}")
        print("[SILVER] COMPLETE")
    except Exception as exc:
        print(f"[SILVER] Verification failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())