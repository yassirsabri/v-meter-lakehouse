#!/usr/bin/env python3
"""
Bronze ingestion pipeline.

Purpose
- Read the raw source file from the mounted data directory.
- Preserve the data with minimal transformation.
- Add technical metadata for lineage and traceability.
- Write the result as Parquet to the Bronze zone in MinIO.

Why this stage matters
- Bronze is the raw landing layer of the medallion architecture.
- It keeps the original source values as intact as possible.
- Metadata fields help identify when, how, and from which file the data was loaded.
"""

import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

DATA_SOURCE_PATH = os.getenv(
    "DATA_SOURCE_PATH",
    "/opt/spark-app/data/data_barley_videometer.csv",
)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio_password")

BRONZE_BUCKET = os.getenv("BRONZE_BUCKET", "bronze")
BRONZE_KEY = os.getenv("BRONZE_KEY", "videometer/raw_measurements")
BRONZE_S3_PATH = f"s3://{BRONZE_BUCKET}/{BRONZE_KEY}"

# Explicit S3 options are required to avoid anonymous access attempts.
S3_STORAGE_OPTIONS = {
    "key": MINIO_ACCESS_KEY,
    "secret": MINIO_SECRET_KEY,
    "client_kwargs": {"endpoint_url": MINIO_ENDPOINT},
    "anon": False,
}


# ============================================================================
# Helper functions
# ============================================================================

def log(message: str) -> None:
    """Print a standard log message for this stage."""
    print(f"[BRONZE] {message}")


def fail(message: str) -> int:
    """Print an error message and return a non-zero exit code."""
    print(f"[BRONZE] ERROR: {message}")
    return 1


def calculate_file_checksum(filepath: str) -> str:
    """
    Compute an MD5 checksum for the source file.

    The checksum is stored as metadata to support traceability and to detect
    whether the source file changed between ingestion runs.
    """
    md5 = hashlib.md5()
    with open(filepath, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def load_source_file(filepath: str) -> pd.DataFrame:
    """
    Load the source file as strings to preserve raw values.

    Bronze should avoid semantic casting so the source data remains close to
    its original representation.
    """
    lower = filepath.lower()
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(filepath, dtype=str)
    return pd.read_csv(filepath, dtype=str, low_memory=False)


def print_null_report(df: pd.DataFrame) -> None:
    """
    Print a non-blocking null-value report.

    This step is informational only. Bronze ingestion should continue even if
    some source columns contain missing values.
    """
    total_rows = len(df)
    log(f"Quality report: total_rows={total_rows}")
    for column in df.columns:
        null_count = int(df[column].isna().sum())
        if null_count > 0:
            percentage = (100.0 * null_count / total_rows) if total_rows else 0.0
            log(f"  NULLS {column}: {null_count} ({percentage:.2f}%)")


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    """
    Execute Bronze ingestion end to end.

    Steps:
    1. Validate that the source file exists.
    2. Load the raw dataset.
    3. Add technical metadata columns.
    4. Print a null-value report.
    5. Normalize object columns to pandas string dtype.
    6. Write the result to MinIO as Parquet.
    7. Read the data back to verify the write.
    """
    log("Starting Bronze ingestion")

    source_path = Path(DATA_SOURCE_PATH)
    if not source_path.exists():
        return fail(f"Source file not found: {DATA_SOURCE_PATH}")

    log(f"Source file: {DATA_SOURCE_PATH}")

    try:
        df = load_source_file(DATA_SOURCE_PATH)
    except Exception as exc:
        return fail(f"Failed to load source file: {exc}")

    if df.empty:
        return fail("Source dataframe is empty; aborting Bronze write")

    log(f"Loaded rows={len(df)}, cols={len(df.columns)}")

    try:
        ingestion_timestamp = datetime.now(timezone.utc).isoformat()
        source_filename = source_path.name
        source_checksum = calculate_file_checksum(DATA_SOURCE_PATH)
        load_id = f"LOAD_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        df["ingestion_date"] = ingestion_timestamp
        df["source_filename"] = source_filename
        df["source_checksum"] = source_checksum
        df["load_id"] = load_id

        log("Technical metadata added")
        log(f"  ingestion_date={ingestion_timestamp}")
        log(f"  source_filename={source_filename}")
        log(f"  source_checksum={source_checksum}")
        log(f"  load_id={load_id}")
    except Exception as exc:
        return fail(f"Failed to add metadata: {exc}")

    print_null_report(df)

    try:
        object_columns = df.select_dtypes(include=["object"]).columns
        for column in object_columns:
            df[column] = df[column].astype("string")
    except Exception as exc:
        return fail(f"Failed to normalize object columns: {exc}")

    log(f"Writing Bronze to {BRONZE_S3_PATH}")
    try:
        df.to_parquet(BRONZE_S3_PATH, index=False, storage_options=S3_STORAGE_OPTIONS)
    except Exception as exc:
        return fail(f"Failed to write Bronze parquet: {exc}")

    try:
        verify = pd.read_parquet(BRONZE_S3_PATH, storage_options=S3_STORAGE_OPTIONS)
    except Exception as exc:
        return fail(f"Failed to verify Bronze read-back: {exc}")

    if len(verify) != len(df):
        return fail(
            f"Verification mismatch rows: written={len(df)} read_back={len(verify)}"
        )

    log(f"Success: verified rows={len(verify)}, cols={len(verify.columns)}")
    log("COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())