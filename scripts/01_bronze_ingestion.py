#!/usr/bin/env python3
# - Auto-discovers CSV/XLSX files in data dir
# - Loads each file with Spark, adds technical metadata
# - Writes a Parquet per crop under the Bronze path

"""
Bronze ingestion pipeline.

Auto-discovers CSV/XLSX files in the data directory, extracts crop name from
filename (e.g., barley.csv -> barley), and writes each to a separate S3 path.
"""

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================================
# Configuration
# ============================================================================
# (configuration constants are unchanged)

DATA_DIR = "/opt/spark-app/data"
VALID_EXTENSIONS = {".csv", ".xlsx", ".xls"}

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio_password")

BRONZE_BUCKET = os.getenv("BRONZE_BUCKET", "bronze")
BRONZE_BASE = f"s3a://{BRONZE_BUCKET}/videometer"

CSV_SEPARATOR = os.getenv("CSV_SEPARATOR", "")


# ============================================================================
# Helpers
# ============================================================================
# Small utility functions used by main()

def log(message: str) -> None:
    """Print a simple log message."""
    print(f"[BRONZE] {message}")


def fail(message: str) -> int:
    """Print an error and return a failure code."""
    print(f"[BRONZE] ERROR: {message}")
    return 1


def extract_crop_name(filename: str) -> str:
    """Extract crop name from filename: 'barley.csv' -> 'barley'."""
    return Path(filename).stem.lower()


def parse_args() -> argparse.Namespace:
    """Read the data directory from the command line."""
    parser = argparse.ArgumentParser(description="Bronze ingestion for Videometer data")
    parser.add_argument("--dir", dest="data_dir", default=DATA_DIR, help="Input data directory")
    return parser.parse_args()


def calculate_file_checksum(filepath: str) -> str:
    """Return an MD5 checksum for the source file."""
    md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                md5.update(chunk)
        return md5.hexdigest()
    except Exception as exc:
        log(f"WARNING: checksum skipped: {exc}")
        return "unknown"


def detect_separator(filepath: str) -> str:
    """Detect the CSV separator from the first line."""
    if CSV_SEPARATOR:
        return CSV_SEPARATOR

    separator = ","
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as stream:
            first_line = stream.readline()
            if "\t" in first_line:
                separator = "\t"
            elif ";" in first_line:
                separator = ";"
    except Exception as exc:
        log(f"WARNING: separator detection failed: {exc}")

    return separator


def create_spark_session() -> SparkSession:
    """Create the Spark session with MinIO settings."""
    return (
        SparkSession.builder
        .appName("bronze-ingestion")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.2")
        .config("spark.driver.memory", "3g")
        .config("spark.executor.memory", "3g")
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", True)
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )


def load_source_file(spark: SparkSession, file_path: str):
    """Load CSV or Excel as raw data."""
    lower = file_path.lower()

    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        import pandas as pd
        df_pandas = pd.read_excel(file_path, dtype=str).fillna("")
        df_pandas.columns = df_pandas.columns.astype(str)
        return spark.createDataFrame(df_pandas)

    separator = detect_separator(file_path)
    log(f"Detected CSV separator: '{separator}'")

    return (
        spark.read
        .option("header", True)
        .option("sep", separator)
        .option("inferSchema", False)
        .csv(file_path)
    )


# ============================================================================
# Main
# ============================================================================
# The main flow processes every valid file found under the data directory.

def main() -> int:
    """Run the Bronze pipeline end to end."""
    args = parse_args()
    data_path = Path(args.data_dir)

    log("Starting Bronze ingestion")

    if not data_path.exists() or not data_path.is_dir():
        return fail(f"Data directory not found: {args.data_dir}")

    # Find all valid files (barley.csv, chickpea.csv, etc.)
    files_to_process = sorted([
        f for f in data_path.iterdir() 
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ])

    if not files_to_process:
        return fail(f"No valid data files found in {args.data_dir}")

    log(f"Found {len(files_to_process)} files to process")
    for f in files_to_process:
        log(f"  - {f.name}")

    spark = create_spark_session()
    success_count = 0

    # Process each file separately by crop
    for file_path in files_to_process:
        crop_name = extract_crop_name(file_path.name)
        log(f"\n--- Processing {crop_name} from {file_path.name} ---")

        try:
            df = load_source_file(spark, str(file_path))
        except Exception as exc:
            log(f"WARNING: Failed to load {file_path.name}: {exc}")
            continue

        rows_loaded = df.count()
        cols_loaded = len(df.columns)

        if rows_loaded == 0:
            log(f"WARNING: Skipping {crop_name} (empty file)")
            continue

        log(f"Loaded rows={rows_loaded}, cols={cols_loaded}")

        # Add technical metadata
        run_time = datetime.now(timezone.utc)
        source_filename = file_path.name
        source_checksum = calculate_file_checksum(str(file_path))
        load_id = f"LOAD_{run_time.strftime('%Y%m%d_%H%M%S')}"

        df = df.withColumn("ingestion_date", F.lit(run_time.isoformat()))
        df = df.withColumn("ingestion_timestamp", F.current_timestamp())
        df = df.withColumn("source_filename", F.lit(source_filename))
        df = df.withColumn("source_checksum", F.lit(source_checksum))
        df = df.withColumn("load_id", F.lit(load_id))
        df = df.withColumn("crop_type", F.lit(crop_name))

        log("Technical metadata added")
        log(f"  crop_type={crop_name}")
        log(f"  source_filename={source_filename}")
        log(f"  load_id={load_id}")

        # Write Bronze for this crop
        bronze_path = f"{BRONZE_BASE}/{crop_name}/"
        log(f"Writing Bronze to {bronze_path}")
        try:
            df.write.mode("overwrite").parquet(bronze_path)
        except Exception as exc:
            log(f"ERROR writing {crop_name}: {exc}")
            continue

        # Verify the write
        try:
            verify = spark.read.parquet(bronze_path)
            rows_verify = verify.count()
            cols_verify = len(verify.columns)

            if rows_verify != rows_loaded:
                log(f"WARNING: Verification mismatch for {crop_name}: written={rows_loaded}, read_back={rows_verify}")
            else:
                log(f"✓ {crop_name} verified: rows={rows_verify}, cols={cols_verify}")
                success_count += 1
        except Exception as exc:
            log(f"WARNING: Failed to verify {crop_name}: {exc}")

    log(f"\nBronze ingestion COMPLETE: {success_count}/{len(files_to_process)} files processed successfully")
    spark.stop()
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())