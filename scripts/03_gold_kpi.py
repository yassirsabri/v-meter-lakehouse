#!/usr/bin/env python3
"""
Gold KPI aggregation pipeline.

Accepts --crop argument to read from crop-specific Silver and write to crop-specific Gold.
Groups by date and source, computes generic KPIs from numeric columns, and adds
shape_index and quality_status metrics.
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import List, Optional

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================================
# Configuration
# ============================================================================

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio_password")

SILVER_BUCKET = os.getenv("SILVER_BUCKET", "silver")
GOLD_BUCKET = os.getenv("GOLD_BUCKET", "gold")

SILVER_BASE = f"s3a://{SILVER_BUCKET}/videometer"
GOLD_BASE = f"s3a://{GOLD_BUCKET}/videometer"


# ============================================================================
# Helpers
# ============================================================================

def log(message: str) -> None:
    """Print a simple log message."""
    print(f"[GOLD] {message}")


def fail(message: str) -> int:
    """Print an error and return a failure code."""
    print(f"[GOLD] ERROR: {message}")
    return 1


def parse_args() -> argparse.Namespace:
    """Read the crop type from the command line."""
    parser = argparse.ArgumentParser(description="Gold KPI aggregation")
    parser.add_argument("--crop", dest="crop_type", default="barley",
                       help="Crop type (e.g., barley, chickpea)")
    return parser.parse_args()


def create_spark_session() -> SparkSession:
    """Create the Spark session with MinIO settings."""
    return (
        SparkSession.builder
        .appName("gold-kpi")
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


def first_existing(columns: List[str], candidates: List[str]) -> Optional[str]:
    """Return the first column that exists."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def detect_source_group_col(columns: List[str]) -> Optional[str]:
    """Find the best source column for grouping."""
    return first_existing(
        columns,
        [
            "source_image_id",
            "sourceimageid",
            "source_image_capture_id",
            "sourceimagecaptureid",
            "source_filename",
            "filename",
            "filepath",
        ],
    )


def detect_time_col(columns: List[str]) -> Optional[str]:
    """Find the best timestamp column."""
    return first_existing(
        columns,
        [
            "ingestion_timestamp",
            "processed_date",
            "ingestion_date",
        ],
    )


def numeric_columns(df) -> List[str]:
    """Return the numeric columns we can aggregate (core morphological features only)."""
    # Only allow core morphological features and quality metrics
    allowed_prefixes = (
        "area", "length", "width", "volume", "compactness",
        "perimeter", "eccentricity", "data_quality", "ratiowidthlength"
    )

    numeric_types = {"double", "float", "int", "bigint", "smallint", "tinyint", "long", "decimal"}
    cols = []
    
    for name, dtype in df.dtypes:
        if dtype in numeric_types and name.startswith(allowed_prefixes):
            # Exclude highly dimensional spectral bands
            if not ("_" in name and name.split("_")[-1].isdigit() and name != "area_mm2"):
                cols.append(name)
    
    return sorted(cols)


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    """Run the Gold KPI pipeline."""
    args = parse_args()
    crop_type = args.crop_type.lower()
    
    # Construct crop-specific paths
    SILVER_PATH = f"{SILVER_BASE}/{crop_type}/"
    GOLD_PATH = f"{GOLD_BASE}/{crop_type}/"
    
    log(f"Starting Gold KPI aggregation for crop: {crop_type}")
    log(f"Silver input: {SILVER_PATH}")
    log(f"Gold output: {GOLD_PATH}")

    spark = create_spark_session()

    # Read Silver.
    log(f"Reading Silver from {SILVER_PATH}")
    try:
        df = spark.read.parquet(SILVER_PATH)
    except Exception as exc:
        return fail(f"Failed to read Silver: {exc}")

    if df.count() == 0:
        return fail("Silver dataframe is empty")

    columns = df.columns
    source_col = detect_source_group_col(columns)
    time_col = detect_time_col(columns)

    log(f"Detected source column: {source_col}")
    log(f"Detected time column: {time_col}")

    if time_col is None:
        return fail("No timestamp column found in Silver")

    # Create a date bucket.
    if time_col == "ingestion_timestamp":
        df = df.withColumn("date_bucket", F.to_date(F.col("ingestion_timestamp")))
    else:
        df = df.withColumn("date_bucket", F.to_date(F.to_timestamp(F.col(time_col))))

    # Find numeric columns dynamically.
    kpi_numeric_cols = numeric_columns(df)
    if not kpi_numeric_cols:
        return fail("No numeric columns found for KPI aggregation")

    log(f"Numeric columns found: {len(kpi_numeric_cols)}")
    if kpi_numeric_cols:
        log(f"First numeric columns: {kpi_numeric_cols[:10]}")

    # Group keys.
    if source_col:
        group_cols = ["date_bucket", source_col]
    else:
        group_cols = ["date_bucket"]

    log(f"Grouping by: {group_cols}")

    # Build the aggregation.
    agg_spec = [F.count(F.lit(1)).alias("measurement_count")]

    for col_name in kpi_numeric_cols:
        agg_spec.append(F.mean(F.col(col_name)).alias(f"{col_name}_mean"))
        agg_spec.append(F.stddev(F.col(col_name)).alias(f"{col_name}_std"))

        if col_name in {"areamm2", "area_mm2", "lengthmm", "length_mm", "widthmm", "width_mm", "reflectancemean", "volume"}:
            agg_spec.append(F.expr(f"percentile_approx({col_name}, 0.5, 1000)").alias(f"{col_name}_p50"))

    if "data_quality_score" in columns:
        agg_spec.append(F.mean(F.col("data_quality_score")).alias("avg_quality_score"))

    kpi = df.groupBy(*group_cols).agg(*agg_spec)

    # Add a simple shape index when the columns exist.
    area_mean_cols = [c for c in kpi.columns if c.endswith("_mean") and "area" in c]
    length_mean_cols = [c for c in kpi.columns if c.endswith("_mean") and "length" in c]

    if area_mean_cols and length_mean_cols:
        kpi = kpi.withColumn(
            "shape_index",
            F.round(F.col(area_mean_cols[0]) / F.col(length_mean_cols[0]), 4)
        )

    # Add a simple quality status when the score exists.
    if "avg_quality_score" in kpi.columns:
        kpi = kpi.withColumn(
            "quality_status",
            F.when(F.col("avg_quality_score") >= 95, F.lit("all_pass"))
             .when(F.col("avg_quality_score") >= 70, F.lit("with_warnings"))
             .otherwise(F.lit("low_quality"))
        )

    # Add the processing timestamp.
    kpi = kpi.withColumn("updated_timestamp", F.current_timestamp())
    kpi = kpi.withColumn("crop_type", F.lit(crop_type))

    rows_out = kpi.count()
    cols_out = len(kpi.columns)
    log(f"Generated KPI rows={rows_out}, cols={cols_out}")

    # Write Gold.
    log(f"Writing Gold to {GOLD_PATH}")
    try:
        kpi.write.mode("overwrite").parquet(GOLD_PATH)
    except Exception as exc:
        return fail(f"Failed to write Gold: {exc}")

    # Verify the write.
    try:
        verify = spark.read.parquet(GOLD_PATH)
        rows_verify = verify.count()
        cols_verify = len(verify.columns)
        log(f"✓ Verification: rows={rows_verify}, cols={cols_verify}")
        log("COMPLETE")
        spark.stop()
        return 0
    except Exception as exc:
        return fail(f"Failed to verify Gold write: {exc}")


if __name__ == "__main__":
    sys.exit(main())