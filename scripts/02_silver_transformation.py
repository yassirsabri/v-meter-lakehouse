#!/usr/bin/env python3
# - Read Bronze Parquet per crop, clean headers, infer numeric cols (sample-based)
# - Remove duplicate columns/rows, apply Tukey filters, compute quality score
# - Write cleaned Silver Parquet and metadata to MinIO/local fallback

"""
Silver transformation pipeline.

Accepts --crop argument to read from crop-specific Bronze and write to crop-specific Silver.
Cleans headers, detects key columns, builds id_measurement, removes duplicates,
applies Tukey outlier filters, and computes quality metrics.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType


# ============================================================================
# Configuration
# ============================================================================
# Environment-driven configuration values used by Spark sessions and paths.

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio_password")

BRONZE_BUCKET = os.getenv("BRONZE_BUCKET", "bronze")
SILVER_BUCKET = os.getenv("SILVER_BUCKET", "silver")

BRONZE_BASE = f"s3a://{BRONZE_BUCKET}/videometer"
SILVER_BASE = f"s3a://{SILVER_BUCKET}/videometer"

LOCAL_METADATA_PATH = "/tmp/silver_metadata.json"


# ============================================================================
# Helpers
# ============================================================================
# Utility helpers: logging, argument parsing, Spark session creation, header normalization, etc.

def log(message: str) -> None:
    """Print a simple log message."""
    print(f"[SILVER] {message}")


def fail(message: str) -> int:
    """Print an error and return a failure code."""
    print(f"[SILVER] ERROR: {message}")
    return 1


def parse_args() -> argparse.Namespace:
    """Read the crop type from the command line."""
    parser = argparse.ArgumentParser(description="Silver transformation")
    parser.add_argument("--crop", dest="crop_type", default="barley",
                       help="Crop type (e.g., barley, chickpea)")
    return parser.parse_args()


def create_spark_session() -> SparkSession:
    """Create the Spark session with MinIO, Iceberg, and Nessie catalog settings."""
    NESSIE_URI = os.getenv("NESSIE_URI", "http://nessie:19120/api/v1")
    WAREHOUSE_PATH = f"s3a://{os.getenv('BRONZE_BUCKET', 'bronze')}/warehouse"
    
    return (
        SparkSession.builder
        .appName("lakehouse-pipeline")
        # --- Packages ---
        .config("spark.jars.packages", 
                "org.apache.hadoop:hadoop-aws:3.3.4,"
                "org.apache.iceberg:iceberg-spark-runtime-3.3_2.12:1.4.3,"
                "org.projectnessie.nessie-integrations:nessie-spark-extensions-3.3_2.12:0.76.6")
        
        # --- Extensions Iceberg/Nessie ---
        .config("spark.sql.extensions", 
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions,"
                "org.projectnessie.spark.extensions.NessieSparkSessionExtensions")
        
        # --- Configuration S3 (MinIO) ---
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        
        # --- Configuration du Catalogue Nessie ---
        .config("spark.sql.catalog.nessie", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.nessie.catalog-impl", "org.apache.iceberg.nessie.NessieCatalog")
        .config("spark.sql.catalog.nessie.uri", NESSIE_URI)
        .config("spark.sql.catalog.nessie.ref", "main")
        .config("spark.sql.catalog.nessie.authentication.type", "NONE")
        .config("spark.sql.catalog.nessie.warehouse", WAREHOUSE_PATH)
        .config("spark.sql.catalog.nessie.s3.endpoint", MINIO_ENDPOINT)
        
        # --- Optimisations ---
        .config("spark.driver.memory", "3g")
        .config("spark.executor.memory", "3g")
        .getOrCreate()
    )


def clean_name(name: str) -> str:
    """Clean a header in a readable way, keeping underscores for readability."""
    value = str(name).strip().lower()
    value = re.sub(r"\(unknown\)", "", value, flags=re.IGNORECASE)
    value = value.replace("%", "pct")
    value = re.sub(r"\[([0-9]+)\]", r"_\1", value)
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "col"


def normalize_headers(df: DataFrame) -> Tuple[DataFrame, Dict[str, str]]:
    """Rename all columns and keep a mapping."""
    mapping: Dict[str, str] = {}
    final_names: List[str] = []
    seen: Dict[str, int] = {}

    for original in df.columns:
        base = clean_name(original)
        count = seen.get(base, 0)
        final_name = base if count == 0 else f"{base}_{count}"
        seen[base] = count + 1
        mapping[original] = final_name
        final_names.append(final_name)

    return df.toDF(*final_names), mapping


def first_existing(columns: List[str], candidates: List[str]) -> Optional[str]:
    """Return the first matching column."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def detect_blob_cols(columns: List[str]) -> List[str]:
    """Find all blob id columns."""
    candidates = ["blobid", "blob_id", "blobindex", "blob_idx", "blob"]
    return [c for c in candidates if c in columns]


def detect_source_cols(columns: List[str]) -> List[str]:
    """Find all source image columns."""
    candidates = [
        "sourceimageid", "source_image_id", "sourceimagecaptureid",
        "source_image_capture_id", "sourceid", "filename", "filepath", "sourcefilename"
    ]
    return [c for c in candidates if c in columns]


def detect_area_cols(columns: List[str]) -> List[str]:
    """Find all area columns."""
    candidates = ["areamm2", "area_mm2", "area", "morphologyarea"]
    return [c for c in candidates if c in columns]


def infer_numeric_columns(df: DataFrame, exclude_cols: Optional[List[str]] = None, 
                          sample_ratio: float = 0.1, numeric_threshold: float = 0.95) -> Tuple[List[str], List[str]]:
    """
    Infer numeric columns using sample-based detection.
    Returns (numeric_cols, text_cols).
    """
    if exclude_cols is None:
        exclude_cols = []
    
    string_cols = [c for c, t in df.dtypes if c not in exclude_cols and t == "string"]
    
    if not string_cols:
        numeric_cols = [c for c, t in df.dtypes if c not in exclude_cols and t != "string"]
        return numeric_cols, []

    # Sample-based detection to avoid full scans per column
    sample_df = df.select(string_cols).sample(withReplacement=False, fraction=sample_ratio, seed=42)
    pandas_sample = sample_df.toPandas()

    numeric_cols = []
    for col in string_cols:
        s = pandas_sample[col].dropna()
        if s.empty:
            continue
        converted = pd.to_numeric(s, errors="coerce")
        ratio = converted.notna().sum() / float(len(s))
        if ratio >= numeric_threshold:
            numeric_cols.append(col)

    # Add non-string typed numeric columns
    for c, t in df.dtypes:
        if c not in exclude_cols and t != "string" and t not in {"array", "struct", "binary"}:
            numeric_cols.append(c)

    text_cols = [c for c in string_cols if c not in numeric_cols]
    
    return sorted(numeric_cols), sorted(text_cols)


def build_measurement_id(df: DataFrame, blob_cols: List[str], source_cols: List[str]) -> DataFrame:
    """Build the measurement id from blob and source columns."""
    df_work = df

    if blob_cols:
        blob_expr = F.coalesce(*[F.col(c).cast(StringType()) for c in blob_cols])
        df_work = df_work.withColumn("blob_id", F.trim(blob_expr))
    else:
        df_work = df_work.withColumn("blob_id", F.lit("unknown_blob"))

    if source_cols:
        source_expr = F.coalesce(*[F.col(c).cast(StringType()) for c in source_cols])
        df_work = df_work.withColumn("source_image_id", F.trim(source_expr))
    else:
        df_work = df_work.withColumn("source_image_id", F.lit("unknown_source"))

    return df_work.withColumn(
        "id_measurement",
        F.concat_ws("|", F.col("blob_id"), F.col("source_image_id"))
    )


def drop_duplicate_columns(df: DataFrame) -> Tuple[DataFrame, List[str]]:
    """
    Drop duplicate columns using sample-based candidate detection + full verification.
    Safe for timestamp columns by casting to string before pandas conversion.
    """
    sample = df.limit(5)
    
    # Cast all columns to string to avoid pandas datetime conversion issues
    for c in sample.columns:
        sample = sample.withColumn(c, sample[c].cast(StringType()))
    
    pdf = sample.toPandas()
    signatures = {}
    
    for col in pdf.columns:
        sig = tuple(map(lambda x: None if pd.isna(x) else str(x), pdf[col].tolist()))
        signatures.setdefault(sig, []).append(col)

    cols_to_drop = []
    for sig, cols in signatures.items():
        if len(cols) <= 1:
            continue
        
        # Verify duplicates across full column using eqNullSafe on master
        master = cols[0]
        for other in cols[1:]:
            diff_count = df.filter(~(F.col(master).eqNullSafe(F.col(other)))).limit(1).count()
            if diff_count == 0:
                cols_to_drop.append(other)
    
    result_df = df.drop(*cols_to_drop) if cols_to_drop else df
    return result_df, cols_to_drop


def apply_area_rule(df: DataFrame, area_cols: List[str]) -> Tuple[DataFrame, int]:
    """Remove rows where area is less than or equal to zero."""
    existing_area_cols = [c for c in area_cols if c in df.columns]
    if not existing_area_cols:
        return df, 0

    before = df.count()
    area_expr = F.coalesce(*[F.col(c).cast(DoubleType()) for c in existing_area_cols])
    df_work = df.withColumn("__area_numeric", area_expr)
    df_work = df_work.filter(F.col("__area_numeric") > 0).drop("__area_numeric")
    after = df_work.count()

    return df_work, before - after


def deduplicate_on_id(df: DataFrame) -> Tuple[DataFrame, int]:
    """Keep only one row per id_measurement (most recent by ingestion_timestamp)."""
    if "id_measurement" not in df.columns:
        return df, 0

    if "ingestion_timestamp" in df.columns:
        order_col = F.col("ingestion_timestamp").desc_nulls_last()
    else:
        order_col = F.lit(1).desc()

    window_spec = Window.partitionBy("id_measurement").orderBy(order_col)
    df_ranked = df.withColumn("__row_rank", F.row_number().over(window_spec))

    before = df.count()
    df_clean = df_ranked.filter(F.col("__row_rank") == 1).drop("__row_rank")
    after = df_clean.count()

    return df_clean, before - after


def apply_tukey_filter(df: DataFrame, column_name: str, rel_error: float = 0.01) -> Tuple[DataFrame, Dict[str, Any]]:
    """Apply Tukey IQR outlier filter on a single column."""
    if column_name not in df.columns:
        return df, {
            "column": column_name,
            "applied": False,
            "reason": "missing_column",
            "removed": 0,
            "lower_fence": None,
            "upper_fence": None,
        }

    df_work = df.withColumn("__tukey_numeric", F.col(column_name).cast(DoubleType()))
    non_null_count = df_work.filter(F.col("__tukey_numeric").isNotNull()).limit(1).count()

    if non_null_count == 0:
        return df, {
            "column": column_name,
            "applied": False,
            "reason": "not_numeric",
            "removed": 0,
            "lower_fence": None,
            "upper_fence": None,
        }

    quantiles = df_work.approxQuantile("__tukey_numeric", [0.25, 0.75], rel_error)
    if len(quantiles) != 2 or quantiles[0] is None or quantiles[1] is None:
        return df, {
            "column": column_name,
            "applied": False,
            "reason": "quantile_unavailable",
            "removed": 0,
            "lower_fence": None,
            "upper_fence": None,
        }

    q1 = float(quantiles[0])
    q3 = float(quantiles[1])
    iqr = q3 - q1
    lower_fence = q1 - (1.5 * iqr)
    upper_fence = q3 + (1.5 * iqr)

    before = df.count()
    df_clean = (
        df_work
        .filter((F.col("__tukey_numeric") >= lower_fence) & (F.col("__tukey_numeric") <= upper_fence))
        .drop("__tukey_numeric")
    )
    after = df_clean.count()

    return df_clean, {
        "column": column_name,
        "applied": True,
        "reason": "ok",
        "removed": int(before - after),
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
    }


def pick_tukey_targets(columns: List[str]) -> List[str]:
    """Pick the notebook outlier columns if they exist."""
    targets: List[str] = []

    for candidate in ["areamm2", "area_mm2", "lengthmm", "length_mm", "widthmm", "width_mm"]:
        if candidate in columns and candidate not in targets:
            targets.append(candidate)

    spectral = first_existing(columns, ["multicolormean9", "spectralstatistics9"])
    if spectral is None:
        for col_name in columns:
            if ("multicolor" in col_name or "spectral" in col_name) and col_name.endswith("9"):
                spectral = col_name
                break

    if spectral and spectral not in targets:
        targets.append(spectral)

    return targets


def apply_eda_outlier_filters(df: DataFrame) -> Tuple[DataFrame, List[Dict[str, Any]]]:
    """Apply notebook Tukey logic on the right columns."""
    reports: List[Dict[str, Any]] = []
    targets = pick_tukey_targets(df.columns)

    for column_name in targets:
        df, report = apply_tukey_filter(df, column_name)
        reports.append(report)
        if report.get("applied"):
            log(
                f"Tukey on {column_name}: removed={report['removed']}, "
                f"fence=[{report['lower_fence']:.4f}, {report['upper_fence']:.4f}]"
            )

    return df, reports


def compute_quality_score(df: DataFrame, core_cols: List[str]) -> DataFrame:
    """Compute a simple quality score from null values."""
    valid_cols = [c for c in core_cols if c in df.columns]

    if not valid_cols:
        return (
            df.withColumn("data_quality_score", F.lit(100.0))
              .withColumn("dq_check_status", F.lit("all_pass"))
        )

    null_expr = None
    for col_name in valid_cols:
        one_null = F.when(F.col(col_name).isNull(), F.lit(1)).otherwise(F.lit(0))
        null_expr = one_null if null_expr is None else (null_expr + one_null)

    total_cols = float(len(valid_cols))
    score_expr = F.lit(100.0) - ((null_expr.cast(DoubleType()) / F.lit(total_cols)) * F.lit(100.0))

    return (
        df.withColumn("data_quality_score", F.round(score_expr, 2))
          .withColumn(
              "dq_check_status",
              F.when(F.col("data_quality_score") >= 95, F.lit("all_pass"))
               .when(F.col("data_quality_score") >= 70, F.lit("with_warnings"))
               .otherwise(F.lit("low_quality"))
          )
    )


def build_pattern_groups(columns: List[str], numeric_cols: List[str]) -> Dict[str, Any]:
    """Group numeric columns into spectral and morphology lists."""
    spectral_keywords = [
        "multicolor", "spectral", "reflectance", "transmittance", "absorbance",
        "cie", "cielab", "srgb", "ihs", "colorband", "successivebanddiff",
    ]

    spectral_cols: List[str] = []
    morphology_cols: List[str] = []
    indexed_groups: Dict[str, List[str]] = {}

    for col_name in numeric_cols:
        match = re.match(r"^(.*)_([0-9]+)$", col_name)
        if match:
            base = match.group(1)
            indexed_groups.setdefault(base, []).append(col_name)

        if any(keyword in col_name for keyword in spectral_keywords):
            spectral_cols.append(col_name)
        else:
            morphology_cols.append(col_name)

    for key in indexed_groups:
        indexed_groups[key] = sorted(indexed_groups[key])

    return {
        "spectral_cols": sorted(spectral_cols),
        "morphology_cols": sorted(morphology_cols),
        "indexed_groups": indexed_groups,
    }


def write_metadata_to_minio(spark: SparkSession, metadata: Dict[str, Any], target_path: str) -> None:
    """Write the metadata JSON to MinIO as a small file."""
    metadata_df = spark.createDataFrame(
        [(json.dumps(metadata, ensure_ascii=False, indent=2),)],
        ["json_text"],
    )
    metadata_df.coalesce(1).write.mode("overwrite").text(target_path)


# ============================================================================
# Main
# ============================================================================
# Main flow: read Bronze, process, write Silver, save metadata.

def main() -> int:
    """Run the Silver pipeline."""
    args = parse_args()
    crop_type = args.crop_type.lower()
    
    # Construct crop-specific paths
    BRONZE_PATH = f"{BRONZE_BASE}/{crop_type}/"
    SILVER_PATH = f"{SILVER_BASE}/{crop_type}/"
    
    log(f"Starting Silver transformation for crop: {crop_type}")
    log(f"Bronze input: {BRONZE_PATH}")
    log(f"Silver output: {SILVER_PATH}")

    spark = create_spark_session()

    # Read Bronze.
    log(f"Reading Bronze from {BRONZE_PATH}")
    try:
        df_bronze = spark.read.format("iceberg").load(f"nessie.bronze.{crop_type}")
    except Exception as exc:
        return fail(f"Failed to read Bronze: {exc}")

    rows_bronze = df_bronze.count()
    cols_bronze = len(df_bronze.columns)

    if rows_bronze == 0:
        return fail("Bronze dataframe is empty")

    log(f"Bronze loaded: rows={rows_bronze}, cols={cols_bronze}")

    # Cache the dataframe to avoid re-reading
    df_bronze = df_bronze.persist(StorageLevel.MEMORY_AND_DISK)

    # Clean the headers.
    log("Normalizing headers")
    df, header_mapping = normalize_headers(df_bronze)
    log(f"Headers normalized: {len(header_mapping)} columns")

    # Ensure time columns exist.
    if "ingestion_date" not in df.columns:
        df = df.withColumn("ingestion_date", F.lit(datetime.now(timezone.utc).isoformat()))

    if "ingestion_timestamp" not in df.columns:
        df = df.withColumn("ingestion_timestamp", F.current_timestamp())
    else:
        df = df.withColumn("ingestion_timestamp", 
                          F.coalesce(F.to_timestamp(F.col("ingestion_timestamp")), F.current_timestamp()))

    # Find the important columns.
    columns = df.columns
    blob_cols = detect_blob_cols(columns)
    source_cols = detect_source_cols(columns)
    area_cols = detect_area_cols(columns)

    log(f"Detected blob columns: {blob_cols}")
    log(f"Detected source columns: {source_cols}")
    log(f"Detected area columns: {area_cols}")

    if not blob_cols or not source_cols:
        return fail(f"Minimal contract violated: blob_cols={blob_cols}, source_cols={source_cols}")

    if not area_cols:
        return fail("Minimal contract violated: area columns not found")

    # Build the measurement id.
    log("Building id_measurement")
    df = build_measurement_id(df, blob_cols, source_cols)

    # Detect numeric columns using sample-based approach (much faster).
    log("Detecting numeric columns (sample-based)")
    exclude_set = {
        "blob_id", "source_image_id", "id_measurement", "ingestion_date",
        "ingestion_timestamp", "processed_date", "source_filename",
        "source_checksum", "load_id", "crop_type",
    }
    numeric_cols, text_cols = infer_numeric_columns(df, exclude_set, sample_ratio=0.1, numeric_threshold=0.90)
    log(f"Numeric columns: {len(numeric_cols)}")
    log(f"Text columns: {len(text_cols)}")

    # Drop duplicated columns (safe for timestamps).
    log("Dropping duplicated columns")
    df, duplicate_columns_dropped = drop_duplicate_columns(df)
    log(f"Duplicate columns dropped: {len(duplicate_columns_dropped)}")

    # Apply the area rule.
    log("Applying area rule")
    df, area_removed = apply_area_rule(df, area_cols)
    log(f"Rows removed by area rule: {area_removed}")

    # Drop duplicated rows.
    log("Dropping duplicated rows")
    df, duplicate_rows_removed = deduplicate_on_id(df)
    log(f"Rows removed by id_measurement deduplication: {duplicate_rows_removed}")

    # Apply the notebook Tukey logic.
    log("Applying Tukey filters")
    df, tukey_reports = apply_eda_outlier_filters(df)
    tukey_removed = sum(int(r.get("removed", 0)) for r in tukey_reports if r.get("applied"))
    log(f"Rows removed by Tukey filters: {tukey_removed}")

    # Compute a simple quality score.
    core_candidates = [
        "areamm2", "area_mm2", "lengthmm", "length_mm", "widthmm", "width_mm",
        "reflectancemean", "volume",
    ]
    df = compute_quality_score(df, core_candidates)

    # Add the processing timestamp.
    df = df.withColumn("processed_date", F.current_timestamp())

    rows_final = df.count()
    cols_final = len(df.columns)
    log(f"Final dataset: rows={rows_final}, cols={cols_final}")

    # Build pattern groups for the next stage.
    pattern_groups = build_pattern_groups(df.columns, numeric_cols)
    log(f"Spectral columns: {len(pattern_groups['spectral_cols'])}")
    log(f"Morphology columns: {len(pattern_groups['morphology_cols'])}")

    # Write Silver.
    # Write Silver to Nessie / Iceberg
    log("Création du namespace (base de données) dans Nessie si inexistant")
    spark.sql("CREATE NAMESPACE IF NOT EXISTS nessie.silver")
    
    table_name = f"nessie.silver.{crop_type}"
    log(f"Writing Iceberg table: {table_name}")
    try:
        df.write.format("iceberg").mode("overwrite").saveAsTable(table_name)
    except Exception as exc:
        return fail(f"Failed to write Silver to Nessie: {exc}")

    # Save metadata in MinIO and keep a local fallback.
    metadata = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "crop_type": crop_type,
        "bronze_path": BRONZE_PATH,
        "silver_path": SILVER_PATH,
        "raw_shape": {"rows": rows_bronze, "cols": cols_bronze},
        "final_shape": {"rows": rows_final, "cols": cols_final},
        "header_mapping": header_mapping,
        "contract": {
            "blob_cols": blob_cols,
            "source_cols": source_cols,
            "area_cols": area_cols,
        },
        "quality": {
            "area_rows_removed": area_removed,
            "duplicate_rows_removed": duplicate_rows_removed,
            "duplicate_columns_dropped": duplicate_columns_dropped,
            "tukey_rows_removed": tukey_removed,
            "tukey_reports": tukey_reports,
        },
        "typing": {
            "numeric_count": len(numeric_cols),
            "text_count": len(text_cols),
            "numeric_cols": sorted(numeric_cols),
            "text_cols": sorted(text_cols),
        },
        "groups": {
            "spectral_count": len(pattern_groups["spectral_cols"]),
            "morphology_count": len(pattern_groups["morphology_cols"]),
            "indexed_group_count": len(pattern_groups["indexed_groups"]),
            "spectral_cols": pattern_groups["spectral_cols"],
            "morphology_cols": pattern_groups["morphology_cols"],
            "indexed_groups": pattern_groups["indexed_groups"],
        },
    }

    try:
        write_metadata_to_minio(
            spark,
            metadata,
            f"s3a://{SILVER_BUCKET}/videometer/{crop_type}/metadata/silver_metadata.json",
        )
        log(f"Metadata saved in MinIO for {crop_type}")
    except Exception as exc:
        log(f"WARNING: metadata upload failed, using local fallback: {exc}")
        try:
            with open(LOCAL_METADATA_PATH, "w", encoding="utf-8") as stream:
                json.dump(metadata, stream, ensure_ascii=False, indent=2)
            log(f"Metadata saved locally: {LOCAL_METADATA_PATH}")
        except Exception as exc2:
            log(f"WARNING: local metadata write failed: {exc2}")

    log("COMPLETE")
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())