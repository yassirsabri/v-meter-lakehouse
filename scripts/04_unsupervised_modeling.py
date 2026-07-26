#!/usr/bin/env python3
"""
Automated Unsupervised Machine Learning Pipeline.

Respecte scrupuleusement la logique du rapport PFE :
1. Lecture depuis Nessie/Iceberg (Couche Silver).
2. Filtre géométrique bivarié (Mahalanobis sur Area/Width).
3. Séparation des features (Spectrales vs Morphologiques).
4. Compression (PCA à 95% de variance) uniquement sur le spectre.
5. Standardisation et concaténation avec la morphologie.
6. Clustering K-Means avec évaluation par Méthode du Coude & Silhouette.
7. Détection d'anomalies globales via Isolation Forest.
8. Logging automatisé des modèles, métriques et graphiques dans MLflow.
"""

import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from pyspark.sql import SparkSession

import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio_password")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT
os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY

# Constantes métier
VARIANCE_THRESHOLD = float(os.getenv("VARIANCE_THRESHOLD", 0.95))
N_CLUSTERS = int(os.getenv("N_CLUSTERS", 3))

METADATA_COLS = {
    "id_measurement", "cluster_label", "is_anomaly", "is_geometric_outlier",
    "crop_type", "source_filename", "source_checksum", "load_id",
    "ingestion_date", "ingestion_timestamp", "processed_date",
    "filename", "file_path", "blob_id", "blobid", "source_image_id",
    "sourceimage_id", "sourceimageid", "dq_check_status"
}

def log(msg: str) -> None:
    print(f"[UNSUPERVISED] {msg}", flush=True)

def parse_args():
    p = argparse.ArgumentParser(description="Unsupervised ML on Silver Data")
    p.add_argument("--crop", required=True, help="Crop name (e.g. chickpea, barley)")
    p.add_argument("--contamination", type=float, default=0.05, help="Proportion of anomalies")
    return p.parse_args()

def create_spark_session() -> SparkSession:
    NESSIE_URI = os.getenv("NESSIE_URI", "http://nessie:19120/api/v1")
    WAREHOUSE_PATH = f"s3a://{os.getenv('BRONZE_BUCKET', 'bronze')}/warehouse"
    
    return (
        SparkSession.builder
        .appName("lakehouse-unsupervised")
        .config("spark.jars.packages", 
                "org.apache.hadoop:hadoop-aws:3.3.4,"
                "org.apache.iceberg:iceberg-spark-runtime-3.3_2.12:1.4.3,"
                "org.projectnessie.nessie-integrations:nessie-spark-extensions-3.3_2.12:0.76.6")
        .config("spark.sql.extensions", 
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions,"
                "org.projectnessie.spark.extensions.NessieSparkSessionExtensions")
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.sql.catalog.nessie", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.nessie.catalog-impl", "org.apache.iceberg.nessie.NessieCatalog")
        .config("spark.sql.catalog.nessie.uri", NESSIE_URI)
        .config("spark.sql.catalog.nessie.ref", "main")
        .config("spark.sql.catalog.nessie.authentication.type", "NONE")
        .config("spark.sql.catalog.nessie.warehouse", WAREHOUSE_PATH)
        .config("spark.sql.catalog.nessie.s3.endpoint", MINIO_ENDPOINT)
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )

def main():
    args = parse_args()
    crop = args.crop.lower()
    
    log(f"Starting Unsupervised ML pipeline for crop: {crop}")
    
    # 1. Extraction des données via PySpark (Iceberg/Nessie)
    spark = create_spark_session()
    table_name = f"nessie.silver.{crop}"
    
    try:
        log(f"Reading data from {table_name}")
        df_spark = spark.read.format("iceberg").load(table_name)
        
        # On supprime les dates côté Spark pour éviter le bug de conversion Pandas
        columns_to_drop = [c for c in df_spark.columns if c in METADATA_COLS]
        df_spark_filtered = df_spark.drop(*columns_to_drop)
        df = df_spark_filtered.toPandas()
    except Exception as exc:
        log(f"ERROR: Could not read Iceberg table '{table_name}': {exc}")
        spark.stop()
        return 1
    finally:
        spark.stop()

    if df.empty:
        log("ERROR: Dataframe is empty")
        return 1

    # Typage des données
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Séparation Spectrales / Morphologiques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    spectral_keywords = ["multicolor", "spectral", "reflectance", "transmittance", 
                         "cielab", "cie", "srgb", "colorband", "successivebanddiff"]
    
    spectral_cols = [c for c in numeric_cols if any(k in c.lower() for k in spectral_keywords)]
    morphology_cols = [c for c in numeric_cols if c not in spectral_cols]

    log(f"Identified {len(spectral_cols)} spectral and {len(morphology_cols)} morphological columns.")

    # Imputation globale
    log("Imputing missing values...")
    imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # 3. Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("automated_mlops_unsupervised")
    run_name = f"Unsupervised_{crop}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("crop", crop)
        mlflow.log_param("contamination", args.contamination)
        mlflow.log_metric("total_samples", len(df))

        # --- A. MAHALANOBIS (Outliers Géométriques Bivariés) ---
        area_col = next((c for c in morphology_cols if c in ["area_mm2", "area"]), None)
        width_col = next((c for c in morphology_cols if c in ["width_mm", "width"]), None)

        if area_col and width_col:
            log(f"Running Elliptic Envelope (Mahalanobis) on [{area_col}, {width_col}]...")
            geom_features = df[[area_col, width_col]].copy()
            envelope = EllipticEnvelope(contamination=args.contamination, random_state=42)
            maha_preds = envelope.fit_predict(geom_features)
            maha_outliers = int((maha_preds == -1).sum())
            mlflow.log_metric("geometric_outliers_mahalanobis", maha_outliers)
            log(f"Detected {maha_outliers} geometric outliers (broken seeds).")
        else:
            log("Area or Width column not found. Skipping Mahalanobis filter.")

        # --- B. COMPRESSION PCA (Spectre uniquement) ---
        if len(spectral_cols) > 0:
            log("Scaling and applying dynamic PCA to spectral features...")
            scaler_spec = MinMaxScaler()
            X_spec_scaled = scaler_spec.fit_transform(df[spectral_cols])

            pca = PCA(n_components=VARIANCE_THRESHOLD, random_state=42)
            X_spec_pca = pca.fit_transform(X_spec_scaled)

            actual_components = pca.n_components_
            explained_variance = float(np.sum(pca.explained_variance_ratio_))
            mlflow.log_metric("pca_components", actual_components)
            mlflow.log_metric("pca_variance_explained", explained_variance)
            log(f"PCA reduced {len(spectral_cols)} spectral cols to {actual_components} components.")
        else:
            X_spec_pca = np.empty((len(df), 0))

        # --- C. STANDARDISATION (Morphologie) ---
        if len(morphology_cols) > 0:
            scaler_morph = StandardScaler()
            X_morph_scaled = scaler_morph.fit_transform(df[morphology_cols])
        else:
            X_morph_scaled = np.empty((len(df), 0))

        # Fusion des Features
        X_combined = np.hstack((X_spec_pca, X_morph_scaled))
        
        if X_combined.shape[1] == 0:
            log("ERROR: No features available for clustering.")
            return 1

        # --- D. K-MEANS & ELBOW METHOD ---
        log("Generating Elbow Curve for K-Means...")
        inertias = []
        K_range = range(1, min(11, len(df)))
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            km.fit(X_combined)
            inertias.append(km.inertia_)
            
        plt.figure(figsize=(8, 5))
        plt.plot(K_range, inertias, marker='o', linestyle='--', color='b')
        plt.title(f"Méthode du Coude (Elbow Method) - {crop.capitalize()}")
        plt.xlabel("Nombre de clusters (K)")
        plt.ylabel("Inertie Intra-Classe (WCSS)")
        plt.grid(True, linestyle=":", alpha=0.6)
        elbow_path = f"/tmp/elbow_{crop}.png"
        plt.savefig(elbow_path, dpi=300, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(elbow_path)

        log(f"Running K-Means Clustering (k={N_CLUSTERS})...")
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(X_combined)
        
        # Silhouette Score (si le dataset n'est pas gigantesque)
        if len(X_combined) < 20000:
            sil_score = silhouette_score(X_combined, cluster_labels)
            mlflow.log_metric("silhouette_score", float(sil_score))
            log(f"Silhouette Score: {sil_score:.4f}")

        mlflow.sklearn.log_model(kmeans, "kmeans_model")

        # --- E. ISOLATION FOREST ---
        log(f"Running Isolation Forest (Contamination: {args.contamination})...")
        iso_forest = IsolationForest(contamination=args.contamination, random_state=42, n_jobs=-1)
        iso_preds = iso_forest.fit_predict(X_combined)
        iso_outliers = int((iso_preds == -1).sum())
        mlflow.log_metric("isolation_forest_anomalies", iso_outliers)
        log(f"Detected {iso_outliers} anomalies globally.")
        mlflow.sklearn.log_model(iso_forest, "isolation_forest_model")

        # Cleanup temp files
        if os.path.exists(elbow_path):
            os.remove(elbow_path)

    log("Unsupervised ML pipeline COMPLETE.")
    return 0

if __name__ == "__main__":
    sys.exit(main())