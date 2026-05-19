#!/usr/bin/env python3
"""
Unsupervised Modeling & MLOps Pipeline.

Purpose:
- Read curated Silver data from MinIO.
- Separate morphology and spectral features.
- Apply PCA on spectral features.
- Perform K-Means Clustering (Population Discovery).
- Perform Isolation Forest (Anomaly/Viability Detection).
- Log hyperparameters, metrics, and artifact plots to MLflow.
"""

import os
import sys
import argparse
import shutil
import urllib.request
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import silhouette_score, calinski_harabasz_score


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio_password")

SILVER_BUCKET = os.getenv("SILVER_BUCKET", "silver")
SILVER_KEY = os.getenv("SILVER_KEY", "videometer/cleaned_measurements")
SILVER_S3_PATH = f"s3://{SILVER_BUCKET}/{SILVER_KEY}"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

VARIANCE_THRESHOLD = float(os.getenv("VARIANCE_THRESHOLD", 0.95))
N_CLUSTERS = int(os.getenv("N_CLUSTERS", 3))
ANOMALY_CONTAMINATION = float(os.getenv("ANOMALY_CONTAMINATION", 0.05))


def log(message: str) -> None:
    print(f"[MODELING] {message}")


def wait_for_mlflow() -> None:
    try:
        urllib.request.urlopen(MLFLOW_TRACKING_URI, timeout=5)
        log("MLflow server is reachable.")
    except Exception as exc:
        log(f"WARNING: MLflow server at {MLFLOW_TRACKING_URI} might be unreachable: {exc}")


def main():
    log("Starting Unsupervised Modeling Pipeline")

    parser = argparse.ArgumentParser(description="Unsupervised Modeling Pipeline")
    parser.add_argument(
        "--crop",
        required=True,
        choices=["barley", "chickpea"],
        help="Crop type to process (barley or chickpea)",
    )
    args = parser.parse_args()

    log(f"Starting Unsupervised Modeling Pipeline for crop: {args.crop.upper()}")
    wait_for_mlflow()

    silver_key = f"videometer/{args.crop}/"
    silver_s3_path = f"s3://{SILVER_BUCKET}/{silver_key}"

    log(f"Reading Silver data from {silver_s3_path}")
    storage_options = {
        "client_kwargs": {"endpoint_url": MINIO_ENDPOINT},
        "key": MINIO_ACCESS_KEY,
        "secret": MINIO_SECRET_KEY,
    }

    try:
        df = pd.read_parquet(silver_s3_path, storage_options=storage_options)
        log(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as exc:
        log(f"ERROR: Failed to read Silver data. {exc}")
        return 1

    metadata_cols = {
        "id_measurement",
        "cluster_label",
        "is_anomaly",
        "is_geometric_outlier",
        "crop_type",
        "source_filename",
        "source_checksum",
        "load_id",
        "ingestion_date",
        "ingestion_timestamp",
        "processed_date",
        "filename",
        "file_path",
        "blob_id",
        "blobid",
        "source_image_id",
        "sourceimage_id",
        "sourceimageid",
    }

    candidate_cols = [c for c in df.columns if c not in metadata_cols]
    for col in candidate_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in {"is_geometric_outlier", "is_anomaly", "cluster_label"}]

    spectral_keywords = [
        "multicolor",
        "spectral",
        "reflectance",
        "transmittance",
        "cielab",
        "cie",
        "srgb",
        "colorband",
        "successivebanddiff",
        "autocorrelation",
        "graylevel",
        "phirad",
    ]

    spectral_cols = [
        c for c in numeric_cols
        if any(k in c.lower() for k in spectral_keywords) and c != "data_quality_score"
    ]
    morphology_cols = [
        c for c in numeric_cols
        if c not in spectral_cols and c != "data_quality_score"
    ]

    log(f"Identified {len(spectral_cols)} spectral columns and {len(morphology_cols)} morphological columns.")

    model_cols = [c for c in spectral_cols + morphology_cols if c in df.columns]
    if not model_cols:
        log("ERROR: No usable model columns found after type conversion.")
        return 1

    keep_cols = model_cols + ([c for c in ["id_measurement"] if c in df.columns])
    df = df[keep_cols].copy()

    medians = df[model_cols].median(numeric_only=True)
    df[model_cols] = df[model_cols].fillna(medians)

    df = df.copy()

    if len(df) < 2:
        log("ERROR: Not enough rows left after preprocessing.")
        return 1

    log(f"Shape after preprocessing: {df.shape}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_name = f"Videometer_{args.crop.capitalize()}_Unsupervised"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{args.crop}_unsupervised_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
        log(f"MLflow Run ID: {run.info.run_id}")

        mlflow.log_param("pca_variance_threshold", VARIANCE_THRESHOLD)
        mlflow.log_param("n_clusters", N_CLUSTERS)
        mlflow.log_param("anomaly_contamination", ANOMALY_CONTAMINATION)
        mlflow.log_param("dataset_size", len(df))

        area_col = next((c for c in morphology_cols if c in ["area_mm2", "area"]), None)
        width_col = next((c for c in morphology_cols if c in ["width_mm", "width"]), None)

        if area_col and width_col:
            geom_features = df[[area_col, width_col]].dropna().copy()

            if len(geom_features) < 2:
                log("Not enough valid rows for geometric outlier detection, skipping.")
                df["is_geometric_outlier"] = False
            else:
                log(f"Fitting Ellipse de fidélité on {{{area_col}, {width_col}}}...")
                envelope = EllipticEnvelope(contamination=ANOMALY_CONTAMINATION, random_state=42)
                df["is_geometric_outlier"] = False
                df.loc[geom_features.index, "is_geometric_outlier"] = envelope.fit_predict(geom_features) == -1
                geom_outliers = int(df["is_geometric_outlier"].sum())
                log(f"Identified {geom_outliers} geometric outliers (broken seeds/artefacts).")
                mlflow.log_metric("geometric_outliers", geom_outliers)
        else:
            log("Area/Width columns not found, skipping bivariate Mahalanobis filter.")
            df["is_geometric_outlier"] = False

        if len(spectral_cols) == 0:
            log("WARNING: No spectral columns found, skipping PCA.")
            X_spec_pca = np.empty((len(df), 0))
            actual_components = 0
            explained_variance = 0.0
            pca = None
        else:
            log("Applying Min-Max scaling and dynamic PCA to spectral features...")
            scaler_spec = MinMaxScaler()
            X_spec_scaled = scaler_spec.fit_transform(df[spectral_cols])

            pca = PCA(n_components=VARIANCE_THRESHOLD, random_state=42)
            X_spec_pca = pca.fit_transform(X_spec_scaled)

            actual_components = pca.n_components_
            explained_variance = float(np.sum(pca.explained_variance_ratio_))
            mlflow.log_metric("pca_explained_variance_ratio", explained_variance)
            mlflow.log_metric("pca_actual_components", actual_components)
            log(f"Dynamic PCA selected {actual_components} components to explain {explained_variance:.2%} of variance.")

        if len(morphology_cols) == 0:
            log("WARNING: No morphology columns found, skipping morphology scaling.")
            X_morph_scaled = np.empty((len(df), 0))
        else:
            morph_variance = df[morphology_cols].var(numeric_only=True)
            morphology_cols_filtered = morph_variance[morph_variance > 0].index.tolist()
            removed = len(morphology_cols) - len(morphology_cols_filtered)
            log(f"Removed {removed} zero-variance morphology columns.")

            if len(morphology_cols_filtered) == 0:
                log("WARNING: No morphology columns left after variance filtering, skipping morphology scaling.")
                X_morph_scaled = np.empty((len(df), 0))
            else:
                scaler_morph = StandardScaler()
                X_morph_scaled = scaler_morph.fit_transform(df[morphology_cols_filtered])

        X_combined = np.hstack((X_spec_pca, X_morph_scaled))

        if X_combined.shape[1] == 0:
            log("ERROR: No features available for clustering.")
            return 1

        X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)

        log(f"Running K-Means Clustering (k={N_CLUSTERS})...")
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(X_combined)
        df["cluster_label"] = cluster_labels

        sil_score = silhouette_score(X_combined, cluster_labels)
        ch_score = calinski_harabasz_score(X_combined, cluster_labels)

        mlflow.log_metric("silhouette_score", sil_score)
        mlflow.log_metric("calinski_harabasz_score", ch_score)
        log(f"Clustering Quality - Silhouette: {sil_score:.4f}, Calinski-Harabasz: {ch_score:.2f}")

        kmeans_model_path = f"/opt/spark-app/data/models/{args.crop}_kmeans_model"
        if os.path.exists(kmeans_model_path):
            shutil.rmtree(kmeans_model_path)
        os.makedirs("/opt/spark-app/data/models", exist_ok=True)
        mlflow.sklearn.save_model(kmeans, kmeans_model_path)
        log(f"K-Means model saved locally to {kmeans_model_path}")

        log("Running Isolation Forest for Anomaly Detection...")
        iso_forest = IsolationForest(contamination=ANOMALY_CONTAMINATION, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X_combined)
        df["is_anomaly"] = (anomaly_labels == -1)

        anomaly_count = int(df["is_anomaly"].sum())
        anomaly_ratio = anomaly_count / len(df)
        mlflow.log_metric("anomaly_count", anomaly_count)
        log(f"Detected {anomaly_count} anomalies ({anomaly_ratio:.2%}).")

        iso_model_path = f"/opt/spark-app/data/models/{args.crop}_isolation_forest_model"
        if os.path.exists(iso_model_path):
            shutil.rmtree(iso_model_path)
        mlflow.sklearn.save_model(iso_forest, iso_model_path)
        log(f"Isolation Forest model saved locally to {iso_model_path}")

        artifacts_dir = "/tmp/mlflow_artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)

        if actual_components > 0 and pca is not None:
            plt.figure(figsize=(8, 5))
            plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o", linestyle="--")
            plt.title("Cumulative Explained Variance by PCA Components")
            plt.xlabel("Number of Components")
            plt.ylabel("Cumulative Explained Variance")
            plt.grid(True)
            pca_plot_path = f"{artifacts_dir}/pca_variance.png"
            plt.savefig(pca_plot_path)
            plt.close()
            try:
                mlflow.log_artifact(pca_plot_path)
            except Exception as exc:
                log(f"WARNING: Could not log PCA plot to MLflow: {exc}")

        if X_spec_pca.shape[1] >= 2:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                x=X_spec_pca[:, 0],
                y=X_spec_pca[:, 1],
                hue=df["cluster_label"],
                palette="viridis",
                style=df["is_anomaly"],
                markers={False: "o", True: "X"},
                s=60,
                alpha=0.7,
            )
            plt.title("Seed Population Clusters & Anomalies (PC1 vs PC2)")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.legend(title="Cluster / Is Anomaly")
            cluster_plot_path = f"{artifacts_dir}/cluster_scatter.png"
            plt.savefig(cluster_plot_path)
            plt.close()
            try:
                mlflow.log_artifact(cluster_plot_path)
            except Exception as exc:
                log(f"WARNING: Could not log cluster plot to MLflow: {exc}")

        output_path = f"/opt/spark-app/data/silver_{args.crop}_unsupervised_labels.csv"
        try:
            cols = df.columns.tolist()
            ordered_cols = [
                "id_measurement",
                "cluster_label",
                "is_anomaly",
                "is_geometric_outlier",
            ] + [
                c for c in cols
                if c not in {"id_measurement", "cluster_label", "is_anomaly", "is_geometric_outlier"}
            ]
            df[ordered_cols].to_csv(output_path, index=False)
            log(f"Saved local copy with labels to {output_path}")
            try:
                mlflow.log_artifact(output_path)
            except Exception as exc:
                log(f"WARNING: Could not log output CSV to MLflow: {exc}")
        except Exception as exc:
            log(f"WARNING: Could not save output csv: {exc}")

        log("Unsupervised Pipeline Complete! Check MLflow UI (http://localhost:5000) for results.")

    return 0


if __name__ == "__main__":
    sys.exit(main())