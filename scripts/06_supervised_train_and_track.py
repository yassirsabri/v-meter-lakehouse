#!/usr/bin/env python3
r"""
Supervised training + MLflow tracking script.

Final logic from Untitled3.html:
- Viability: binary target with threshold 80, XGBoost + class weighting
- Origin: grouped scenarios, not exact ORI country classification
- MLflow tracking, optional Nessie tag capture, local artifact export

Usage examples (PowerShell):
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"

python .\scripts\06_supervised_train_and_track.py `
  --task viability `
  --data-path ".\data_for_supervised_ml\ready for ml\dataset_complet_viabilite.csv" `
  --output-model-path ".\data\models"

python .\scripts\06_supervised_train_and_track.py `
  --task origin `
  --data-path ".\data_for_supervised_ml\ready for ml\dataset_passport_ready_for_supervised_modeling.csv" `
  --output-model-path ".\data\models"
"""

import argparse
import hashlib
import json
import os
import uuid
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
except Exception as exc:
    xgb = None
    print(f"XGBoost import failed: {exc}")

try:
    import mlflow
except Exception as exc:
    print(f"mlflow not installed or not importable: {exc}")
    raise


C_GREEN = "#2E8B57"
C_ORANGE = "#FF8C00"
C_YELLOW = "#FFD700"


def sha256_of_file(path, block_size=65536):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def numeric_matrix(df, exclude_columns):
    existing = [column for column in exclude_columns if column in df.columns]
    X_raw = df.drop(columns=existing, errors="ignore")
    X_raw = X_raw.select_dtypes(
        include=["float64", "float32", "int64", "int32", "uint8", "uint16", "uint32", "uint64"]
    )
    return X_raw


def impute_numeric_frame(X_raw):
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)
    X_imputed = pd.DataFrame(X_imputed, columns=X_raw.columns)
    return imputer, X_imputed


def split_with_safe_stratify(X, y, test_size=0.2, random_state=42):
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


def plot_confusion_matrix(y_true, y_pred, labels, title, out_path, cmap):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=18)
    ax.set_xlabel("Prédiction", fontsize=12, fontweight="bold")
    ax.set_ylabel("Réalité", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_feature_importance(importances, feature_names, title, out_path, top_n=15, color=C_GREEN):
    if importances is None:
        return None

    importance_df = pd.DataFrame(
        {
            "Variable": feature_names,
            "Importance": importances,
        }
    ).sort_values("Importance", ascending=False)

    top_df = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(top_df["Variable"][::-1], top_df["Importance"][::-1], color=color, edgecolor="black", alpha=0.85)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("Variables", fontsize=12)
    ax.grid(axis="x", linestyle=":", alpha=0.7)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return top_df


def log_text_artifact(text, out_path):
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return out_path


def save_bundle(bundle, out_dir, file_name="model_bundle.joblib"):
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, file_name)
    try:
        import joblib

        joblib.dump(bundle, out_path)
    except Exception:
        import pickle

        with open(out_path, "wb") as handle:
            pickle.dump(bundle, handle)
    return out_path


def build_origin_scenarios():
    def as_map(grouped_codes):
        mapping = {}
        for group_name, codes in grouped_codes.items():
            for code in codes:
                mapping[code] = group_name
        return mapping

    dict_cwana = as_map(
        {
            "CWANA": [
                "AFG", "ARM", "AZE", "GEO", "IRN", "IRQ", "ISR", "JOR", "KAZ", "KGZ",
                "LBN", "OMN", "PAK", "PSE", "SAU", "SYR", "TJK", "TKM", "TUR", "UZB",
                "YEM", "DZA", "EGY", "LBY", "MAR", "TUN"
            ],
            "Europe": [
                "ALB", "AUT", "BGR", "BIH", "CHE", "ESP", "GBR", "GRC", "ITA", "MKD",
                "MNE", "NLD", "NOR", "POL", "PRT", "RUS", "SRB", "SWE", "UKR", "XKX"
            ],
            "Americas_Oceania": [
                "BOL", "CAN", "MEX", "PER", "PRY", "URY", "USA", "VEN", "AUS"
            ],
            "Asia_East_South": [
                "BTN", "CHN", "IND", "JPN", "KOR", "MNG", "NPL", "PRK"
            ],
            "Sub_Saharan_Africa": [
                "ERI", "ETH", "ZAF"
            ],
        }
    )

    dict_climate = as_map(
        {
            "Mediterranean": [
                "ESP", "PRT", "ITA", "GRC", "TUR", "SYR", "LBN", "ISR", "PSE", "JOR",
                "MAR", "DZA", "TUN", "LBY", "EGY", "ALB", "MKD", "MNE", "SRB", "BIH", "XKX"
            ],
            "Continental_Temperate": [
                "RUS", "UKR", "KAZ", "POL", "SWE", "NOR", "GBR", "CHE", "AUT", "NLD",
                "BGR", "CAN", "USA", "GEO", "ARM", "AZE"
            ],
            "Arid_SemiArid": [
                "SAU", "OMN", "YEM", "IRQ", "IRN", "TKM", "UZB", "PAK", "AFG"
            ],
            "Highland_Mountain": [
                "NPL", "BTN", "MNG", "CHN", "PER", "BOL", "KGZ", "TJK"
            ],
            "Tropical_Other": [
                "IND", "MEX", "VEN", "PRY", "URY", "AUS", "ZAF", "ERI", "ETH", "JPN", "KOR", "PRK"
            ],
        }
    )

    dict_genetic = as_map(
        {
            "Fertile_Crescent": [
                "SYR", "IRQ", "TUR", "LBN", "ISR", "JOR", "PSE", "IRN"
            ],
            "Mediterranean_Basin": [
                "MAR", "DZA", "TUN", "LBY", "EGY", "ESP", "PRT", "ITA", "GRC", "ALB",
                "MNE", "MKD", "SRB", "BIH", "XKX"
            ],
            "Horn_of_Africa": [
                "ETH", "ERI"
            ],
            "Central_East_Asia": [
                "AFG", "KAZ", "KGZ", "TJK", "UZB", "TKM", "CHN", "MNG", "JPN", "KOR",
                "PRK", "IND", "PAK", "BTN", "NPL", "ARM", "AZE", "GEO"
            ],
            "Introduced_Regions": [
                "BOL", "CAN", "MEX", "PER", "PRY", "URY", "USA", "VEN", "AUS", "AUT",
                "BGR", "CHE", "GBR", "NLD", "NOR", "POL", "RUS", "SWE", "UKR", "ZAF"
            ],
        }
    )

    return {
        "A. Régions ICARDA (CWANA)": dict_cwana,
        "B. Zones Climatiques": dict_climate,
        "C. Centres Génétiques": dict_genetic,
    }


def run_viability_task(df, args, run_id):
    if "ViabilityRate" not in df.columns:
        raise ValueError("ViabilityRate column not found in dataset.")

    df = df.copy()
    df["ViabilityRate"] = pd.to_numeric(df["ViabilityRate"], errors="coerce")
    df = df.dropna(subset=["ViabilityRate"]).reset_index(drop=True)

    df["Viability_Class"] = (df["ViabilityRate"] < args.viability_threshold).astype(int)
    y = df["Viability_Class"]

    cols_to_exclude = [
        "IG",
        "id_measurement",
        "cluster_label",
        "is_anomaly",
        "is_geometric_outlier",
        "sourceimage_captureid",
        "sourceimageid",
        "dq_check_status",
        "TAX_NAME",
        "POP_TYPE",
        "ORI",
        "REGION",
        "MACRO_REGION",
        "ViabilityRate",
        "Viability_Class",
        "filename",
        "blobid",
    ]

    X_raw = numeric_matrix(df, cols_to_exclude)
    if X_raw.shape[1] == 0:
        raise ValueError("No numeric features found for viability task.")

    imputer, X_imputed = impute_numeric_frame(X_raw)
    X_train, X_test, y_train, y_test = split_with_safe_stratify(X_imputed, y, test_size=args.test_size, random_state=42)

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    if pos_count == 0:
        raise ValueError("No positive samples found for viability task.")
    ratio_penalite = neg_count / pos_count

    if xgb is None:
        raise RuntimeError("xgboost is required for the viability task.")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        tree_method="hist",
        scale_pos_weight=ratio_penalite,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Accepté (>=80%)", "Rejeté (<80%) - CIBLE"],
        digits=4,
        zero_division=0,
    )

    print("\n" + "=" * 55)
    print(f" RÉSULTAT XGBOOST PÉNALISÉ : F1-Score Macro = {f1_macro:.4f}")
    print("=" * 55 + "\n")
    print(report)

    ensure_dir(args.output_model_path)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_dir = os.path.join(args.output_model_path, f"viability_{timestamp}_{run_id}")
    ensure_dir(model_dir)

    bundle = {
        "task": "viability",
        "threshold": args.viability_threshold,
        "model": model,
        "imputer": imputer,
        "feature_names": X_raw.columns.tolist(),
        "class_positive_definition": "ViabilityRate < threshold",
    }
    bundle_path = save_bundle(bundle, model_dir)

    cm_path = os.path.join(model_dir, "acte3_matrice_viabilite_penalise.png")
    plot_confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
        title="Matrice de Confusion : XGBoost Pénalisé (Détection d'Anomalie)",
        out_path=cm_path,
        cmap=sns.light_palette(C_ORANGE, as_cmap=True),
    )

    fi_path = os.path.join(model_dir, "acte3_feature_importance_viabilite.png")
    top_df = plot_feature_importance(
        model.feature_importances_,
        X_raw.columns.tolist(),
        title="Ce que voit la machine : Les 15 variables clés du VideometerLab",
        out_path=fi_path,
        top_n=15,
        color=C_GREEN,
    )

    report_path = os.path.join(model_dir, "classification_report_viability.txt")
    log_text_artifact(report, report_path)

    metrics = {
        "task": "viability",
        "f1_macro": float(f1_macro),
        "ratio_penalite": float(ratio_penalite),
        "n_samples": int(df.shape[0]),
        "n_features": int(X_raw.shape[1]),
        "bundle_path": bundle_path,
    }

    if top_df is not None:
        metrics["top_feature"] = top_df.iloc[0]["Variable"]
        metrics["top_feature_importance"] = float(top_df.iloc[0]["Importance"])

    return metrics, model_dir


def run_origin_task(df, args, run_id):
    if "ORI" not in df.columns:
        raise ValueError("ORI column not found in dataset.")

    df = df.copy()
    df = df.dropna(subset=["ORI"]).reset_index(drop=True)

    scenarios = build_origin_scenarios()

    cols_to_exclude = [
        "IG",
        "id_measurement",
        "cluster_label",
        "is_anomaly",
        "is_geometric_outlier",
        "sourceimage_captureid",
        "sourceimageid",
        "dq_check_status",
        "TAX_NAME",
        "POP_TYPE",
        "ORI",
        "REGION",
        "MACRO_REGION",
        "filename",
        "blobid",
        "ViabilityRate",
        "Viability_Class",
    ]

    X_raw = numeric_matrix(df, cols_to_exclude)
    if X_raw.shape[1] == 0:
        raise ValueError("No numeric features found for origin task.")

    imputer, X_imputed = impute_numeric_frame(X_raw)

    results = {}

    for scenario_name, mapping in scenarios.items():
        print(f"\n[Test en cours] Scénario : {scenario_name}")

        y_temp = df["ORI"].map(mapping)
        mask = y_temp.notna()

        X_model = X_imputed.loc[mask].reset_index(drop=True)
        y_model = y_temp.loc[mask].reset_index(drop=True)

        if y_model.nunique() < 2:
            print("-> Scénario ignoré: une seule classe après mapping.")
            continue

        X_train, X_test, y_train, y_test = split_with_safe_stratify(
            X_model, y_model, test_size=args.test_size, random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        results[scenario_name] = {
            "f1_macro": float(f1),
            "n_samples": int(X_model.shape[0]),
            "n_features": int(X_model.shape[1]),
            "classes": sorted(y_model.unique().tolist()),
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "mapping": mapping,
        }

        print(f"-> F1-Score Macro : {f1:.4f}")

    if not results:
        raise RuntimeError("No origin scenario could be trained successfully.")

    best_scenario = max(results.items(), key=lambda item: item[1]["f1_macro"])[0]
    best = results[best_scenario]

    print("\n-> Meilleur scénario :", best_scenario)
    print(f"-> F1-Score Macro retenu : {best['f1_macro']:.4f}")

    ensure_dir(args.output_model_path)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_dir = os.path.join(args.output_model_path, f"origin_{timestamp}_{run_id}")
    ensure_dir(model_dir)

    bundle = {
        "task": "origin",
        "best_scenario": best_scenario,
        "scenarios": {
            name: {
                "f1_macro": data["f1_macro"],
                "n_samples": data["n_samples"],
                "n_features": data["n_features"],
                "classes": data["classes"],
            }
            for name, data in results.items()
        },
        "imputer": imputer,
        "model": best["model"],
        "feature_names": X_raw.columns.tolist(),
        "origin_mapping": best["mapping"],
    }
    bundle_path = save_bundle(bundle, model_dir)

    comparison_labels = ["Baseline (66 Pays exacts)"] + list(results.keys())
    comparison_scores = [0.0641] + [results[name]["f1_macro"] for name in results.keys()]
    comparison_colors = ["gray", C_GREEN, C_ORANGE, C_YELLOW][: len(comparison_labels)]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        comparison_labels,
        comparison_scores,
        color=comparison_colors,
        edgecolor="black",
        alpha=0.85,
    )
    ax.set_title(
        "L'Impact du Pivot Métier : Performance selon les zones climatiques",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("F1-Score Macro", fontsize=12)
    ax.set_ylim(0, max(comparison_scores) + 0.15)
    ax.grid(axis="y", linestyle=":", alpha=0.7)
    plt.xticks(rotation=15, ha="right", fontsize=11)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    plt.tight_layout()
    comparison_path = os.path.join(model_dir, "acte3_comparaison_scenarios.png")
    fig.savefig(comparison_path, dpi=300)
    plt.close(fig)

    y_test = best["y_test"]
    y_pred = best["y_pred"]
    labels = sorted(pd.Index(y_test).unique().tolist())

    cm_path = os.path.join(model_dir, "acte3_origin_confusion_best.png")
    plot_confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
        title=f"Confusion Matrix - {best_scenario}",
        out_path=cm_path,
        cmap=sns.light_palette(C_GREEN, as_cmap=True),
    )

    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    report_path = os.path.join(model_dir, "classification_report_origin_best.txt")
    log_text_artifact(report, report_path)

    fi_path = os.path.join(model_dir, "acte3_feature_importance_origin_best.png")
    top_df = plot_feature_importance(
        best["model"].feature_importances_,
        X_raw.columns.tolist(),
        title=f"Top 15 variables - {best_scenario}",
        out_path=fi_path,
        top_n=15,
        color=C_GREEN,
    )

    metrics = {
        "task": "origin",
        "best_scenario": best_scenario,
        "best_f1_macro": float(best["f1_macro"]),
        "scenarios": {name: float(data["f1_macro"]) for name, data in results.items()},
        "n_samples_total": int(df.shape[0]),
        "n_features": int(X_raw.shape[1]),
        "bundle_path": bundle_path,
    }

    if top_df is not None:
        metrics["top_feature"] = top_df.iloc[0]["Variable"]
        metrics["top_feature_importance"] = float(top_df.iloc[0]["Importance"])

    return metrics, model_dir


def maybe_log_nessie_info(nessie_url):
    if not nessie_url:
        return {}

    try:
        import requests

        response = requests.get(nessie_url.rstrip("/") + "/api/v2/config", timeout=5)
        if response.ok:
            return response.json()
    except Exception as exc:
        print(f"Could not query Nessie: {exc}")

    return {}


def main():
    parser = argparse.ArgumentParser(description="Train supervised models and log to MLflow")
    parser.add_argument("--task", choices=["viability", "origin"], required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="MLflow tracking URI (env MLFLOW_TRACKING_URI if omitted)",
    )
    parser.add_argument(
        "--output-model-path",
        default="./data/models",
        help="Local folder to write the best model bundle",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--viability-threshold", type=float, default=80.0)
    parser.add_argument("--experiment-name", default="pfe_supervised_final")
    parser.add_argument("--nessie-url", default=None, help="Optional Nessie base URL (e.g. http://localhost:19120)")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(args.experiment_name)

    dataset_checksum = sha256_of_file(args.data_path)
    run_id = str(uuid.uuid4())[:8]
    nessie_info = maybe_log_nessie_info(args.nessie_url)

    df = pd.read_csv(args.data_path, low_memory=False)

    with mlflow.start_run(run_name=f"{args.task}_{run_id}") as run:
        mlflow.set_tag("task", args.task)
        mlflow.set_tag("logic", "final_notebook_pivot")
        mlflow.set_tag("dataset_checksum", dataset_checksum)
        mlflow.log_param("data_path", args.data_path)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("viability_threshold", args.viability_threshold)
        mlflow.log_param("n_rows", int(df.shape[0]))
        mlflow.log_param("n_columns", int(df.shape[1]))

        if nessie_info:
            if "defaultBranch" in nessie_info:
                mlflow.set_tag("nessie.defaultBranch", nessie_info.get("defaultBranch"))

            nessie_tmp = os.path.join(os.getcwd(), f"nessie_config_{run_id}.json")
            with open(nessie_tmp, "w", encoding="utf-8") as handle:
                json.dump(nessie_info, handle, indent=2, ensure_ascii=False)
            mlflow.log_artifact(nessie_tmp)
            try:
                os.remove(nessie_tmp)
            except Exception:
                pass

        if args.task == "viability":
            metrics, model_dir = run_viability_task(df, args, run_id)
        else:
            metrics, model_dir = run_origin_task(df, args, run_id)

        for key, value in metrics.items():
            if isinstance(value, (int, float, np.floating, np.integer)):
                mlflow.log_metric(key, float(value))
            elif isinstance(value, str):
                mlflow.set_tag(key, value)

        summary_path = os.path.join(model_dir, "run_summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_id": run.info.run_id,
                    "task": args.task,
                    "dataset_checksum": dataset_checksum,
                    "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
                    "created_at_utc": datetime.utcnow().isoformat() + "Z",
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

        mlflow.log_artifact(summary_path)

        bundle_path = os.path.join(model_dir, "model_bundle.joblib")
        if os.path.exists(bundle_path):
            mlflow.log_artifact(bundle_path)

        for artifact_name in os.listdir(model_dir):
            artifact_path = os.path.join(model_dir, artifact_name)
            if os.path.isfile(artifact_path) and artifact_name not in {"model_bundle.joblib", "run_summary.json"}:
                mlflow.log_artifact(artifact_path)

        print("Run finished.")
        print("Run id:", run.info.run_id)
        print("Task:", args.task)
        if args.task == "origin":
            print("Best scenario:", metrics["best_scenario"])
            print("Best F1 macro:", f"{metrics['best_f1_macro']:.4f}")
            print("All scenarios:", metrics["scenarios"])
        else:
            print("F1 macro:", f"{metrics['f1_macro']:.4f}")
            print("Ratio penalite:", f"{metrics['ratio_penalite']:.2f}")
        print("Model stored at:", model_dir)
        print("Dataset checksum:", dataset_checksum)


if __name__ == "__main__":
    main()