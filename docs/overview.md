# Project Overview

This repository is part of the final-year engineering project:

**Design and Implementation of a Containerized Lakehouse Architecture for Videometer Big Data and Predictive Modeling.**

The project focuses on building a practical, automated open-source lakehouse stack for high-dimensional data, seamlessly integrating data engineering preparation with MLOps workflows.

## Main Scope

- Architect a fully containerized, production-ready environment using Docker Compose with strict database isolation for transactional safety.
- Implement a robust Medallion data pipeline orchestrated by Apache Airflow:
  - **Bronze**: Raw data ingestion
  - **Silver**: Data cleaning, deduplication, and standardization
  - **Gold**: Aggregated key performance indicators (KPIs)
- Prepare scenario-based predictive modeling use cases, with a primary focus on crop viability prediction.
- Automate the end-to-end machine learning lifecycle using MLflow, including:
  - Dynamic dataset discovery and supervised training.
  - Model evaluation, hyperparameter logging, and feature importance tracking.
  - Automated model registration and artifact versioning.
- Deploy a unified, zero-downtime Prediction API capable of hot-reloading active models directly from the registry without dropping requests.

## Current Repository Status

The repository represents a complete, decoupled production deliverable. It fully integrates the containerized data platform, the automated Medallion data preparation pipeline (DAG 1), the MLOps training and tracking pipeline (DAG 2), and the real-time inference serving layer.