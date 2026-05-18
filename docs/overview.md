
# Project Overview

This repository is part of the final year internship project:

Design and Implementation of a Containerized Lakehouse Architecture for Videometer Big Data and Predictive Modeling.

The project focuses on building a practical open source lakehouse stack for high-dimensional data, then using it for data preparation and predictive modeling workflows.

## Main Scope

- Benchmark open source solutions for a feature lakehouse context.
- Build a containerized environment with Docker Compose.
- Implement medallion ingestion layers:
  - Bronze for raw ingestion
  - Silver for cleaning and standardization
  - Gold for aggregated key performance indicators
- Prepare scenario-based modeling use cases, including:
  - population type classification
  - viability prediction
- Support model lifecycle management with MLflow:
  - hyperparameter logging
  - model version tracking
  - feature importance tracking
- Prepare a unified prediction API and analytical dashboards as project targets.

## Current Repository Focus

At this stage, the repository contains the containerized data platform and the Bronze, Silver, and Gold pipeline implementation, with Airflow orchestration for execution flow.



