# Pipeline

## Bronze

The Bronze stage reads the raw source file and writes it to MinIO with technical metadata.

This stage keeps the data as close as possible to the original source.

## Silver

The Silver stage reads the Bronze data and performs the following steps:
- validates required columns
- parses timestamps
- normalizes the business key
- removes duplicates
- casts numeric columns
- applies simple sanity rules
- computes a row-level quality score

## Gold

The Gold stage reads the Silver data and creates aggregated key performance indicators.

This stage groups the data by image and date, then calculates:
- measurement count
- mean values
- standard deviation values
- shape index
- quality status

## Orchestration

Apache Airflow runs the pipeline in this order:

Bronze -> Silver -> Gold
