# Limitations

This project is currently running in a local environment and is not yet deployed to a cloud platform.

## Current limitations

- The documentation is still being finalized.
- The current pipeline is designed for a single local environment.
- Data is stored in MinIO buckets and versioning is not yet fully implemented in practice.
- Bronze, Silver, and Gold outputs are overwritten when the pipeline is executed again.
- There is no automatic partitioning strategy yet for historical loads.
- Monitoring is currently limited to local Airflow and service logs.
- The machine learning workstreams are not yet integrated into this repository.

## Silver transformation limitations

- The Silver processing logic must be revised because some transformations are not necessary.
- Processing multiple files from the data folder into the s3:/bronze zone has not been tested yet.
- The current Silver logic assumes that database columns are known in advance; schema-agnostic processing is not implemented yet.
