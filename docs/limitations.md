
# Limitations

This project is working locally and is not deployed to a cloud environment.

## Current limitations

- The documentation is still incomplete.
- The current pipeline is designed for a single local environment.
- The data is stored in MinIO buckets and is not versioned yet in practice.
- The Bronze, Silver, and Gold outputs are overwritten when the pipeline is run again.
- There is no automatic partitioning strategy for historical loads.
- Monitoring is limited to the local Airflow and service logs.
- The project does not yet include the machine learning parts of the future workstreams.

