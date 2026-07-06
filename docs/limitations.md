# Platform Boundaries

This project serves as a fully featured, production-ready local deliverable. However, standard architectural boundaries apply for its current iteration:

## Current Operational Limits

- **Cluster Scalability**: The Apache Spark component is currently configured as a single-node cluster (one Master, one Worker). True horizontal scaling would require migrating to Kubernetes or a managed cloud environment.
- **Storage Redundancy**: Persistent storage relies on local Docker volumes (`minio_data` and `postgres_data`). Disaster recovery relies on manual backups, as distributed failover is not implemented.
- **Schema Evolution**: The Silver transformation logic infers data types via sampling. Processing highly mutated schemas across datasets without warning may cause downstream Spark typing errors.
- **Security**: The platform utilizes default HTTP routing and basic authentication. It is not currently hardened with SSL/TLS certificates for deployment on public-facing external networks.

