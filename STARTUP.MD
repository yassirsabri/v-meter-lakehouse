
# Workstream A - Infrastructure Startup and Operations Runbook

## Quick Start

```bash
# Start all services
docker compose up -d --build

# Verify all services are healthy
docker compose ps -a

# Check all endpoints are responding
curl http://localhost:5432 > /dev/null 2>&1 && echo "PostgreSQL OK"
curl http://localhost:9000 > /dev/null 2>&1 && echo "MinIO API OK"
curl http://localhost:9001 > /dev/null 2>&1 && echo "MinIO Console OK"
curl http://localhost:8080 > /dev/null 2>&1 && echo "Spark Master UI OK"
curl http://localhost:5000 > /dev/null 2>&1 && echo "MLflow OK"
curl http://localhost:19120/api/v2/config > /dev/null 2>&1 && echo "Nessie OK"

