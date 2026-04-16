FROM python:3.11-slim

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir \
    mlflow==2.14.3 \
    #serveur de suivi d'experiences ML
    psycopg2-binary==2.9.9 \  
    #driver pour se connecter a postgresql
    boto3==1.34.162
    #driver AWS pour parler a S3

EXPOSE 5000
#port d'ecoute du serveur MLflow
CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri ${BACKEND_STORE_URI} \
    #postgresql pour les metadonnees
    --default-artifact-root ${ARTIFACT_ROOT}
    #minio pour les modeles