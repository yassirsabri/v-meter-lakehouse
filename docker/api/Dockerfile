FROM python:3.11-slim

WORKDIR /opt/spark-app

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir --default-timeout=2000 --retries=20 \
    fastapi \
    uvicorn \
    pandas==2.2.3 \
    pyarrow==15.0.2 \
    mlflow==2.14.3 \
    boto3==1.34.162 \
    scikit-learn==1.5.2 \
    xgboost

EXPOSE 8000

CMD ["python", "scripts/05_prediction_api.py"]