FROM python:3.11-slim

WORKDIR /opt/spark-app

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pandas==2.2.3 \
    mlflow==2.14.3 \
    boto3==1.34.162

EXPOSE 8000

CMD ["python", "scripts/05_prediction_api.py"]