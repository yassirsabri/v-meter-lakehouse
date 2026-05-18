FROM bitnamilegacy/spark:3.3.4

USER root

RUN python3 -m pip install --no-cache-dir --upgrade pip \
  && python3 -m pip install --no-cache-dir \
    pandas==2.2.3 \
    openpyxl==3.1.2 \
    mlflow==2.14.3 \
    s3fs==2023.12.0 \
    boto3==1.34.162

WORKDIR /opt/spark-app
USER 1001