FROM apache/airflow:2.9.3-python3.11

USER root

RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    openjdk-17-jdk-headless \
    procps \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

USER airflow

RUN pip install --no-cache-dir \
    pandas==2.1.4 \
    pyarrow==15.0.2 \
    s3fs==2023.12.2 \
    openpyxl==3.1.5 \
    boto3==1.34.162 \
    scikit-learn==1.5.2 \
    pyspark==3.3.4 \
    seaborn \
    matplotlib \
    mlflow==2.14.3 \
    fastapi \
    uvicorn