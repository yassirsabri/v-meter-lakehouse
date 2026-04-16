FROM bitnamilegacy/spark:3.3.4
# Image de base: Spark 3.3.4 (distribution Bitnami = production-ready)
# Legacy = version antérieure, stable et testée
USER root
#passer a l'utilisateur root pour installer des paquets
RUN python3 -m pip install --no-cache-dir --upgrade pip \
  && python3 -m pip install --no-cache-dir pandas openpyxl mlflow s3fs boto3

WORKDIR /opt/spark-app
#repertoire de travail (ou les commandes s'executent)
USER 1001
#revenir a un utilisateur non-root pour la securite
#(les conteneurs ne doivent pas tourner en root)


