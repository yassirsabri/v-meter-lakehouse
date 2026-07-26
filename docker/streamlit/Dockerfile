FROM python:3.11-slim

WORKDIR /opt/spark-app

# Installation des dépendances pour le dashboard
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir \
    streamlit==1.36.0 \
    requests \
    pandas \
    plotly \
    seaborn

EXPOSE 8501

# Lancement de Streamlit
CMD ["streamlit", "run", "scripts/07_streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]