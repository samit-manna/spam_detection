FROM python:3.10-slim

LABEL maintainer="ml-platform-team"
LABEL description="MLflow model evaluation and registry operations"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    pandas==1.5.3 \
    pyarrow==14.0.2 \
    adlfs==2024.2.0 \
    azure-storage-blob==12.19.0 \
    azure-identity==1.15.0 \
    scikit-learn==1.4.0 \
    xgboost==2.0.3 \
    mlflow==2.10.0 \
    feast==0.36.0 \
    grpcio==1.60.0 \
    matplotlib==3.8.2 \
    seaborn==0.13.2

# Create non-root user
RUN useradd -m -u 1000 mluser
USER mluser

ENTRYPOINT ["python"]
