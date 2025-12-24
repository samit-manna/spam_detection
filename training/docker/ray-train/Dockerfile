FROM rayproject/ray:2.9.0-py310

LABEL maintainer="ml-platform-team"
LABEL description="Ray worker image for ML training and HPO"

WORKDIR /app

# Install system dependencies (as root)
USER root
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
    ray[tune]==2.9.0 \
    ray[train]==2.9.0 \
    hyperopt==0.2.7 \
    optuna==3.5.0

# Copy training scripts
COPY scripts/ /app/scripts/

# Switch back to ray user
USER ray

ENV PYTHONPATH=/app:$PYTHONPATH
