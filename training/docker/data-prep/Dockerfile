FROM python:3.10-slim

LABEL maintainer="ml-platform-team"
LABEL description="Data preparation and Feast feature retrieval"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Install feast first (with its dependencies including older pandas)
RUN pip install --no-cache-dir \
    feast>=0.40.0 \
    grpcio==1.60.0 \
    protobuf==4.23.3

# Then upgrade pandas to ensure we have the correct version
RUN pip install --no-cache-dir --upgrade \
    pandas>=2.0.1 \
    pyarrow==14.0.2 \
    adlfs==2024.2.0 \
    azure-storage-blob==12.19.0 \
    azure-identity==1.15.0 \
    scikit-learn==1.4.0

# Create non-root user
RUN useradd -m -u 1000 mluser
USER mluser

ENTRYPOINT ["python"]
