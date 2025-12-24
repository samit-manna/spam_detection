# Model Export Docker Image
# Converts MLflow XGBoost model to ONNX and uploads to Azure Blob Storage

FROM python:3.10-slim

LABEL maintainer="ML Team"
LABEL description="MLflow to ONNX model export for Triton Inference Server"

# Set working directory
WORKDIR /app

# Install system dependencies
# libgomp1 is required by onnxruntime for OpenMP support
# Additional libs needed for onnxruntime native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    libstdc++6 \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip for better SSL handling and install dependencies with retries
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --retries 5 --timeout 60 -r requirements.txt

# Copy application code
COPY export_model.py .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Environment variables (to be overridden at runtime)
# Note: MLFLOW_TRACKING_URI, AZURE_STORAGE_ACCOUNT_NAME, and AZURE_STORAGE_ACCOUNT_KEY
# must be provided at runtime via -e flags or Kubernetes secrets
ENV AZURE_STORAGE_CONTAINER="models"
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import mlflow; import onnx; print('OK')" || exit 1

# Default command
ENTRYPOINT ["python", "export_model.py"]
CMD ["--model-name", "spam-detector", "--model-stage", "Staging"]
