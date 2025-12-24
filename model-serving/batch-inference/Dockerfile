# Batch Inference Docker Image
# Ray-based batch prediction for spam detection

FROM rayproject/ray:2.9.0-py310

LABEL maintainer="ML Team"
LABEL description="Ray batch inference for spam detection"

WORKDIR /app

# Install system dependencies (as root)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install additional dependencies on top of Ray image
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY batch_predict.py .

# Use the existing ray user (UID 1000 already exists in base image)
RUN chown -R 1000:1000 /app
USER 1000

# Environment variables (to be overridden at runtime)
# Note: MLFLOW_TRACKING_URI, AZURE_STORAGE_ACCOUNT_NAME, and AZURE_STORAGE_ACCOUNT_KEY
# must be provided at runtime via -e flags or Kubernetes secrets
ENV AZURE_STORAGE_CONTAINER="datasets"
ENV FEAST_CONTAINER="feast"
ENV MODELS_CONTAINER="models"
ENV NUM_WORKERS="4"
ENV PYTHONUNBUFFERED=1

# NOTE: No ENTRYPOINT set - RayJob needs to be able to run 'ray job submit'
# The job submission is handled by KubeRay operator
# For standalone execution, use: docker run ... python batch_predict.py --args
CMD ["python", "batch_predict.py", "--help"]
