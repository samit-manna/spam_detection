# Feature Transformer Service Docker Image
# Real-time feature extraction for spam detection inference

FROM python:3.10-slim

LABEL maintainer="ML Team"
LABEL description="Feature transformer service for spam detection"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip for better SSL handling and install dependencies with retries
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --retries 5 --timeout 60 -r requirements.txt

# Copy application code
COPY transformer.py .
COPY feast_client.py .

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Create feature_repo directory for Feast config
RUN mkdir -p /feature_repo && chown -R appuser:appuser /feature_repo

USER appuser

# Environment variables (to be overridden at runtime)
# Note: AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY
# must be provided at runtime via -e flags or Kubernetes secrets
ENV PORT=8080
ENV AZURE_STORAGE_CONTAINER="feast"
ENV TFIDF_VECTORIZER_PATH="features/artifacts/tfidf_vectorizer.pkl"
ENV FEAST_ENABLED="false"
ENV FEAST_REDIS_HOST="feast-redis.feast.svc.cluster.local"
ENV FEAST_REDIS_PORT="6379"
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "transformer:app", "--host", "0.0.0.0", "--port", "8080"]
