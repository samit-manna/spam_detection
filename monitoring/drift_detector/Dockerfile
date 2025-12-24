FROM python:3.10-slim

LABEL maintainer="ML Platform Team"
LABEL description="Drift Detector for Model Monitoring"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from monitoring directory
COPY monitoring/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy monitoring module
COPY monitoring/ /app/monitoring/

# Set Python path
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import monitoring.drift_detector" || exit 1

# Default command - runs drift detection
CMD ["python", "-m", "monitoring.drift_detector.detector", "--help"]
