FROM python:3.10-slim

LABEL maintainer="ML Platform Team"
LABEL description="Baseline Generator for Model Monitoring"

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

# Default command
CMD ["python", "-m", "monitoring.baseline.generator", "--help"]
