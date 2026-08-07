# Multi-stage build - optimized for Railway's constraints
FROM python:3.11-slim AS builder

WORKDIR /app

# Install only minimal build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements early
COPY requirements.txt .

# Install Python packages with aggressive optimization
# Use only pre-built wheels, skip pip cache, increase timeout
RUN pip install --no-cache-dir \
    --only-binary :all: \
    --default-timeout=1000 \
    --retries 3 \
    -r requirements.txt

# Final stage - minimal runtime
FROM python:3.11-slim

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application
COPY app ./app
COPY models ./models

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=2 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Start
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
