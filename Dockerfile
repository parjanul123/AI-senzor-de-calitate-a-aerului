# Multi-stage build
FROM python:3.11-slim as builder

WORKDIR /app

# Install minimal build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --user --no-cache-dir -r /app/requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy Python packages
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY app /app/app
COPY models /app/models

# Set environment
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=2 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Start app
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
