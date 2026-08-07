# Multi-stage build pentru optimizare
FROM python:3.11-slim AS builder

WORKDIR /app

# Instaleaza dependențe de build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie requirements și instaleaza dependențele
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage final
FROM python:3.11-slim

WORKDIR /app

# Instaleaza dependențe runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copie Python packages din builder
COPY --from=builder /root/.local /root/.local

# Copie aplicația
COPY app ./app
COPY models ./models

# Setează PATH pentru pip packages
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Expune portul (PORT var din Railway)
EXPOSE 8000

# Comanda de start - acceptă PORT din variabile de mediu
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
