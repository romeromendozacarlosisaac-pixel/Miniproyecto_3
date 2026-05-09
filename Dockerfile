# =============================================================================
# TELCO CUSTOMER CHURN — DOCKERFILE
# =============================================================================
 
FROM python:3.11-slim AS builder
 
WORKDIR /install
 
COPY requirements.txt .
 
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install/packages -r requirements.txt
 
 
FROM python:3.11-slim AS runtime
 
LABEL maintainer="equipo-mlops"
LABEL project="telco-churn"
LABEL version="2.0.0"
 
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/app/data/models \
    PORT=8000
 
COPY --from=builder /install/packages /usr/local

# Dependencia del sistema requerida por LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app
RUN chown appuser:appuser /app
USER appuser
 
# Copiar código de la aplicación
COPY --chown=appuser:appuser app/ ./app/
 
# Copiar los 4 modelos entrenados
COPY --chown=appuser:appuser data/models/ ./data/models/
 
EXPOSE 8000
 
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
 
CMD ["uvicorn", "app.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]