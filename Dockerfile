# =====================================================================
# STAGE 1: Dependency Compiler Build Layer
# =====================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# ✅ UPGRADE: Compile wheels directly into a shareable system site-packages folder
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# =====================================================================
# STAGE 2: Lightweight Production Execution Layer
# =====================================================================
FROM python:3.11-slim AS runner

WORKDIR /workspace

# ✅ UPGRADE: Copy installed dependencies into system paths to bypass /root isolation blocks
COPY --from=builder /install /usr/local
COPY requirements.txt .

ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1

# Copy application layers and required machine learning binary objects
COPY app/ ./app/
COPY models/ ./models/

# Expose FastAPI application server ports
EXPOSE 8000

# ✅ UPGRADE: Swapped requests library dependency for native python urllib tracking
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; exit(0 if urllib.request.urlopen('http://localhost:8000/health').getcode() == 200 else 1)"]

# ✅ UPGRADE: Expose system directory permissions so OpenShift non-root users can read workloads
RUN chmod -R g+w /workspace

# Boot the uvicorn server mapping to all network interfaces for container visibility
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
