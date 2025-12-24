# ====================================
# Stage 1: Builder
# ====================================
FROM python:3.11-slim as builder

# Set build-time environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=2.1.1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry (pinned version) and export plugin
RUN curl -sSL https://install.python-poetry.org | python3 - --version $POETRY_VERSION && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry && \
    poetry self add poetry-plugin-export

# Set working directory
WORKDIR /build

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Export dependencies to requirements.txt (no dev deps)
RUN poetry config virtualenvs.create false && \
    poetry export -f requirements.txt --output requirements.txt --without-hashes --without dev

# Build wheels for all dependencies
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ====================================
# Stage 2: Runtime
# ====================================
FROM python:3.11-slim

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install runtime dependencies (PostgreSQL client libs only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser -u 1000 -m -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy wheels and requirements from builder
COPY --from=builder /wheels /wheels
COPY --from=builder /build/requirements.txt .

# Install dependencies from wheels using requirements.txt (controlled install)
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt && \
    rm -rf /wheels

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port (documentation only)
EXPOSE 8000

# Health check using socket (faster, no DNS issues)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', 8000)); s.close()" || exit 1

# Run with Uvicorn
# For production with multiple workers, override in docker-compose or k8s:
#   uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
# Or use Gunicorn:
#   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
