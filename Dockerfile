# Dockerfile for Nebulift Training - Simplified build for reliability
FROM python:3.12-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv package manager
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy all necessary files for package building
COPY pyproject.toml uv.lock README.md ./
COPY nebulift/ ./nebulift/

# Install dependencies
RUN uv sync --frozen --no-dev

# Create non-root user for security
RUN useradd -m -u 1000 nebulift && chown -R nebulift:nebulift /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

# Switch to non-root user
USER nebulift

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; import nebulift; print('Nebulift ready')" || exit 1

# Default command
CMD ["python", "-m", "nebulift.distributed.k8s_trainer"]
