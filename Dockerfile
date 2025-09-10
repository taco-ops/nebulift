# Dockerfile for Nebulift Training
FROM python:3.12-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies for RPi5 compatibility
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY nebulift/ ./nebulift/
COPY README.md ./

# Install Python dependencies
RUN uv sync --frozen

# Create non-root user for security
RUN useradd -m -u 1000 nebulift && chown -R nebulift:nebulift /app
USER nebulift

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; import nebulift; print('Nebulift ready')"

# Default command
CMD ["python", "-m", "nebulift.distributed.k8s_trainer"]
