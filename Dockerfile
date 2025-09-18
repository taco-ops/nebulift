# Dockerfile for Nebulift Training - Multi-stage build for optimization
FROM python:3.12-slim-bookworm AS builder

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

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Copy README.md as it's required by pyproject.toml for package building
COPY README.md ./

# Install dependencies (this layer will be cached if dependencies don't change)
RUN uv sync --frozen --no-dev

# Production stage
FROM python:3.12-slim-bookworm AS runtime

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy the virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY nebulift/ ./nebulift/
COPY README.md ./

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
