# syntax=docker/dockerfile:1.7
#
# Baked MLflow tracking server image with boto3 for MinIO-backed artifact
# storage. Replaces the Phase 1a runtime `pip install mlflow boto3` shim that
# ran on top of a vanilla python:3.11-slim Deployment.
#
# Why we maintain our own image instead of using ghcr.io/mlflow/mlflow:
#   - The upstream image does not bundle boto3, so S3/MinIO artifact storage
#     requires an init container or a runtime pip install.
#   - Phase 1a worked around this by installing mlflow + boto3 at pod start.
#     That added ~20-30s of cold-start latency, made the platform vulnerable
#     to PyPI outages, and prevented `readOnlyRootFilesystem: true`.
#   - Baking the image lets us pin versions deterministically, lock down the
#     container filesystem, and pull from GHCR (same registry as the trainer).
#
# Version coupling:
#   MLFLOW_VERSION here MUST stay compatible with the trainer client floor in
#   pyproject.toml (`mlflow>=2.20.0,<3.0.0`). Bumping requires a coordinated
#   trainer + image release; see docs/PHASE_1B_PLATFORM.md for the procedure.

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim-bookworm

ARG MLFLOW_VERSION=2.20.0
# boto3 is unpinned within the 1.x line so security patches flow on rebuild
# without breaking the MLflow S3 artifact store contract.
ARG BOTO3_SPEC="boto3>=1.34,<2"
ARG APP_UID=1000
ARG APP_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create a real non-root user matching the Deployment's runAsUser/runAsGroup
# so `docker run` ergonomics line up with in-cluster behavior. K8s overrides
# USER via securityContext, but having a real /etc/passwd entry avoids any
# tool that resolves UID -> name from blowing up.
RUN groupadd --system --gid "${APP_GID}" mlflow \
 && useradd  --system --uid "${APP_UID}" --gid "${APP_GID}" \
             --home-dir /home/mlflow --create-home \
             --shell /usr/sbin/nologin mlflow

# Install mlflow + boto3 into the system site-packages so the non-root runtime
# user does not need a writable HOME. This is what enables
# `readOnlyRootFilesystem: true` in the Deployment.
RUN pip install --no-cache-dir \
        "mlflow==${MLFLOW_VERSION}" \
        "${BOTO3_SPEC}"

USER mlflow:mlflow
WORKDIR /home/mlflow

EXPOSE 5000

# Image-level healthcheck mirrors the K8s probe so `docker run` users get the
# same signal locally. In-cluster readiness/liveness probes remain
# authoritative; kubelet ignores HEALTHCHECK directives.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3).status == 200 else 1)" \
    || exit 1

# ENTRYPOINT lets the Deployment pass clean `args: [server, --backend-store-uri, ...]`
# without a bash wrapper. Override at runtime if a different mlflow subcommand
# is ever needed (e.g. `mlflow db upgrade`).
ENTRYPOINT ["mlflow"]
CMD ["--help"]
