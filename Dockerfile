# Use a specific Python version for reproducibility
FROM python:3.12-slim AS builder

# Set environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.7.1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    libc6-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    # Create a non-root user
    && adduser --disabled-password --gecos "" app

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Set working directory
WORKDIR /app

# Copy only requirements to cache them in docker layer
COPY requirements.txt /app/

# Project initialization:
RUN pip install --no-cache-dir wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt \
    && pip install gunicorn \
    # Install security packages
    && pip install argon2-cffi \
    bandit \
    safety \
    trivy-python

# For the actual application, use a clean base image
FROM python:3.12-slim

LABEL maintainer="Vikas Sahani <vikassahani17@gmail.com>" \
      org.opencontainers.image.source="https://github.com/VIKAS9793/CustomerAI_Project" \
      org.opencontainers.image.description="CustomerAI Insights Platform: Enterprise-ready AI for customer analytics" \
      org.opencontainers.image.licenses="MIT"

# Set environment variables 
ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
    # Set Python to run in production mode
    PYTHON_ENV=production \
    # Set timezone
    TZ=UTC

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    # Set timezone
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Create a non-root user and working directory
RUN adduser --disabled-password --gecos "" app
WORKDIR /app
USER app

# Copy Python dependencies from builder image
COPY --from=builder --chown=app:app /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder --chown=app:app /usr/local/bin /usr/local/bin

# Copy the application code
COPY --chown=app:app . /app/

# Security: remove unnecessary files
RUN find /app -type d -name __pycache__ -exec rm -rf {} +

# Create necessary directories with proper permissions
RUN mkdir -p /app/data/cache /app/logs /app/models

# Switch back to root to run the entry script
USER root

# Set file permissions
RUN chmod -R 755 /app && \
    chmod 444 /app/requirements.txt && \
    chmod 444 /app/Dockerfile && \
    chmod +x /app/docker-entrypoint.sh && \
    # Run security checks
    bandit -r /app -x /app/tests/ || true && \
    safety check || true

# Switch back to non-root user for running the app
USER app

# Set container health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose the application port
EXPOSE 8000

# Use tini as entrypoint to properly handle signals and zombie processes
ENTRYPOINT ["/usr/bin/tini", "--", "/app/docker-entrypoint.sh"]

# Default to running API server
CMD ["api"]

# -------------------- GPU SUPPORT (Optional) --------------------
# FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS gpu-runtime

# # Copy everything from the base image
# COPY --from=0 / /

# # Set environment variables for GPU
# ENV NVIDIA_VISIBLE_DEVICES=all \
#     NVIDIA_DRIVER_CAPABILITIES=compute,utility \
#     NVIDIA_REQUIRE_CUDA="cuda>=12.0"

# # Re-expose port and set entrypoint/cmd from the base image
# EXPOSE 8000
# ENTRYPOINT ["/usr/bin/tini", "--", "/app/docker-entrypoint.sh"]
# CMD ["api"]
# ----------------------------------------------------------------- 