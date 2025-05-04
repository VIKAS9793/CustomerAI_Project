# Use a specific Python version for reproducibility
FROM python:3.10-slim AS builder

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

# Copy only the dependency files first for better caching
COPY pyproject.toml poetry.lock* ./

# Configure poetry to not use a virtual environment and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-dev

# Now copy the rest of the application
COPY . .

# Build stage for production
FROM python:3.10-slim AS production

# Set environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies including those for onnxruntime-gpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Required for OpenAI's tiktoken
    build-essential \
    # Required for PostgreSQL
    libpq-dev \
    # For Redis
    redis-tools \
    # For security
    ca-certificates \
    # For node-based frontends
    curl \
    # For modern cryptography
    libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    # Install node for the modern web interface (human review dashboard)
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    # Create a non-root user
    && adduser --disabled-password --gecos "" app

# Set working directory
WORKDIR /app

# Copy from builder stage
COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install additional packages for human-in-the-loop and AI safety
RUN pip install --no-cache-dir \
    tiktoken>=0.5.1 \
    pypdf>=3.15.1 \
    slack-sdk>=3.21.3 \
    # Database dependencies for the review system
    databases[postgresql,sqlite]>=0.8.0 \
    psycopg2-binary>=2.9.7 \
    asyncpg>=0.28.0 \
    # Modern cryptography for token handling
    cryptography>=42.0.0 \
    # Explainability tools
    shap>=0.42.1 \
    lime>=0.2.0.1 \
    # Frontend for human review dashboard
    streamlit>=1.32.0 \
    # Healthcheck
    pytest-timeout>=2.1.0

# Set up the model cards directory
RUN mkdir -p /app/data/model_cards /app/data/human_feedback && \
    chown -R app:app /app/data

# Application port
EXPOSE 8000
# Review dashboard port
EXPOSE 8501

# Switch to non-root user
USER app

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set up entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# GPU-enabled build for inference
FROM production AS gpu

USER root

# Install CUDA dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg2 \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-minimal-build-11-8 \
        libcudnn8 \
    && rm -rf /var/lib/apt/lists/* \
    && rm cuda-keyring_1.0-1_all.deb

# Install GPU-enabled packages
RUN pip install --no-cache-dir \
    torch>=2.2.0 --extra-index-url https://download.pytorch.org/whl/cu118 \
    onnxruntime-gpu>=1.16.0 \
    tensorflow>=2.15.0 \
    triton>=2.1.0

# Switch back to app user
USER app

# Set environment variables for GPU use
ENV CUDA_VISIBLE_DEVICES=0

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