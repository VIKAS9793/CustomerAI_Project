# Use a specific Python version for reproducibility
FROM python:3.12-slim AS builder

# Set environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.5.1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    libc6-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    # Create a non-root user
    && adduser --disabled-password --gecos "" app

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Set working directory
WORKDIR /app

# Copy only requirements to cache them in docker layer
COPY requirements.txt /app/

# Install dependencies with no additional packages
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install gunicorn \
    # Install security packages
    && pip install argon2-cffi \
    bandit \
    safety

# For the actual application, use a smaller base image
FROM python:3.12-slim

# Set environment variables 
ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
    # Set Python to run in production mode
    PYTHON_ENV=production

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

# Security: Set file permissions
RUN chmod -R 755 /app

# Create data directory with proper permissions
RUN mkdir -p /app/data/cache && \
    chmod -R 755 /app/data

# Switch back to root to run the entry script
USER root

# Make the entry script executable and run security scans
RUN chmod +x /app/docker-entrypoint.sh && \
    # Run security checks
    bandit -r /app -x /app/tests/ || true && \
    safety check || true

# Switch back to non-root user for running the app
USER app

# Expose the application port
EXPOSE 8000

# Run the script that selects between FastAPI and Streamlit
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default to running API server
CMD ["api"] 