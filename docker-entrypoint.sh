#!/bin/bash
set -e

# Print environment info
echo "Starting CustomerAI Insights Platform"
echo "Environment: $ENVIRONMENT"
echo "Python: $(python --version)"

# Verify Python version is at least 3.10
if ! python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "ERROR: Python 3.10 or higher is required. Current version: $(python --version)"
    echo "Please update your Python installation or use a compatible container image."
    exit 1
fi

# Create required directories
mkdir -p /var/log/customerai
mkdir -p /data
mkdir -p /exports

# Check for required environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY is not set. AI features may not work correctly."
fi

if [ -z "$JWT_SECRET_KEY" ]; then
    echo "WARNING: JWT_SECRET_KEY is not set. Using an auto-generated key (not suitable for production clusters)."
fi

# Check for database connection
if [ "$ENVIRONMENT" = "production" ]; then
    echo "Checking database connection..."
    python -c "
import os, sys, time
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

db_uri = os.environ.get('DATABASE_URI')
if not db_uri:
    print('DATABASE_URI environment variable not set.')
    sys.exit(1)

# Try to connect to the database with retries
max_retries = 5
retry_interval = 5

for i in range(max_retries):
    try:
        engine = create_engine(db_uri)
        conn = engine.connect()
        conn.close()
        print('Database connection successful.')
        break
    except OperationalError as e:
        print(f'Database connection failed: {e}')
        if i < max_retries - 1:
            print(f'Retrying in {retry_interval} seconds...')
            time.sleep(retry_interval)
        else:
            print('Failed to connect to database after multiple attempts.')
            sys.exit(1)
"
fi

# Apply database migrations if needed
if [ -f "migrations/alembic.ini" ]; then
    echo "Applying database migrations..."
    alembic upgrade head
fi

# Collect static files if needed
if [ "$COLLECT_STATIC" = "true" ]; then
    echo "Collecting static files..."
    python -m app.manage collectstatic --noinput
fi

# Run custom startup script if it exists
if [ -f "/app/startup.sh" ]; then
    echo "Running custom startup script..."
    bash /app/startup.sh
fi

# Execute the main command
echo "Starting main process..."
exec "$@"
