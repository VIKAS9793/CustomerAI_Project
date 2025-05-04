# CustomerAI Insights Platform - Troubleshooting Guide

This document provides solutions to common issues you might encounter while setting up, running, or using the CustomerAI Insights Platform.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [API Connection Issues](#api-connection-issues)
3. [Authentication Problems](#authentication-problems)
4. [Data Processing Issues](#data-processing-issues)
5. [AI Model Performance](#ai-model-performance)
6. [Docker Deployment](#docker-deployment)
7. [Database Problems](#database-problems)
8. [Dashboard Issues](#dashboard-issues)
9. [Error Codes Reference](#error-codes-reference)

## Installation Issues

### Package Installation Failures

**Problem**: `pip install -r requirements.txt` fails with dependency conflicts.

**Solution**:
1. Create a fresh virtual environment: `python -m venv venv_new`
2. Activate it: `source venv_new/bin/activate` (Linux/Mac) or `venv_new\Scripts\activate` (Windows)
3. Install packages one group at a time:
   ```
   pip install pandas numpy scikit-learn
   pip install nltk openai
   pip install fastapi uvicorn
   ```
4. If a specific package fails, try installing an earlier version.

### OpenAI Integration Issues

**Problem**: `ModuleNotFoundError: No module named 'openai'` despite installing requirements.

**Solution**:
1. Ensure your virtual environment is activated
2. Install OpenAI specifically: `pip install openai==0.27.0`
3. Verify with `pip list | grep openai`

## API Connection Issues

### Connection Refused

**Problem**: API returns "Connection refused" errors.

**Solution**:
1. Verify the API server is running: `ps aux | grep uvicorn`
2. Check if the port is in use: `netstat -tuln | grep 8000`
3. Ensure firewall settings allow connections: `sudo ufw status`
4. Try changing the port: `uvicorn api.main:app --port 8001`

### Timeout Issues

**Problem**: API requests timeout after 30 seconds.

**Solution**:
1. Check network connectivity
2. Increase client timeout settings
3. For heavy processing endpoints, implement asynchronous processing:
   ```python
   # In your API client
   response = requests.post(url, json=data, timeout=60)
   ```

## Authentication Problems

### Invalid Token Errors

**Problem**: Getting "Invalid token" or "Token expired" errors.

**Solution**:
1. Verify your JWT token hasn't expired
2. Check that your `JWT_SECRET_KEY` environment variable is consistent across environments
3. Ensure clock synchronization between servers
4. Request a new token and try again

### Permission Denied

**Problem**: Getting "Insufficient permissions" errors despite valid token.

**Solution**:
1. Verify your user account has the required roles
2. Check token payload for correct role claims:
   ```python
   import jwt
   
   token = "your.jwt.token"
   # Use the same secret key used for token generation
   payload = jwt.decode(token, "your-secret-key", algorithms=["HS256"])
   print(payload)  # Check 'roles' field
   ```
3. Contact your administrator for role assignment

## Data Processing Issues

### Memory Errors During Batch Processing

**Problem**: Large dataset processing causes out-of-memory errors.

**Solution**:
1. Reduce batch size in API requests
2. Implement chunking in your client code:
   ```python
   def process_in_chunks(data, chunk_size=1000):
       results = []
       for i in range(0, len(data), chunk_size):
           chunk = data[i:i+chunk_size]
           response = requests.post(url, json={"data": chunk})
           results.extend(response.json()["data"]["results"])
       return results
   ```

### Sentiment Analysis Inaccuracy

**Problem**: Sentiment analysis doesn't correctly capture domain-specific sentiment.

**Solution**:
1. Set `use_ai: true` in your request for financial domain expertise
2. Provide context in your requests if available
3. Consider fine-tuning models with your own labeled data

## AI Model Performance

### Slow Response Generation

**Problem**: AI-powered response generation is taking too long.

**Solution**:
1. Check your OpenAI API key quota and limits
2. Use caching for common queries
3. Implement retry logic with exponential backoff:
   ```python
   from tenacity import retry, stop_after_attempt, wait_exponential
   
   @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
   def generate_response_with_retry(query):
       # Your API call here
       pass
   ```

### OpenAI API Key Issues

**Problem**: "Invalid API key" or "API key quota exceeded" errors.

**Solution**:
1. Verify your OpenAI API key in environment variables
2. Check billing status in OpenAI dashboard
3. Use API key rotation for high-volume applications

## Docker Deployment

### Container Won't Start

**Problem**: Docker container exits immediately after starting.

**Solution**:
1. Check container logs: `docker logs <container-id>`
2. Verify environment variables are properly set
3. Check if DB connection is available
4. Ensure docker-entrypoint.sh has execute permissions: `chmod +x docker-entrypoint.sh`

### Memory Limits

**Problem**: Container crashes with OOM (Out of Memory) errors.

**Solution**:
1. Increase container memory limit:
   ```
   docker run --memory=2g --memory-swap=2g customerai-insights
   ```
2. Or in docker-compose.yml:
   ```yaml
   services:
     app:
       image: customerai-insights
       deploy:
         resources:
           limits:
             memory: 2G
   ```

## Database Problems

### Migration Errors

**Problem**: Database migration fails with "Table already exists" errors.

**Solution**:
1. Check if you're running migrations twice
2. For development, you might want to drop and recreate the database:
   ```
   # SQLite
   rm customerai_dev.db
   python -m scripts.init_db
   
   # PostgreSQL
   psql -U postgres -c "DROP DATABASE customerai; CREATE DATABASE customerai;"
   alembic upgrade head
   ```

### Connection Pool Exhaustion

**Problem**: "Too many connections" or connection timeout errors.

**Solution**:
1. Check for connection leaks in your code
2. Adjust pool settings in config:
   ```python
   # In config/config.py
   DATABASE_POOL_SIZE = 10
   DATABASE_MAX_OVERFLOW = 20
   ```
3. Implement connection release in your exception handlers

## Dashboard Issues

### Dashboard Not Loading Data

**Problem**: Streamlit dashboard shows loading spinner but never displays data.

**Solution**:
1. Check browser console for CORS errors
2. Verify API is accessible from dashboard host
3. Check authentication tokens in dashboard session
4. Try clearing browser cache or using incognito mode

### Visualization Rendering Issues

**Problem**: Charts or graphs not rendering correctly.

**Solution**:
1. Update Plotly and Streamlit to latest versions
2. Check for data format issues
3. Verify JavaScript is enabled in browser
4. Try a different browser

## Error Codes Reference

| Error Code | Description | Troubleshooting |
|------------|-------------|-----------------|
| `AUTH001`  | Invalid credentials | Check username/password |
| `AUTH002`  | Token expired | Request a new token |
| `AUTH003`  | Insufficient permissions | Request role elevation |
| `API001`   | Rate limit exceeded | Reduce request frequency |
| `API002`   | Invalid request format | Check request JSON format |
| `DATA001`  | Invalid data format | Check input data structure |
| `AI001`    | OpenAI API error | Check API key and quota |
| `AI002`    | Model generation timeout | Retry or simplify request |
| `DB001`    | Database connection error | Check DB credentials and availability |
| `DB002`    | Query timeout | Optimize query or add indexes |

## Getting Additional Help

If you're still experiencing issues after trying these solutions, please:

1. Check the project's GitHub Issues page for similar problems
2. Open a new issue with detailed information:
   - Error messages and stack traces
   - Steps to reproduce
   - Environment details (OS, Python version, package versions)
3. For urgent support, contact the development team at support@customerai-insights.example.com 