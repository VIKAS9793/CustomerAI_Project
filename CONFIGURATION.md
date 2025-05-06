# CustomerAI Insights Platform - Configuration Guide
*Last Updated: May 5, 2025*

## Overview

This guide details the configuration options and best practices for the CustomerAI project. The project uses a hierarchical configuration system that supports multiple sources (environment variables, configuration files, and code-based settings).

## Core Configuration

### Environment Variables

Copy the `env.example` file to `.env` and configure the following required variables:

```bash
# API Configuration
PORT=8000
DEBUG=false
API_KEY=your_secure_api_key

# Security Settings
SECRET_KEY=your_secure_secret_key
TOKEN_EXPIRATION=3600
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=customerai
DB_USER=your_db_user
DB_PASSWORD=your_db_password

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Fairness Framework Configuration

The fairness framework can be configured through `config/fairness_config.py`:

```python
FAIRNESS_CONFIG = {
    'bias_detection': {
        'threshold': 0.15,
        'significance_level': 0.05,
        'metrics': ['disparate_impact', 'equal_opportunity']
    },
    'mitigation': {
        'strategies': ['reweighing', 'prejudice_remover'],
        'max_iterations': 100
    },
    'dashboard': {
        'page_size': 1000,
        'cache_timeout': 3600
    }
}
```

## Security Configuration

### Rate Limiting

Rate limiting is configured through environment variables and can be customized per endpoint:

```python
RATE_LIMIT_CONFIG = {
    'default': {
        'requests': 100,
        'window': 3600
    },
    '/api/v1/generate': {
        'requests': 50,
        'window': 3600
    }
}
```

### Device Security

Device validation is enforced through headers:
- X-Device-Id
- X-Device-Model
- X-OS-Version
- X-App-Version

Configure validation rules in `config/security_config.py`.

## Performance Tuning

### Dashboard Configuration

Optimize dashboard performance through:

```python
DASHBOARD_CONFIG = {
    'cache_enabled': True,
    'cache_timeout': 3600,
    'page_size': 1000,
    'max_memory_usage': '2GB'
}
```

### API Performance

Tune API performance with:

```python
API_CONFIG = {
    'worker_count': 4,
    'timeout': 30,
    'keep_alive': True,
    'max_request_size': '10MB'
}
```

## Logging Configuration

Configure logging through `config/logging_config.py`:

```python
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(timestamp)s %(level)s %(name)s %(message)s'
        }
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'level': 'INFO',
            'formatter': 'json',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/customerai.log',
            'maxBytes': 10485760,
            'backupCount': 5
        }
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}
```

## Testing Configuration

Configure test settings in `config/test_config.py`:

```python
TEST_CONFIG = {
    'mock_data_size': 1000,
    'test_timeout': 30,
    'coverage_threshold': 80,
    'parallel_tests': True
}
```

## Production Deployment

### Docker Configuration

Customize `docker-compose.yml` for production:

```yaml
version: '3.8'
services:
  api:
    build: .
    environment:
      - PORT=8000
      - DEBUG=false
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Best Practices

1. **Security**
   - Never commit sensitive data
   - Rotate secrets regularly
   - Use strong, unique keys
   - Enable rate limiting

2. **Performance**
   - Monitor memory usage
   - Configure appropriate cache settings
   - Tune database connections
   - Set reasonable timeouts

3. **Logging**
   - Use appropriate log levels
   - Implement log rotation
   - Configure structured logging
   - Monitor log storage

4. **Testing**
   - Maintain high coverage
   - Use realistic test data
   - Configure CI/CD properly
   - Regular security scans

> For local development, add credentials to your own untracked files. Do not commit secret-like patterns.

This document provides detailed information about configuring and customizing the CustomerAI Insights Platform for various environments and use cases.

## Configuration Systems

The platform uses a multi-layered configuration system:

1. **Environment Variables**: For sensitive data and environment-specific settings
2. **Configuration Files**: For detailed application settings
3. **Database Configuration**: For runtime configurable settings
4. **Feature Flags**: For enabling/disabling features

## Environment Variables

### Core Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Deployment environment (development, testing, production, docker) | `development` | No |
| `HOST` | API server host | `0.0.0.0` | No |
| `PORT` | API server port | `8000` | No |
| `WORKERS` | Number of worker processes | `4` | No |

### Database Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|

### AI Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `AI_MAX_TOKENS` | Maximum tokens for AI responses | `500` | No |
| `AI_REQUEST_TIMEOUT` | Timeout for AI API requests (seconds) | `30` | No |
| `AI_MOCK` | Use mock AI responses (for testing) | `false` | No |

## Configuration File

The platform also supports a JSON configuration file that can be specified with the `CONFIG_FILE` environment variable:

```json
{
  "APP_NAME": "CustomerAI Insights Platform",
  "APP_VERSION": "1.0.0",
  "DEBUG": false,
  "API_PREFIX": "/api/v1",
  "API_RATE_LIMIT": 100,

  "WORKER_THREADS": 4,
  "FEATURES": {
    "human_in_loop": true,
    "bias_detection": true,
    "privacy_features": true,
    "advanced_analytics": true,
    "export_reports": true
  }
}
```

## Environment-Specific Configurations

The platform includes pre-defined configurations for different environments:

### Development Environment

Optimized for local development with:
- Debug mode enabled
- Verbose logging
- Relaxed security settings

To activate:
```
export ENVIRONMENT=development
```

### Testing Environment

Optimized for automated testing with:
- In-memory database
- Mock AI responses
- Test settings
- All features enabled

To activate:
```
export ENVIRONMENT=testing
```

### Production Environment

Optimized for production deployment with:
- Performance optimizations
- Strict security settings
- Warning-level logging in JSON format

To activate:
```
export ENVIRONMENT=production
```

### Docker Environment

Extends production configuration with Docker-specific settings:
- Paths configured for container volumes
- Log files directed to mounted volumes

## Advanced Configuration

### Database Connection Pool

Fine-tune database connection pooling for optimal performance:

```python
# In config/config.py
DATABASE_POOL_SIZE = 20  # Increase for high-concurrency environments
DATABASE_MAX_OVERFLOW = 30  # Allow more temporary connections during spikes
```

### API Rate Limiting

Configure rate limiting to prevent abuse:

```python
# In config/config.py
API_RATE_LIMIT = 200  # Requests per hour per client
```

You can also set different rate limits for specific endpoints:

```python
# In api/endpoints/sentiment.py
@app.post("/analyze/sentiment")
@rate_limit(limit=50, window=3600)  # 50 requests per hour
def analyze_sentiment():
    # ...
```

### Sentiment Analysis Thresholds

Customize sentiment classification thresholds:

```python
# In config/config.py
SENTIMENT_THRESHOLDS = {
    "positive": 0.6,  # Text with score >= 0.6 is positive
    "negative": 0.4,  # Text with score <= 0.4 is negative
    # Between 0.4 and 0.6 is neutral
}
```

### Customizing AI Models

Configure which AI models to use:

```python
# In config/config.py
AI_MODELS = {
    "sentiment": "",
    "response_generation": "",
    "compliance_validation": "",
    "bias_detection": ""
}
```

### Human Review Configuration

Configure the human review workflow:

```python
# In config/config.py
HUMAN_REVIEW = {
    "queue_limit": 1000,  # Maximum items in review queue
    "auto_approve_threshold": 0.95,  # Auto-approve items with confidence >= 0.95
    "high_priority_threshold": 0.3,  # Items with confidence <= 0.3 are high priority
}
```

## Feature Flag System

The platform uses a feature flag system to enable/disable features without code changes:

```python
# Check if a feature is enabled
from config.config import current_config

if current_config.FEATURES.get("human_in_loop"):
    # Enable human review functionality
    # ...
```

Available feature flags:
- `human_in_loop`: Human review workflow
- `bias_detection`: Bias detection and fairness analysis
- `privacy_features`: PII detection and anonymization
- `advanced_analytics`: Advanced analytics dashboard
- `export_reports`: Report export functionality

## Logging Configuration

Configure logging for different components:

```python
# In your module
from src.utils.logger import get_logger

# Component-specific logger with custom settings
logger = get_logger(
    'sentiment_analyzer',
    log_level=logging.DEBUG,
    log_file='logs/sentiment.log',
    json_format=True
)

# Use the logger
logger.info("Analyzing sentiment", extra={"query_id": "123"})
```

## Monitoring and Metrics

Configure application metrics collection:

```python
# In config/config.py
METRICS = {
    "enabled": True,
    "statsd_host": "localhost",
    "statsd_port": 8125,
    "prefix": "customerai"
}
```

## Configuration Best Practices

1. **Never commit secrets** to version control
2. Use **environment variables** for sensitive information
3. Use **feature flags** for enabling/disabling functionality
4. Implement **validation** for all configuration values
5. Use **reasonable defaults** to minimize required configuration
6. Provide **clear error messages** for misconfiguration
7. Include **configuration documentation** in code comments

## Troubleshooting Configuration Issues

### Missing Environment Variables

If the application fails to start with an error about missing environment variables:

1. Check that your `.env` file exists and is in the correct location
2. Verify that environment variables are properly set
3. For Docker, ensure environment variables are passed to the container

### Database Connection Issues

If you encounter database connection errors:

1. Verify the database server is running
2. Check firewall settings
3. Verify database credentials

### AI and ML Configuration

### JAX Configuration

JAX is used for high-performance numerical computing. Configure JAX specific settings in your `.env` file:

```

```

## AI and ML Configuration

### LLM Provider Configuration

The CustomerAI Insights Platform supports multiple LLM providers that can be configured in various ways:

> **Disclaimer for Developers**: The LLM integration system is designed to be highly customizable according to your specific business requirements and use cases. Developers have complete flexibility to configure token limits, model selection, temperature settings, system prompts, and other parameters for each provider. This allows for optimizing different aspects such as cost management, inference speed, output quality, and regulatory compliance based on your organization's priorities.

#### Configuration File

Create a JSON file at `config/llm_config.json`:

```json
{
  "default_client": "gpt4o_financial",
  "clients": {
    "gpt4o_financial": {
      "provider": "openai",
      "model_name": "gpt-4o",
      "compliance_level": "financial",
      "capabilities": ["text_generation", "embeddings", "classification"],
      "additional_params": {
        "user": "financial-services",
        "safe_mode": true
      }
    },
    "claude_sonnet": {
      "provider": "anthropic",
      "model_name": "claude-3-5-sonnet-20240620",
      "compliance_level": "financial",
      "capabilities": ["text_generation", "classification"]
    },
    "gemini_pro": {
      "provider": "google",
      "model_name": "gemini-1.5-pro",
      "compliance_level": "standard",
      "capabilities": ["text_generation", "embeddings", "classification"]
    }
  }
}
```

#### Environment Variables

Configure your `.env` file with the following variables:

```
# LLM Provider API Keys
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_ORGANIZATION=your_openai_org_id  # Optional

ANTHROPIC_API_KEY=your_anthropic_api_key_here

GOOGLE_API_KEY=your_google_ai_api_key_here

# Azure OpenAI (Optional)
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
AZURE_OPENAI_API_VERSION=2023-12-01-preview

# LLM Configuration Path
LLM_CONFIG_PATH=config/llm_config.json

# LLM Security Settings
LLM_ALLOW_USER_DATA=false
LLM_LOG_PROMPTS=false
LLM_LOG_COMPLETIONS=false
LLM_MAX_TOKENS=4000
LLM_REQUEST_TIMEOUT=60
LLM_ENABLE_CONTENT_FILTERING=true
```

#### Runtime Configuration

You can also configure LLM providers programmatically:

```python
from cloud.ai.llm_provider import LLMProvider, LLMConfig, LLMComplianceLevel, LLMCapability
from cloud.ai.llm_manager import get_llm_manager

# Get the LLM manager
llm_manager = get_llm_manager()

# Register a financial compliance client
llm_manager.register_financial_client(
    client_id="financial_compliance",
    provider=LLMProvider.ANTHROPIC,
    model_name="claude-3-5-sonnet-20240620"
)

# Set as default
llm_manager.set_default_client("financial_compliance")
```

### JAX Configuration

JAX is used for high-performance numerical computing. Configure JAX specific settings in your `.env` file:

```
