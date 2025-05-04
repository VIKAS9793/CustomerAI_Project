# CustomerAI Insights Platform - Configuration Guide

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
| `DATABASE_URI` | Database connection URI | `sqlite:///customerai_dev.db` | Yes |
| `DATABASE_POOL_SIZE` | Connection pool size | `5` | No |
| `DATABASE_MAX_OVERFLOW` | Maximum pool overflow | `10` | No |

### API Keys and Security

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key | None | Yes |
| `JWT_SECRET_KEY` | Secret key for JWT token generation | Auto-generated (not for production) | Yes (for production) |
| `ENCRYPTION_KEY` | Key for sensitive data encryption | None | Yes (for production) |
| `CORS_ORIGINS` | Allowed CORS origins (comma-separated) | `*` | No |

### Logging

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | `INFO` | No |
| `LOG_FORMAT` | Log format (standard, json) | `standard` | No |
| `LOG_DIR` | Directory for log files | `logs` | No |

### Feature Flags

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENABLE_HUMAN_REVIEW` | Enable human review workflow | `true` | No |
| `ENABLE_BIAS_DETECTION` | Enable bias detection | `true` | No |
| `ENABLE_PRIVACY_FEATURES` | Enable privacy protection | `true` | No |

### AI Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `AI_MAX_TOKENS` | Maximum tokens for AI responses | `500` | No |
| `AI_TEMPERATURE` | Temperature for response generation | `0.7` | No |
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
  "DATABASE_URI": "postgresql://username:password@localhost:5432/customerai",
  "OPENAI_API_KEY": "your-api-key",
  "JWT_SECRET_KEY": "your-secret-key-here",
  "LOG_LEVEL": "INFO",
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
- SQLite database
- Relaxed security settings

To activate:
```
export ENVIRONMENT=development
```

### Testing Environment

Optimized for automated testing with:
- In-memory database
- Mock AI responses
- Test secret keys
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
- Required environment variables for secrets

To activate:
```
export ENVIRONMENT=production
export DATABASE_URI=postgresql://user:password@dbhost:5432/customerai
export OPENAI_API_KEY=your-api-key
export JWT_SECRET_KEY=your-jwt-secret
export ENCRYPTION_KEY=your-encryption-key
```

### Docker Environment

Extends production configuration with Docker-specific settings:
- Paths configured for container volumes
- Database connection to container services
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

Configure which OpenAI models to use:

```python
# In config/config.py
AI_MODELS = {
    "sentiment": "gpt-4",
    "response_generation": "gpt-4",
    "compliance_validation": "gpt-4",
    "bias_detection": "gpt-4"
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
    "notification_email": "reviewers@example.com"
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

1. Verify the `DATABASE_URI` is correct
2. Ensure the database server is running
3. Check firewall settings
4. Verify database credentials

### OpenAI API Errors

If you experience OpenAI API errors:

1. Verify your `OPENAI_API_KEY` is valid
2. Check your API usage and limits
3. Check if you have billing enabled on your OpenAI account
4. Set `AI_MOCK=true` for testing without API calls

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