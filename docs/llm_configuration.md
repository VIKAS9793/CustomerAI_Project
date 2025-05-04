# LLM Configuration Guide

This document provides detailed information about configuring Large Language Models (LLMs) in the CustomerAI Insights Platform.

## Supported LLM Providers

The platform currently supports the following LLM providers:

| Provider | Models | Best Use Cases | Requirements |
|----------|--------|----------------|--------------|
| **OpenAI** | GPT-4o, GPT-4-turbo, text-embedding-3-large | General text generation, embeddings, code generation | API key |
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Opus | Financial compliance, long document processing | API key |
| **Google** | Gemini 1.5 Pro, Gemini 1.5 Flash | Multimodal processing, cost-effective deployments | API key |
| **Azure OpenAI** | All OpenAI models via Azure | Enterprise deployments with data residency requirements | API key, endpoint URL |
| **Cohere** | Command, Embed | Embeddings, multilingual support | API key |

## Configuration Methods

You can configure LLMs in three ways:

1. **Environment Variables**: Set API keys and basic configuration in .env file
2. **Configuration File**: Create a detailed `config/llm_config.json` file
3. **Programmatic Configuration**: Configure at runtime using the LLM Manager API

### 1. Environment Variables

In your `.env` file, set the following variables:

```
# OpenAI
OPENAI_API_KEY=your_key_here
OPENAI_ORGANIZATION=your_org_id  # Optional

# Anthropic
ANTHROPIC_API_KEY=your_key_here

# Google
GOOGLE_API_KEY=your_key_here

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2023-12-01-preview

# LLM Configuration
LLM_CONFIG_PATH=config/llm_config.json  # Path to config file
```

### 2. Configuration File

Create a JSON file at `config/llm_config.json` with the following structure:

```json
{
    "default_client": "client_id_for_default",
    "clients": {
        "client_id_1": {
            "provider": "openai",
            "model_name": "gpt-4o",
            "compliance_level": "financial",
            "capabilities": ["text_generation", "embeddings", "classification"],
            "timeout_seconds": 30,
            "max_retries": 3,
            "additional_headers": {},
            "additional_params": {
                "user": "financial-services",
                "safe_mode": true
            }
        },
        "client_id_2": {
            "provider": "anthropic",
            "model_name": "claude-3-5-sonnet-20240620",
            "compliance_level": "healthcare",
            "capabilities": ["text_generation", "classification"]
        }
        // Add more clients as needed
    }
}
```

### 3. Programmatic Configuration

You can configure LLMs at runtime using the `LLMManager` class:

```python
from cloud.ai.llm_provider import LLMProvider, LLMConfig, LLMComplianceLevel, LLMCapability
from cloud.ai.llm_manager import get_llm_manager

# Get the global LLM manager
llm_manager = get_llm_manager()

# Register a new client
llm_config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model_name="gpt-4o",
    compliance_level=LLMComplianceLevel.FINANCIAL,
    capabilities=[
        LLMCapability.TEXT_GENERATION,
        LLMCapability.CLASSIFICATION
    ],
    additional_params={"user": "financial-services"}
)

llm_manager.register_client("my_gpt4_client", llm_config)

# Register a financial client (shorthand)
llm_manager.register_financial_client(
    client_id="my_financial_client",
    provider=LLMProvider.ANTHROPIC,
    model_name="claude-3-5-sonnet-20240620"
)

# Set default client
llm_manager.set_default_client("my_financial_client")
```

## Compliance Levels

The platform supports different compliance levels for LLMs:

| Level | Description | Best Providers |
|-------|-------------|----------------|
| `STANDARD` | Default level with no special compliance requirements | All providers |
| `FINANCIAL` | Optimized for financial services compliance | OpenAI, Anthropic |
| `HEALTHCARE` | Optimized for healthcare and HIPAA compliance | OpenAI, Anthropic |
| `GOVERNMENT` | Optimized for government and public sector requirements | Azure OpenAI |

## Using LLMs in Your Code

Once configured, you can use LLMs in your code:

```python
from cloud.ai.llm_manager import get_llm_manager

async def analyze_customer_feedback(feedback_text):
    llm_manager = get_llm_manager()
    
    # Use the default client
    result = await llm_manager.generate_text(
        prompt=f"Analyze this customer feedback: {feedback_text}",
        temperature=0.1,
        max_tokens=500
    )
    
    # Use a specific client
    compliance_result = await llm_manager.generate_text(
        prompt=f"Check if this text contains regulatory issues: {feedback_text}",
        client_id="compliance_checker",  # A specific client for compliance
        system_prompt="You are a financial compliance expert",
        temperature=0.0
    )
    
    return {
        "analysis": result["text"],
        "compliance_check": compliance_result["text"]
    }
```

## Handling API Failures

The LLM system includes built-in retry logic and failover capabilities:

- Each client has configurable `max_retries` (default: 3)
- Exponential backoff between retries
- If a specific client is unavailable, you can catch the exception and use a different client

Example of handling failures:

```python
async def generate_with_fallback(prompt):
    llm_manager = get_llm_manager()
    
    try:
        # Try primary provider
        return await llm_manager.generate_text(
            prompt=prompt,
            client_id="primary_provider",
            max_tokens=500
        )
    except Exception as e:
        logger.warning(f"Primary provider failed: {str(e)}")
        
        # Fall back to secondary provider
        return await llm_manager.generate_text(
            prompt=prompt,
            client_id="fallback_provider",
            max_tokens=500
        )
```

## Security Best Practices

When using LLMs, follow these security best practices:

1. **Never hard-code API keys** in your code
2. **Validate and sanitize all user inputs** before sending to LLMs
3. **Set appropriate context windows** to avoid token wastage
4. **Implement content filtering** for both inputs and outputs
5. **Use compliance-specific models** for regulated industries
6. **Monitor usage patterns** to detect anomalies
7. **Set timeouts** to prevent hanging requests

## Monitoring and Observability

The platform includes monitoring for LLM usage:

- All requests are logged (configurable detail level)
- Token usage is tracked for cost management
- Integration with platform's observability system
- Error tracking and alerting

## LLM Provider-Specific Notes

### OpenAI

- Most versatile provider with multiple model options
- Strongest embedding capabilities
- Recommended for general-purpose use cases

### Anthropic

- Excellent for compliance and safety-critical applications
- Strong performance on financial regulatory tasks
- Limited embedding capabilities (use OpenAI for embeddings)

### Google Gemini

- Strong multimodal capabilities
- Good cost-performance ratio
- Particularly strong in document analysis

### Azure OpenAI

- Same capabilities as OpenAI but with Azure's security and compliance
- Data residency options for regulated industries
- Recommended for enterprise deployments 