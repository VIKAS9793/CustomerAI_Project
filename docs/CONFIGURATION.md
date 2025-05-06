# CustomerAI Configuration Guide

Last updated: May 6, 2025

## Overview

This document outlines the configuration system for the CustomerAI project, which uses a hierarchical approach to manage settings across different components.

## Configuration Structure

```
config/
├── config.yaml           # Main configuration file
├── fairness_config.json  # Fairness framework settings
├── guardrails_config.json# AI safety guardrails
└── llm_config.json      # Language model configurations
```

## Standard Configurations

### 1. Financial Services
- Stricter fairness thresholds
- Enhanced audit logging
- Compliance-focused guardrails

### 2. Healthcare
- Focus on false negative reduction
- HIPAA-compliant data handling
- Enhanced privacy controls

### 3. General Business
- Balanced fairness metrics
- Standard logging
- Default privacy settings

## Environment Variables

Required environment variables:
- `CUSTOMERAI_ENV`: Development environment (dev/staging/prod)
- `PYTHON_VERSION`: Must be set to "3.10"
- `FAIRNESS_CONFIG_PATH`: Path to fairness configuration
- `LLM_API_KEY`: API key for language model access

## Usage

```python
from config import FairnessConfig

# Load configuration
config = FairnessConfig.load()

# Access settings
threshold = config.bias_detection_threshold
metrics = config.enabled_metrics
```

## Customization

See `examples/` directory for industry-specific configuration examples and customization guides.
