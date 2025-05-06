# CustomerAI Fairness Framework Customization Guide

This guide explains how to customize the fairness framework to meet your organization's specific requirements, policies, and business needs.

## Table of Contents

1. [Configuration System](#configuration-system)
2. [Customizing Fairness Thresholds](#customizing-fairness-thresholds)
3. [Customizing Mitigation Strategies](#customizing-mitigation-strategies)
4. [Customizing Visualization](#customizing-visualization)
5. [Customizing API Settings](#customizing-api-settings)
6. [Environment Variables](#environment-variables)
7. [Advanced Customization](#advanced-customization)

## Configuration System

The fairness framework uses a centralized configuration system that allows for easy customization without modifying the code. There are three ways to customize the configuration:

1. **Configuration File**: Create a JSON file based on the provided template
2. **Environment Variables**: Set environment variables to override specific settings
3. **Direct Code Configuration**: Pass configuration dictionaries to component constructors

### Using the Configuration File

1. Copy the example configuration file:
   ```bash
   cp config/fairness_config.json.example config/fairness_config.json
   ```

2. Edit the configuration file to match your requirements:
   ```json
   {
     "thresholds": {
       "disparate_impact": 0.85,
       "statistical_parity_difference": 0.05
     }
   }
   ```

3. The framework will automatically load this configuration at startup.

### Setting the Configuration Path

By default, the framework looks for the configuration file at `config/fairness_config.json`. You can specify a different location using the `FAIRNESS_CONFIG_PATH` environment variable:

```bash
export FAIRNESS_CONFIG_PATH=/path/to/your/config.json
```

## Customizing Fairness Thresholds

Fairness thresholds determine when a disparity is considered significant enough to be flagged as potential bias. You can customize these thresholds to align with your organization's policies or regulatory requirements.

### Available Threshold Settings

```json
{
  "thresholds": {
    "disparate_impact": 0.8,         // Standard 80% rule
    "statistical_parity_difference": 0.1,
    "equal_opportunity_difference": 0.1,
    "predictive_parity_difference": 0.1
  },
  "significance": {
    "pvalue_threshold": 0.05,        // Statistical significance level
    "confidence_interval": 0.95
  }
}
```

### Example: Stricter Thresholds for Financial Services

Financial services organizations often need stricter fairness thresholds due to regulatory requirements:

```json
{
  "thresholds": {
    "disparate_impact": 0.9,         // Stricter than the standard 80% rule
    "statistical_parity_difference": 0.05,
    "equal_opportunity_difference": 0.05,
    "predictive_parity_difference": 0.05
  },
  "significance": {
    "pvalue_threshold": 0.01,        // More stringent significance level
    "confidence_interval": 0.99
  }
}
```

## Customizing Mitigation Strategies

You can customize which bias mitigation strategies are available and their default parameters.

### Available Mitigation Settings

```json
{
  "mitigation": {
    "default_strategy": "reweighing",
    "available_strategies": [
      "reweighing",
      "disparate_impact_remover",
      "equalized_odds",
      "calibrated_equalized_odds",
      "reject_option_classification",
      "balanced_sampling"
    ]
  }
}
```

### Example: Healthcare-Specific Configuration

For healthcare applications, you might want to prioritize certain mitigation strategies:

```json
{
  "mitigation": {
    "default_strategy": "calibrated_equalized_odds",
    "available_strategies": [
      "calibrated_equalized_odds",
      "equalized_odds",
      "reweighing"
    ],
    "strategy_params": {
      "reweighing": {
        "weight_bound": 5.0  // Limit weight adjustments for healthcare data
      },
      "calibrated_equalized_odds": {
        "cost_constraint": "fnr"  // Prioritize reducing false negatives
      }
    }
  }
}
```

## Customizing Visualization

You can customize the appearance and behavior of the fairness dashboard.

### Available Visualization Settings

```json
{
  "visualization": {
    "color_palette": "colorblind",  // Accessible color palette
    "chart_types": ["bar", "heatmap", "scatter", "line"],
    "default_chart": "bar",
    "decimal_places": 3
  }
}
```

### Example: Corporate Brand Colors

To align with your organization's brand guidelines:

```json
{
  "visualization": {
    "color_palette": "custom",
    "custom_colors": ["#1A73E8", "#EA4335", "#FBBC04", "#34A853", "#5F6368"],
    "default_chart": "bar",
    "decimal_places": 2,
    "show_logo": true,
    "logo_path": "/static/company_logo.png"
  }
}
```

## Customizing API Settings

You can customize API behavior for performance and security.

### Available API Settings

```json
{
  "api": {
    "rate_limit": 100,  // Requests per minute
    "max_payload_size_mb": 10,
    "cache_ttl_seconds": 300
  }
}
```

### Example: High-Volume Production Environment

For high-traffic production environments:

```json
{
  "api": {
    "rate_limit": 500,
    "max_payload_size_mb": 50,
    "cache_ttl_seconds": 600,
    "enable_compression": true,
    "compression_threshold_kb": 100
  }
}
```

## Environment Variables

You can override any configuration setting using environment variables with the `FAIRNESS_` prefix followed by the configuration path with underscores.

### Examples

```bash
# Set disparate impact threshold to 0.85
export FAIRNESS_THRESHOLDS_DISPARATE_IMPACT=0.85

# Set default mitigation strategy
export FAIRNESS_MITIGATION_DEFAULT_STRATEGY=equalized_odds

# Set API rate limit
export FAIRNESS_API_RATE_LIMIT=200
```

## Advanced Customization

For more advanced customization needs, you can extend the framework by:

1. **Custom Metrics**: Create custom fairness metrics by extending the `BiasDetector` class
2. **Custom Mitigation Strategies**: Add new mitigation strategies to the `FairnessMitigation` class
3. **Custom Visualizations**: Add new visualization types to the `FairnessDashboard` class

### Example: Adding a Custom Fairness Metric

```python
from src.fairness.bias_detector import BiasDetector

class CustomBiasDetector(BiasDetector):
    def __init__(self, config=None):
        super().__init__(config)
        # Add your custom metric to the list of metrics
        self.metrics.append('my_custom_metric')

    def calculate_my_custom_metric(self, group1_data, group2_data):
        # Implement your custom metric calculation
        # ...
        return metric_value
```

---

By following this guide, you can customize the fairness framework to meet your organization's specific requirements while maintaining the core functionality and benefits of the system.
