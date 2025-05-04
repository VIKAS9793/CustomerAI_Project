# Fairness API Documentation

This document provides comprehensive documentation for the enhanced fairness framework API endpoints in the CustomerAI Insights Platform.

## Overview

The Fairness API provides endpoints for detecting bias in datasets and model predictions, visualizing fairness metrics, and applying bias mitigation strategies. It supports both batch analysis and real-time fairness evaluation.

## Base URL

```
http://your-deployment/api/v1/fairness
```

## Authentication

All API endpoints require authentication. Include your API token in the Authorization header:

```
Authorization: Bearer YOUR_API_TOKEN
```

## Endpoints

### 1. Analyze Fairness

Analyzes fairness in a dataset across protected attributes.

**Endpoint:** `POST /analyze`

**Request Body:**

```json
{
  "data": [
    {"attribute1": "value1", "attribute2": "value2", "outcome": "outcome_value"},
    ...
  ],
  "attributes": ["attribute1", "attribute2"],
  "outcome_columns": ["outcome"],
  "positive_outcome_value": "positive_value",  // Optional
  "threshold": 0.8  // Optional fairness threshold
}
```

**Response:**

```json
{
  "status": "success",
  "code": 200,
  "timestamp": "2025-05-04T22:12:32+05:30",
  "data": {
    "timestamp": "2025-05-04T22:12:32+05:30",
    "dataset_info": {
      "size": 1000,
      "attributes_analyzed": ["attribute1", "attribute2"],
      "outcomes_analyzed": ["outcome"]
    },
    "summary": {
      "fairness_score": 0.85,
      "bias_detected": true,
      "threshold": 0.8,
      "significant_findings": [
        {
          "attribute": "attribute1",
          "outcome": "outcome",
          "metric": "disparate_impact_value1_vs_value2",
          "value": 0.75,
          "threshold": 0.8,
          "p_value": 0.03,
          "severity": "medium"
        }
      ],
      "recommendations": [
        "Review data collection and processing for 'attribute1' to identify potential sources of bias.",
        "Consider rebalancing training data for 'attribute1' to address disparate impact."
      ]
    },
    "detailed_findings": [
      {
        "attribute": "attribute1",
        "outcome": "outcome",
        "metric": "disparate_impact_value1_vs_value2",
        "value": 0.75,
        "threshold": 0.8,
        "p_value": 0.03,
        "severity": "medium"
      }
    ],
    "mitigation_recommendations": {
      "pre_processing": [
        {
          "attribute": "attribute1",
          "strategy": "reweigh_samples",
          "description": "Apply sample reweighting to balance outcomes across attribute1 groups",
          "severity": "medium"
        }
      ],
      "in_processing": [],
      "post_processing": [
        {
          "attribute": "attribute1",
          "strategy": "equalized_odds_postprocessing",
          "description": "Adjust prediction thresholds to achieve similar error rates across attribute1 groups",
          "severity": "medium"
        }
      ]
    }
  }
}
```

### 2. Upload and Analyze

Upload a CSV file and analyze fairness across protected attributes.

**Endpoint:** `POST /upload-analyze`

**Form Data:**
- `file`: CSV file to analyze
- `attributes`: Comma-separated list of protected attribute column names
- `outcome_columns`: Comma-separated list of outcome column names
- `threshold`: (Optional) Fairness threshold value

**Response:**
Same format as the `/analyze` endpoint, with additional dataset summary information.

### 3. Mitigate Bias in Dataset

Apply bias mitigation strategies to a dataset.

**Endpoint:** `POST /mitigate`

**Request Body:**

```json
{
  "data": [
    {"attribute1": "value1", "attribute2": "value2", "outcome": "outcome_value"},
    ...
  ],
  "attribute": "attribute1",
  "outcome_column": "outcome",
  "strategy": "reweigh_samples",
  "strategy_params": {
    "positive_outcome_value": "positive_value"
  }
}
```

**Available Strategies:**
- `reweigh_samples`: Assigns weights to training examples
- `balanced_sampling`: Creates a balanced dataset through resampling
- `disparate_impact_remover`: Transforms features to reduce correlation with protected attribute

**Response:**
The response format varies based on the selected strategy:

#### For reweigh_samples:
```json
{
  "status": "success",
  "code": 200,
  "timestamp": "2025-05-04T22:12:32+05:30",
  "data": {
    "strategy": "reweigh_samples",
    "weights": [1.2, 0.8, 1.0, ...],
    "original_data": [
      {"attribute1": "value1", "attribute2": "value2", "outcome": "outcome_value"},
      ...
    ]
  }
}
```

#### For balanced_sampling:
```json
{
  "status": "success",
  "code": 200,
  "timestamp": "2025-05-04T22:12:32+05:30",
  "data": {
    "strategy": "balanced_sampling",
    "original_size": 1000,
    "balanced_size": 800,
    "balanced_data": [
      {"attribute1": "value1", "attribute2": "value2", "outcome": "outcome_value"},
      ...
    ]
  }
}
```

### 4. Mitigate Bias in Predictions

Apply post-processing bias mitigation strategies to model predictions.

**Endpoint:** `POST /mitigate-predictions`

**Request Body:**

```json
{
  "predictions": [0.7, 0.3, 0.8, ...],
  "true_values": [1, 0, 1, ...],
  "protected_attributes": ["group1", "group2", "group1", ...],
  "strategy": "equalized_odds_postprocessing",
  "strategy_params": {
    "threshold": 0.5
  }
}
```

**Available Strategies:**
- `equalized_odds_postprocessing`: Adjusts prediction thresholds for different groups
- `calibrated_equalized_odds`: Applies calibrated equalized odds to prediction probabilities

**Response:**

```json
{
  "status": "success",
  "code": 200,
  "timestamp": "2025-05-04T22:12:32+05:30",
  "data": {
    "strategy": "equalized_odds_postprocessing",
    "adjusted_predictions": [1, 0, 1, ...]
  }
}
```

### 5. Get Available Mitigation Strategies

Get information about available bias mitigation strategies.

**Endpoint:** `GET /strategies`

**Response:**

```json
{
  "status": "success",
  "code": 200,
  "timestamp": "2025-05-04T22:12:32+05:30",
  "data": {
    "pre_processing": [
      {
        "id": "reweigh_samples",
        "name": "Sample Reweighting",
        "description": "Assigns weights to training examples to ensure fairness across protected attribute groups",
        "parameters": [
          {"name": "protected_attribute", "type": "string", "required": true, "description": "Protected attribute column name"},
          {"name": "outcome_column", "type": "string", "required": true, "description": "Outcome column name"},
          {"name": "positive_outcome_value", "type": "any", "required": false, "description": "Value considered as positive outcome"}
        ]
      },
      ...
    ],
    "post_processing": [
      {
        "id": "equalized_odds_postprocessing",
        "name": "Equalized Odds Post-processing",
        "description": "Adjusts prediction thresholds to achieve similar error rates across protected attribute groups",
        "parameters": [
          {"name": "threshold", "type": "number", "required": false, "description": "Initial classification threshold"}
        ]
      },
      ...
    ]
  }
}
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- `400 Bad Request`: Invalid input parameters
- `401 Unauthorized`: Authentication failure
- `403 Forbidden`: Insufficient permissions
- `500 Internal Server Error`: Server-side error

Error responses follow this format:

```json
{
  "status": "error",
  "code": 400,
  "timestamp": "2025-05-04T22:12:32+05:30",
  "message": "Invalid input parameters",
  "details": {
    "error": "Attributes are required"
  }
}
```

## Usage Examples

### Python Example: Analyzing Fairness

```python
import requests
import json

API_URL = "http://your-deployment/api/v1/fairness"
API_TOKEN = "your_api_token"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

data = {
    "data": [
        {"age_group": "18-30", "gender": "male", "satisfaction_score": 4},
        {"age_group": "31-45", "gender": "female", "satisfaction_score": 3},
        # More data...
    ],
    "attributes": ["age_group", "gender"],
    "outcome_columns": ["satisfaction_score"],
    "threshold": 0.8
}

response = requests.post(
    f"{API_URL}/analyze",
    headers=headers,
    json=data
)

result = response.json()
print(json.dumps(result, indent=2))
```

### Python Example: Applying Bias Mitigation

```python
import requests
import json

API_URL = "http://your-deployment/api/v1/fairness"
API_TOKEN = "your_api_token"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

data = {
    "data": [
        {"age_group": "18-30", "gender": "male", "satisfaction_score": 4},
        {"age_group": "31-45", "gender": "female", "satisfaction_score": 3},
        # More data...
    ],
    "attribute": "gender",
    "outcome_column": "satisfaction_score",
    "strategy": "reweigh_samples"
}

response = requests.post(
    f"{API_URL}/mitigate",
    headers=headers,
    json=data
)

result = response.json()
print(json.dumps(result, indent=2))
```

## Integration with the Dashboard

The fairness API can be used in conjunction with the Fairness Dashboard for interactive visualization:

1. Use the `/analyze` endpoint to perform fairness analysis
2. Pass the results to the FairnessDashboard component
3. Visualize the results and explore mitigation options
4. Apply selected mitigation strategies using the `/mitigate` endpoint

For more information on the dashboard integration, see the [Fairness Dashboard Documentation](FAIRNESS_DASHBOARD.md).
