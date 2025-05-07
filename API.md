# CustomerAI Insights Platform API Documentation

This document provides detailed information about the REST API endpoints provided by the CustomerAI Insights Platform.

## Authentication

All API endpoints require authentication using JWT tokens, except for the `/auth/login` endpoint.

**Headers**:
```
Authorization: Bearer <your_jwt_token>
```

## API Endpoints

### Authentication

#### POST /api/v1/auth/login

Authenticates a user and returns a JWT token.

**Request Body**:
```json
{
  "username": "user@example.com",
  "password": "secure_password" # pragma: allowlist secret
}
```

**Response**:
```json
{
  "error": false,
  "status_code": 200,
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." # pragma: allowlist secret,
    "user_id": "123",
    "roles": ["analyst"]
  },
  "timestamp": "2023-08-15T14:22:31.123456"
}
```

### Sentiment Analysis

#### POST /api/v1/analyze/sentiment

Analyzes the sentiment of customer interactions.

**Request Body**:
```json
{
  "text": "I'm very happy with your service and prompt response to my query.",
  "use_ai": true
}
```

**Response**:
```json
{
  "error": false,
  "status_code": 200,
  "data": {
    "sentiment": "positive",
    "positive": 0.92,
    "negative": 0.03,
    "neutral": 0.05,
    "analysis": {
      "satisfaction_score": 9,
      "key_positives": ["prompt response", "service quality"]
    }
  },
  "timestamp": "2023-08-15T14:23:45.123456"
}
```

#### POST /api/v1/analyze/batch

Performs batch sentiment analysis on multiple conversations.

**Request Body**:
```json
{
  "conversations": [
    {
      "id": "conv-123",
      "text": "I've been trying to contact support for days with no response!"
    },
    {
      "id": "conv-124",
      "text": "Thank you for resolving my issue so quickly."
    }
  ]
}
```

**Response**:
```json
{
  "error": false,
  "status_code": 200,
  "data": {
    "results": [
      {
        "id": "conv-123",
        "sentiment": "negative",
        "positive": 0.05,
        "negative": 0.85,
        "neutral": 0.10
      },
      {
        "id": "conv-124",
        "sentiment": "positive",
        "positive": 0.89,
        "negative": 0.04,
        "neutral": 0.07
      }
    ],
    "summary": {
      "positive_count": 1,
      "negative_count": 1,
      "neutral_count": 0
    }
  },
  "timestamp": "2023-08-15T14:25:12.123456"
}
```

### Response Generation

#### POST /api/v1/generate/response

Generates AI-powered responses for customer queries.

**Request Body**:
```json
{
  "query": "Can I get a personal loan with a credit score of 650?",
  "customer_id": "cust-456",
  "context": {
    "previous_interactions": 3,
    "account_type": "savings"
  }
}
```

**Response**:
```json
{
  "error": false,
  "status_code": 200,
  "data": {
    "response": "Based on the information provided, a credit score of 650 is generally considered fair and may qualify for a personal loan. However, loan approval and terms depend on multiple factors including income, employment history, and existing debt obligations. I recommend applying through our online portal where you can receive a pre-approval decision without affecting your credit score. Would you like me to provide the link to our application page?",
    "category": "loan_inquiry",
    "requires_human_review": false,
    "confidence": 0.87,
    "alternative_responses": [
      "Thank you for your interest in a personal loan. With a credit score of 650, you may qualify for our standard personal loan options."
    ]
  },
  "timestamp": "2023-08-15T14:27:33.123456"
}
```

### Privacy Management

#### POST /api/v1/privacy/anonymize

Anonymizes personally identifiable information (PII) in text.

**Request Body**:
```json
{
  "text": "Hi, my name is John Smith and my account number is 1234567890. My phone number is 555-123-4567.",
  "keep_mapping": true
}
```

**Response**:
```json
{
  "error": false,
  "status_code": 200,
  "data": {
    "anonymized_text": "Hi, my name is [NAME_1] and my account number is [ACCOUNT_NUMBER_1]. My phone number is [PHONE_1].",
    "mapping": {
      "[NAME_1]": "John Smith",
      "[ACCOUNT_NUMBER_1]": "XXXXXX7890",
      "[PHONE_1]": "XXX-XXX-4567"
    },
    "pii_detected": ["name", "account_number", "phone_number"]
  },
  "timestamp": "2023-08-15T14:29:04.123456"
}
```

### Fairness Analysis

#### POST /api/v1/fairness/analyze

Analyzes a dataset for bias across demographic groups.

**Request Body**:
```json
{
  "data": [
    {"age_group": "18-30", "gender": "female", "satisfaction_score": 4, "resolved": true},
    {"age_group": "31-45", "gender": "male", "satisfaction_score": 5, "resolved": true},
    {"age_group": "46-60", "gender": "non-binary", "satisfaction_score": 3, "resolved": false}
  ],
  "attributes": ["age_group", "gender"],
  "outcome_columns": ["satisfaction_score", "resolved"]
}
```

**Response**:
```json
{
  "error": false,
  "status_code": 200,
  "data": {
    "fairness_score": 0.82,
    "detailed_findings": [
      {
        "attribute": "age_group",
        "outcome": "satisfaction_score",
        "disparities": {
          "18-30": {"average": 4.2, "sample_size": 100},
          "31-45": {"average": 4.0, "sample_size": 150},
          "46-60": {"average": 3.1, "sample_size": 125},
          "60+": {"average": 2.8, "sample_size": 75}
        },
        "disparity_score": 0.67,
        "concern_level": "medium"
      }
    ],
    "recommendations": [
      "Review and improve response quality for customers in the 46-60 and 60+ age groups",
      "Implement specialized training for agents handling older demographic inquiries"
    ]
  },
  "timestamp": "2023-08-15T14:31:17.123456"
}
```

### Human Review

#### POST /api/v1/review/queue

Adds a response to the human review queue.

**Request Body**:
```json
{
  "query": "What stocks should I invest in for maximum returns?",
  "response": "Based on market trends, diversifying your portfolio across technology, healthcare, and energy sectors may provide balanced growth potential...",
  "category": "investment_advice",
  "priority": 2,
  "metadata": {
    "customer_id": "cust-789",
    "interaction_id": "int-456"
  }
}
```

**Response**:
```json
{
  "error": false,
  "status_code": 200,
  "data": {
    "item_id": "rev-123456",
    "status": "queued",
    "estimated_review_time": "2023-08-15T16:30:00.000000"
  },
  "timestamp": "2023-08-15T14:33:42.123456"
}
```

#### POST /api/v1/review/decision

Records a human reviewer's decision on a queued item.

**Request Body**:
```json
{
  "item_id": "rev-123456",
  "approved": true,
  "feedback": "Response includes proper disclaimers and balanced advice",
  "edits": {
    "modified_response": "Based on market trends, diversifying your portfolio across technology, healthcare, and energy sectors may provide balanced growth potential. However, please note that all investments carry risk and past performance is not indicative of future results."
  }
}
```

**Response**:
```json
{
  "error": false,
  "status_code": 200,
  "data": {
    "item_id": "rev-123456",
    "status": "reviewed",
    "review_time": "2023-08-15T14:35:12.123456",
    "reviewer_id": "user-345"
  },
  "timestamp": "2023-08-15T14:35:12.123456"
}
```

### Dashboard Analytics

#### GET /api/v1/analytics/summary
Retrieves summary analytics for the dashboard.

**Query Parameters**:
- `start_date`: ISO date string (optional)
- `end_date`: ISO date string (optional)
- `granularity`: "daily", "weekly", "monthly" (optional, default: "daily")

**Response**:
```json
{
  "error": false,
  "status_code": 200,
  "data": {
    "total_conversations": 12542,
    "sentiment_distribution": {
      "positive": 0.65,
      "negative": 0.18,
      "neutral": 0.17
    },
    "average_satisfaction": 4.2,
    "response_time_avg": 3.7,
    "resolution_rate": 0.92,
    "timeline_data": [
      {"date": "2023-08-01", "conversations": 412, "avg_sentiment": 0.68},
      {"date": "2023-08-02", "conversations": 389, "avg_sentiment": 0.71}
    ],
    "top_issues": [
      {"issue": "account_access", "count": 342, "sentiment": 0.42},
      {"issue": "transaction_disputes", "count": 289, "sentiment": 0.37}
    ]
  },
  "timestamp": "2023-08-15T14:38:05.123456"
}

### Generative AI and Advanced ML
#### POST /api/v1/ai/generate
Generates text using state-of-the-art generative AI models.

**Request Body**:
```json
{
  "prompt": "Write a response to a customer who is asking about our premium credit card benefits.",
  "model": "gpt4o_financial",
  "max_tokens": 500,
  "temperature": 0.7,
  "system_prompt": "You are a helpful financial assistant with expertise in credit card benefits.",
  "stream": false,
  "context": {
    "customer_segment": "premium",
    "risk_profile": "low"
  }
}
```
**Response**:
```json
{
  "error": false,
  "status_code": 200,
  "data": {
    "generated_text": "Thank you for your interest in our premium credit card benefits...",
    "usage": {
      "prompt_tokens": 25,
      "completion_tokens": 130,
      "total_tokens": 155
    },
    "model": "gpt-4o",
    "compliance_check": {
      "passed": true,
      "flags": []
    }
  }
}

#### POST /api/v1/ai/embed
Generates embeddings for text using specified model.

**Request Body**:
```json
{
  "texts": [
    "What are the benefits of your premium credit card?",
    "How do I dispute a charge on my statement?",
    "Can I increase my credit limit?"
  ],
  "model": "embeddings",
  "dimensions": 1536
}
```
**Response**:
```json
{
  "error": false,
  "status_code": 200,
  "data": {
    "embeddings": [
      [0.0023729, 0.0078125, -0.0029297, ...],
      [-0.0012817, 0.0035400, 0.0046387, ...],
      [0.0018921, -0.0063477, 0.0012207, ...]
    ],
    "model": "text-embedding-3-large",
    "dimensions": 1536
  }
}

#### POST /api/v1/ai/classify
Classifies text into provided categories.

**Request Body**:
```json
{
  "text": "I'm having trouble accessing my account online after multiple attempts.",
  "categories": [
    "account_access",
    "transaction_issue",
    "balance_inquiry",
    "fee_dispute",
    "technical_support"
  ],
  "model": "claude_sonnet"
}
```
**Response**:
```json
{
  "error": false,
  "status_code": 200,
  "data": {
    "classifications": {
      "account_access": 0.82,
      "technical_support": 0.15,
      "transaction_issue": 0.02,
      "balance_inquiry": 0.01,
      "fee_dispute": 0.0
    },
    "top_category": "account_access",
    "model": "claude-3-5-sonnet-20240620"
  }
}

#### GET /api/v1/ai/providers
Gets information about available LLM providers and models.

**Response**:
```json
{
  "error": false,
  "status_code": 200,
  "data": {
    "providers": [
      {
        "id": "gpt4o_financial",
        "provider": "openai",
        "model": "gpt-4o",
        "compliance_level": "financial",
        "capabilities": ["text_generation", "embeddings", "classification"],
        "is_default": true
      },
      {
        "id": "claude_sonnet",
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20240620",
        "compliance_level": "financial",
        "capabilities": ["text_generation", "classification"],
        "is_default": false
      },
      {
        "id": "gemini_pro",
        "provider": "google",
        "model": "gemini-1.5-pro",
        "compliance_level": "financial",
        "capabilities": ["text_generation", "embeddings", "classification"],
        "is_default": false
      }
    ],
    "default_provider": "gpt4o_financial"
  }
}

#### POST /api/v1/ai/distributed-training
Creates a distributed training job for model fine-tuning.

**Request Body**:
```json
{
  "base_model": "customerai-classifier",
  "training_data_uri": "s3://customerai-data/financial-sentiment-dataset.csv",
  "validation_data_uri": "s3://customerai-data/financial-sentiment-validation.csv",
  "hyperparameters": {
    "learning_rate": 2e-5,
    "epochs": 3,
    "batch_size": 16
  },
  "instance_count": 4,
  "instance_type": "ml.g4dn.xlarge"
}
```
**Response**:
```json
{
  "error": false,
  "status_code": 200,
  "data": {
    "job_id": "train-2025-05-04-12345",
    "status": "STARTING",
    "estimated_time": "25 minutes",
    "model_output_uri": "s3://customerai-models/financial-sentiment-2025-05-04/",
    "logs_uri": "s3://customerai-logs/training/train-2025-05-04-12345/"
  },
  "timestamp": "2025-05-04T15:20:45.123456"
}

#### GET /api/v1/ai/training-job/{job_id}
Gets the status of a training job.

**Response**:
```json
{
  "error": false,
  "status_code": 200,
  "data": {
    "job_id": "train-2025-05-04-12345",
    "status": "IN_PROGRESS",
    "progress": {
      "current_epoch": 2,
      "total_epochs": 3,
      "current_step": 1500,
      "total_steps": 2000,
      "metrics": {
        "train_loss": 0.1823,
        "validation_accuracy": 0.9234
      }
    },
    "estimated_completion": "2025-05-04T15:45:23Z"
  },
  "timestamp": "2025-05-04T15:30:45.123456"
}

#### POST /api/v1/ai/deploy
Deploys a trained model to the inference endpoint.

**Request Body**:
```json
{
  "model_uri": "s3://customerai-models/financial-sentiment-2025-05-04/",
  "deployment_name": "financial-sentiment-prod",
  "deployment_strategy": "kubernetes",
  "replicas": 3,
  "auto_scaling": {
    "min_replicas": 2,
    "max_replicas": 10,
    "target_utilization": 70
  },
  "resource_requirements": {
    "cpu": "2",
    "memory": "4Gi",
    "gpu": "1"
  }
}
```
**Response**:
```json
{
  "error": false,
  "status_code": 200,
  "data": {
    "deployment_id": "deploy-2025-05-04-67890",
    "status": "CREATING",
    "endpoint": "https://api.customerai.com/v1/predict/financial-sentiment-prod",
    "metrics_endpoint": "https://metrics.customerai.com/deployments/deploy-2025-05-04-67890"
  },
  "timestamp": "2025-05-04T16:01:12.123456"
}

#### POST /api/v1/ai/explain
Gets model explanation for predictions.

**Request Body**:
```json
{
  "model_name": "financial-sentiment-prod",
  "data": {
    "text": "I am very satisfied with the quick approval process for my loan application."
  },
  "explanation_method": "shap"
}
```
**Response**:
```json
{
  "error": false,
  "status_code": 200,
  "data": {
    "prediction": {
      "sentiment": "positive",
      "confidence": 0.95
    },
    "explanation": {
      "method": "shap",
      "feature_importance": [
        {"feature": "satisfied", "importance": 0.42},
        {"feature": "quick approval", "importance": 0.38},
        {"feature": "very", "importance": 0.12},
        {"feature": "loan application", "importance": 0.08}
      ],
      "visualization_url": "https://api.customerai.com/visualizations/exp-12345"
    }
  },
  "timestamp": "2025-05-04T16:10:27.123456"
}

## Error Responses
The API uses standard HTTP status codes and returns detailed error information:

**Example Error Response**:
```json
{
  "error": true,
  "status_code": 400,
  "message": "Validation error",
  "details": {
    "text": "Field is required"
  },
  "timestamp": "2023-08-15T14:40:22.123456"
}

## Rate Limiting
API endpoints are rate-limited to prevent abuse. Rate limit information is provided in response headers:
```
X-Rate-Limit-Limit: 100
X-Rate-Limit-Remaining: 95
X-Rate-Limit-Reset: 1692105600
```

## WebSocket APIs
Real-time updates are available through WebSocket connections at:
`wss://api.customerai.example.com/websocket`

### Events
- `queue_update`: Notification when the human review queue changes
- `sentiment_alert`: Real-time alert for extremely negative customer interactions
- `system_status`: Service health and performance metrics

## API Client Libraries
Official client libraries are available for:
- Python 3.10: `pip install customerai-insights-client`
- JavaScript: `npm install @customerai/insights-client`
- Java: Available as Maven dependency
