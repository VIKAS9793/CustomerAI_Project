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
  "password": "secure_password"
}
```

**Response**:
```json
{
  "error": false,
  "status_code": 200,
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
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
      "Thank you for your interest in a personal loan. With a credit score of 650, you may qualify for our standard personal loan options..."
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
      "[ACCOUNT_NUMBER_1]": "1234567890",
      "[PHONE_1]": "555-123-4567"
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
```

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
```

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
- Python: `pip install customerai-client`
- JavaScript: `npm install customerai-client`
- Java: Available as Maven dependency 