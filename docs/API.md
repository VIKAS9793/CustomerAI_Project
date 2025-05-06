# API Documentation

## Authentication

### Login
```http
POST /api/auth/login
Content-Type: application/json

{
  "username": "string",
  "password": "string"
}
```

### Refresh Token
```http
POST /api/auth/refresh
Authorization: Bearer <access_token>
```

## Chat API

### Send Message
```http
POST /api/chat
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "message": "string",
  "context": {
    "customer_id": "string",
    "session_id": "string"
  }
}
```

### Get Conversation History
```http
GET /api/chat/history
Authorization: Bearer <access_token>

Parameters:
- customer_id: string
- session_id: string
```

## Sentiment Analysis

### Analyze Sentiment
```http
POST /api/sentiment
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "text": "string"
}
```

## Human Review

### Submit for Review
```http
POST /api/review
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "conversation_id": "string",
  "reason": "string",
  "priority": "high|medium|low"
}
```

### Get Review Status
```http
GET /api/review/status
Authorization: Bearer <access_token>

Parameters:
- conversation_id: string
```

## File Upload

### Upload File
```http
POST /api/upload
Authorization: Bearer <access_token>
Content-Type: multipart/form-data

Fields:
- file: file
- type: string ("image"|"document")
```

## Security Headers

All responses include the following security headers:
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection: 1; mode=block
- Strict-Transport-Security: max-age=31536000; includeSubDomains
- Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline';
