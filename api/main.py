import json
import os
import sys
import time
from datetime import timedelta
from typing import Any, Dict, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm,
)
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.date_provider import DateProvider

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
from dotenv import load_dotenv
from validation.domain_validator import FinancialDomainValidator
from validation.human_in_loop import HumanReviewSystem

from api.cloud_services import router as cloud_router
from api.fairness import router as fairness_router
from api.privacy import router as privacy_router
from api.response_gen import router as response_gen_router
from api.review import router as review_router

# Import API routers
from api.sentiment import router as sentiment_router
from api.users import router as users_router
from fairness.bias_detector import BiasDetector
from privacy.anonymizer import DataAnonymizer
from src.response_generator import ResponseGenerator

# Import project components
from src.sentiment_analyzer import SentimentAnalyzer
from src.utils.error_handler import CustomException
from src.utils.logger import setup_logger
from src.utils.security import (
    DeviceSecurityValidator,
    authentication,
    create_jwt_token,
    rate_limiter,
    validate_jwt_token,
)

load_dotenv()

# Setup logging
logger = setup_logger("api")

# Create FastAPI app
app = FastAPI(
    title="CustomerAI Insights Platform API",
    description="API for financial customer service insights with privacy, fairness, and compliance features",
    version="1.0.0",
)

# Initialize device security validator
device_security = DeviceSecurityValidator()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8501").split(
        ","
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count", "X-Rate-Limit-Remaining"],
)

# Add trusted hosts middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=[os.getenv("ALLOWED_HOST", "*")])


# Rate limiting middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()

        # Clean old records
        self.clients = {
            ip: data
            for ip, data in self.clients.items()
            if current_time - data["first_request"] < self.period
        }

        # Check if client exists
        if client_ip not in self.clients:
            self.clients[client_ip] = {"first_request": current_time, "count": 1}
        else:
            client = self.clients[client_ip]

            # If period has passed, reset count
            if current_time - client["first_request"] > self.period:
                client["first_request"] = current_time
                client["count"] = 1
            else:
                client["count"] += 1

            # If rate limit exceeded
            if client["count"] > self.calls:
                return JSONResponse(
                    status_code=429,
                    content=create_response(
                        error=True,
                        status_code=429,
                        message="Rate limit exceeded. Please try again later.",
                    ),
                )

        response = await call_next(request)

        # Add rate limit headers
        if client_ip in self.clients:
            client = self.clients[client_ip]
            response.headers["X-Rate-Limit-Limit"] = str(self.calls)
            response.headers["X-Rate-Limit-Remaining"] = str(max(0, self.calls - client["count"]))
            response.headers["X-Rate-Limit-Reset"] = str(int(client["first_request"] + self.period))

        return response


# Add rate limit middleware - adjust values based on expected load
app.add_middleware(RateLimitMiddleware, calls=100, period=60)


# Enhance security headers middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response


app.add_middleware(SecurityHeadersMiddleware)

# Security
security = HTTPBearer()

# Initialize components
sentiment_analyzer = SentimentAnalyzer()
response_generator = ResponseGenerator()
domain_validator = FinancialDomainValidator()
data_anonymizer = DataAnonymizer()
bias_detector = BiasDetector()
human_review = HumanReviewSystem()


# API response wrapper
def create_response(
    data: Any = None,
    error: bool = False,
    status_code: int = 200,
    message: str = None,
    details: Dict = None,
) -> Dict:
    """Create a standardized API response."""
    response = {
        "error": error,
        "status_code": status_code,
        "timestamp": DateProvider.get_instance().iso_format(),
    }

    if data is not None:
        response["data"] = data

    if message:
        response["message"] = message

    if details:
        response["details"] = details

    return response


# Middleware for request timing and logging
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"Status: {response.status_code} "
            f"Time: {process_time:.4f}s"
        )

        return response
    except Exception as e:
        logger.error(f"Request error: {str(e)}")
        process_time = time.time() - start_time
        return JSONResponse(
            status_code=500,
            content=create_response(
                error=True, status_code=500, message=f"Internal server error: {str(e)}"
            ),
        )


# Exception handlers
@app.exception_handler(CustomException)
async def custom_exception_handler(request: Request, exc: CustomException):
    return JSONResponse(
        status_code=exc.status_code,
        content=create_response(
            error=True,
            status_code=exc.status_code,
            message=exc.message,
            details=exc.details,
        ),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=create_response(error=True, status_code=500, message="Internal server error"),
    )


# Authentication dependency
async def validate_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        if not token or token.count(".") != 2:  # Simple JWT structure validation
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token format",
                headers={"WWW-Authenticate": "Bearer"},
            )

        payload = validate_jwt_token(token)

        # Check token expiration - double check even if validate_jwt_token should do this
        if "exp" in payload and payload["exp"] < time.time():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return payload
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to CustomerAI Insights API",
        "version": "1.0.0",
        "docs_url": "/docs",
    }


# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": DateProvider.get_instance().iso_format(),
        "version": "1.0.0",
    }


# Authentication endpoints
@app.post("/api/v1/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authentication.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token_expires = timedelta(
        minutes=int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 1440))
    )
    access_token = create_jwt_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=access_token_expires,
    )

    return {"access_token": access_token, "token_type": "bearer"}


# Sentiment Analysis endpoints
@app.post("/api/v1/analyze/sentiment")
async def analyze_sentiment(request_data: Dict, token=Depends(validate_token)):
    try:
        text = request_data.get("text")
        if not text:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Text is required")

        # Analyze sentiment
        result = sentiment_analyzer.analyze_sentiment(text)
        return create_response(result)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Sentiment analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sentiment analysis failed",
        )


@app.post("/api/v1/analyze/batch")
async def analyze_batch(request_data: Dict, token=Depends(validate_token)):
    try:
        conversations = request_data.get("conversations", [])
        if not conversations:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No conversations provided",
            )

        # Extract texts
        texts = [conv.get("text", "") for conv in conversations]
        ids = [conv.get("id", i) for i, conv in enumerate(conversations)]

        # Analyze sentiments
        results = sentiment_analyzer.analyze_batch(texts)

        # Create response with IDs
        response_data = {
            "results": [
                {
                    "id": id,
                    "sentiment": result["sentiment"],
                    "positive": result["positive"],
                    "negative": result["negative"],
                    "neutral": result["neutral"],
                }
                for id, result in zip(ids, results)
            ],
            "summary": {
                "positive_count": sum(1 for r in results if r["sentiment"] == "positive"),
                "negative_count": sum(1 for r in results if r["sentiment"] == "negative"),
                "neutral_count": sum(1 for r in results if r["sentiment"] == "neutral"),
            },
        }

        return create_response(response_data)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Batch analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch analysis failed",
        )


# Response Generation endpoints
@app.post("/api/v1/generate/response")
async def generate_response(request_data: Dict, token=Depends(validate_token)):
    try:
        query = request_data.get("query")
        if not query:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query is required")

        # Optional parameters
        customer_id = request_data.get("customer_id")
        context = request_data.get("context", {})

        # Generate response
        result = response_generator.generate_response(query, context=context)

        # Validate compliance
        validation = domain_validator.validate_response(query, result["response"])

        # Add to human review if necessary
        if validation.get("requires_review", False) or result.get("requires_human_review", False):
            human_review.add_to_queue(
                item_id=f"resp-{int(time.time())}",
                query=query,
                response=result["response"],
                category=result["category"],
                priority=(2 if result["category"] in ["investment_advice", "loan_approval"] else 1),
                metadata={"customer_id": customer_id} if customer_id else {},
            )
            result["requires_human_review"] = True

        return create_response(result)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Response generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Response generation failed",
        )


# Privacy Management endpoints
@app.post("/api/v1/privacy/anonymize")
async def anonymize_text(request_data: Dict, token=Depends(validate_token)):
    try:
        text = request_data.get("text")
        if not text:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Text is required")

        keep_mapping = request_data.get("keep_mapping", False)

        # Anonymize text
        result = data_anonymizer.anonymize_text(text, keep_mapping=keep_mapping)
        return create_response(result)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Anonymization error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Anonymization failed",
        )


# Fairness Analysis endpoints
@app.post("/api/v1/fairness/analyze")
async def analyze_fairness(request_data: Dict, token=Depends(validate_token)):
    try:
        data = request_data.get("data", [])
        if not data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Data is required")

        attributes = request_data.get("attributes", [])
        outcome_columns = request_data.get("outcome_columns", [])

        if not attributes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Attributes are required",
            )

        if not outcome_columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Outcome columns are required",
            )

        # Convert to DataFrame
        import pandas as pd

        df = pd.DataFrame(data)

        # Analyze fairness
        results = bias_detector.detect_outcome_bias(
            df, attributes=attributes, outcome_columns=outcome_columns
        )

        # Generate report
        fairness_report = bias_detector.generate_fairness_report(results)
        return create_response(fairness_report)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Fairness analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fairness analysis failed",
        )


# Human Review endpoints
@app.post("/api/v1/review/queue")
async def queue_for_review(request_data: Dict, token=Depends(validate_token)):
    try:
        query = request_data.get("query")
        response = request_data.get("response")
        category = request_data.get("category")
        priority = request_data.get("priority", 1)
        metadata = request_data.get("metadata", {})

        if not query or not response:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query and response are required",
            )

        # Add to review queue
        result = human_review.add_to_queue(
            item_id=f"rev-{int(time.time())}",
            query=query,
            response=response,
            category=category,
            priority=priority,
            metadata=metadata,
        )

        return create_response(
            {
                "item_id": result["item_id"],
                "status": "queued",
                "estimated_review_time": result.get("estimated_review_time"),
            }
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Queue addition error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add to review queue",
        )


@app.post("/api/v1/review/decision")
async def record_review_decision(request_data: Dict, token=Depends(validate_token)):
    try:
        item_id = request_data.get("item_id")
        approved = request_data.get("approved")
        feedback = request_data.get("feedback", "")
        edits = request_data.get("edits", {})

        if item_id is None or approved is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Item ID and approval decision are required",
            )

        # Record review decision
        result = human_review.record_review(
            item_id=item_id, approved=approved, feedback=feedback, edits=edits
        )

        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "Failed to record review"),
            )

        return create_response(
            {
                "item_id": item_id,
                "status": "reviewed",
                "review_time": DateProvider.get_instance().iso_format(),
                "reviewer_id": token.get("sub", "unknown"),
            }
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Review decision error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record review decision",
        )


# Dashboard Analytics endpoints
@app.get("/api/v1/analytics/summary")
async def get_analytics_summary(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    granularity: str = "daily",
    token=Depends(validate_token),
):
    try:
        # In a real implementation, this would query a database
        # For demo purposes, return sample data
        return create_response(
            {
                "total_conversations": 12542,
                "sentiment_distribution": {
                    "positive": 0.65,
                    "negative": 0.18,
                    "neutral": 0.17,
                },
                "average_satisfaction": 4.2,
                "response_time_avg": 3.7,
                "resolution_rate": 0.92,
                "timeline_data": [
                    {"date": "2023-08-01", "conversations": 412, "avg_sentiment": 0.68},
                    {"date": "2023-08-02", "conversations": 389, "avg_sentiment": 0.71},
                    {"date": "2023-08-03", "conversations": 425, "avg_sentiment": 0.65},
                    {"date": "2023-08-04", "conversations": 401, "avg_sentiment": 0.62},
                    {"date": "2023-08-05", "conversations": 378, "avg_sentiment": 0.69},
                ],
                "top_issues": [
                    {"issue": "account_access", "count": 342, "sentiment": 0.42},
                    {"issue": "transaction_disputes", "count": 289, "sentiment": 0.37},
                    {"issue": "loan_inquiries", "count": 187, "sentiment": 0.67},
                    {"issue": "mobile_app", "count": 156, "sentiment": 0.51},
                    {"issue": "fee_questions", "count": 134, "sentiment": 0.39},
                ],
            }
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Analytics summary error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analytics summary",
        )


# Add security middleware
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"

    # Apply rate limiting
    remaining_requests = rate_limiter.check_rate_limit(client_ip)
    if remaining_requests < 0:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again later."},
        )

    # Check device security (for relevant endpoints)
    if request.url.path.startswith(("/api/v1/user", "/api/v1/sentiment", "/api/v1/generate")):
        device_info = {}

        # Extract device info from headers
        for header in [
            "X-Device-Id",
            "X-Device-Model",
            "X-OS-Version",
            "X-App-Version",
        ]:
            if header in request.headers:
                device_info[header.replace("X-", "").lower()] = request.headers.get(header)

        # Check device security if info provided
        if device_info:
            security_result = device_security.validate_device(device_info)
            if not security_result["secure"]:
                return JSONResponse(
                    status_code=403,
                    content={
                        "detail": "Device security check failed",
                        "reason": security_result["reason"],
                    },
                )

    # Continue with request
    response = await call_next(request)

    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-Rate-Limit-Remaining"] = str(remaining_requests)

    return response


# Device security check middleware
@app.middleware("http")
async def validate_device_headers(request: Request, call_next):
    # Skip validation for authentication endpoints
    if request.url.path in ["/api/auth/login", "/api/auth/token"]:
        return await call_next(request)

    # Extract device information
    user_agent = request.headers.get("User-Agent", "")

    # Try to get device fingerprint from headers or cookies
    device_fingerprint = {}
    fingerprint_header = request.headers.get("X-Device-Fingerprint")

    if fingerprint_header:
        try:
            device_fingerprint = json.loads(fingerprint_header)
        except json.JSONDecodeError:
            # Invalid fingerprint format
            pass

    # Prepare request data for security validation
    request_data = {
        "user_agent": user_agent,
        "device_fingerprint": device_fingerprint,
        "device_id": request.headers.get("X-Device-ID", "unknown"),
        "ip_address": request.client.host if request.client else "unknown",
    }

    # Check device security
    is_secure, reason = device_security.is_device_secure(request_data)

    if not is_secure:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"detail": "Security requirements not met", "reason": reason},
        )

    # Continue with the request if device is secure
    return await call_next(request)


# OAuth2 password flow for token based authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Register API routers
app.include_router(fairness_router)
app.include_router(sentiment_router, prefix="/api/v1")
app.include_router(response_gen_router, prefix="/api/v1")
app.include_router(privacy_router, prefix="/api/v1")
app.include_router(users_router, prefix="/api/v1")
app.include_router(review_router, prefix="/api/v1")
app.include_router(cloud_router, prefix="/api/v1")

# Run the application
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
