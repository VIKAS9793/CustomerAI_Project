import functools
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Union

from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("error_handler", log_file="logs/errors.log")


class APIError(Exception):
    """Base exception for API errors"""

    def __init__(self, message: str, status_code: int = 500, details: Optional[Any] = None):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)


class BadRequestError(APIError):
    """Exception for 400 Bad Request errors"""

    def __init__(self, message: str = "Bad request", details: Any = None):
        super().__init__(message=message, status_code=400, details=details)


class UnauthorizedError(APIError):
    """Exception for 401 Unauthorized errors"""

    def __init__(self, message: str = "Unauthorized", details: Any = None):
        super().__init__(message=message, status_code=401, details=details)


class ForbiddenError(APIError):
    """Exception for 403 Forbidden errors"""

    def __init__(self, message: str = "Forbidden", details: Any = None):
        super().__init__(message=message, status_code=403, details=details)


class NotFoundError(APIError):
    """Exception for 404 Not Found errors"""

    def __init__(self, message: str = "Resource not found", details: Any = None):
        super().__init__(message=message, status_code=404, details=details)


class ConflictError(APIError):
    """Exception for 409 Conflict errors"""

    def __init__(self, message: str = "Resource conflict", details: Any = None):
        super().__init__(message=message, status_code=409, details=details)


class RateLimitError(APIError):
    """Exception for 429 Too Many Requests errors"""

    def __init__(self, message: str = "Rate limit exceeded", details: Any = None):
        super().__init__(message=message, status_code=429, details=details)


class ValidationError(BadRequestError):
    """Exception for validation errors"""

    def __init__(self, message: str = "Validation error", details: Any = None):
        super().__init__(message=message, details=details)


def handle_exceptions(func: Callable) -> Callable:
    """
    Decorator to handle exceptions in API endpoints

    Args:
        func (Callable): Function to decorate

    Returns:
        Callable: Decorated function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except APIError as e:
            # Log API errors with appropriate level based on status code
            if e.status_code >= 500:
                logger.error(f"API Error: {e.message}", exc_info=True)
            elif e.status_code >= 400:
                logger.warning(f"API Error: {e.message}", exc_info=True)

            # Return standardized error response
            return {
                "error": True,
                "status_code": e.status_code,
                "message": e.message,
                "details": e.details,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            # Log unexpected errors
            logger.exception(f"Unexpected error in {func.__name__}: {str(e)}")

            # Return generic error response
            return {
                "error": True,
                "status_code": 500,
                "message": "Internal server error",
                "timestamp": datetime.utcnow().isoformat(),
            }

    return wrapper


def validate_request(schema: Dict[str, Any]) -> Callable:
    """
    Decorator to validate request data against schema

    Args:
        schema (Dict[str, Any]): Validation schema

    Returns:
        Callable: Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get request data from args/kwargs (depends on framework)
            # This example assumes the request data is in kwargs['data']
            data = kwargs.get("data")

            if not data:
                # Handle case where framework passes data differently
                # e.g., for FastAPI the first arg might be the data model
                if args and hasattr(args[0], "dict"):
                    data = args[0].dict()

            if not data:
                raise BadRequestError("Missing request data")

            # Validate data against schema
            errors = {}
            for field, rules in schema.items():
                # Check required fields
                if rules.get("required", False) and field not in data:
                    errors[field] = "Field is required"
                    continue

                # Skip validation if field is not in data
                if field not in data:
                    continue

                value = data[field]

                # Check type
                if "type" in rules:
                    expected_type = rules["type"]
                    if expected_type == "string" and not isinstance(value, str):
                        errors[field] = "Field must be a string"
                    elif expected_type == "number" and not isinstance(value, (int, float)):
                        errors[field] = "Field must be a number"
                    elif expected_type == "boolean" and not isinstance(value, bool):
                        errors[field] = "Field must be a boolean"
                    elif expected_type == "array" and not isinstance(value, list):
                        errors[field] = "Field must be an array"
                    elif expected_type == "object" and not isinstance(value, dict):
                        errors[field] = "Field must be an object"

                # Check string pattern
                if isinstance(value, str) and "pattern" in rules:
                    from src.utils.security import InputValidator

                    if not InputValidator.validate_custom(value, rules["pattern"]):
                        errors[field] = rules.get(
                            "error", "Field does not match the required pattern"
                        )

                # Check minimum/maximum length for strings
                if isinstance(value, str):
                    if "minLength" in rules and len(value) < rules["minLength"]:
                        errors[field] = f"Field must be at least {rules['minLength']} characters"
                    if "maxLength" in rules and len(value) > rules["maxLength"]:
                        errors[field] = f"Field must be at most {rules['maxLength']} characters"

                # Check minimum/maximum for numbers
                if isinstance(value, (int, float)):
                    if "minimum" in rules and value < rules["minimum"]:
                        errors[field] = f"Field must be at least {rules['minimum']}"
                    if "maximum" in rules and value > rules["maximum"]:
                        errors[field] = f"Field must be at most {rules['maximum']}"

                # Check enum values
                if "enum" in rules and value not in rules["enum"]:
                    errors[field] = (
                        f"Field must be one of: {', '.join(str(v) for v in rules['enum'])}"
                    )

            # Raise validation error if any errors
            if errors:
                raise ValidationError(details=errors)

            # Call original function with validated data
            return func(*args, **kwargs)

        return wrapper

    return decorator


def handle_api_response(func: Callable) -> Callable:
    """
    Decorator to standardize API responses

    Args:
        func (Callable): Function to decorate

    Returns:
        Callable: Decorated function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        # If result is already an error response, return it as is
        if isinstance(result, dict) and result.get("error", False):
            return result

        # Return standardized success response
        return {
            "error": False,
            "status_code": 200,
            "data": result,
            "timestamp": datetime.utcnow().isoformat(),
        }

    return wrapper


def rate_limit(limit: int = 100, window: int = 3600) -> Callable:
    """
    Decorator to apply rate limiting to API endpoints

    Args:
        limit (int): Maximum requests allowed in window
        window (int): Time window in seconds

    Returns:
        Callable: Decorator function
    """
    from src.utils.security import RateLimiter

    # Create rate limiter instance specific to this decorator
    rate_limiter = RateLimiter(limit=limit, window=window)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get client identifier (IP or user ID)
            # This example assumes the client ID is in kwargs['client_id']
            client_id = kwargs.get("client_id")

            if not client_id:
                # Try to get client ID from request object
                request = kwargs.get("request")
                if request and hasattr(request, "client"):
                    client_id = getattr(request.client, "host", "unknown")
                elif request and hasattr(request, "remote_addr"):
                    client_id = getattr(request, "remote_addr", "unknown")

            # Default to 'unknown' if client ID cannot be determined
            client_id = client_id or "unknown"

            # Check rate limit
            if not rate_limiter.is_allowed(client_id):
                remaining = 0
                reset_time = int(datetime.utcnow().timestamp()) + window

                # Add rate limit headers to response if kwargs has 'response'
                response = kwargs.get("response")
                if response and hasattr(response, "headers"):
                    response.headers["X-Rate-Limit-Limit"] = str(limit)
                    response.headers["X-Rate-Limit-Remaining"] = "0"
                    response.headers["X-Rate-Limit-Reset"] = str(reset_time)

                raise RateLimitError(
                    details={
                        "limit": limit,
                        "remaining": remaining,
                        "reset": reset_time,
                    }
                )

            # Add rate limit headers to response if kwargs has 'response'
            response = kwargs.get("response")
            if response and hasattr(response, "headers"):
                remaining = rate_limiter.get_remaining(client_id)
                reset_time = int(datetime.utcnow().timestamp()) + window

                response.headers["X-Rate-Limit-Limit"] = str(limit)
                response.headers["X-Rate-Limit-Remaining"] = str(remaining)
                response.headers["X-Rate-Limit-Reset"] = str(reset_time)

            # Call original function
            return func(*args, **kwargs)

        return wrapper

    return decorator


def require_auth(roles: Union[str, list] = None) -> Callable:
    """
    Decorator to require authentication for API endpoints

    Args:
        roles (Union[str, list], optional): Required roles. Defaults to None.

    Returns:
        Callable: Decorator function
    """
    from src.utils.security import Authentication

    # Create authentication instance
    auth = Authentication()

    # Convert single role to list
    if isinstance(roles, str):
        roles = [roles]

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get token from request
            # This example assumes the token is in kwargs['token'] or request headers
            token = kwargs.get("token")

            if not token:
                # Try to get token from request object
                request = kwargs.get("request")
                if request and hasattr(request, "headers"):
                    auth_header = request.headers.get("Authorization", "")
                    if auth_header.startswith("Bearer "):
                        token = auth_header[7:]

            if not token:
                raise UnauthorizedError("Missing authentication token")

            # Verify token
            payload = auth.verify_token(token)
            if not payload:
                raise UnauthorizedError("Invalid or expired token")

            # Check roles if required
            if roles and not auth.has_role(token, roles):
                raise ForbiddenError("Insufficient permissions")

            # Add user info to kwargs for the endpoint
            kwargs["user_id"] = payload.get("sub")
            kwargs["user_roles"] = payload.get("roles", [])

            # Call original function
            return func(*args, **kwargs)

        return wrapper

    return decorator
