"""
Security configuration and utilities for the CustomerAI project.
"""

from typing import Dict, List

from pydantic import BaseModel, Field


class SecurityConfig(BaseModel):
    """Configuration for security settings."""

    # Authentication settings
    jwt_secret_key: str = Field(..., description="Secret key for JWT token signing")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Password settings
    password_min_length: int = 8
    password_max_length: int = 128
    required_password_chars: List[str] = ["lower", "upper", "digit", "special"]

    # Rate limiting
    request_limit: int = 100
    time_window_seconds: int = 60

    # CORS settings
    allowed_origins: List[str] = ["http://localhost:3000"]
    allowed_methods: List[str] = ["GET", "POST", "PUT", "DELETE"]
    allowed_headers: List[str] = ["*"]

    # Security headers
    security_headers: Dict[str, str] = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    }

    # Input validation
    max_input_length: int = 1000
    allowed_content_types: List[str] = ["application/json", "text/plain"]

    # File upload settings
    max_file_size_mb: int = 10
    allowed_file_types: List[str] = [
        "image/jpeg",
        "image/png",
        "application/pdf",
        "text/csv",
    ]


class SecuritySettings:
    """Security settings and utilities."""

    def __init__(self, config: SecurityConfig):
        self.config = config

    def validate_password(self, password: str) -> bool:
        """Validate password strength."""
        if len(password) < self.config.password_min_length:
            return False
        if len(password) > self.config.password_max_length:
            return False

        # Check required character types
        if "lower" in self.config.required_password_chars and not any(
            c.islower() for c in password
        ):
            return False
        if "upper" in self.config.required_password_chars and not any(
            c.isupper() for c in password
        ):
            return False
        if "digit" in self.config.required_password_chars and not any(
            c.isdigit() for c in password
        ):
            return False
        if "special" in self.config.required_password_chars and not any(
            not c.isalnum() for c in password
        ):
            return False

        return True

    def validate_input(self, data: str) -> bool:
        """Validate input data length."""
        return len(data) <= self.config.max_input_length

    def validate_file(self, file: bytes, content_type: str) -> bool:
        """Validate file upload."""
        if len(file) > self.config.max_file_size_mb * 1024 * 1024:
            return False
        if content_type not in self.config.allowed_file_types:
            return False

        return True


# Default security configuration
class BaseConfig(SecurityConfig):
    pass


DEFAULT_SECURITY_CONFIG = BaseConfig(
    jwt_secret_key=JWT_SECRET_KEY,
    allowed_origins=["http://localhost:3000", "https://your-domain.com"],
    security_headers={
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline';",
    },
)
