"""
Security utility functions for input validation and sanitization.

This module provides reusable security functions that can be used across the application.
All functions are designed to be pure and testable.
"""

import json
import re
from typing import Any, List
from urllib.parse import quote


class SecurityError(Exception):
    """Base class for security-related errors."""

    pass


class ValidationError(SecurityError):
    """Raised when input validation fails."""

    pass


class InjectionError(SecurityError):
    """Raised when potential injection is detected."""

    pass


def validate_input(data: Any, max_length: int = 1024) -> None:
    """
    Validate input data for security.

    Args:
        data: Input data to validate
        max_length: Maximum allowed length for string inputs

    Raises:
        ValidationError: If validation fails
    """
    if isinstance(data, str):
        if len(data) > max_length:
            raise ValidationError(f"Input too long (max {max_length} characters)")
        if any(char in data for char in ["\0", "\n", "\r"]):
            raise ValidationError("Input contains invalid characters")

    elif isinstance(data, (list, dict)):
        if isinstance(data, list):
            for item in data:
                validate_input(item, max_length)
        else:
            for value in data.values():
                validate_input(value, max_length)


def sanitize_input(data: Any) -> Any:
    """
    Sanitize input data to prevent injection attacks.

    Args:
        data: Input data to sanitize

    Returns:
        Sanitized data
    """
    if isinstance(data, str):
        # Escape special characters
        return quote(data)
    elif isinstance(data, (list, dict)):
        if isinstance(data, list):
            return [sanitize_input(item) for item in data]
        return {k: sanitize_input(v) for k, v in data.items()}
    return data


def validate_path(path: str, base_path: str) -> str:
    """
    Validate that a path is safe and within the base path.

    Args:
        path: Path to validate
        base_path: Base path that the path must be within

    Returns:
        Validated path

    Raises:
        ValidationError: If path validation fails
    """
    if not path:
        raise ValidationError("Path cannot be empty")

    if ".." in path:
        raise ValidationError("Path traversal detected")

    if path.startswith("~"):
        raise ValidationError("Home directory expansion not allowed")

    # Normalize paths
    path = path.strip().replace("\\", "/")
    base_path = base_path.strip().replace("\\", "/")

    # Check if path is within base path
    if not path.startswith(base_path):
        raise ValidationError("Path is outside of base directory")

    return path


def validate_json(data: Any) -> None:
    """
    Validate that data is valid JSON.

    Args:
        data: Data to validate

    Raises:
        ValidationError: If JSON validation fails
    """
    try:
        if isinstance(data, str):
            json.loads(data)
        else:
            json.dumps(data)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON: {str(e)}")


def validate_command_args(args: List[str]) -> List[str]:
    """
    Validate and sanitize command arguments to prevent injection.

    Args:
        args: List of command arguments

    Returns:
        Validated and sanitized arguments

    Raises:
        InjectionError: If potential injection is detected
    """
    dangerous_chars = "<>|&;(){}[]$`\n\r"
    dangerous_patterns = ["cd ", "rm ", "mv ", "cp "]

    sanitized = []
    for arg in args:
        if not arg:
            raise InjectionError("Empty argument detected")

        if any(pattern in arg.lower() for pattern in dangerous_patterns):
            raise InjectionError("Dangerous command pattern detected")

        if any(c in dangerous_chars for c in arg):
            raise InjectionError(f"Invalid character in argument: {arg}")

        # Additional checks for shell injection
        if arg.startswith("-c") or arg.startswith("-e"):
            raise InjectionError("Shell injection attempt detected")

        sanitized.append(arg)

    return sanitized


def is_safe_url(url: str) -> bool:
    """
    Validate that a URL is safe to use.

    Args:
        url: URL to validate

    Returns:
        True if URL is safe, False otherwise
    """
    # Basic URL validation
    if not url.startswith(("http://", "https://")):
        return False

    # Check for dangerous protocols
    dangerous_protocols = ["ftp:", "file:", "javascript:"]
    if any(proto in url.lower() for proto in dangerous_protocols):
        return False

    # Check for suspicious patterns
    dangerous_chars = '[<>"\0\n\r]'
    if re.search(dangerous_chars, url):
        return False

    return True
