"""
Error handling utilities for cloud services.

This module provides standardized error handling for cloud service operations,
with consistent error types and proper error mapping.

Copyright (c) 2025 Vikas Sahani
GitHub: https://github.com/VIKAS9793
Email: vikassahani17@gmail.com

Licensed under MIT License - see LICENSE file for details
"""

import logging
import traceback
from typing import Dict, Any, Optional, Union, Type
from enum import Enum

from cloud.config import CloudProvider

# Configure logging
logger = logging.getLogger(__name__)

# Error categories
class ErrorCategory(Enum):
    """Categories of cloud service errors."""
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication" 
    AUTHORIZATION = "authorization"
    RESOURCE_NOT_FOUND = "resource_not_found"
    RESOURCE_EXISTS = "resource_exists"
    SERVICE_UNAVAILABLE = "service_unavailable"
    QUOTA_EXCEEDED = "quota_exceeded"
    INVALID_REQUEST = "invalid_request"
    NETWORK = "network"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

class CloudError(Exception):
    """Base exception for all cloud-related errors."""
    
    def __init__(
        self, 
        message: str, 
        provider: Optional[Union[CloudProvider, str]] = None,
        service: Optional[str] = None,
        category: Optional[Union[ErrorCategory, str]] = None,
        original_exception: Optional[Exception] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize cloud error.
        
        Args:
            message: Error message
            provider: Cloud provider (AWS, Azure, GCP)
            service: Service name (e.g., 's3', 'blob_storage')
            category: Error category
            original_exception: Original exception that was caught
            error_code: Provider-specific error code
            details: Additional error details
        """
        self.message = message
        
        # Set provider
        if isinstance(provider, str):
            try:
                self.provider = CloudProvider(provider.lower())
            except ValueError:
                self.provider = None
        else:
            self.provider = provider
        
        # Set service
        self.service = service
        
        # Set category
        if isinstance(category, str):
            try:
                self.category = ErrorCategory(category.lower())
            except ValueError:
                self.category = ErrorCategory.UNKNOWN
        else:
            self.category = category or ErrorCategory.UNKNOWN
        
        # Set original exception
        self.original_exception = original_exception
        
        # Set error code
        self.error_code = error_code
        
        # Set details
        self.details = details or {}
        
        # Generate traceback for debugging
        self.traceback = traceback.format_exc() if original_exception else None
        
        # Call parent constructor
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        result = {
            "message": self.message,
            "category": self.category.value if self.category else None,
        }
        
        if self.provider:
            result["provider"] = self.provider.value
        
        if self.service:
            result["service"] = self.service
        
        if self.error_code:
            result["error_code"] = self.error_code
        
        if self.details:
            result["details"] = self.details
        
        return result
    
    def log(self, level: int = logging.ERROR) -> None:
        """Log the error with appropriate level."""
        error_dict = self.to_dict()
        
        # Add provider-specific information
        if self.provider and self.service:
            prefix = f"{self.provider.value.upper()}:{self.service}"
        elif self.provider:
            prefix = self.provider.value.upper()
        elif self.service:
            prefix = self.service
        else:
            prefix = "CLOUD"
        
        # Create log message
        log_message = f"{prefix} Error: {self.message}"
        
        # Add error code if available
        if self.error_code:
            log_message += f" (Code: {self.error_code})"
        
        # Log with appropriate level
        if level == logging.DEBUG:
            logger.debug(log_message, extra={"error_details": error_dict})
        elif level == logging.INFO:
            logger.info(log_message, extra={"error_details": error_dict})
        elif level == logging.WARNING:
            logger.warning(log_message, extra={"error_details": error_dict})
        elif level == logging.ERROR:
            logger.error(log_message, extra={"error_details": error_dict})
        elif level == logging.CRITICAL:
            logger.critical(log_message, extra={"error_details": error_dict})
        
        # Log original exception traceback
        if self.original_exception and level >= logging.ERROR:
            logger.debug(f"Original exception: {type(self.original_exception).__name__}: {str(self.original_exception)}")
            if self.traceback:
                logger.debug(f"Traceback: {self.traceback}")


# Specific error types
class CloudConfigurationError(CloudError):
    """Error in cloud service configuration."""
    
    def __init__(self, message: str, **kwargs):
        """Initialize with configuration error category."""
        kwargs["category"] = ErrorCategory.CONFIGURATION
        super().__init__(message, **kwargs)


class CloudAuthenticationError(CloudError):
    """Error in cloud service authentication."""
    
    def __init__(self, message: str, **kwargs):
        """Initialize with authentication error category."""
        kwargs["category"] = ErrorCategory.AUTHENTICATION
        super().__init__(message, **kwargs)


class CloudAuthorizationError(CloudError):
    """Error in cloud service authorization."""
    
    def __init__(self, message: str, **kwargs):
        """Initialize with authorization error category."""
        kwargs["category"] = ErrorCategory.AUTHORIZATION
        super().__init__(message, **kwargs)


class CloudResourceNotFoundError(CloudError):
    """Error when cloud resource is not found."""
    
    def __init__(self, message: str, **kwargs):
        """Initialize with resource not found error category."""
        kwargs["category"] = ErrorCategory.RESOURCE_NOT_FOUND
        super().__init__(message, **kwargs)


class CloudResourceExistsError(CloudError):
    """Error when cloud resource already exists."""
    
    def __init__(self, message: str, **kwargs):
        """Initialize with resource exists error category."""
        kwargs["category"] = ErrorCategory.RESOURCE_EXISTS
        super().__init__(message, **kwargs)


class CloudServiceUnavailableError(CloudError):
    """Error when cloud service is unavailable."""
    
    def __init__(self, message: str, **kwargs):
        """Initialize with service unavailable error category."""
        kwargs["category"] = ErrorCategory.SERVICE_UNAVAILABLE
        super().__init__(message, **kwargs)


class CloudQuotaExceededError(CloudError):
    """Error when cloud service quota is exceeded."""
    
    def __init__(self, message: str, **kwargs):
        """Initialize with quota exceeded error category."""
        kwargs["category"] = ErrorCategory.QUOTA_EXCEEDED
        super().__init__(message, **kwargs)


class CloudInvalidRequestError(CloudError):
    """Error when cloud service request is invalid."""
    
    def __init__(self, message: str, **kwargs):
        """Initialize with invalid request error category."""
        kwargs["category"] = ErrorCategory.INVALID_REQUEST
        super().__init__(message, **kwargs)


class CloudNetworkError(CloudError):
    """Error in cloud service network communication."""
    
    def __init__(self, message: str, **kwargs):
        """Initialize with network error category."""
        kwargs["category"] = ErrorCategory.NETWORK
        super().__init__(message, **kwargs)


class CloudTimeoutError(CloudError):
    """Error when cloud service operation times out."""
    
    def __init__(self, message: str, **kwargs):
        """Initialize with timeout error category."""
        kwargs["category"] = ErrorCategory.TIMEOUT
        super().__init__(message, **kwargs)


# Error handler functions
def handle_aws_error(e: Exception, service: str, operation: str) -> CloudError:
    """
    Convert AWS exception to standardized CloudError.
    
    Args:
        e: AWS exception
        service: AWS service name
        operation: Operation being performed
        
    Returns:
        Standardized CloudError
    """
    from botocore.exceptions import ClientError, BotoCoreError
    
    if isinstance(e, ClientError):
        error_response = getattr(e, 'response', {})
        error_code = error_response.get('Error', {}).get('Code', 'Unknown')
        error_message = error_response.get('Error', {}).get('Message', str(e))
        
        # Map common AWS error codes to error categories
        if error_code in ('AccessDenied', 'Forbidden'):
            return CloudAuthorizationError(
                f"Access denied for {operation} on {service}",
                provider=CloudProvider.AWS,
                service=service,
                original_exception=e,
                error_code=error_code,
                details={"operation": operation, "message": error_message}
            )
        elif error_code in ('InvalidAccessKeyId', 'InvalidSecurity', 'InvalidToken', 'MissingAuthenticationToken'):
            return CloudAuthenticationError(
                f"Authentication failed for {operation} on {service}",
                provider=CloudProvider.AWS,
                service=service,
                original_exception=e,
                error_code=error_code,
                details={"operation": operation, "message": error_message}
            )
        elif error_code == 'NoSuchBucket' or error_code == 'NoSuchKey' or error_code == '404':
            return CloudResourceNotFoundError(
                f"Resource not found for {operation} on {service}",
                provider=CloudProvider.AWS,
                service=service,
                original_exception=e,
                error_code=error_code,
                details={"operation": operation, "message": error_message}
            )
        elif error_code == 'BucketAlreadyExists' or error_code == 'BucketAlreadyOwnedByYou':
            return CloudResourceExistsError(
                f"Resource already exists for {operation} on {service}",
                provider=CloudProvider.AWS,
                service=service,
                original_exception=e,
                error_code=error_code,
                details={"operation": operation, "message": error_message}
            )
        elif error_code == 'ServiceUnavailable':
            return CloudServiceUnavailableError(
                f"Service unavailable for {operation} on {service}",
                provider=CloudProvider.AWS,
                service=service,
                original_exception=e,
                error_code=error_code,
                details={"operation": operation, "message": error_message}
            )
        elif error_code == 'ThrottlingException' or error_code == 'RequestLimitExceeded':
            return CloudQuotaExceededError(
                f"Rate limit exceeded for {operation} on {service}",
                provider=CloudProvider.AWS,
                service=service,
                original_exception=e,
                error_code=error_code,
                details={"operation": operation, "message": error_message}
            )
        elif error_code == 'InvalidRequest' or error_code == 'ValidationError':
            return CloudInvalidRequestError(
                f"Invalid request for {operation} on {service}: {error_message}",
                provider=CloudProvider.AWS,
                service=service,
                original_exception=e,
                error_code=error_code,
                details={"operation": operation, "message": error_message}
            )
        else:
            return CloudError(
                f"Error in {operation} on {service}: {error_message}",
                provider=CloudProvider.AWS,
                service=service,
                original_exception=e,
                error_code=error_code,
                details={"operation": operation, "message": error_message}
            )
    elif isinstance(e, BotoCoreError):
        return CloudNetworkError(
            f"Network error in {operation} on {service}: {str(e)}",
            provider=CloudProvider.AWS,
            service=service,
            original_exception=e,
            details={"operation": operation}
        )
    else:
        return CloudError(
            f"Unknown error in {operation} on {service}: {str(e)}",
            provider=CloudProvider.AWS,
            service=service,
            original_exception=e,
            details={"operation": operation}
        )


def handle_azure_error(e: Exception, service: str, operation: str) -> CloudError:
    """
    Convert Azure exception to standardized CloudError.
    
    Args:
        e: Azure exception
        service: Azure service name
        operation: Operation being performed
        
    Returns:
        Standardized CloudError
    """
    # Import Azure SDK exceptions here to avoid dependency if not using Azure
    try:
        from azure.core.exceptions import (
            AzureError, ResourceNotFoundError, ResourceExistsError,
            ClientAuthenticationError, HttpResponseError
        )
        
        if isinstance(e, ResourceNotFoundError):
            return CloudResourceNotFoundError(
                f"Resource not found for {operation} on {service}",
                provider=CloudProvider.AZURE,
                service=service,
                original_exception=e,
                details={"operation": operation, "message": str(e)}
            )
        elif isinstance(e, ResourceExistsError):
            return CloudResourceExistsError(
                f"Resource already exists for {operation} on {service}",
                provider=CloudProvider.AZURE,
                service=service,
                original_exception=e,
                details={"operation": operation, "message": str(e)}
            )
        elif isinstance(e, ClientAuthenticationError):
            return CloudAuthenticationError(
                f"Authentication failed for {operation} on {service}",
                provider=CloudProvider.AZURE,
                service=service,
                original_exception=e,
                details={"operation": operation, "message": str(e)}
            )
        elif isinstance(e, HttpResponseError):
            status_code = getattr(e, 'status_code', None)
            
            if status_code == 401:
                return CloudAuthenticationError(
                    f"Authentication failed for {operation} on {service}",
                    provider=CloudProvider.AZURE,
                    service=service,
                    original_exception=e,
                    error_code=str(status_code),
                    details={"operation": operation, "message": str(e)}
                )
            elif status_code == 403:
                return CloudAuthorizationError(
                    f"Authorization failed for {operation} on {service}",
                    provider=CloudProvider.AZURE,
                    service=service,
                    original_exception=e,
                    error_code=str(status_code),
                    details={"operation": operation, "message": str(e)}
                )
            elif status_code == 404:
                return CloudResourceNotFoundError(
                    f"Resource not found for {operation} on {service}",
                    provider=CloudProvider.AZURE,
                    service=service,
                    original_exception=e,
                    error_code=str(status_code),
                    details={"operation": operation, "message": str(e)}
                )
            elif status_code == 409:
                return CloudResourceExistsError(
                    f"Resource conflict for {operation} on {service}",
                    provider=CloudProvider.AZURE,
                    service=service,
                    original_exception=e,
                    error_code=str(status_code),
                    details={"operation": operation, "message": str(e)}
                )
            elif status_code == 429:
                return CloudQuotaExceededError(
                    f"Rate limit exceeded for {operation} on {service}",
                    provider=CloudProvider.AZURE,
                    service=service,
                    original_exception=e,
                    error_code=str(status_code),
                    details={"operation": operation, "message": str(e)}
                )
            elif status_code >= 400 and status_code < 500:
                return CloudInvalidRequestError(
                    f"Invalid request for {operation} on {service}",
                    provider=CloudProvider.AZURE,
                    service=service,
                    original_exception=e,
                    error_code=str(status_code),
                    details={"operation": operation, "message": str(e)}
                )
            elif status_code >= 500:
                return CloudServiceUnavailableError(
                    f"Service unavailable for {operation} on {service}",
                    provider=CloudProvider.AZURE,
                    service=service,
                    original_exception=e,
                    error_code=str(status_code),
                    details={"operation": operation, "message": str(e)}
                )
            else:
                return CloudError(
                    f"Error in {operation} on {service}: {str(e)}",
                    provider=CloudProvider.AZURE,
                    service=service,
                    original_exception=e,
                    error_code=str(status_code) if status_code else None,
                    details={"operation": operation, "message": str(e)}
                )
        elif isinstance(e, AzureError):
            return CloudError(
                f"Azure error in {operation} on {service}: {str(e)}",
                provider=CloudProvider.AZURE,
                service=service,
                original_exception=e,
                details={"operation": operation, "message": str(e)}
            )
        else:
            return CloudError(
                f"Unknown error in {operation} on {service}: {str(e)}",
                provider=CloudProvider.AZURE,
                service=service,
                original_exception=e,
                details={"operation": operation, "message": str(e)}
            )
    except ImportError:
        # Azure SDK not installed
        return CloudError(
            f"Unknown Azure error in {operation} on {service}: {str(e)}",
            provider=CloudProvider.AZURE,
            service=service,
            original_exception=e,
            details={"operation": operation, "message": str(e)}
        )


def handle_gcp_error(e: Exception, service: str, operation: str) -> CloudError:
    """
    Convert GCP exception to standardized CloudError.
    
    Args:
        e: GCP exception
        service: GCP service name
        operation: Operation being performed
        
    Returns:
        Standardized CloudError
    """
    # Import Google Cloud exceptions here to avoid dependency if not using GCP
    try:
        from google.api_core.exceptions import GoogleAPIError, NotFound, AlreadyExists
        from google.auth.exceptions import GoogleAuthError
        
        if isinstance(e, NotFound):
            return CloudResourceNotFoundError(
                f"Resource not found for {operation} on {service}",
                provider=CloudProvider.GCP,
                service=service,
                original_exception=e,
                details={"operation": operation, "message": str(e)}
            )
        elif isinstance(e, AlreadyExists):
            return CloudResourceExistsError(
                f"Resource already exists for {operation} on {service}",
                provider=CloudProvider.GCP,
                service=service,
                original_exception=e,
                details={"operation": operation, "message": str(e)}
            )
        elif isinstance(e, GoogleAuthError):
            return CloudAuthenticationError(
                f"Authentication failed for {operation} on {service}",
                provider=CloudProvider.GCP,
                service=service,
                original_exception=e,
                details={"operation": operation, "message": str(e)}
            )
        elif isinstance(e, GoogleAPIError):
            # Extract error code if available
            error_code = getattr(e, 'code', None)
            
            if error_code == 401:
                return CloudAuthenticationError(
                    f"Authentication failed for {operation} on {service}",
                    provider=CloudProvider.GCP,
                    service=service,
                    original_exception=e,
                    error_code=str(error_code),
                    details={"operation": operation, "message": str(e)}
                )
            elif error_code == 403:
                return CloudAuthorizationError(
                    f"Authorization failed for {operation} on {service}",
                    provider=CloudProvider.GCP,
                    service=service,
                    original_exception=e,
                    error_code=str(error_code),
                    details={"operation": operation, "message": str(e)}
                )
            elif error_code == 404:
                return CloudResourceNotFoundError(
                    f"Resource not found for {operation} on {service}",
                    provider=CloudProvider.GCP,
                    service=service,
                    original_exception=e,
                    error_code=str(error_code),
                    details={"operation": operation, "message": str(e)}
                )
            elif error_code == 409:
                return CloudResourceExistsError(
                    f"Resource conflict for {operation} on {service}",
                    provider=CloudProvider.GCP,
                    service=service,
                    original_exception=e,
                    error_code=str(error_code),
                    details={"operation": operation, "message": str(e)}
                )
            elif error_code == 429:
                return CloudQuotaExceededError(
                    f"Rate limit exceeded for {operation} on {service}",
                    provider=CloudProvider.GCP,
                    service=service,
                    original_exception=e,
                    error_code=str(error_code),
                    details={"operation": operation, "message": str(e)}
                )
            elif error_code == 400:
                return CloudInvalidRequestError(
                    f"Invalid request for {operation} on {service}",
                    provider=CloudProvider.GCP,
                    service=service,
                    original_exception=e,
                    error_code=str(error_code),
                    details={"operation": operation, "message": str(e)}
                )
            elif error_code == 503:
                return CloudServiceUnavailableError(
                    f"Service unavailable for {operation} on {service}",
                    provider=CloudProvider.GCP,
                    service=service,
                    original_exception=e,
                    error_code=str(error_code),
                    details={"operation": operation, "message": str(e)}
                )
            else:
                return CloudError(
                    f"Error in {operation} on {service}: {str(e)}",
                    provider=CloudProvider.GCP,
                    service=service,
                    original_exception=e,
                    error_code=str(error_code) if error_code else None,
                    details={"operation": operation, "message": str(e)}
                )
        else:
            return CloudError(
                f"Unknown error in {operation} on {service}: {str(e)}",
                provider=CloudProvider.GCP,
                service=service,
                original_exception=e,
                details={"operation": operation, "message": str(e)}
            )
    except ImportError:
        # Google Cloud SDK not installed
        return CloudError(
            f"Unknown GCP error in {operation} on {service}: {str(e)}",
            provider=CloudProvider.GCP,
            service=service,
            original_exception=e,
            details={"operation": operation, "message": str(e)}
        )


def handle_cloud_error(e: Exception, provider: Union[CloudProvider, str], service: str, operation: str) -> CloudError:
    """
    Convert any cloud provider exception to standardized CloudError.
    
    Args:
        e: Provider-specific exception
        provider: Cloud provider
        service: Service name
        operation: Operation being performed
        
    Returns:
        Standardized CloudError
    """
    # Convert string provider to enum
    if isinstance(provider, str):
        try:
            provider = CloudProvider(provider.lower())
        except ValueError:
            provider = None
    
    # Handle based on provider
    if provider == CloudProvider.AWS:
        return handle_aws_error(e, service, operation)
    elif provider == CloudProvider.AZURE:
        return handle_azure_error(e, service, operation)
    elif provider == CloudProvider.GCP:
        return handle_gcp_error(e, service, operation)
    else:
        # Generic error handling for unknown provider
        return CloudError(
            f"Error in {operation} on {service}: {str(e)}",
            provider=provider,
            service=service,
            original_exception=e,
            details={"operation": operation}
        ) 