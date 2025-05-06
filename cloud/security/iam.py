"""
IAM (Identity and Access Management) integration for cloud services.

This module provides integration with enterprise IAM solutions for
authentication and authorization with cloud services.

Copyright (c) 2025 Vikas Sahani
GitHub: https://github.com/VIKAS9793
Email: vikassahani17@gmail.com

Licensed under MIT License - see LICENSE file for details
This copyright and license applies only to the original code in this file,
not to any third-party libraries or dependencies used.
"""

import functools
import json
import logging
import os
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from cloud.config import CloudProvider
from cloud.errors import CloudAuthenticationError, CloudAuthorizationError
from src.utils.date_provider import DateProvider

logger = logging.getLogger(__name__)

# Type variable for function return type
T = TypeVar("T")


class IAMProvider(Enum):
    """Supported IAM providers."""

    AWS_IAM = "aws_iam"
    AZURE_AD = "azure_ad"
    GCP_IAM = "gcp_iam"
    OKTA = "okta"
    AUTH0 = "auth0"
    KEYCLOAK = "keycloak"
    CUSTOM = "custom"


class Role:
    """
    Represents an IAM role with associated permissions.

    Attributes:
        name: Role name
        permissions: List of permissions
        scope: Scope of the role (e.g., global, resource-specific)
    """

    def __init__(
        self,
        name: str,
        permissions: List[str],
        scope: str = "global",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a role.

        Args:
            name: Role name
            permissions: List of permissions
            scope: Scope of the role
            metadata: Additional metadata
        """
        self.name = name
        self.permissions = permissions
        self.scope = scope
        self.metadata = metadata or {}

    def has_permission(self, permission: str) -> bool:
        """
        Check if the role has a specific permission.

        Args:
            permission: Permission to check

        Returns:
            True if the role has the permission, False otherwise
        """
        # Check for exact match
        if permission in self.permissions:
            return True

        # Check for wildcard
        for perm in self.permissions:
            if perm.endswith("*"):
                prefix = perm[:-1]
                if permission.startswith(prefix):
                    return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert role to dictionary representation."""
        return {
            "name": self.name,
            "permissions": self.permissions,
            "scope": self.scope,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Role":
        """Create role from dictionary representation."""
        return cls(
            name=data["name"],
            permissions=data["permissions"],
            scope=data.get("scope", "global"),
            metadata=data.get("metadata", {}),
        )


class Credential:
    """
    Represents a credential for authentication with cloud services.

    Attributes:
        provider: Cloud provider
        type: Credential type (e.g., access_key, oauth_token)
        value: Credential value
        expiry: Expiry time
    """

    def __init__(
        self,
        provider: Union[CloudProvider, str],
        type: str,
        value: Any,
        expiry: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a credential.

        Args:
            provider: Cloud provider
            type: Credential type
            value: Credential value
            expiry: Expiry time
            metadata: Additional metadata
        """
        # Normalize provider
        if isinstance(provider, str):
            try:
                self.provider = CloudProvider(provider.lower())
            except ValueError:
                self.provider_str = provider.lower()
                self.provider = None
        else:
            self.provider = provider
            self.provider_str = provider.value

        self.type = type
        self.value = value
        self.expiry = expiry
        self.metadata = metadata or {}
        self.created_at = DateProvider.get_instance().now()

    def is_expired(self) -> bool:
        """
        Check if the credential is expired.

        Returns:
            True if the credential is expired, False otherwise
        """
        if self.expiry is None:
            return False

        return DateProvider.get_instance().now() > self.expiry

    def time_to_expiry(self) -> Optional[timedelta]:
        """
        Get the time until expiry.

        Returns:
            Time until expiry, or None if no expiry
        """
        if self.expiry is None:
            return None

        return self.expiry - DateProvider.get_instance().now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert credential to dictionary representation."""
        result = {
            "provider": self.provider_str,
            "type": self.type,
            "value": self.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

        if self.expiry:
            result["expiry"] = self.expiry.isoformat()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Credential":
        """Create credential from dictionary representation."""
        expiry = None
        if "expiry" in data and data["expiry"]:
            expiry = datetime.fromisoformat(data["expiry"])

        return cls(
            provider=data["provider"],
            type=data["type"],
            value=data["value"],
            expiry=expiry,
            metadata=data.get("metadata", {}),
        )


class IAMManager:
    """
    Manager for IAM (Identity and Access Management) operations.

    This class provides methods for authentication, authorization,
    and credential management for cloud services.
    """

    _instance = None
    _lock = None

    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            import threading

            cls._lock = threading.Lock()
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(IAMManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize IAM manager (only once for singleton)."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        # Storage
        self.roles: Dict[str, Role] = {}
        self.credentials: Dict[str, Dict[str, Credential]] = {}  # {provider: {name: credential}}
        self.tokens: Dict[str, Dict[str, Any]] = {}  # {token_id: {token_data}}

        # Provider-specific connectors
        self.iam_connectors: Dict[IAMProvider, Any] = {}

        # Configuration
        self.default_iam_provider = IAMProvider.CUSTOM
        self.token_expiry = timedelta(hours=1)
        self.credential_refresh_threshold = timedelta(minutes=10)

        # Thread safety
        import threading

        self.lock = threading.RLock()

        # Credential refresh thread
        self._refresh_thread_running = False

        self._initialized = True

        # Start credential refresh thread
        self._start_credential_refresh_thread()

    def _start_credential_refresh_thread(self):
        """Start the credential refresh background thread."""
        if not self._refresh_thread_running:
            import threading

            refresh_thread = threading.Thread(target=self._credential_refresh_task, daemon=True)
            refresh_thread.start()
            self._refresh_thread_running = True

    def _credential_refresh_task(self):
        """Background task for refreshing credentials."""
        while True:
            try:
                self._refresh_expiring_credentials()
            except Exception as e:
                logger.error(f"Error refreshing credentials: {str(e)}")

            # Sleep for 1 minute
            time.sleep(60)

    def _refresh_expiring_credentials(self):
        """Refresh credentials that are about to expire."""
        with self.lock:
            now = DateProvider.get_instance().now()

            for provider, creds in self.credentials.items():
                for name, credential in list(creds.items()):
                    # Skip credentials without expiry
                    if credential.expiry is None:
                        continue

                    # Check if credential is about to expire
                    if credential.expiry - now <= self.credential_refresh_threshold:
                        try:
                            # Try to refresh the credential
                            self.refresh_credential(provider, name)
                        except Exception as e:
                            logger.warning(
                                f"Failed to refresh credential {provider}/{name}: {str(e)}"
                            )

    def register_iam_connector(self, provider: IAMProvider, connector: Any) -> None:
        """
        Register an IAM connector for a specific provider.

        Args:
            provider: IAM provider
            connector: Provider-specific connector
        """
        with self.lock:
            self.iam_connectors[provider] = connector

    def set_default_iam_provider(self, provider: IAMProvider) -> None:
        """
        Set the default IAM provider.

        Args:
            provider: IAM provider
        """
        with self.lock:
            self.default_iam_provider = provider

    def add_role(self, role: Role) -> None:
        """
        Add a role.

        Args:
            role: Role to add
        """
        with self.lock:
            self.roles[role.name] = role

    def get_role(self, role_name: str) -> Optional[Role]:
        """
        Get a role by name.

        Args:
            role_name: Role name

        Returns:
            Role, or None if not found
        """
        with self.lock:
            return self.roles.get(role_name)

    def remove_role(self, role_name: str) -> None:
        """
        Remove a role.

        Args:
            role_name: Role name
        """
        with self.lock:
            if role_name in self.roles:
                del self.roles[role_name]

    def add_credential(
        self,
        provider: Union[CloudProvider, str],
        name: str,
        credential_type: str,
        value: Any,
        expiry: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Credential:
        """
        Add a credential.

        Args:
            provider: Cloud provider
            name: Credential name
            credential_type: Credential type
            value: Credential value
            expiry: Expiry time
            metadata: Additional metadata

        Returns:
            The created credential
        """
        # Create credential
        credential = Credential(
            provider=provider,
            type=credential_type,
            value=value,
            expiry=expiry,
            metadata=metadata,
        )

        # Normalize provider
        provider_str = credential.provider_str

        with self.lock:
            # Ensure provider exists in credentials
            if provider_str not in self.credentials:
                self.credentials[provider_str] = {}

            # Add credential
            self.credentials[provider_str][name] = credential

        return credential

    def get_credential(
        self, provider: Union[CloudProvider, str], name: str
    ) -> Optional[Credential]:
        """
        Get a credential by provider and name.

        Args:
            provider: Cloud provider
            name: Credential name

        Returns:
            Credential, or None if not found
        """
        # Normalize provider
        if isinstance(provider, CloudProvider):
            provider_str = provider.value
        else:
            provider_str = str(provider).lower()

        with self.lock:
            if provider_str not in self.credentials:
                return None

            return self.credentials[provider_str].get(name)

    def remove_credential(self, provider: Union[CloudProvider, str], name: str) -> None:
        """
        Remove a credential.

        Args:
            provider: Cloud provider
            name: Credential name
        """
        # Normalize provider
        if isinstance(provider, CloudProvider):
            provider_str = provider.value
        else:
            provider_str = str(provider).lower()

        with self.lock:
            if provider_str in self.credentials and name in self.credentials[provider_str]:
                del self.credentials[provider_str][name]

    def refresh_credential(
        self, provider: Union[CloudProvider, str], name: str
    ) -> Optional[Credential]:
        """
        Refresh a credential.

        Args:
            provider: Cloud provider
            name: Credential name

        Returns:
            Refreshed credential, or None if refresh failed
        """
        # Normalize provider
        if isinstance(provider, CloudProvider):
            provider_str = provider.value
            provider_enum = provider
        else:
            provider_str = str(provider).lower()
            try:
                provider_enum = CloudProvider(provider_str)
            except ValueError:
                provider_enum = None

        with self.lock:
            # Check if credential exists
            if provider_str not in self.credentials or name not in self.credentials[provider_str]:
                logger.warning(f"Credential not found: {provider_str}/{name}")
                return None

            credential = self.credentials[provider_str][name]

            # Try to refresh using provider-specific connector
            if provider_enum == CloudProvider.AWS:
                return self._refresh_aws_credential(credential, name)
            elif provider_enum == CloudProvider.AZURE:
                return self._refresh_azure_credential(credential, name)
            elif provider_enum == CloudProvider.GCP:
                return self._refresh_gcp_credential(credential, name)
            else:
                logger.warning(f"No credential refresh implementation for provider: {provider_str}")
                return None

    def _refresh_aws_credential(self, credential: Credential, name: str) -> Optional[Credential]:
        """Refresh an AWS credential."""
        # Check credential type
        if credential.type == "iam_role":
            try:
                # Import boto3 only when needed
                import boto3

                # Get role ARN from credential metadata
                role_arn = credential.metadata.get("role_arn")
                session_name = credential.metadata.get("session_name", "cloud-infra-session")

                if not role_arn:
                    logger.warning(
                        "Cannot refresh AWS IAM role credential: role_arn not found in metadata"
                    )
                    return None

                # Create STS client
                sts_client = boto3.client("sts")

                # Assume role
                response = sts_client.assume_role(
                    RoleArn=role_arn,
                    RoleSessionName=session_name,
                    DurationSeconds=3600,  # 1 hour
                )

                # Extract credentials
                credentials = response["Credentials"]

                # Create new credential
                new_credential = Credential(
                    provider=CloudProvider.AWS,
                    type="temporary_credentials",
                    value={
                        "AccessKeyId": credentials["AccessKeyId"],
                        "SecretAccessKey": credentials["SecretAccessKey"],
                        "SessionToken": credentials["SessionToken"],
                    },
                    expiry=credentials["Expiration"],
                    metadata={
                        "source_credential": name,
                        "role_arn": role_arn,
                        "session_name": session_name,
                    },
                )

                # Store the new credential
                self.credentials[credential.provider_str][name] = new_credential

                logger.info(f"Refreshed AWS IAM role credential: {name}")

                return new_credential

            except Exception as e:
                logger.error(f"Error refreshing AWS IAM role credential: {str(e)}")
                return None
        else:
            logger.warning(f"Cannot refresh AWS credential of type: {credential.type}")
            return None

    def _refresh_azure_credential(self, credential: Credential, name: str) -> Optional[Credential]:
        """Refresh an Azure credential."""
        # Check credential type
        if credential.type == "service_principal":
            try:
                # Import Azure SDK only when needed
                from azure.identity import ClientSecretCredential

                # Get client details from credential metadata
                tenant_id = credential.metadata.get("tenant_id")
                client_id = credential.metadata.get("client_id")
                client_secret = credential.value.get("secret")

                if not all([tenant_id, client_id, client_secret]):
                    logger.warning(
                        "Cannot refresh Azure service principal: missing required metadata"
                    )
                    return None

                # Create credential
                azure_credential = ClientSecretCredential(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    client_secret=client_secret,
                )

                # Get token

                token = azure_credential.get_token("https://management.azure.com/.default")

                # Create new credential
                new_credential = Credential(
                    provider=CloudProvider.AZURE,
                    type="access_token",
                    value={
                        "token": token.token,
                    },
                    expiry=datetime.fromtimestamp(token.expires_on),
                    metadata={
                        "source_credential": name,
                        "tenant_id": tenant_id,
                        "client_id": client_id,
                    },
                )

                # Store the new credential
                self.credentials[credential.provider_str][name] = new_credential

                logger.info(f"Refreshed Azure service principal credential: {name}")

                return new_credential

            except Exception as e:
                logger.error(f"Error refreshing Azure service principal credential: {str(e)}")
                return None
        else:
            logger.warning(f"Cannot refresh Azure credential of type: {credential.type}")
            return None

    def _refresh_gcp_credential(self, credential: Credential, name: str) -> Optional[Credential]:
        """Refresh a GCP credential."""
        # Check credential type
        if credential.type == "service_account":
            try:
                # Import Google Cloud SDK only when needed
                import google.auth.transport.requests
                from google.oauth2 import service_account

                # Get service account key from credential value
                key_data = credential.value

                if not key_data:
                    logger.warning("Cannot refresh GCP service account: key data not found")
                    return None

                # Create credentials
                credentials = service_account.Credentials.from_service_account_info(
                    key_data, scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )

                # Refresh token
                request = google.auth.transport.requests.Request()
                credentials.refresh(request)

                # Create new credential
                new_credential = Credential(
                    provider=CloudProvider.GCP,
                    type="access_token",
                    value={
                        "token": credentials.token,
                    },
                    expiry=credentials.expiry,
                    metadata={
                        "source_credential": name,
                        "project_id": key_data.get("project_id"),
                    },
                )

                # Store the new credential
                self.credentials[credential.provider_str][name] = new_credential

                logger.info(f"Refreshed GCP service account credential: {name}")

                return new_credential

            except Exception as e:
                logger.error(f"Error refreshing GCP service account credential: {str(e)}")
                return None
        else:
            logger.warning(f"Cannot refresh GCP credential of type: {credential.type}")
            return None

    def create_token(
        self,
        user_id: str,
        roles: List[str],
        expiry: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create an authentication token.

        Args:
            user_id: User ID
            roles: List of role names
            expiry: Token expiry time
            metadata: Additional metadata

        Returns:
            Token ID
        """
        import uuid

        # Generate token ID
        token_id = str(uuid.uuid4())

        # Set expiry if not provided
        if expiry is None:
            expiry = DateProvider.get_instance().now() + self.token_expiry

        # Create token data
        token_data = {
            "id": token_id,
            "user_id": user_id,
            "roles": roles,
            "created_at": DateProvider.get_instance().iso_format(),
            "expiry": expiry.isoformat(),
            "metadata": metadata or {},
        }

        with self.lock:
            self.tokens[token_id] = token_data

        return token_id

    def validate_token(self, token_id: str) -> Optional[Dict[str, Any]]:
        """
        Validate an authentication token.

        Args:
            token_id: Token ID

        Returns:
            Token data if valid, None otherwise
        """
        with self.lock:
            # Check if token exists
            if token_id not in self.tokens:
                return None

            token_data = self.tokens[token_id]

            # Check if token is expired
            expiry = datetime.fromisoformat(token_data["expiry"])
            if DateProvider.get_instance().now() > expiry:
                # Remove expired token
                del self.tokens[token_id]
                return None

            return token_data

    def revoke_token(self, token_id: str) -> None:
        """
        Revoke an authentication token.

        Args:
            token_id: Token ID
        """
        with self.lock:
            if token_id in self.tokens:
                del self.tokens[token_id]

    def has_permission(
        self, token_id: str, permission: str, resource: Optional[str] = None
    ) -> bool:
        """
        Check if a token has a specific permission.

        Args:
            token_id: Token ID
            permission: Permission to check
            resource: Optional resource to check permission against

        Returns:
            True if the token has the permission, False otherwise
        """
        # Validate token
        token_data = self.validate_token(token_id)
        if token_data is None:
            return False

        # Get roles
        role_names = token_data["roles"]

        with self.lock:
            # Check each role
            for role_name in role_names:
                if role_name not in self.roles:
                    continue

                role = self.roles[role_name]

                # Check if role has permission
                if role.has_permission(permission):
                    # If resource is specified, check if role scope matches
                    if resource and role.scope != "global" and role.scope != resource:
                        continue

                    return True

        return False


# Global IAM manager instance
_iam_manager = IAMManager()


def get_iam_manager() -> IAMManager:
    """Get the global IAM manager instance."""
    return _iam_manager


def with_auth(
    permission: str, resource: Optional[str] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to require authentication and authorization for a function.

    Args:
        permission: Required permission
        resource: Optional resource to check permission against

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Get token ID from kwargs
            token_id = kwargs.pop("token_id", None)

            if token_id is None:
                raise CloudAuthenticationError("Authentication token required")

            # Check permission
            if not _iam_manager.has_permission(token_id, permission, resource):
                raise CloudAuthorizationError(f"Permission denied: {permission}")

            # Call the function
            return func(*args, **kwargs)

        return wrapper

    return decorator


def load_aws_credentials_from_environment() -> None:
    """
    Load AWS credentials from environment variables.

    This loads credentials from the standard AWS environment variables:
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_SESSION_TOKEN
    """
    # Check if required environment variables are set
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_session_token = os.environ.get("AWS_SESSION_TOKEN")

    if not aws_access_key_id or not aws_secret_access_key:
        logger.warning("AWS credentials not found in environment variables")
        return

    # Create credential value
    value = {
        "access_key_id": aws_access_key_id,
        "secret_access_key": aws_secret_access_key,
    }

    if aws_session_token:
        value["session_token"] = aws_session_token

    # Add credential
    _iam_manager.add_credential(
        provider=CloudProvider.AWS,
        name="env_credentials",
        credential_type="access_key",
        value=value,
        metadata={"source": "environment"},
    )

    logger.info("Loaded AWS credentials from environment variables")


def load_azure_credentials_from_environment() -> None:
    """
    Load Azure credentials from environment variables.

    This loads credentials from the standard Azure environment variables:
    - AZURE_TENANT_ID
    - AZURE_CLIENT_ID
    - AZURE_CLIENT_SECRET
    """
    # Check if required environment variables are set
    azure_tenant_id = os.environ.get("AZURE_TENANT_ID")
    azure_client_id = os.environ.get("AZURE_CLIENT_ID")
    azure_client_secret = os.environ.get("AZURE_CLIENT_SECRET")

    if not azure_tenant_id or not azure_client_id or not azure_client_secret:
        logger.warning("Azure credentials not found in environment variables")
        return

    # Create credential value and metadata
    value = {"secret": azure_client_secret}

    metadata = {
        "tenant_id": azure_tenant_id,
        "client_id": azure_client_id,
        "source": "environment",
    }

    # Add credential
    _iam_manager.add_credential(
        provider=CloudProvider.AZURE,
        name="env_credentials",
        credential_type="service_principal",
        value=value,
        metadata=metadata,
    )

    logger.info("Loaded Azure credentials from environment variables")


def load_gcp_credentials_from_environment() -> None:
    """
    Load GCP credentials from environment variables.

    This loads credentials from the standard GCP environment variable:
    - GOOGLE_APPLICATION_CREDENTIALS
    """
    # Check if required environment variable is set
    gcp_credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    if not gcp_credentials_path:
        logger.warning("GCP credentials not found in environment variables")
        return

    try:
        # Read service account key file
        with open(gcp_credentials_path, "r") as f:
            key_data = json.load(f)

        # Add credential
        _iam_manager.add_credential(
            provider=CloudProvider.GCP,
            name="env_credentials",
            credential_type="service_account",
            value=key_data,
            metadata={"source": "environment", "path": gcp_credentials_path},
        )

        logger.info("Loaded GCP credentials from environment variables")

    except Exception as e:
        logger.error(f"Error loading GCP credentials: {str(e)}")


def load_credentials_from_environment() -> None:
    """Load all available credentials from environment variables."""
    load_aws_credentials_from_environment()
    load_azure_credentials_from_environment()
    load_gcp_credentials_from_environment()
