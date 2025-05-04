"""
Cloud configuration management for CustomerAI platform.

This module handles configuration for different cloud providers
and ensures credentials and settings are properly managed.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class CloudProvider(Enum):
    """Supported cloud service providers."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    NONE = "none"

class CloudConfig:
    """
    Configuration manager for cloud service providers.
    
    This class handles loading and providing access to configuration
    settings for different cloud providers.
    """
    
    def __init__(self, provider: Union[CloudProvider, str] = None):
        """
        Initialize cloud configuration.
        
        Args:
            provider: Cloud provider to use (AWS, Azure, GCP)
        """
        self.configs = {}
        
        # Determine provider
        if provider is None:
            # Try to get from environment
            provider_str = os.getenv("CLOUD_PROVIDER", "")
            try:
                self.provider = CloudProvider(provider_str.lower()) if provider_str else CloudProvider.NONE
            except ValueError:
                logger.warning(f"Unknown cloud provider: {provider_str}, defaulting to none")
                self.provider = CloudProvider.NONE
        elif isinstance(provider, CloudProvider):
            self.provider = provider
        else:
            try:
                self.provider = CloudProvider(provider.lower())
            except ValueError:
                logger.warning(f"Unknown cloud provider: {provider}, defaulting to none")
                self.provider = CloudProvider.NONE
        
        # Load configurations
        self._load_configs()
    
    def _load_configs(self) -> None:
        """Load configuration for all providers from environment and config files."""
        # AWS Configuration
        self.configs[CloudProvider.AWS] = {
            "region": os.getenv("AWS_REGION", "us-east-1"),
            "access_key_id": os.getenv("AWS_ACCESS_KEY_ID", ""),
            "secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            "s3_bucket": os.getenv("AWS_S3_BUCKET", "customerai-data"),
            "dynamodb_table": os.getenv("AWS_DYNAMODB_TABLE", "customerai-data"),
            "lambda_function": os.getenv("AWS_LAMBDA_FUNCTION", "customerai-function"),
            "cognito_user_pool": os.getenv("AWS_COGNITO_USER_POOL", ""),
            "cognito_client_id": os.getenv("AWS_COGNITO_CLIENT_ID", ""),
            "kms_key_id": os.getenv("AWS_KMS_KEY_ID", ""),
            "sagemaker_endpoint": os.getenv("AWS_SAGEMAKER_ENDPOINT", ""),
            "cloudwatch_log_group": os.getenv("AWS_CLOUDWATCH_LOG_GROUP", "/customerai/logs"),
        }
        
        # Azure Configuration
        self.configs[CloudProvider.AZURE] = {
            "tenant_id": os.getenv("AZURE_TENANT_ID", ""),
            "client_id": os.getenv("AZURE_CLIENT_ID", ""),
            "client_secret": os.getenv("AZURE_CLIENT_SECRET", ""),
            "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID", ""),
            "resource_group": os.getenv("AZURE_RESOURCE_GROUP", "customerai-resources"),
            "storage_account": os.getenv("AZURE_STORAGE_ACCOUNT", "customeraidata"),
            "storage_container": os.getenv("AZURE_STORAGE_CONTAINER", "data"),
            "cosmos_db_account": os.getenv("AZURE_COSMOS_DB_ACCOUNT", ""),
            "cosmos_db_database": os.getenv("AZURE_COSMOS_DB_DATABASE", "customerai"),
            "function_app_name": os.getenv("AZURE_FUNCTION_APP_NAME", "customerai-functions"),
            "key_vault_name": os.getenv("AZURE_KEY_VAULT_NAME", "customerai-vault"),
            "ml_workspace": os.getenv("AZURE_ML_WORKSPACE", "customerai-ml"),
            "app_insights_key": os.getenv("AZURE_APP_INSIGHTS_KEY", ""),
        }
        
        # GCP Configuration
        self.configs[CloudProvider.GCP] = {
            "project_id": os.getenv("GCP_PROJECT_ID", ""),
            "credentials_file": os.getenv("GCP_CREDENTIALS_FILE", ""),
            "storage_bucket": os.getenv("GCP_STORAGE_BUCKET", "customerai-data"),
            "firestore_collection": os.getenv("GCP_FIRESTORE_COLLECTION", "customerai-data"),
            "cloud_function_name": os.getenv("GCP_CLOUD_FUNCTION_NAME", "customerai-function"),
            "ml_model_name": os.getenv("GCP_ML_MODEL_NAME", "customerai-model"),
            "kms_key_ring": os.getenv("GCP_KMS_KEY_RING", "customerai-keyring"),
            "kms_key_name": os.getenv("GCP_KMS_KEY_NAME", "customerai-key"),
            "logging_name": os.getenv("GCP_LOGGING_NAME", "customerai-logs"),
        }
        
        # Try to load from config file if it exists
        config_file = os.getenv("CLOUD_CONFIG_FILE", "cloud_config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_configs = json.load(f)
                    
                # Update configs with file values
                for provider_name, config in file_configs.items():
                    try:
                        provider = CloudProvider(provider_name.lower())
                        if provider in self.configs:
                            self.configs[provider].update(config)
                    except ValueError:
                        logger.warning(f"Unknown provider in config file: {provider_name}")
                        
            except Exception as e:
                logger.error(f"Error loading cloud config file: {str(e)}")
    
    def get_config(self, provider: Union[CloudProvider, str] = None) -> Dict[str, Any]:
        """
        Get configuration for specified provider.
        
        Args:
            provider: Cloud provider (defaults to the initialized provider)
            
        Returns:
            Dict of configuration values
        """
        if provider is None:
            provider = self.provider
        elif isinstance(provider, str):
            try:
                provider = CloudProvider(provider.lower())
            except ValueError:
                logger.warning(f"Unknown provider: {provider}, returning empty config")
                return {}
        
        return self.configs.get(provider, {})
    
    def is_configured(self, provider: Union[CloudProvider, str] = None) -> bool:
        """
        Check if a provider is properly configured with required credentials.
        
        Args:
            provider: Cloud provider to check
            
        Returns:
            True if properly configured, False otherwise
        """
        config = self.get_config(provider)
        
        if provider is None:
            provider = self.provider
        elif isinstance(provider, str):
            try:
                provider = CloudProvider(provider.lower())
            except ValueError:
                return False
        
        # Check required fields for each provider
        if provider == CloudProvider.AWS:
            return bool(config.get("access_key_id")) and bool(config.get("secret_access_key"))
        elif provider == CloudProvider.AZURE:
            return (bool(config.get("tenant_id")) and 
                    bool(config.get("client_id")) and 
                    bool(config.get("client_secret")))
        elif provider == CloudProvider.GCP:
            return bool(config.get("project_id")) and (
                bool(config.get("credentials_file")) or 
                os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            )
        
        return False
    
    def get_current_provider(self) -> CloudProvider:
        """Get the currently configured cloud provider."""
        return self.provider
    
    def get_provider_name(self) -> str:
        """Get the name of the currently configured cloud provider."""
        return self.provider.value 