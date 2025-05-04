"""
Cloud service factory for CustomerAI platform.

This module provides a factory for creating cloud service clients
for various providers (AWS, Azure, GCP).
"""

import logging
from typing import Any, Dict, Optional, Union, Type

from cloud.config import CloudConfig, CloudProvider
from cloud.errors import CloudError, CloudConfigurationError, handle_cloud_error

logger = logging.getLogger(__name__)

class CloudServiceFactory:
    """
    Factory for creating cloud service clients.
    
    This class provides methods for creating clients for various
    cloud services across different providers.
    """
    
    def __init__(self, config: Optional[CloudConfig] = None):
        """
        Initialize the cloud service factory.
        
        Args:
            config: Cloud configuration to use
        """
        self.config = config or CloudConfig()
        self.clients = {}
    
    def get_storage_client(self, provider: Union[CloudProvider, str] = None):
        """
        Get a client for cloud storage services.
        
        Args:
            provider: Cloud provider to use (defaults to configured provider)
            
        Returns:
            Storage client for the specified provider
            
        Raises:
            CloudConfigurationError: If the provider is not supported or not configured
        """
        try:
            if provider is None:
                provider = self.config.get_current_provider()
            elif isinstance(provider, str):
                try:
                    provider = CloudProvider(provider.lower())
                except ValueError:
                    raise CloudConfigurationError(
                        f"Unknown provider: {provider}",
                        service="storage"
                    )
            
            # Check if client already exists
            client_key = f"storage_{provider.value}"
            if client_key in self.clients:
                return self.clients[client_key]
            
            # Check if provider is configured
            if not self.config.is_configured(provider):
                raise CloudConfigurationError(
                    f"Provider {provider.value} is not properly configured",
                    provider=provider,
                    service="storage"
                )
            
            # Create new client based on provider
            if provider == CloudProvider.AWS:
                from cloud.storage.aws import AWSS3Client
                client = AWSS3Client(self.config)
            elif provider == CloudProvider.AZURE:
                from cloud.storage.azure import AzureBlobClient
                client = AzureBlobClient(self.config)
            elif provider == CloudProvider.GCP:
                from cloud.storage.gcp import GCPStorageClient
                client = GCPStorageClient(self.config)
            else:
                raise CloudConfigurationError(
                    f"Unsupported provider for storage: {provider.value}",
                    provider=provider,
                    service="storage"
                )
            
            # Cache and return client
            self.clients[client_key] = client
            return client
            
        except Exception as e:
            # Handle any unexpected errors
            if not isinstance(e, CloudError):
                cloud_error = handle_cloud_error(
                    e, 
                    provider or self.config.get_current_provider(),
                    "storage",
                    "get_client"
                )
                cloud_error.log()
                raise cloud_error
            else:
                e.log()
                raise
    
    def get_database_client(self, provider: Union[CloudProvider, str] = None):
        """
        Get a client for cloud database services.
        
        Args:
            provider: Cloud provider to use (defaults to configured provider)
            
        Returns:
            Database client for the specified provider
            
        Raises:
            CloudConfigurationError: If the provider is not supported or not configured
        """
        try:
            if provider is None:
                provider = self.config.get_current_provider()
            elif isinstance(provider, str):
                try:
                    provider = CloudProvider(provider.lower())
                except ValueError:
                    raise CloudConfigurationError(
                        f"Unknown provider: {provider}",
                        service="database"
                    )
            
            # Check if client already exists
            client_key = f"database_{provider.value}"
            if client_key in self.clients:
                return self.clients[client_key]
            
            # Check if provider is configured
            if not self.config.is_configured(provider):
                raise CloudConfigurationError(
                    f"Provider {provider.value} is not properly configured",
                    provider=provider,
                    service="database"
                )
            
            # Create new client based on provider
            if provider == CloudProvider.AWS:
                from cloud.database.aws import AWSDynamoDBClient
                client = AWSDynamoDBClient(self.config)
            elif provider == CloudProvider.AZURE:
                from cloud.database.azure import AzureCosmosDBClient
                client = AzureCosmosDBClient(self.config)
            elif provider == CloudProvider.GCP:
                from cloud.database.gcp import GCPFirestoreClient
                client = GCPFirestoreClient(self.config)
            else:
                raise CloudConfigurationError(
                    f"Unsupported provider for database: {provider.value}",
                    provider=provider,
                    service="database"
                )
            
            # Cache and return client
            self.clients[client_key] = client
            return client
            
        except Exception as e:
            # Handle any unexpected errors
            if not isinstance(e, CloudError):
                cloud_error = handle_cloud_error(
                    e, 
                    provider or self.config.get_current_provider(),
                    "database",
                    "get_client"
                )
                cloud_error.log()
                raise cloud_error
            else:
                e.log()
                raise
    
    def get_ai_client(self, provider: Union[CloudProvider, str] = None):
        """
        Get a client for cloud AI/ML services.
        
        Args:
            provider: Cloud provider to use (defaults to configured provider)
            
        Returns:
            AI/ML client for the specified provider
            
        Raises:
            CloudConfigurationError: If the provider is not supported or not configured
        """
        try:
            if provider is None:
                provider = self.config.get_current_provider()
            elif isinstance(provider, str):
                try:
                    provider = CloudProvider(provider.lower())
                except ValueError:
                    raise CloudConfigurationError(
                        f"Unknown provider: {provider}",
                        service="ai"
                    )
            
            # Check if client already exists
            client_key = f"ai_{provider.value}"
            if client_key in self.clients:
                return self.clients[client_key]
            
            # Check if provider is configured
            if not self.config.is_configured(provider):
                raise CloudConfigurationError(
                    f"Provider {provider.value} is not properly configured",
                    provider=provider,
                    service="ai"
                )
            
            # Create new client based on provider
            if provider == CloudProvider.AWS:
                from cloud.ai.aws import AWSSageMakerClient
                client = AWSSageMakerClient(self.config)
            elif provider == CloudProvider.AZURE:
                from cloud.ai.azure import AzureMLClient
                client = AzureMLClient(self.config)
            elif provider == CloudProvider.GCP:
                from cloud.ai.gcp import GCPVertexAIClient
                client = GCPVertexAIClient(self.config)
            else:
                raise CloudConfigurationError(
                    f"Unsupported provider for AI/ML: {provider.value}",
                    provider=provider,
                    service="ai"
                )
            
            # Cache and return client
            self.clients[client_key] = client
            return client
            
        except Exception as e:
            # Handle any unexpected errors
            if not isinstance(e, CloudError):
                cloud_error = handle_cloud_error(
                    e, 
                    provider or self.config.get_current_provider(),
                    "ai",
                    "get_client"
                )
                cloud_error.log()
                raise cloud_error
            else:
                e.log()
                raise
    
    def get_security_client(self, provider: Union[CloudProvider, str] = None):
        """
        Get a client for cloud security services.
        
        Args:
            provider: Cloud provider to use (defaults to configured provider)
            
        Returns:
            Security client for the specified provider
            
        Raises:
            CloudConfigurationError: If the provider is not supported or not configured
        """
        try:
            if provider is None:
                provider = self.config.get_current_provider()
            elif isinstance(provider, str):
                try:
                    provider = CloudProvider(provider.lower())
                except ValueError:
                    raise CloudConfigurationError(
                        f"Unknown provider: {provider}",
                        service="security"
                    )
            
            # Check if client already exists
            client_key = f"security_{provider.value}"
            if client_key in self.clients:
                return self.clients[client_key]
            
            # Check if provider is configured
            if not self.config.is_configured(provider):
                raise CloudConfigurationError(
                    f"Provider {provider.value} is not properly configured",
                    provider=provider,
                    service="security"
                )
            
            # Create new client based on provider
            if provider == CloudProvider.AWS:
                from cloud.security.aws import AWSSecurityClient
                client = AWSSecurityClient(self.config)
            elif provider == CloudProvider.AZURE:
                from cloud.security.azure import AzureKeyVaultClient
                client = AzureKeyVaultClient(self.config)
            elif provider == CloudProvider.GCP:
                from cloud.security.gcp import GCPSecurityClient
                client = GCPSecurityClient(self.config)
            else:
                raise CloudConfigurationError(
                    f"Unsupported provider for security: {provider.value}",
                    provider=provider,
                    service="security"
                )
            
            # Cache and return client
            self.clients[client_key] = client
            return client
            
        except Exception as e:
            # Handle any unexpected errors
            if not isinstance(e, CloudError):
                cloud_error = handle_cloud_error(
                    e, 
                    provider or self.config.get_current_provider(),
                    "security",
                    "get_client"
                )
                cloud_error.log()
                raise cloud_error
            else:
                e.log()
                raise
    
    def get_serverless_client(self, provider: Union[CloudProvider, str] = None):
        """
        Get a client for cloud serverless/functions services.
        
        Args:
            provider: Cloud provider to use (defaults to configured provider)
            
        Returns:
            Serverless client for the specified provider
            
        Raises:
            CloudConfigurationError: If the provider is not supported or not configured
        """
        try:
            if provider is None:
                provider = self.config.get_current_provider()
            elif isinstance(provider, str):
                try:
                    provider = CloudProvider(provider.lower())
                except ValueError:
                    raise CloudConfigurationError(
                        f"Unknown provider: {provider}",
                        service="serverless"
                    )
            
            # Check if client already exists
            client_key = f"serverless_{provider.value}"
            if client_key in self.clients:
                return self.clients[client_key]
            
            # Check if provider is configured
            if not self.config.is_configured(provider):
                raise CloudConfigurationError(
                    f"Provider {provider.value} is not properly configured",
                    provider=provider,
                    service="serverless"
                )
            
            # Create new client based on provider
            if provider == CloudProvider.AWS:
                from cloud.serverless.aws import AWSLambdaClient
                client = AWSLambdaClient(self.config)
            elif provider == CloudProvider.AZURE:
                from cloud.serverless.azure import AzureFunctionsClient
                client = AzureFunctionsClient(self.config)
            elif provider == CloudProvider.GCP:
                from cloud.serverless.gcp import GCPFunctionsClient
                client = GCPFunctionsClient(self.config)
            else:
                raise CloudConfigurationError(
                    f"Unsupported provider for serverless: {provider.value}",
                    provider=provider,
                    service="serverless"
                )
            
            # Cache and return client
            self.clients[client_key] = client
            return client
            
        except Exception as e:
            # Handle any unexpected errors
            if not isinstance(e, CloudError):
                cloud_error = handle_cloud_error(
                    e, 
                    provider or self.config.get_current_provider(),
                    "serverless",
                    "get_client"
                )
                cloud_error.log()
                raise cloud_error
            else:
                e.log()
                raise
    
    def get_monitoring_client(self, provider: Union[CloudProvider, str] = None):
        """
        Get a client for cloud monitoring/logging services.
        
        Args:
            provider: Cloud provider to use (defaults to configured provider)
            
        Returns:
            Monitoring client for the specified provider
            
        Raises:
            CloudConfigurationError: If the provider is not supported or not configured
        """
        try:
            if provider is None:
                provider = self.config.get_current_provider()
            elif isinstance(provider, str):
                try:
                    provider = CloudProvider(provider.lower())
                except ValueError:
                    raise CloudConfigurationError(
                        f"Unknown provider: {provider}",
                        service="monitoring"
                    )
            
            # Check if client already exists
            client_key = f"monitoring_{provider.value}"
            if client_key in self.clients:
                return self.clients[client_key]
            
            # Check if provider is configured
            if not self.config.is_configured(provider):
                raise CloudConfigurationError(
                    f"Provider {provider.value} is not properly configured",
                    provider=provider,
                    service="monitoring"
                )
            
            # Create new client based on provider
            if provider == CloudProvider.AWS:
                from cloud.monitoring.aws import AWSCloudWatchClient
                client = AWSCloudWatchClient(self.config)
            elif provider == CloudProvider.AZURE:
                from cloud.monitoring.azure import AzureMonitorClient
                client = AzureMonitorClient(self.config)
            elif provider == CloudProvider.GCP:
                from cloud.monitoring.gcp import GCPMonitoringClient
                client = GCPMonitoringClient(self.config)
            else:
                raise CloudConfigurationError(
                    f"Unsupported provider for monitoring: {provider.value}",
                    provider=provider,
                    service="monitoring"
                )
            
            # Cache and return client
            self.clients[client_key] = client
            return client
            
        except Exception as e:
            # Handle any unexpected errors
            if not isinstance(e, CloudError):
                cloud_error = handle_cloud_error(
                    e, 
                    provider or self.config.get_current_provider(),
                    "monitoring",
                    "get_client"
                )
                cloud_error.log()
                raise cloud_error
            else:
                e.log()
                raise 