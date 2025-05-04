"""
LLM Manager

This module provides a flexible manager for LLM clients that allows developers
to easily configure and use different LLM providers based on their specific
requirements and use cases.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Any, Union, Type

from .llm_provider import (
    LLMProvider,
    LLMConfig,
    LLMComplianceLevel,
    LLMCapability,
    BaseLLMClient,
    LLMClientFactory
)

logger = logging.getLogger(__name__)

class LLMManager:
    """
    Manager for LLM clients that provides a flexible way to configure and use
    different LLM providers for different use cases.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize LLM Manager
        
        Args:
            config_path: Path to LLM configuration file (optional)
        """
        self.clients: Dict[str, BaseLLMClient] = {}
        self.default_client_id: Optional[str] = None
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> None:
        """
        Load LLM configurations from a JSON file
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Configure clients based on the config file
            for client_id, client_config in config_data.get("clients", {}).items():
                provider = LLMProvider(client_config.get("provider", "openai"))
                
                # Create LLM config
                llm_config = LLMConfig(
                    provider=provider,
                    api_key=client_config.get("api_key"),
                    api_endpoint=client_config.get("api_endpoint"),
                    model_name=client_config.get("model_name"),
                    compliance_level=LLMComplianceLevel(client_config.get("compliance_level", "standard")),
                    capabilities=[LLMCapability(cap) for cap in client_config.get("capabilities", ["text_generation"])],
                    timeout_seconds=client_config.get("timeout_seconds", 30),
                    max_retries=client_config.get("max_retries", 3),
                    additional_headers=client_config.get("additional_headers", {}),
                    additional_params=client_config.get("additional_params", {})
                )
                
                # Register the client
                self.register_client(client_id, llm_config)
            
            # Set default client if specified
            if "default_client" in config_data:
                self.default_client_id = config_data["default_client"]
                logger.info(f"Set default LLM client to: {self.default_client_id}")
                
        except Exception as e:
            logger.error(f"Error loading LLM configuration: {str(e)}")
    
    def register_client(self, client_id: str, config: Union[LLMConfig, Dict[str, Any]]) -> None:
        """
        Register a new LLM client
        
        Args:
            client_id: Unique identifier for the client
            config: LLM configuration or dict with configuration parameters
        """
        # Convert dict to LLMConfig if needed
        if isinstance(config, dict):
            provider = LLMProvider(config.get("provider", "openai"))
            
            config = LLMConfig(
                provider=provider,
                api_key=config.get("api_key"),
                api_endpoint=config.get("api_endpoint"),
                model_name=config.get("model_name"),
                compliance_level=LLMComplianceLevel(config.get("compliance_level", "standard")),
                capabilities=[LLMCapability(cap) for cap in config.get("capabilities", ["text_generation"])],
                timeout_seconds=config.get("timeout_seconds", 30),
                max_retries=config.get("max_retries", 3),
                additional_headers=config.get("additional_headers", {}),
                additional_params=config.get("additional_params", {})
            )
        
        # Create and register the client
        client = LLMClientFactory.create_client(config)
        self.clients[client_id] = client
        
        # If this is the first client, set it as default
        if self.default_client_id is None:
            self.default_client_id = client_id
            logger.info(f"Set default LLM client to: {client_id}")
    
    def register_financial_client(
        self, 
        client_id: str, 
        provider: LLMProvider = LLMProvider.OPENAI,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> None:
        """
        Register a specialized LLM client for financial services
        
        Args:
            client_id: Unique identifier for the client
            provider: LLM provider
            api_key: API key for the provider
            model_name: Model name to use
        """
        client = LLMClientFactory.create_financial_client(
            provider=provider,
            api_key=api_key,
            model_name=model_name
        )
        
        self.clients[client_id] = client
        
        # If this is the first client, set it as default
        if self.default_client_id is None:
            self.default_client_id = client_id
            logger.info(f"Set default LLM client to: {client_id}")
    
    def get_client(self, client_id: Optional[str] = None) -> BaseLLMClient:
        """
        Get an LLM client by ID
        
        Args:
            client_id: Client ID or None to get the default client
        
        Returns:
            LLM client instance
        
        Raises:
            ValueError: If client_id is not registered or no default client is set
        """
        # Use default client if client_id is not provided
        if client_id is None:
            if self.default_client_id is None:
                raise ValueError("No default LLM client is set")
            client_id = self.default_client_id
        
        # Check if client exists
        if client_id not in self.clients:
            raise ValueError(f"LLM client '{client_id}' is not registered")
        
        return self.clients[client_id]
    
    def set_default_client(self, client_id: str) -> None:
        """
        Set the default LLM client
        
        Args:
            client_id: Client ID to set as default
            
        Raises:
            ValueError: If client_id is not registered
        """
        if client_id not in self.clients:
            raise ValueError(f"LLM client '{client_id}' is not registered")
        
        self.default_client_id = client_id
        logger.info(f"Set default LLM client to: {client_id}")
    
    def list_clients(self) -> List[str]:
        """
        List all registered client IDs
        
        Returns:
            List of client IDs
        """
        return list(self.clients.keys())
    
    def get_client_info(self, client_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about an LLM client
        
        Args:
            client_id: Client ID or None to get information about the default client
            
        Returns:
            Dictionary with client information
            
        Raises:
            ValueError: If client_id is not registered or no default client is set
        """
        client = self.get_client(client_id)
        
        return {
            "provider": client.config.provider.value,
            "model_name": client.config.model_name,
            "compliance_level": client.config.compliance_level.value,
            "capabilities": [cap.value for cap in client.config.capabilities],
            "is_default": (client_id or self.default_client_id) == self.default_client_id
        }
    
    async def generate_text(
        self,
        prompt: str,
        client_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using an LLM client
        
        Args:
            prompt: Input prompt for generation
            client_id: Client ID or None to use the default client
            **kwargs: Additional parameters for the generate_text method
            
        Returns:
            Generation result
        """
        client = self.get_client(client_id)
        return await client.generate_text(prompt=prompt, **kwargs)
    
    async def get_embeddings(
        self,
        texts: List[str],
        client_id: Optional[str] = None,
        **kwargs
    ) -> List[List[float]]:
        """
        Get embeddings for texts
        
        Args:
            texts: Input texts for embedding
            client_id: Client ID or None to use the default client
            **kwargs: Additional parameters for the get_embeddings method
            
        Returns:
            List of embeddings
        """
        client = self.get_client(client_id)
        return await client.get_embeddings(texts=texts, **kwargs)
    
    async def classify_text(
        self,
        text: str,
        categories: List[str],
        client_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Classify text into categories
        
        Args:
            text: Input text for classification
            categories: List of categories
            client_id: Client ID or None to use the default client
            **kwargs: Additional parameters for the classify_text method
            
        Returns:
            Classification result with confidence scores
        """
        client = self.get_client(client_id)
        return await client.classify_text(text=text, categories=categories, **kwargs)

# Create a global instance of LLM Manager
llm_manager = LLMManager()

def get_llm_manager() -> LLMManager:
    """Get the global LLM manager instance"""
    return llm_manager 