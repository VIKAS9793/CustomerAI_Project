"""
CustomerAI Cloud Service Integrations.

This package provides integrations with major cloud service providers
including AWS, Azure, and Google Cloud Platform.
"""

from cloud.config import CloudConfig
from cloud.factory import CloudServiceFactory

__all__ = ['CloudConfig', 'CloudServiceFactory'] 