"""
Base class for cloud AI clients.

This module defines the base interface for cloud AI services
across different providers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, BinaryIO

from cloud.config import CloudConfig

logger = logging.getLogger(__name__)

class CloudAIClient(ABC):
    """
    Abstract base class for cloud AI/ML clients.
    
    This class defines the common interface for interacting with
    cloud AI/ML services like AWS SageMaker, Azure ML, and GCP Vertex AI.
    """
    
    def __init__(self, config: CloudConfig):
        """
        Initialize the AI client.
        
        Args:
            config: Cloud configuration
        """
        self.config = config
    
    @abstractmethod
    def predict(self, data: Any, model_name: str, **kwargs) -> Dict[str, Any]:
        """
        Make a prediction using a deployed model.
        
        Args:
            data: Input data for prediction
            model_name: Name of the deployed model
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Dictionary with prediction results
        """
        pass
    
    @abstractmethod
    def deploy_model(self, model_path: str, model_name: str, **kwargs) -> Dict[str, Any]:
        """
        Deploy a model to the cloud provider.
        
        Args:
            model_path: Path to the model file
            model_name: Name for the deployed model
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Dictionary with deployment result information
        """
        pass
    
    @abstractmethod
    def list_models(self, **kwargs) -> List[Dict[str, Any]]:
        """
        List deployed models.
        
        Args:
            **kwargs: Additional provider-specific arguments
            
        Returns:
            List of model information dictionaries
        """
        pass
    
    @abstractmethod
    def delete_model(self, model_name: str, **kwargs) -> Dict[str, Any]:
        """
        Delete a deployed model.
        
        Args:
            model_name: Name of the deployed model
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Dictionary with deletion result information
        """
        pass
    
    @abstractmethod
    def get_model_metrics(self, model_name: str, **kwargs) -> Dict[str, Any]:
        """
        Get metrics for a deployed model.
        
        Args:
            model_name: Name of the deployed model
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Dictionary with model metrics
        """
        pass
    
    @abstractmethod
    def get_model_status(self, model_name: str, **kwargs) -> Dict[str, Any]:
        """
        Get the status of a deployed model.
        
        Args:
            model_name: Name of the deployed model
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Dictionary with model status information
        """
        pass
    
    @abstractmethod
    def create_training_job(self, training_data: str, model_name: str, 
                          hyperparameters: Dict[str, str], **kwargs) -> Dict[str, Any]:
        """
        Create a training job.
        
        Args:
            training_data: Path to training data
            model_name: Name for the resulting model
            hyperparameters: Training hyperparameters
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Dictionary with training job information
        """
        pass
    
    @abstractmethod
    def get_training_job_status(self, job_id: str, **kwargs) -> Dict[str, Any]:
        """
        Get status of a training job.
        
        Args:
            job_id: Training job identifier
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Dictionary with training job status
        """
        pass 