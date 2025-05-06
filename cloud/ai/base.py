"""
Base class for cloud AI clients.

This module defines the base interface for cloud AI services
across different providers.
"""

import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from cloud.config import CloudConfig

logger = logging.getLogger(__name__)

# Type variables for generics
T = TypeVar("T")


class ModelType(str, Enum):
    """Enum for different types of AI models."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATIVE = "generative"
    LANGUAGE = "language"
    VISION = "vision"
    MULTIMODAL = "multimodal"
    EMBEDDING = "embedding"
    REINFORCEMENT = "reinforcement"


class ModelFramework(str, Enum):
    """Enum for different ML frameworks."""

    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SCIKIT_LEARN = "scikit-learn"
    XGBOOST = "xgboost"
    HUGGINGFACE = "huggingface"
    JAX = "jax"
    ONNX = "onnx"
    CUSTOM = "custom"


class DeploymentStrategy(str, Enum):
    """Enum for different deployment strategies."""

    SERVERLESS = "serverless"
    DEDICATED = "dedicated"
    AUTO_SCALING = "auto-scaling"
    SPOT_INSTANCES = "spot-instances"
    EDGE = "edge"
    KUBERNETES = "kubernetes"


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
    def deploy_model(
        self,
        model_path: str,
        model_name: str,
        model_type: ModelType = None,
        framework: ModelFramework = None,
        strategy: DeploymentStrategy = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Deploy a model to the cloud provider.

        Args:
            model_path: Path to the model file
            model_name: Name for the deployed model
            model_type: Type of the model (classification, regression, etc.)
            framework: Framework used for the model
            strategy: Deployment strategy to use
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
    def create_training_job(
        self,
        training_data: str,
        model_name: str,
        hyperparameters: Dict[str, str],
        **kwargs,
    ) -> Dict[str, Any]:
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

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        model_name: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text using a generative AI model.

        Args:
            prompt: Input prompt for text generation
            model_name: Name of the generative model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (creativity control)
            **kwargs: Additional provider-specific arguments

        Returns:
            Dictionary with generated text and metadata
        """
        pass

    @abstractmethod
    def create_fine_tuning_job(
        self,
        base_model: str,
        training_data: str,
        validation_data: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a fine-tuning job for a pre-trained model.

        Args:
            base_model: Base model to fine-tune
            training_data: Path to training data
            validation_data: Path to validation data (optional)
            **kwargs: Additional provider-specific arguments

        Returns:
            Dictionary with fine-tuning job information
        """
        pass

    @abstractmethod
    def batch_transform(
        self, data_path: str, model_name: str, output_path: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Run batch inference on a dataset.

        Args:
            data_path: Path to input data
            model_name: Name of the deployed model
            output_path: Path for output predictions
            **kwargs: Additional provider-specific arguments

        Returns:
            Dictionary with batch transform job information
        """
        pass

    @abstractmethod
    def get_model_explanation(self, data: Any, model_name: str, **kwargs) -> Dict[str, Any]:
        """
        Get model explanations for predictions.

        Args:
            data: Input data for explanation
            model_name: Name of the deployed model
            **kwargs: Additional provider-specific arguments

        Returns:
            Dictionary with model explanations
        """
        pass

    @abstractmethod
    def create_distributed_training_job(
        self,
        training_data: str,
        model_name: str,
        hyperparameters: Dict[str, str],
        instance_count: int = 2,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a distributed training job.

        Args:
            training_data: Path to training data
            model_name: Name for the resulting model
            hyperparameters: Training hyperparameters
            instance_count: Number of instances for distributed training
            **kwargs: Additional provider-specific arguments

        Returns:
            Dictionary with training job information
        """
        pass

    @abstractmethod
    def create_hyperparameter_tuning_job(
        self,
        training_data: str,
        model_name: str,
        parameter_ranges: Dict[str, Dict[str, Any]],
        objective_metric: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a hyperparameter tuning job.

        Args:
            training_data: Path to training data
            model_name: Name for the resulting model
            parameter_ranges: Dictionary of parameter ranges for tuning
            objective_metric: Metric to optimize
            **kwargs: Additional provider-specific arguments

        Returns:
            Dictionary with hyperparameter tuning job information
        """
        pass

    def with_retry(
        self,
        operation: Callable,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        retry_on_exceptions: Tuple[Exception, ...] = (Exception,),
    ) -> Any:
        """
        Execute an operation with retry logic.

        Args:
            operation: The function to execute
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
            backoff_factor: Factor to increase delay between retries
            retry_on_exceptions: Exceptions that should trigger a retry

        Returns:
            Result of the operation

        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        current_delay = retry_delay

        for attempt in range(max_retries + 1):
            try:
                return operation()
            except retry_on_exceptions as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(
                        f"Operation failed on attempt {attempt + 1}/{max_retries + 1}: {str(e)}"
                    )
                    logger.info(f"Retrying in {current_delay:.2f} seconds...")
                    time.sleep(current_delay)
                    current_delay *= backoff_factor

        raise last_exception
