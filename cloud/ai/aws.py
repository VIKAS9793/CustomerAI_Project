"""
AWS SageMaker client implementation for CustomerAI platform.

This module provides integration with AWS SageMaker service.
"""

import os
import json
import logging
import boto3
import sagemaker
from botocore.exceptions import ClientError
from typing import Any, Dict, List, Optional, Union, BinaryIO
import numpy as np
import pandas as pd
import time
import uuid

from cloud.ai.base import CloudAIClient
from cloud.config import CloudConfig

logger = logging.getLogger(__name__)

class AWSSageMakerClient(CloudAIClient):
    """
    AWS SageMaker client implementation.
    
    This class provides methods for interacting with AWS SageMaker service
    for model training, deployment, and inference.
    """
    
    def __init__(self, config: CloudConfig):
        """
        Initialize the AWS SageMaker client.
        
        Args:
            config: Cloud configuration
        """
        super().__init__(config)
        
        # Get AWS configuration
        aws_config = config.get_config("aws")
        self.region = aws_config.get("region", "us-east-1")
        self.sagemaker_role = aws_config.get("sagemaker_role")
        self.sagemaker_endpoint = aws_config.get("sagemaker_endpoint")
        
        # Initialize SageMaker session
        try:
            self.boto_session = boto3.Session(
                region_name=self.region,
                aws_access_key_id=aws_config.get("access_key_id"),
                aws_secret_access_key=aws_config.get("secret_access_key")
            )
            
            self.sagemaker_client = self.boto_session.client('sagemaker')
            self.sagemaker_runtime = self.boto_session.client('sagemaker-runtime')
            
            # Create SageMaker session
            self.sagemaker_session = sagemaker.Session(
                boto_session=self.boto_session,
                sagemaker_client=self.sagemaker_client,
                sagemaker_runtime_client=self.sagemaker_runtime
            )
            
            # Default S3 bucket
            self.default_bucket = aws_config.get(
                "s3_bucket", 
                self.sagemaker_session.default_bucket()
            )
            
            logger.info(f"AWS SageMaker client initialized in {self.region}")
        except Exception as e:
            logger.error(f"Error initializing AWS SageMaker client: {e}")
            self.sagemaker_session = None
    
    def predict(self, data: Any, model_name: str, **kwargs) -> Dict[str, Any]:
        """
        Make a prediction using a deployed SageMaker endpoint.
        
        Args:
            data: Input data for prediction
            model_name: Name of the deployed model/endpoint
            **kwargs: Additional arguments
                content_type: Content type of the request
                accept_type: Content type of the response
                
        Returns:
            Dictionary with prediction results
        """
        if not self.sagemaker_session:
            return {"error": "SageMaker session not initialized"}
        
        try:
            # Get endpoint name
            endpoint_name = kwargs.get("endpoint_name", model_name)
            
            # Prepare data
            if isinstance(data, (pd.DataFrame, pd.Series)):
                if kwargs.get("content_type") == "application/json":
                    payload = data.to_json(orient="records")
                else:
                    payload = data.to_csv(index=False)
            elif isinstance(data, np.ndarray):
                payload = json.dumps(data.tolist())
            elif isinstance(data, (dict, list)):
                payload = json.dumps(data)
            elif isinstance(data, str):
                payload = data
            else:
                payload = str(data)
            
            # Set content and accept types
            content_type = kwargs.get("content_type", "application/json")
            accept_type = kwargs.get("accept_type", "application/json")
            
            # Make prediction
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType=content_type,
                Accept=accept_type,
                Body=payload
            )
            
            # Parse response
            result = response['Body'].read().decode('utf-8')
            
            if accept_type == "application/json":
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse response as JSON: {result}")
            
            return {
                "success": True,
                "prediction": result,
                "model": model_name
            }
            
        except ClientError as e:
            logger.error(f"Error making prediction with SageMaker: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def deploy_model(self, model_path: str, model_name: str, **kwargs) -> Dict[str, Any]:
        """
        Deploy a model to SageMaker.
        
        Args:
            model_path: Path to the model file or S3 URI
            model_name: Name for the deployed model
            **kwargs: Additional arguments
                instance_type: EC2 instance type for deployment
                instance_count: Number of instances
                framework: ML framework (pytorch, tensorflow, etc.)
                
        Returns:
            Dictionary with deployment result information
        """
        if not self.sagemaker_session:
            return {"success": False, "error": "SageMaker session not initialized"}
        
        try:
            # Get parameters
            instance_type = kwargs.get("instance_type", "ml.m5.large")
            instance_count = kwargs.get("instance_count", 1)
            framework = kwargs.get("framework", "pytorch").lower()
            framework_version = kwargs.get("framework_version", "1.13.1")
            
            # Create model path in S3 if local file
            if model_path.startswith("s3://"):
                model_data = model_path
            else:
                if os.path.exists(model_path):
                    # Upload to S3
                    model_data = self.sagemaker_session.upload_data(
                        path=model_path,
                        bucket=self.default_bucket,
                        key_prefix=f"models/{model_name}"
                    )
                else:
                    return {
                        "success": False,
                        "error": f"Model file not found: {model_path}"
                    }
            
            # Create model
            if framework == "pytorch":
                from sagemaker.pytorch import PyTorchModel
                model = PyTorchModel(
                    model_data=model_data,
                    role=self.sagemaker_role,
                    framework_version=framework_version,
                    py_version="py3",
                    entry_point=kwargs.get("entry_point", "inference.py"),
                    source_dir=kwargs.get("source_dir"),
                    name=model_name
                )
            elif framework == "tensorflow":
                from sagemaker.tensorflow import TensorFlowModel
                model = TensorFlowModel(
                    model_data=model_data,
                    role=self.sagemaker_role,
                    framework_version=framework_version,
                    py_version="py3",
                    entry_point=kwargs.get("entry_point", "inference.py"),
                    source_dir=kwargs.get("source_dir"),
                    name=model_name
                )
            elif framework == "sklearn":
                from sagemaker.sklearn import SKLearnModel
                model = SKLearnModel(
                    model_data=model_data,
                    role=self.sagemaker_role,
                    framework_version=kwargs.get("framework_version", "1.0-1"),
                    py_version="py3",
                    entry_point=kwargs.get("entry_point", "inference.py"),
                    source_dir=kwargs.get("source_dir"),
                    name=model_name
                )
            else:
                return {
                    "success": False,
                    "error": f"Unsupported framework: {framework}"
                }
            
            # Deploy model
            endpoint_name = kwargs.get("endpoint_name", model_name)
            
            # Check for existing endpoint
            try:
                self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                # Endpoint exists, update it
                logger.info(f"Updating existing endpoint: {endpoint_name}")
                predictor = model.deploy(
                    initial_instance_count=instance_count,
                    instance_type=instance_type,
                    endpoint_name=endpoint_name,
                    update_endpoint=True
                )
            except ClientError as e:
                if "ValidationException" in str(e):
                    # Endpoint doesn't exist, create it
                    logger.info(f"Creating new endpoint: {endpoint_name}")
                    predictor = model.deploy(
                        initial_instance_count=instance_count,
                        instance_type=instance_type,
                        endpoint_name=endpoint_name
                    )
                else:
                    raise
            
            return {
                "success": True,
                "model_name": model_name,
                "endpoint_name": endpoint_name,
                "instance_type": instance_type,
                "instance_count": instance_count,
                "framework": framework,
                "status": "Creating"
            }
            
        except Exception as e:
            logger.error(f"Error deploying model to SageMaker: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_models(self, **kwargs) -> List[Dict[str, Any]]:
        """
        List deployed SageMaker models.
        
        Args:
            **kwargs: Additional arguments
                max_results: Maximum number of results to return
                name_contains: Filter models by name substring
                
        Returns:
            List of model information dictionaries
        """
        if not self.sagemaker_session:
            return []
        
        try:
            # Prepare parameters
            params = {}
            
            max_results = kwargs.get("max_results")
            if max_results:
                params["MaxResults"] = max_results
            
            name_contains = kwargs.get("name_contains")
            if name_contains:
                params["NameContains"] = name_contains
            
            # List models
            response = self.sagemaker_client.list_models(**params)
            
            # Process results
            models = []
            for model in response.get("Models", []):
                models.append({
                    "name": model.get("ModelName"),
                    "arn": model.get("ModelArn"),
                    "creation_time": model.get("CreationTime"),
                    "status": "Deployed"
                })
            
            return models
            
        except Exception as e:
            logger.error(f"Error listing SageMaker models: {e}")
            return []
    
    def delete_model(self, model_name: str, **kwargs) -> Dict[str, Any]:
        """
        Delete a SageMaker model and its endpoint.
        
        Args:
            model_name: Name of the deployed model
            **kwargs: Additional arguments
                delete_endpoint: Whether to delete the endpoint
                
        Returns:
            Dictionary with deletion result information
        """
        if not self.sagemaker_session:
            return {"success": False, "error": "SageMaker session not initialized"}
        
        try:
            # Delete endpoint (if exists and requested)
            delete_endpoint = kwargs.get("delete_endpoint", True)
            endpoint_name = kwargs.get("endpoint_name", model_name)
            
            if delete_endpoint:
                try:
                    self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                    # Delete endpoint configuration first
                    try:
                        self.sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
                        logger.info(f"Deleted endpoint configuration: {endpoint_name}")
                    except ClientError as e:
                        if "ValidationException" not in str(e):
                            logger.warning(f"Error deleting endpoint configuration: {e}")
                    
                    # Delete endpoint
                    self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
                    logger.info(f"Deleted endpoint: {endpoint_name}")
                except ClientError as e:
                    if "ValidationException" not in str(e):
                        logger.warning(f"Error deleting endpoint: {e}")
            
            # Delete model
            self.sagemaker_client.delete_model(ModelName=model_name)
            
            return {
                "success": True,
                "model_name": model_name,
                "endpoint_deleted": delete_endpoint
            }
            
        except ClientError as e:
            logger.error(f"Error deleting SageMaker model: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_model_metrics(self, model_name: str, **kwargs) -> Dict[str, Any]:
        """
        Get metrics for a deployed SageMaker model.
        
        Args:
            model_name: Name of the deployed model
            **kwargs: Additional arguments
                start_time: Start time for metrics
                end_time: End time for metrics
                period: Period for metrics in seconds
                
        Returns:
            Dictionary with model metrics
        """
        if not self.sagemaker_session:
            return {"error": "SageMaker session not initialized"}
        
        try:
            # Get parameters
            endpoint_name = kwargs.get("endpoint_name", model_name)
            start_time = kwargs.get("start_time", time.time() - 3600)  # Default: last hour
            end_time = kwargs.get("end_time", time.time())
            period = kwargs.get("period", 60)  # Default: 1 minute
            
            # Initialize CloudWatch client
            cloudwatch = self.boto_session.client('cloudwatch')
            
            # Get metrics
            metrics = {}
            
            # Get invocation metrics
            invocation_metrics = cloudwatch.get_metric_data(
                MetricDataQueries=[
                    {
                        'Id': 'invocations',
                        'MetricStat': {
                            'Metric': {
                                'Namespace': 'AWS/SageMaker',
                                'MetricName': 'Invocations',
                                'Dimensions': [
                                    {
                                        'Name': 'EndpointName',
                                        'Value': endpoint_name
                                    }
                                ]
                            },
                            'Period': period,
                            'Stat': 'Sum'
                        },
                        'ReturnData': True
                    },
                    {
                        'Id': 'invocation_errors',
                        'MetricStat': {
                            'Metric': {
                                'Namespace': 'AWS/SageMaker',
                                'MetricName': 'ModelLatency',
                                'Dimensions': [
                                    {
                                        'Name': 'EndpointName',
                                        'Value': endpoint_name
                                    }
                                ]
                            },
                            'Period': period,
                            'Stat': 'Average'
                        },
                        'ReturnData': True
                    }
                ],
                StartTime=start_time,
                EndTime=end_time
            )
            
            # Process metrics
            metrics["invocations"] = {
                "timestamps": [t.timestamp() for t in invocation_metrics["MetricDataResults"][0]["Timestamps"]],
                "values": invocation_metrics["MetricDataResults"][0]["Values"]
            }
            
            metrics["model_latency"] = {
                "timestamps": [t.timestamp() for t in invocation_metrics["MetricDataResults"][1]["Timestamps"]],
                "values": invocation_metrics["MetricDataResults"][1]["Values"]
            }
            
            # Get endpoint description
            endpoint_desc = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            
            return {
                "success": True,
                "model_name": model_name,
                "endpoint_name": endpoint_name,
                "status": endpoint_desc.get("EndpointStatus"),
                "metrics": metrics
            }
            
        except ClientError as e:
            logger.error(f"Error getting SageMaker model metrics: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_model_status(self, model_name: str, **kwargs) -> Dict[str, Any]:
        """
        Get the status of a deployed SageMaker model.
        
        Args:
            model_name: Name of the deployed model
            **kwargs: Additional arguments
                
        Returns:
            Dictionary with model status information
        """
        if not self.sagemaker_session:
            return {"success": False, "error": "SageMaker session not initialized"}
        
        try:
            # Get endpoint name
            endpoint_name = kwargs.get("endpoint_name", model_name)
            
            # Get model description
            try:
                model_desc = self.sagemaker_client.describe_model(ModelName=model_name)
                model_exists = True
            except ClientError as e:
                if "ValidationException" in str(e):
                    model_exists = False
                    model_desc = {}
                else:
                    raise
            
            # Get endpoint description
            try:
                endpoint_desc = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                endpoint_exists = True
                status = endpoint_desc.get("EndpointStatus")
            except ClientError as e:
                if "ValidationException" in str(e):
                    endpoint_exists = False
                    endpoint_desc = {}
                    status = "NotExists"
                else:
                    raise
            
            return {
                "success": True,
                "model_name": model_name,
                "model_exists": model_exists,
                "endpoint_name": endpoint_name,
                "endpoint_exists": endpoint_exists,
                "status": status,
                "creation_time": endpoint_desc.get("CreationTime") if endpoint_exists else None,
                "last_modified_time": endpoint_desc.get("LastModifiedTime") if endpoint_exists else None,
                "monitoring_schedules": endpoint_desc.get("MonitoringSchedules", []) if endpoint_exists else []
            }
            
        except ClientError as e:
            logger.error(f"Error getting SageMaker model status: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_training_job(self, training_data: str, model_name: str, 
                          hyperparameters: Dict[str, str], **kwargs) -> Dict[str, Any]:
        """
        Create a SageMaker training job.
        
        Args:
            training_data: Path or S3 URI to training data
            model_name: Name for the resulting model
            hyperparameters: Training hyperparameters
            **kwargs: Additional arguments
                instance_type: EC2 instance type for training
                instance_count: Number of instances
                framework: ML framework (pytorch, tensorflow, etc.)
                
        Returns:
            Dictionary with training job information
        """
        if not self.sagemaker_session:
            return {"success": False, "error": "SageMaker session not initialized"}
        
        try:
            # Get parameters
            instance_type = kwargs.get("instance_type", "ml.m5.large")
            instance_count = kwargs.get("instance_count", 1)
            framework = kwargs.get("framework", "pytorch").lower()
            framework_version = kwargs.get("framework_version", "1.13.1")
            
            # Generate job name
            job_name = kwargs.get("job_name", f"{model_name}-{int(time.time())}")
            
            # Create data paths in S3 if local file/directory
            if training_data.startswith("s3://"):
                train_data = training_data
            else:
                if os.path.exists(training_data):
                    # Upload to S3
                    train_data = self.sagemaker_session.upload_data(
                        path=training_data,
                        bucket=self.default_bucket,
                        key_prefix=f"data/{model_name}/train"
                    )
                else:
                    return {
                        "success": False,
                        "error": f"Training data not found: {training_data}"
                    }
            
            # Validation data
            validation_data = kwargs.get("validation_data")
            if validation_data:
                if validation_data.startswith("s3://"):
                    validate_data = validation_data
                else:
                    if os.path.exists(validation_data):
                        # Upload to S3
                        validate_data = self.sagemaker_session.upload_data(
                            path=validation_data,
                            bucket=self.default_bucket,
                            key_prefix=f"data/{model_name}/validation"
                        )
                    else:
                        return {
                            "success": False,
                            "error": f"Validation data not found: {validation_data}"
                        }
            else:
                validate_data = None
            
            # Create estimator based on framework
            if framework == "pytorch":
                from sagemaker.pytorch import PyTorch
                estimator = PyTorch(
                    entry_point=kwargs.get("entry_point", "train.py"),
                    source_dir=kwargs.get("source_dir"),
                    role=self.sagemaker_role,
                    framework_version=framework_version,
                    py_version="py3",
                    instance_count=instance_count,
                    instance_type=instance_type,
                    hyperparameters=hyperparameters,
                    output_path=f"s3://{self.default_bucket}/models/{model_name}/output",
                    sagemaker_session=self.sagemaker_session
                )
            elif framework == "tensorflow":
                from sagemaker.tensorflow import TensorFlow
                estimator = TensorFlow(
                    entry_point=kwargs.get("entry_point", "train.py"),
                    source_dir=kwargs.get("source_dir"),
                    role=self.sagemaker_role,
                    framework_version=framework_version,
                    py_version="py3",
                    instance_count=instance_count,
                    instance_type=instance_type,
                    hyperparameters=hyperparameters,
                    output_path=f"s3://{self.default_bucket}/models/{model_name}/output",
                    sagemaker_session=self.sagemaker_session
                )
            elif framework == "sklearn":
                from sagemaker.sklearn import SKLearn
                estimator = SKLearn(
                    entry_point=kwargs.get("entry_point", "train.py"),
                    source_dir=kwargs.get("source_dir"),
                    role=self.sagemaker_role,
                    framework_version=kwargs.get("framework_version", "1.0-1"),
                    instance_count=instance_count,
                    instance_type=instance_type,
                    hyperparameters=hyperparameters,
                    output_path=f"s3://{self.default_bucket}/models/{model_name}/output",
                    sagemaker_session=self.sagemaker_session
                )
            else:
                return {
                    "success": False,
                    "error": f"Unsupported framework: {framework}"
                }
            
            # Start training job
            if validate_data:
                estimator.fit({
                    "train": train_data,
                    "validation": validate_data
                }, job_name=job_name)
            else:
                estimator.fit(train_data, job_name=job_name)
            
            return {
                "success": True,
                "job_name": job_name,
                "model_name": model_name,
                "framework": framework,
                "instance_type": instance_type,
                "instance_count": instance_count,
                "status": "InProgress"
            }
            
        except Exception as e:
            logger.error(f"Error creating SageMaker training job: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_training_job_status(self, job_id: str, **kwargs) -> Dict[str, Any]:
        """
        Get status of a SageMaker training job.
        
        Args:
            job_id: Training job identifier
            **kwargs: Additional arguments
                
        Returns:
            Dictionary with training job status
        """
        if not self.sagemaker_session:
            return {"success": False, "error": "SageMaker session not initialized"}
        
        try:
            # Get job description
            job_desc = self.sagemaker_client.describe_training_job(TrainingJobName=job_id)
            
            # Extract relevant information
            result = {
                "success": True,
                "job_name": job_id,
                "status": job_desc.get("TrainingJobStatus"),
                "creation_time": job_desc.get("CreationTime"),
                "start_time": job_desc.get("TrainingStartTime"),
                "end_time": job_desc.get("TrainingEndTime"),
                "instance_type": job_desc.get("ResourceConfig", {}).get("InstanceType"),
                "instance_count": job_desc.get("ResourceConfig", {}).get("InstanceCount"),
                "model_artifacts": job_desc.get("ModelArtifacts", {}).get("S3ModelArtifacts")
            }
            
            # Add secondary status and failure reason if available
            secondary_status = job_desc.get("SecondaryStatus")
            if secondary_status:
                result["secondary_status"] = secondary_status
            
            failure_reason = job_desc.get("FailureReason")
            if failure_reason:
                result["failure_reason"] = failure_reason
            
            return result
            
        except ClientError as e:
            logger.error(f"Error getting SageMaker training job status: {e}")
            return {
                "success": False,
                "error": str(e)
            } 