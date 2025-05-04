"""
Terraform integration for cloud infrastructure deployment.

This module provides tools for automating cloud infrastructure
deployment using Terraform.

Copyright (c) 2025 Vikas Sahani
GitHub: https://github.com/VIKAS9793
Email: vikassahani17@gmail.com

Licensed under MIT License - see LICENSE file for details
This copyright and license applies only to the original code in this file,
not to any third-party libraries or dependencies used.
"""

import os
import logging
import subprocess
import json
import tempfile
import shutil
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar
from enum import Enum
import threading
import time

from cloud.config import CloudProvider
from cloud.errors import CloudDeploymentError

logger = logging.getLogger(__name__)

# Type variable for function return type
T = TypeVar('T')

class TerraformCommand(Enum):
    """Terraform commands."""
    INIT = "init"
    PLAN = "plan"
    APPLY = "apply"
    DESTROY = "destroy"
    VALIDATE = "validate"
    OUTPUT = "output"
    STATE = "state"


class DeploymentStatus(Enum):
    """Deployment status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DESTROYED = "destroyed"


class TerraformDeployer:
    """
    Terraform-based infrastructure deployer.
    
    This class provides methods for deploying and managing
    cloud infrastructure using Terraform.
    """
    
    def __init__(
        self,
        working_directory: str,
        terraform_binary: str = "terraform",
        auto_approve: bool = False,
        variables: Optional[Dict[str, Any]] = None,
        var_files: Optional[List[str]] = None
    ):
        """
        Initialize Terraform deployer.
        
        Args:
            working_directory: Working directory for Terraform
            terraform_binary: Path to Terraform binary
            auto_approve: Whether to auto-approve Terraform commands
            variables: Variables to pass to Terraform
            var_files: Variable files to pass to Terraform
        """
        self.working_directory = working_directory
        self.terraform_binary = terraform_binary
        self.auto_approve = auto_approve
        self.variables = variables or {}
        self.var_files = var_files or []
        
        # Create working directory if it doesn't exist
        os.makedirs(working_directory, exist_ok=True)
        
        # Deployment status
        self.status = DeploymentStatus.PENDING
        self.last_output = ""
        self.last_error = ""
        
        # Thread safety
        self.lock = threading.RLock()
    
    def _run_command(
        self,
        command: TerraformCommand,
        args: Optional[List[str]] = None,
        capture_output: bool = True,
        check: bool = True,
        env: Optional[Dict[str, str]] = None
    ) -> subprocess.CompletedProcess:
        """
        Run a Terraform command.
        
        Args:
            command: Terraform command
            args: Additional arguments
            capture_output: Whether to capture command output
            check: Whether to check return code
            env: Environment variables
            
        Returns:
            Completed process
            
        Raises:
            CloudDeploymentError: If command fails
        """
        # Build command
        cmd = [self.terraform_binary, command.value]
        
        # Add auto-approve for apply and destroy
        if command in [TerraformCommand.APPLY, TerraformCommand.DESTROY] and self.auto_approve:
            cmd.append("-auto-approve")
        
        # Add variables
        for key, value in self.variables.items():
            if isinstance(value, str):
                cmd.extend(["-var", f"{key}={value}"])
            else:
                # Convert non-string values to JSON
                cmd.extend(["-var", f"{key}={json.dumps(value)}"])
        
        # Add variable files
        for var_file in self.var_files:
            cmd.extend(["-var-file", var_file])
        
        # Add additional arguments
        if args:
            cmd.extend(args)
        
        # Set environment
        command_env = os.environ.copy()
        if env:
            command_env.update(env)
        
        # Run command
        logger.info(f"Running Terraform command: {' '.join(cmd)}")
        
        try:
            # Run process
            process = subprocess.run(
                cmd,
                cwd=self.working_directory,
                capture_output=capture_output,
                text=True,
                check=check,
                env=command_env
            )
            
            # Log output
            if capture_output:
                if process.stdout:
                    logger.debug(f"Terraform output: {process.stdout}")
                
                if process.stderr:
                    logger.warning(f"Terraform stderr: {process.stderr}")
            
            return process
            
        except subprocess.CalledProcessError as e:
            # Log error
            if capture_output:
                if e.stdout:
                    logger.debug(f"Terraform output: {e.stdout}")
                
                if e.stderr:
                    logger.error(f"Terraform error: {e.stderr}")
            
            # Store error
            self.last_error = e.stderr if e.stderr else str(e)
            
            raise CloudDeploymentError(f"Terraform command failed: {e}")
    
    def init(
        self,
        backend_config: Optional[Dict[str, str]] = None,
        upgrade: bool = False,
        reconfigure: bool = False
    ) -> None:
        """
        Initialize Terraform.
        
        Args:
            backend_config: Backend configuration
            upgrade: Whether to upgrade modules
            reconfigure: Whether to reconfigure backend
            
        Raises:
            CloudDeploymentError: If initialization fails
        """
        args = []
        
        # Add backend configuration
        if backend_config:
            for key, value in backend_config.items():
                args.extend(["-backend-config", f"{key}={value}"])
        
        # Add upgrade flag
        if upgrade:
            args.append("-upgrade")
        
        # Add reconfigure flag
        if reconfigure:
            args.append("-reconfigure")
        
        # Run command
        with self.lock:
            self.status = DeploymentStatus.RUNNING
            
            try:
                process = self._run_command(TerraformCommand.INIT, args)
                self.last_output = process.stdout
                self.status = DeploymentStatus.COMPLETED
            except Exception as e:
                self.status = DeploymentStatus.FAILED
                raise
    
    def plan(
        self,
        out_file: Optional[str] = None,
        detailed_exitcode: bool = False
    ) -> Union[str, int]:
        """
        Create a Terraform plan.
        
        Args:
            out_file: File to save plan to
            detailed_exitcode: Whether to return detailed exit code
            
        Returns:
            Plan output or exit code if detailed_exitcode is True
            
        Raises:
            CloudDeploymentError: If plan fails
        """
        args = []
        
        # Add output file
        if out_file:
            args.extend(["-out", out_file])
        
        # Add detailed exit code
        if detailed_exitcode:
            args.append("-detailed-exitcode")
        
        # Run command
        with self.lock:
            self.status = DeploymentStatus.RUNNING
            
            try:
                process = self._run_command(TerraformCommand.PLAN, args, check=not detailed_exitcode)
                self.last_output = process.stdout
                self.status = DeploymentStatus.COMPLETED
                
                return process.returncode if detailed_exitcode else process.stdout
            except Exception as e:
                self.status = DeploymentStatus.FAILED
                raise
    
    def apply(
        self,
        plan_file: Optional[str] = None,
        target: Optional[List[str]] = None
    ) -> str:
        """
        Apply Terraform changes.
        
        Args:
            plan_file: Plan file to apply
            target: Resources to target
            
        Returns:
            Apply output
            
        Raises:
            CloudDeploymentError: If apply fails
        """
        args = []
        
        # Add plan file
        if plan_file:
            args.append(plan_file)
        
        # Add targets
        if target:
            for resource in target:
                args.extend(["-target", resource])
        
        # Run command
        with self.lock:
            self.status = DeploymentStatus.RUNNING
            
            try:
                process = self._run_command(TerraformCommand.APPLY, args)
                self.last_output = process.stdout
                self.status = DeploymentStatus.COMPLETED
                
                return process.stdout
            except Exception as e:
                self.status = DeploymentStatus.FAILED
                raise
    
    def destroy(
        self,
        target: Optional[List[str]] = None
    ) -> str:
        """
        Destroy Terraform resources.
        
        Args:
            target: Resources to target
            
        Returns:
            Destroy output
            
        Raises:
            CloudDeploymentError: If destroy fails
        """
        args = []
        
        # Add targets
        if target:
            for resource in target:
                args.extend(["-target", resource])
        
        # Run command
        with self.lock:
            self.status = DeploymentStatus.RUNNING
            
            try:
                process = self._run_command(TerraformCommand.DESTROY, args)
                self.last_output = process.stdout
                self.status = DeploymentStatus.DESTROYED
                
                return process.stdout
            except Exception as e:
                self.status = DeploymentStatus.FAILED
                raise
    
    def validate(self) -> str:
        """
        Validate Terraform configuration.
        
        Returns:
            Validation output
            
        Raises:
            CloudDeploymentError: If validation fails
        """
        # Run command
        with self.lock:
            self.status = DeploymentStatus.RUNNING
            
            try:
                process = self._run_command(TerraformCommand.VALIDATE)
                self.last_output = process.stdout
                self.status = DeploymentStatus.COMPLETED
                
                return process.stdout
            except Exception as e:
                self.status = DeploymentStatus.FAILED
                raise
    
    def output(
        self,
        name: Optional[str] = None,
        json_format: bool = True
    ) -> Union[str, Dict[str, Any]]:
        """
        Get Terraform outputs.
        
        Args:
            name: Output name to get
            json_format: Whether to return output as JSON
            
        Returns:
            Output as string or dictionary
            
        Raises:
            CloudDeploymentError: If output fails
        """
        args = []
        
        # Add output name
        if name:
            args.append(name)
        
        # Add JSON format
        if json_format:
            args.append("-json")
        
        # Run command
        with self.lock:
            try:
                process = self._run_command(TerraformCommand.OUTPUT, args)
                
                if json_format:
                    try:
                        return json.loads(process.stdout)
                    except json.JSONDecodeError:
                        return {}
                else:
                    return process.stdout
            except Exception as e:
                raise
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get Terraform state.
        
        Returns:
            State as dictionary
            
        Raises:
            CloudDeploymentError: If state retrieval fails
        """
        # Run command
        with self.lock:
            try:
                process = self._run_command(TerraformCommand.STATE, ["list", "-json"])
                
                try:
                    return json.loads(process.stdout)
                except json.JSONDecodeError:
                    return {}
            except Exception as e:
                raise
    
    def get_status(self) -> DeploymentStatus:
        """
        Get deployment status.
        
        Returns:
            Deployment status
        """
        with self.lock:
            return self.status
    
    def get_last_output(self) -> str:
        """
        Get last command output.
        
        Returns:
            Last command output
        """
        with self.lock:
            return self.last_output
    
    def get_last_error(self) -> str:
        """
        Get last command error.
        
        Returns:
            Last command error
        """
        with self.lock:
            return self.last_error


class CloudTemplateManager:
    """
    Manager for cloud infrastructure templates.
    
    This class provides methods for generating and managing
    Terraform templates for cloud infrastructure.
    """
    
    def __init__(
        self,
        template_directory: str,
        output_directory: str
    ):
        """
        Initialize template manager.
        
        Args:
            template_directory: Directory containing templates
            output_directory: Directory to write generated templates to
        """
        self.template_directory = template_directory
        self.output_directory = output_directory
        
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        
        # Thread safety
        self.lock = threading.RLock()
    
    def generate_template(
        self,
        provider: Union[CloudProvider, str],
        template_name: str,
        variables: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a Terraform template.
        
        Args:
            provider: Cloud provider
            template_name: Template name
            variables: Template variables
            output_path: Path to write template to
            
        Returns:
            Path to generated template
            
        Raises:
            FileNotFoundError: If template not found
            CloudDeploymentError: If template generation fails
        """
        # Normalize provider
        if isinstance(provider, CloudProvider):
            provider_str = provider.value
        else:
            provider_str = str(provider).lower()
        
        # Compute paths
        template_path = os.path.join(self.template_directory, provider_str, f"{template_name}.tf")
        
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        # Compute output path
        if output_path is None:
            output_name = f"{provider_str}-{template_name}-{int(time.time())}"
            output_path = os.path.join(self.output_directory, output_name)
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Copy template to output directory
        with self.lock:
            try:
                # Copy template
                shutil.copy2(template_path, os.path.join(output_path, f"{template_name}.tf"))
                
                # Create variables file
                with open(os.path.join(output_path, "terraform.tfvars.json"), "w") as f:
                    json.dump(variables, f, indent=2)
                
                # Create backend configuration
                backend_config = variables.get("backend", {})
                with open(os.path.join(output_path, "backend.tf"), "w") as f:
                    if backend_config:
                        f.write("""
terraform {
  backend "s3" {}
}
""")
                
                return output_path
                
            except Exception as e:
                raise CloudDeploymentError(f"Failed to generate template: {str(e)}")
    
    def validate_variables(
        self,
        provider: Union[CloudProvider, str],
        template_name: str,
        variables: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Validate template variables.
        
        Args:
            provider: Cloud provider
            template_name: Template name
            variables: Template variables
            
        Returns:
            Dictionary of validation errors
            
        Raises:
            FileNotFoundError: If template not found
            CloudDeploymentError: If validation fails
        """
        # Normalize provider
        if isinstance(provider, CloudProvider):
            provider_str = provider.value
        else:
            provider_str = str(provider).lower()
        
        # Compute paths
        template_path = os.path.join(self.template_directory, provider_str, f"{template_name}.tf")
        
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        # Generate temporary template
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy template to temporary directory
            shutil.copy2(template_path, os.path.join(temp_dir, f"{template_name}.tf"))
            
            # Create variables file
            with open(os.path.join(temp_dir, "terraform.tfvars.json"), "w") as f:
                json.dump(variables, f, indent=2)
            
            # Create Terraform deployer
            deployer = TerraformDeployer(working_directory=temp_dir)
            
            try:
                # Initialize Terraform
                deployer.init()
                
                # Validate configuration
                deployer.validate()
                
                # No errors
                return {}
                
            except CloudDeploymentError as e:
                # Parse error message
                error_message = str(e)
                
                # Extract variable errors
                errors = {}
                
                for line in error_message.splitlines():
                    if "variable" in line.lower() and "missing" in line.lower():
                        parts = line.split()
                        if len(parts) >= 2:
                            var_name = parts[1].strip('"')
                            if var_name not in errors:
                                errors[var_name] = []
                            errors[var_name].append("Required variable missing")
                    
                    elif "variable" in line.lower() and "invalid" in line.lower():
                        parts = line.split()
                        if len(parts) >= 2:
                            var_name = parts[1].strip('"')
                            if var_name not in errors:
                                errors[var_name] = []
                            errors[var_name].append("Invalid variable value")
                
                return errors


def deploy_infrastructure(
    provider: Union[CloudProvider, str],
    template_name: str,
    variables: Dict[str, Any],
    backend_config: Optional[Dict[str, str]] = None,
    auto_approve: bool = False,
    template_directory: str = "templates",
    output_directory: str = "deployments"
) -> Dict[str, Any]:
    """
    Deploy cloud infrastructure.
    
    Args:
        provider: Cloud provider
        template_name: Template name
        variables: Template variables
        backend_config: Backend configuration
        auto_approve: Whether to auto-approve deployment
        template_directory: Directory containing templates
        output_directory: Directory to write generated templates to
        
    Returns:
        Dictionary of outputs
        
    Raises:
        FileNotFoundError: If template not found
        CloudDeploymentError: If deployment fails
    """
    # Create template manager
    template_manager = CloudTemplateManager(
        template_directory=template_directory,
        output_directory=output_directory
    )
    
    # Generate template
    template_path = template_manager.generate_template(
        provider=provider,
        template_name=template_name,
        variables=variables
    )
    
    # Create Terraform deployer
    deployer = TerraformDeployer(
        working_directory=template_path,
        auto_approve=auto_approve
    )
    
    try:
        # Initialize Terraform
        deployer.init(backend_config=backend_config)
        
        # Create plan
        plan_file = os.path.join(template_path, "terraform.plan")
        deployer.plan(out_file=plan_file)
        
        # Apply plan
        deployer.apply(plan_file=plan_file)
        
        # Get outputs
        outputs = deployer.output()
        
        return outputs
        
    except Exception as e:
        raise CloudDeploymentError(f"Deployment failed: {str(e)}")


def destroy_infrastructure(
    deployment_path: str,
    auto_approve: bool = False
) -> None:
    """
    Destroy cloud infrastructure.
    
    Args:
        deployment_path: Path to deployment
        auto_approve: Whether to auto-approve destruction
        
    Raises:
        CloudDeploymentError: If destruction fails
    """
    # Create Terraform deployer
    deployer = TerraformDeployer(
        working_directory=deployment_path,
        auto_approve=auto_approve
    )
    
    try:
        # Initialize Terraform
        deployer.init()
        
        # Destroy infrastructure
        deployer.destroy()
        
    except Exception as e:
        raise CloudDeploymentError(f"Destruction failed: {str(e)}")


def get_deployment_outputs(
    deployment_path: str
) -> Dict[str, Any]:
    """
    Get outputs from a deployment.
    
    Args:
        deployment_path: Path to deployment
        
    Returns:
        Dictionary of outputs
        
    Raises:
        CloudDeploymentError: If output retrieval fails
    """
    # Create Terraform deployer
    deployer = TerraformDeployer(
        working_directory=deployment_path
    )
    
    try:
        # Initialize Terraform
        deployer.init()
        
        # Get outputs
        outputs = deployer.output()
        
        return outputs
        
    except Exception as e:
        raise CloudDeploymentError(f"Failed to get outputs: {str(e)}")


def create_aws_s3_backend(
    bucket_name: str,
    region: str = "us-east-1",
    key_prefix: str = "terraform/state/",
    dynamodb_table: Optional[str] = None
) -> Dict[str, str]:
    """
    Create AWS S3 backend configuration.
    
    Args:
        bucket_name: S3 bucket name
        region: AWS region
        key_prefix: Key prefix for state files
        dynamodb_table: DynamoDB table for state locking
        
    Returns:
        Backend configuration
    """
    # Create backend configuration
    backend_config = {
        "bucket": bucket_name,
        "region": region,
        "key": f"{key_prefix}terraform.tfstate"
    }
    
    # Add DynamoDB table if provided
    if dynamodb_table:
        backend_config["dynamodb_table"] = dynamodb_table
    
    return backend_config


def create_azure_storage_backend(
    storage_account_name: str,
    container_name: str,
    key: str = "terraform.tfstate",
    resource_group_name: Optional[str] = None
) -> Dict[str, str]:
    """
    Create Azure Storage backend configuration.
    
    Args:
        storage_account_name: Storage account name
        container_name: Container name
        key: State file key
        resource_group_name: Resource group name
        
    Returns:
        Backend configuration
    """
    # Create backend configuration
    backend_config = {
        "storage_account_name": storage_account_name,
        "container_name": container_name,
        "key": key
    }
    
    # Add resource group if provided
    if resource_group_name:
        backend_config["resource_group_name"] = resource_group_name
    
    return backend_config


def create_gcp_storage_backend(
    bucket: str,
    prefix: str = "terraform/state",
    credentials: Optional[str] = None
) -> Dict[str, str]:
    """
    Create GCP Storage backend configuration.
    
    Args:
        bucket: GCS bucket name
        prefix: Prefix for state files
        credentials: Path to credentials file
        
    Returns:
        Backend configuration
    """
    # Create backend configuration
    backend_config = {
        "bucket": bucket,
        "prefix": prefix
    }
    
    # Add credentials if provided
    if credentials:
        backend_config["credentials"] = credentials
    
    return backend_config 