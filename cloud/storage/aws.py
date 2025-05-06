"""
AWS S3 client implementation for CustomerAI platform.

This module provides integration with AWS S3 storage service.
"""

import logging
import os
from typing import Any, Dict, List

import boto3
from botocore.exceptions import ClientError

from cloud.config import CloudConfig, CloudProvider
from cloud.errors import CloudResourceNotFoundError, handle_aws_error
from cloud.storage.base import CloudStorageClient
from cloud.utils.metrics import track_performance
from cloud.utils.retry import retry, retry_aggressive, retry_default

logger = logging.getLogger(__name__)


class AWSS3Client(CloudStorageClient):
    """
    AWS S3 storage client implementation.

    This class provides methods for interacting with AWS S3 storage service.
    """

    def __init__(self, config: CloudConfig):
        """
        Initialize the AWS S3 client.

        Args:
            config: Cloud configuration
        """
        super().__init__(config)

        # Get AWS configuration
        aws_config = config.get_config("aws")
        self.region = aws_config.get("region", "us-east-1")
        self.bucket_name = aws_config.get("s3_bucket", "customerai-data")

        # Initialize S3 client
        self.s3 = boto3.client(
            "s3",
            region_name=self.region,
            aws_access_key_id=aws_config.get("access_key_id"),
            aws_secret_access_key=aws_config.get("secret_access_key"),
        )

        # Initialize S3 resource (for higher-level operations)
        self.s3_resource = boto3.resource(
            "s3",
            region_name=self.region,
            aws_access_key_id=aws_config.get("access_key_id"),
            aws_secret_access_key=aws_config.get("secret_access_key"),
        )

        # Create bucket if it doesn't exist
        self._ensure_bucket_exists()

    @retry_default
    @track_performance(CloudProvider.AWS, "s3", "ensure_bucket_exists")
    def _ensure_bucket_exists(self) -> None:
        """Ensure the S3 bucket exists."""
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            # Convert to standardized error
            cloud_error = handle_aws_error(e, "s3", "head_bucket")

            if isinstance(cloud_error, CloudResourceNotFoundError):
                # Bucket doesn't exist, create it
                logger.info(f"Creating S3 bucket: {self.bucket_name}")
                try:
                    if self.region == "us-east-1":
                        # US East 1 requires special handling
                        self.s3.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={"LocationConstraint": self.region},
                        )
                except ClientError as create_error:
                    # Convert to standardized error and log
                    create_cloud_error = handle_aws_error(create_error, "s3", "create_bucket")
                    create_cloud_error.log()
                    raise create_cloud_error
            else:
                # Log the error
                cloud_error.log()
                raise cloud_error

    @retry(max_retries=3, strategy="exponential")
    @track_performance(CloudProvider.AWS, "s3", "upload_file")
    def upload_file(self, local_path: str, remote_path: str, **kwargs) -> Dict[str, Any]:
        """
        Upload a file to S3.

        Args:
            local_path: Path to local file
            remote_path: Path in S3 bucket
            **kwargs: Additional arguments
                public: Whether to make the file publicly accessible
                metadata: Custom metadata to attach to the file

        Returns:
            Dictionary with upload result information
        """
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")

        try:
            # Prepare extra args
            extra_args = {}

            # Set content type
            content_type = self.get_content_type(local_path)
            if content_type:
                extra_args["ContentType"] = content_type

            # Set ACL if public
            if kwargs.get("public"):
                extra_args["ACL"] = "public-read"

            # Add custom metadata if provided
            metadata = kwargs.get("metadata")
            if metadata and isinstance(metadata, dict):
                extra_args["Metadata"] = metadata

            # Use any additional arguments for S3
            s3_args = kwargs.get("s3_args", {})
            if s3_args:
                extra_args.update(s3_args)

            # Upload the file
            self.s3.upload_file(local_path, self.bucket_name, remote_path, ExtraArgs=extra_args)

            # Generate URL
            url = self.get_file_url(remote_path, **kwargs)

            return {
                "success": True,
                "bucket": self.bucket_name,
                "key": remote_path,
                "url": url,
                "content_type": content_type,
            }

        except ClientError as e:
            # Convert to standardized error and log
            cloud_error = handle_aws_error(e, "s3", "upload_file")
            # Note: We don't log here since retry decorator will handle it

            # Return error response
            return {
                "success": False,
                "error": cloud_error.message,
                "error_details": cloud_error.to_dict(),
            }

    @retry_default
    @track_performance(CloudProvider.AWS, "s3", "download_file")
    def download_file(self, remote_path: str, local_path: str, **kwargs) -> Dict[str, Any]:
        """
        Download a file from S3.

        Args:
            remote_path: Path in S3 bucket
            local_path: Path to save local file
            **kwargs: Additional arguments

        Returns:
            Dictionary with download result information
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)

            # Download the file
            self.s3.download_file(self.bucket_name, remote_path, local_path)

            return {
                "success": True,
                "bucket": self.bucket_name,
                "key": remote_path,
                "local_path": local_path,
                "size": os.path.getsize(local_path),
            }

        except ClientError as e:
            # Convert to standardized error and log
            cloud_error = handle_aws_error(e, "s3", "download_file")
            # Note: We don't log here since retry decorator will handle it

            # Return error response
            return {
                "success": False,
                "error": cloud_error.message,
                "error_details": cloud_error.to_dict(),
                "exists": not isinstance(cloud_error, CloudResourceNotFoundError),
            }

    @retry(max_retries=3, strategy="linear")
    @track_performance(CloudProvider.AWS, "s3", "list_files")
    def list_files(self, prefix: str = "", **kwargs) -> List[Dict[str, Any]]:
        """
        List files in S3 bucket.

        Args:
            prefix: Path prefix to filter files
            **kwargs: Additional arguments
                max_keys: Maximum number of keys to return

        Returns:
            List of file information dictionaries
        """
        try:
            # Prepare parameters
            params = {"Bucket": self.bucket_name}

            if prefix:
                params["Prefix"] = prefix

            max_keys = kwargs.get("max_keys")
            if max_keys:
                params["MaxKeys"] = max_keys

            # List objects
            response = self.s3.list_objects_v2(**params)

            # Process results
            files = []
            for item in response.get("Contents", []):
                files.append(
                    {
                        "key": item.get("Key"),
                        "size": item.get("Size"),
                        "last_modified": item.get("LastModified"),
                        "etag": item.get("ETag", "").strip('"'),
                        "storage_class": item.get("StorageClass"),
                    }
                )

            return files

        except ClientError as e:
            # Convert to standardized error and log
            handle_aws_error(e, "s3", "list_files")
            # Note: We don't log here since retry decorator will handle it

            # Return empty list on error
            return []

    @retry_default
    @track_performance(CloudProvider.AWS, "s3", "delete_file")
    def delete_file(self, remote_path: str, **kwargs) -> Dict[str, Any]:
        """
        Delete a file from S3.

        Args:
            remote_path: Path in S3 bucket
            **kwargs: Additional arguments

        Returns:
            Dictionary with deletion result information
        """
        try:
            # Delete the object
            self.s3.delete_object(Bucket=self.bucket_name, Key=remote_path)

            return {"success": True, "bucket": self.bucket_name, "key": remote_path}

        except ClientError as e:
            # Convert to standardized error and log
            cloud_error = handle_aws_error(e, "s3", "delete_file")
            # Note: We don't log here since retry decorator will handle it

            # Return error response
            return {
                "success": False,
                "error": cloud_error.message,
                "error_details": cloud_error.to_dict(),
            }

    @retry_default
    @track_performance(CloudProvider.AWS, "s3", "file_exists")
    def file_exists(self, remote_path: str, **kwargs) -> bool:
        """
        Check if a file exists in S3.

        Args:
            remote_path: Path in S3 bucket
            **kwargs: Additional arguments

        Returns:
            True if file exists, False otherwise
        """
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=remote_path)
            return True
        except ClientError as e:
            # Convert to standardized error
            cloud_error = handle_aws_error(e, "s3", "head_object")

            # Don't raise exception for not found errors
            if isinstance(cloud_error, CloudResourceNotFoundError):
                return False

            # For other errors, propagate through the retry mechanism
            raise cloud_error

    @retry_default
    @track_performance(CloudProvider.AWS, "s3", "get_file_url")
    def get_file_url(self, remote_path: str, expiration: int = 3600, **kwargs) -> str:
        """
        Get a URL for accessing a file in S3.

        Args:
            remote_path: Path in S3 bucket
            expiration: URL expiration time in seconds (for presigned URLs)
            **kwargs: Additional arguments
                public: Whether to generate a public URL

        Returns:
            File URL
        """
        try:
            # If public URL is requested
            if kwargs.get("public"):
                return f"https://{self.bucket_name}.s3.amazonaws.com/{remote_path}"

            # Generate presigned URL
            url = self.s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": remote_path},
                ExpiresIn=expiration,
            )

            return url

        except ClientError as e:
            # Convert to standardized error and log
            cloud_error = handle_aws_error(e, "s3", "generate_presigned_url")
            # Note: We don't log here since retry decorator will handle it

            # Raise the error for retry mechanism
            raise cloud_error

    @retry_default
    @track_performance(CloudProvider.AWS, "s3", "get_file_metadata")
    def get_file_metadata(self, remote_path: str, **kwargs) -> Dict[str, Any]:
        """
        Get metadata for a file in S3.

        Args:
            remote_path: Path in S3 bucket
            **kwargs: Additional arguments

        Returns:
            Dictionary with file metadata
        """
        try:
            # Get object metadata
            response = self.s3.head_object(Bucket=self.bucket_name, Key=remote_path)

            # Extract metadata
            metadata = {
                "content_type": response.get("ContentType"),
                "content_length": response.get("ContentLength"),
                "last_modified": response.get("LastModified"),
                "etag": response.get("ETag", "").strip('"'),
                "metadata": response.get("Metadata", {}),
            }

            return {"success": True, "metadata": metadata}

        except ClientError as e:
            # Convert to standardized error and log
            cloud_error = handle_aws_error(e, "s3", "head_object")
            # Note: We don't log here since retry decorator will handle it

            # Return error response
            return {
                "success": False,
                "error": cloud_error.message,
                "error_details": cloud_error.to_dict(),
            }

    @retry_default
    @track_performance(CloudProvider.AWS, "s3", "create_folder")
    def create_folder(self, folder_path: str, **kwargs) -> Dict[str, Any]:
        """
        Create a folder in S3.

        Args:
            folder_path: Path for the folder
            **kwargs: Additional arguments

        Returns:
            Dictionary with creation result information
        """
        try:
            # Ensure path ends with trailing slash
            if not folder_path.endswith("/"):
                folder_path = folder_path + "/"

            # Create empty object with folder path (S3 convention for folders)
            self.s3.put_object(Bucket=self.bucket_name, Key=folder_path, Body="")

            return {"success": True, "bucket": self.bucket_name, "key": folder_path}

        except ClientError as e:
            # Convert to standardized error and log
            cloud_error = handle_aws_error(e, "s3", "put_object")
            # Note: We don't log here since retry decorator will handle it

            # Return error response
            return {
                "success": False,
                "error": cloud_error.message,
                "error_details": cloud_error.to_dict(),
            }

    @retry_aggressive
    @track_performance(CloudProvider.AWS, "s3", "bulk_upload")
    def bulk_upload(
        self, local_directory: str, remote_prefix: str = "", **kwargs
    ) -> Dict[str, Any]:
        """
        Upload multiple files from a local directory to S3.

        Args:
            local_directory: Path to local directory
            remote_prefix: Prefix to add to remote paths
            **kwargs: Additional arguments passed to upload_file

        Returns:
            Dictionary with bulk upload result information
        """
        if not os.path.isdir(local_directory):
            return {
                "success": False,
                "error": f"Local directory not found: {local_directory}",
            }

        results = {
            "success": True,
            "files": [],
            "failed": [],
            "total": 0,
            "succeeded": 0,
            "failed_count": 0,
        }

        # Process all files in directory
        for root, _, files in os.walk(local_directory):
            for filename in files:
                # Build local and remote paths
                local_path = os.path.join(root, filename)

                # Calculate relative path from local_directory
                rel_path = os.path.relpath(local_path, local_directory)

                # Normalize path separators for S3
                rel_path = rel_path.replace("\\", "/")

                # Join with remote prefix
                if remote_prefix:
                    remote_path = f"{remote_prefix.rstrip('/')}/{rel_path}"
                else:
                    remote_path = rel_path

                # Upload file
                upload_result = self.upload_file(local_path, remote_path, **kwargs)
                results["total"] += 1

                if upload_result["success"]:
                    results["succeeded"] += 1
                    results["files"].append(
                        {
                            "local_path": local_path,
                            "remote_path": remote_path,
                            "url": upload_result.get("url", ""),
                        }
                    )
                else:
                    results["failed_count"] += 1
                    results["failed"].append(
                        {
                            "local_path": local_path,
                            "remote_path": remote_path,
                            "error": upload_result.get("error", "Unknown error"),
                        }
                    )

        # Update success flag if any failures
        if results["failed_count"] > 0:
            results["success"] = False

        return results

    @retry_aggressive
    @track_performance(CloudProvider.AWS, "s3", "bulk_download")
    def bulk_download(self, remote_prefix: str, local_directory: str, **kwargs) -> Dict[str, Any]:
        """
        Download multiple files from S3 to a local directory.

        Args:
            remote_prefix: Prefix to filter remote paths
            local_directory: Path to local directory
            **kwargs: Additional arguments passed to download_file

        Returns:
            Dictionary with bulk download result information
        """
        # Create local directory if it doesn't exist
        os.makedirs(local_directory, exist_ok=True)

        results = {
            "success": True,
            "files": [],
            "failed": [],
            "total": 0,
            "succeeded": 0,
            "failed_count": 0,
        }

        # List files with prefix
        files = self.list_files(prefix=remote_prefix)

        for file_info in files:
            remote_path = file_info["key"]

            # Calculate local path
            if remote_prefix:
                # Remove prefix to get relative path
                rel_path = (
                    remote_path[len(remote_prefix) :]
                    if remote_path.startswith(remote_prefix)
                    else remote_path
                )
                rel_path = rel_path.lstrip("/")
            else:
                rel_path = remote_path

            local_path = os.path.join(local_directory, rel_path)

            # Create subdirectories if needed
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Download file
            download_result = self.download_file(remote_path, local_path, **kwargs)
            results["total"] += 1

            if download_result["success"]:
                results["succeeded"] += 1
                results["files"].append(
                    {
                        "remote_path": remote_path,
                        "local_path": local_path,
                        "size": file_info.get("size", 0),
                    }
                )
            else:
                results["failed_count"] += 1
                results["failed"].append(
                    {
                        "remote_path": remote_path,
                        "local_path": local_path,
                        "error": download_result.get("error", "Unknown error"),
                    }
                )

        # Update success flag if any failures
        if results["failed_count"] > 0:
            results["success"] = False

        return results
