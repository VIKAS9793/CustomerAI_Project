"""
Google Cloud Storage client implementation for CustomerAI platform.

This module provides integration with Google Cloud Storage service.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List

from google.cloud import storage
from google.cloud.exceptions import NotFound
from google.oauth2 import service_account

from cloud.config import CloudConfig
from cloud.storage.base import CloudStorageClient

logger = logging.getLogger(__name__)


class GCPStorageClient(CloudStorageClient):
    """
    Google Cloud Storage client implementation.

    This class provides methods for interacting with Google Cloud Storage service.
    """

    def __init__(self, config: CloudConfig):
        """
        Initialize the Google Cloud Storage client.

        Args:
            config: Cloud configuration
        """
        super().__init__(config)

        # Get GCP configuration
        gcp_config = config.get_config("gcp")
        self.project_id = gcp_config.get("project_id")
        self.bucket_name = gcp_config.get("storage_bucket", "customerai-data")
        self.credentials_file = gcp_config.get("credentials_file")

        # Initialize Storage client
        try:
            if self.credentials_file and os.path.exists(self.credentials_file):
                # Use provided credentials file
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_file
                )
                self.storage_client = storage.Client(
                    project=self.project_id, credentials=credentials
                )
            else:
                # Use default credentials (environment or application default)
                self.storage_client = storage.Client(project=self.project_id)

            # Create bucket if it doesn't exist
            self._ensure_bucket_exists()
        except Exception as e:
            logger.error(f"Error initializing Google Cloud Storage client: {e}")
            self.storage_client = None

    def _ensure_bucket_exists(self) -> None:
        """Ensure the GCS bucket exists."""
        if not self.storage_client:
            return

        try:
            # Check if bucket exists
            self.storage_client.get_bucket(self.bucket_name)
        except NotFound:
            # Bucket doesn't exist, create it
            logger.info(f"Creating GCS bucket: {self.bucket_name}")
            try:
                self.storage_client.create_bucket(self.bucket_name)
                logger.info(f"Bucket {self.bucket_name} created")
            except Exception as e:
                logger.error(f"Error creating GCS bucket: {e}")
        except Exception as e:
            logger.error(f"Error checking GCS bucket: {e}")

    def upload_file(self, local_path: str, remote_path: str, **kwargs) -> Dict[str, Any]:
        """
        Upload a file to Google Cloud Storage.

        Args:
            local_path: Path to local file
            remote_path: Path in GCS bucket
            **kwargs: Additional arguments
                content_type: Content type of the blob
                metadata: Custom metadata to attach to the blob
                public: Whether to make the file publicly accessible

        Returns:
            Dictionary with upload result information
        """
        if not self.storage_client:
            return {"success": False, "error": "Storage client not initialized"}

        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")

        try:
            # Get bucket and blob
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(remote_path)

            # Set content type
            content_type = kwargs.get("content_type") or self.get_content_type(local_path)
            if content_type:
                blob.content_type = content_type

            # Set metadata if provided
            metadata = kwargs.get("metadata")
            if metadata and isinstance(metadata, dict):
                blob.metadata = metadata

            # Upload file
            blob.upload_from_filename(local_path)

            # Make public if requested
            if kwargs.get("public"):
                blob.make_public()

            # Generate URL
            url = self.get_file_url(remote_path, **kwargs)

            return {
                "success": True,
                "bucket": self.bucket_name,
                "name": remote_path,
                "url": url,
                "content_type": content_type,
            }

        except Exception as e:
            logger.error(f"Error uploading file to Google Cloud Storage: {e}")
            return {"success": False, "error": str(e)}

    def download_file(self, remote_path: str, local_path: str, **kwargs) -> Dict[str, Any]:
        """
        Download a file from Google Cloud Storage.

        Args:
            remote_path: Path in GCS bucket
            local_path: Path to save local file
            **kwargs: Additional arguments

        Returns:
            Dictionary with download result information
        """
        if not self.storage_client:
            return {"success": False, "error": "Storage client not initialized"}

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)

            # Get bucket and blob
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(remote_path)

            # Download blob
            blob.download_to_filename(local_path)

            return {
                "success": True,
                "bucket": self.bucket_name,
                "name": remote_path,
                "local_path": local_path,
                "size": os.path.getsize(local_path),
            }

        except NotFound:
            logger.warning(f"Blob not found in Google Cloud Storage: {remote_path}")
            return {"success": False, "error": "File not found", "exists": False}
        except Exception as e:
            logger.error(f"Error downloading file from Google Cloud Storage: {e}")
            return {"success": False, "error": str(e)}

    def list_files(self, prefix: str = "", **kwargs) -> List[Dict[str, Any]]:
        """
        List files in Google Cloud Storage bucket.

        Args:
            prefix: Path prefix to filter files
            **kwargs: Additional arguments
                max_results: Maximum number of blobs to return

        Returns:
            List of file information dictionaries
        """
        if not self.storage_client:
            return []

        try:
            # Get bucket
            bucket = self.storage_client.bucket(self.bucket_name)

            # List blobs
            blobs = []

            # Prepare list parameters
            max_results = kwargs.get("max_results")

            # List blobs with the specified prefix
            blob_list = bucket.list_blobs(prefix=prefix, max_results=max_results)

            # Process blobs
            for blob in blob_list:
                blobs.append(
                    {
                        "name": blob.name,
                        "size": blob.size,
                        "updated": blob.updated,
                        "content_type": blob.content_type,
                        "md5_hash": blob.md5_hash,
                    }
                )

            return blobs

        except Exception as e:
            logger.error(f"Error listing files in Google Cloud Storage: {e}")
            return []

    def delete_file(self, remote_path: str, **kwargs) -> Dict[str, Any]:
        """
        Delete a file from Google Cloud Storage.

        Args:
            remote_path: Path in GCS bucket
            **kwargs: Additional arguments

        Returns:
            Dictionary with deletion result information
        """
        if not self.storage_client:
            return {"success": False, "error": "Storage client not initialized"}

        try:
            # Get bucket and blob
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(remote_path)

            # Delete blob
            blob.delete()

            return {"success": True, "bucket": self.bucket_name, "name": remote_path}

        except NotFound:
            logger.warning(f"Blob not found in Google Cloud Storage: {remote_path}")
            return {"success": False, "error": "File not found", "exists": False}
        except Exception as e:
            logger.error(f"Error deleting file from Google Cloud Storage: {e}")
            return {"success": False, "error": str(e)}

    def file_exists(self, remote_path: str, **kwargs) -> bool:
        """
        Check if a file exists in Google Cloud Storage.

        Args:
            remote_path: Path in GCS bucket
            **kwargs: Additional arguments

        Returns:
            True if file exists, False otherwise
        """
        if not self.storage_client:
            return False

        try:
            # Get bucket and blob
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(remote_path)

            # Check if blob exists
            return blob.exists()

        except Exception as e:
            logger.error(f"Error checking if file exists in Google Cloud Storage: {e}")
            return False

    def get_file_url(self, remote_path: str, expiration: int = 3600, **kwargs) -> str:
        """
        Get a signed URL for Google Cloud Storage file access.

        Args:
            remote_path: Path in GCS bucket
            expiration: URL expiration time in seconds
            **kwargs: Additional arguments
                public: Whether the file is publicly accessible

        Returns:
            Signed URL string
        """
        if not self.storage_client:
            return ""

        try:
            # Get bucket and blob
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(remote_path)

            # For public files, return a direct URL
            if kwargs.get("public"):
                return blob.public_url

            # Generate signed URL
            url = blob.generate_signed_url(
                version="v4",
                expiration=datetime.utcnow() + timedelta(seconds=expiration),
                method="GET",
            )

            return url

        except Exception as e:
            logger.error(f"Error generating signed URL: {e}")
            return ""

    def get_file_metadata(self, remote_path: str, **kwargs) -> Dict[str, Any]:
        """
        Get metadata for a file in Google Cloud Storage.

        Args:
            remote_path: Path in GCS bucket
            **kwargs: Additional arguments

        Returns:
            Dictionary with file metadata
        """
        if not self.storage_client:
            return {}

        try:
            # Get bucket and blob
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(remote_path)

            # Reload blob to get latest metadata
            blob.reload()

            # Extract metadata
            metadata = {
                "name": blob.name,
                "bucket": blob.bucket.name,
                "size": blob.size,
                "updated": blob.updated,
                "created": blob.time_created,
                "content_type": blob.content_type,
                "content_encoding": blob.content_encoding,
                "content_disposition": blob.content_disposition,
                "cache_control": blob.cache_control,
                "metadata": blob.metadata,
                "md5_hash": blob.md5_hash,
                "storage_class": blob.storage_class,
            }

            return metadata

        except NotFound:
            logger.warning(f"Blob not found in Google Cloud Storage: {remote_path}")
            return {}
        except Exception as e:
            logger.error(f"Error getting file metadata from Google Cloud Storage: {e}")
            return {}

    def create_folder(self, folder_path: str, **kwargs) -> Dict[str, Any]:
        """
        Create a folder in Google Cloud Storage bucket.
        Note: GCS doesn't have actual folders, but uses path prefixes.

        Args:
            folder_path: Path for the folder
            **kwargs: Additional arguments

        Returns:
            Dictionary with folder creation result
        """
        if not self.storage_client:
            return {"success": False, "error": "Storage client not initialized"}

        try:
            # Ensure path ends with slash
            if not folder_path.endswith("/"):
                folder_path += "/"

            # Create empty blob with folder prefix
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(folder_path)

            # Upload empty content
            blob.upload_from_string("")

            return {"success": True, "bucket": self.bucket_name, "folder": folder_path}

        except Exception as e:
            logger.error(f"Error creating folder in Google Cloud Storage: {e}")
            return {"success": False, "error": str(e)}
