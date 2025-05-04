"""
Azure Blob Storage client implementation for CustomerAI platform.

This module provides integration with Azure Blob Storage service.
"""

import os
import logging
from typing import Any, Dict, List, Optional, BinaryIO
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError

from cloud.storage.base import CloudStorageClient
from cloud.config import CloudConfig

logger = logging.getLogger(__name__)

class AzureBlobClient(CloudStorageClient):
    """
    Azure Blob Storage client implementation.
    
    This class provides methods for interacting with Azure Blob Storage service.
    """
    
    def __init__(self, config: CloudConfig):
        """
        Initialize the Azure Blob Storage client.
        
        Args:
            config: Cloud configuration
        """
        super().__init__(config)
        
        # Get Azure configuration
        azure_config = config.get_config("azure")
        self.storage_account = azure_config.get("storage_account", "customeraidata")
        self.container_name = azure_config.get("storage_container", "data")
        
        # Get connection string or create from account key
        self.connection_string = azure_config.get("storage_connection_string")
        
        if not self.connection_string:
            account_key = azure_config.get("storage_account_key")
            if account_key:
                self.connection_string = (
                    f"DefaultEndpointsProtocol=https;"
                    f"AccountName={self.storage_account};"
                    f"AccountKey={account_key};"
                    f"EndpointSuffix=core.windows.net"
                )
        
        # Initialize Blob Service client
        if self.connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            
            # Create container if it doesn't exist
            self._ensure_container_exists()
        else:
            logger.error("Azure Blob Storage connection string not provided")
            self.blob_service_client = None
    
    def _ensure_container_exists(self) -> None:
        """Ensure the blob container exists."""
        if not self.blob_service_client:
            return
            
        try:
            # Check if container exists
            container_client = self.blob_service_client.get_container_client(self.container_name)
            container_client.get_container_properties()
        except ResourceNotFoundError:
            # Container doesn't exist, create it
            logger.info(f"Creating Azure Blob container: {self.container_name}")
            try:
                self.blob_service_client.create_container(self.container_name)
            except ResourceExistsError:
                # Container was created in the meantime
                pass
            except Exception as e:
                logger.error(f"Error creating Azure Blob container: {e}")
        except Exception as e:
            logger.error(f"Error checking Azure Blob container: {e}")
    
    def upload_file(self, local_path: str, remote_path: str, **kwargs) -> Dict[str, Any]:
        """
        Upload a file to Azure Blob Storage.
        
        Args:
            local_path: Path to local file
            remote_path: Path in blob container
            **kwargs: Additional arguments
                content_type: Content type of the blob
                metadata: Custom metadata to attach to the blob
                
        Returns:
            Dictionary with upload result information
        """
        if not self.blob_service_client:
            return {"success": False, "error": "Blob service client not initialized"}
        
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        try:
            # Get blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=remote_path
            )
            
            # Determine content type
            content_type = kwargs.get("content_type") or self.get_content_type(local_path)
            
            # Prepare metadata if provided
            metadata = kwargs.get("metadata")
            
            # Upload file
            with open(local_path, "rb") as data:
                blob_client.upload_blob(
                    data,
                    overwrite=True,
                    content_type=content_type,
                    metadata=metadata
                )
            
            # Generate URL
            url = self.get_file_url(remote_path, **kwargs)
            
            return {
                "success": True,
                "container": self.container_name,
                "blob": remote_path,
                "url": url,
                "content_type": content_type
            }
            
        except Exception as e:
            logger.error(f"Error uploading file to Azure Blob Storage: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def download_file(self, remote_path: str, local_path: str, **kwargs) -> Dict[str, Any]:
        """
        Download a file from Azure Blob Storage.
        
        Args:
            remote_path: Path in blob container
            local_path: Path to save local file
            **kwargs: Additional arguments
                
        Returns:
            Dictionary with download result information
        """
        if not self.blob_service_client:
            return {"success": False, "error": "Blob service client not initialized"}
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
            
            # Get blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=remote_path
            )
            
            # Download blob
            with open(local_path, "wb") as file:
                blob_data = blob_client.download_blob()
                file.write(blob_data.readall())
            
            return {
                "success": True,
                "container": self.container_name,
                "blob": remote_path,
                "local_path": local_path,
                "size": os.path.getsize(local_path)
            }
            
        except ResourceNotFoundError:
            logger.warning(f"Blob not found in Azure Blob Storage: {remote_path}")
            return {
                "success": False,
                "error": "File not found",
                "exists": False
            }
        except Exception as e:
            logger.error(f"Error downloading file from Azure Blob Storage: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_files(self, prefix: str = "", **kwargs) -> List[Dict[str, Any]]:
        """
        List files in Azure Blob container.
        
        Args:
            prefix: Path prefix to filter files
            **kwargs: Additional arguments
                max_results: Maximum number of blobs to return
                
        Returns:
            List of file information dictionaries
        """
        if not self.blob_service_client:
            return []
        
        try:
            # Get container client
            container_client = self.blob_service_client.get_container_client(self.container_name)
            
            # List blobs
            blobs = []
            
            # Prepare list parameters
            name_starts_with = prefix or None
            max_results = kwargs.get("max_results")
            
            # List blobs with the specified prefix
            blob_list = container_client.list_blobs(
                name_starts_with=name_starts_with,
                results_per_page=max_results
            )
            
            # Process blobs
            for blob in blob_list:
                properties = blob.properties
                blobs.append({
                    "name": blob.name,
                    "size": properties.size,
                    "last_modified": properties.last_modified,
                    "content_type": properties.content_settings.content_type,
                    "etag": properties.etag
                })
            
            return blobs
            
        except Exception as e:
            logger.error(f"Error listing files in Azure Blob Storage: {e}")
            return []
    
    def delete_file(self, remote_path: str, **kwargs) -> Dict[str, Any]:
        """
        Delete a file from Azure Blob Storage.
        
        Args:
            remote_path: Path in blob container
            **kwargs: Additional arguments
                
        Returns:
            Dictionary with deletion result information
        """
        if not self.blob_service_client:
            return {"success": False, "error": "Blob service client not initialized"}
        
        try:
            # Get blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=remote_path
            )
            
            # Delete blob
            blob_client.delete_blob()
            
            return {
                "success": True,
                "container": self.container_name,
                "blob": remote_path
            }
            
        except ResourceNotFoundError:
            logger.warning(f"Blob not found in Azure Blob Storage: {remote_path}")
            return {
                "success": False,
                "error": "File not found",
                "exists": False
            }
        except Exception as e:
            logger.error(f"Error deleting file from Azure Blob Storage: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def file_exists(self, remote_path: str, **kwargs) -> bool:
        """
        Check if a file exists in Azure Blob Storage.
        
        Args:
            remote_path: Path in blob container
            **kwargs: Additional arguments
                
        Returns:
            True if file exists, False otherwise
        """
        if not self.blob_service_client:
            return False
        
        try:
            # Get blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=remote_path
            )
            
            # Check if blob exists
            blob_client.get_blob_properties()
            
            return True
            
        except ResourceNotFoundError:
            return False
        except Exception as e:
            logger.error(f"Error checking if file exists in Azure Blob Storage: {e}")
            return False
    
    def get_file_url(self, remote_path: str, expiration: int = 3600, **kwargs) -> str:
        """
        Get a SAS URL for Azure Blob access.
        
        Args:
            remote_path: Path in blob container
            expiration: URL expiration time in seconds
            **kwargs: Additional arguments
                
        Returns:
            SAS URL string
        """
        if not self.blob_service_client:
            return ""
        
        try:
            # Get blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=remote_path
            )
            
            # Generate SAS token for the blob
            sas_token = blob_client.generate_shared_access_signature(
                permission="r",  # Read permission
                expiry=datetime.utcnow() + timedelta(seconds=expiration)
            )
            
            # Construct URL with SAS token
            url = f"{blob_client.url}?{sas_token}"
            
            return url
            
        except Exception as e:
            logger.error(f"Error generating SAS URL: {e}")
            return ""
    
    def get_file_metadata(self, remote_path: str, **kwargs) -> Dict[str, Any]:
        """
        Get metadata for a file in Azure Blob Storage.
        
        Args:
            remote_path: Path in blob container
            **kwargs: Additional arguments
                
        Returns:
            Dictionary with file metadata
        """
        if not self.blob_service_client:
            return {}
        
        try:
            # Get blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=remote_path
            )
            
            # Get blob properties
            properties = blob_client.get_blob_properties()
            
            # Extract metadata
            metadata = {
                "size": properties.size,
                "last_modified": properties.last_modified,
                "content_type": properties.content_settings.content_type,
                "etag": properties.etag,
                "metadata": properties.metadata
            }
            
            return metadata
            
        except ResourceNotFoundError:
            logger.warning(f"Blob not found in Azure Blob Storage: {remote_path}")
            return {}
        except Exception as e:
            logger.error(f"Error getting file metadata from Azure Blob Storage: {e}")
            return {}
    
    def create_folder(self, folder_path: str, **kwargs) -> Dict[str, Any]:
        """
        Create a folder in Azure Blob container.
        Note: Azure Blob Storage doesn't have actual folders, but uses path prefixes.
        
        Args:
            folder_path: Path for the folder
            **kwargs: Additional arguments
                
        Returns:
            Dictionary with folder creation result
        """
        if not self.blob_service_client:
            return {"success": False, "error": "Blob service client not initialized"}
        
        try:
            # Ensure path ends with slash
            if not folder_path.endswith('/'):
                folder_path += '/'
            
            # Create empty blob with folder prefix
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=folder_path
            )
            
            # Upload empty content
            blob_client.upload_blob(b"", overwrite=True)
            
            return {
                "success": True,
                "container": self.container_name,
                "folder": folder_path
            }
            
        except Exception as e:
            logger.error(f"Error creating folder in Azure Blob Storage: {e}")
            return {
                "success": False,
                "error": str(e)
            } 