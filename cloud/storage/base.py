"""
Base class for cloud storage clients.

This module defines the base interface for cloud storage services
across different providers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, BinaryIO, Union, Iterator
import os
import mimetypes

from cloud.config import CloudConfig

logger = logging.getLogger(__name__)

class CloudStorageClient(ABC):
    """
    Abstract base class for cloud storage clients.
    
    This class defines the common interface for interacting with
    cloud storage services like S3, Azure Blob Storage, and GCP Storage.
    """
    
    def __init__(self, config: CloudConfig):
        """
        Initialize the storage client.
        
        Args:
            config: Cloud configuration
        """
        self.config = config
    
    @abstractmethod
    def upload_file(self, local_path: str, remote_path: str, **kwargs) -> Dict[str, Any]:
        """
        Upload a file to cloud storage.
        
        Args:
            local_path: Path to local file
            remote_path: Path in cloud storage
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Dictionary with upload result information
        """
        pass
    
    @abstractmethod
    def download_file(self, remote_path: str, local_path: str, **kwargs) -> Dict[str, Any]:
        """
        Download a file from cloud storage.
        
        Args:
            remote_path: Path in cloud storage
            local_path: Path to save local file
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Dictionary with download result information
        """
        pass
    
    @abstractmethod
    def list_files(self, prefix: str = "", **kwargs) -> List[Dict[str, Any]]:
        """
        List files in cloud storage.
        
        Args:
            prefix: Path prefix to filter files
            **kwargs: Additional provider-specific arguments
            
        Returns:
            List of file information dictionaries
        """
        pass
    
    @abstractmethod
    def delete_file(self, remote_path: str, **kwargs) -> Dict[str, Any]:
        """
        Delete a file from cloud storage.
        
        Args:
            remote_path: Path in cloud storage
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Dictionary with deletion result information
        """
        pass
    
    @abstractmethod
    def file_exists(self, remote_path: str, **kwargs) -> bool:
        """
        Check if a file exists in cloud storage.
        
        Args:
            remote_path: Path in cloud storage
            **kwargs: Additional provider-specific arguments
            
        Returns:
            True if file exists, False otherwise
        """
        pass
    
    @abstractmethod
    def get_file_url(self, remote_path: str, expiration: int = 3600, **kwargs) -> str:
        """
        Get a pre-signed URL for file access.
        
        Args:
            remote_path: Path in cloud storage
            expiration: URL expiration time in seconds
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Pre-signed URL string
        """
        pass
    
    @abstractmethod
    def get_file_metadata(self, remote_path: str, **kwargs) -> Dict[str, Any]:
        """
        Get metadata for a file.
        
        Args:
            remote_path: Path in cloud storage
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Dictionary with file metadata
        """
        pass
    
    @abstractmethod
    def create_folder(self, folder_path: str, **kwargs) -> Dict[str, Any]:
        """
        Create a folder in cloud storage.
        
        Args:
            folder_path: Path for the folder
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Dictionary with folder creation result
        """
        pass
    
    def get_content_type(self, file_path: str) -> str:
        """
        Get the MIME type for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type string
        """
        content_type, _ = mimetypes.guess_type(file_path)
        return content_type or 'application/octet-stream' 