"""
Base class for cloud database clients.

This module defines the base interface for cloud database services
across different providers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from cloud.config import CloudConfig

logger = logging.getLogger(__name__)


class CloudDatabaseClient(ABC):
    """
    Abstract base class for cloud database clients.

    This class defines the common interface for interacting with
    cloud database services like AWS DynamoDB, Azure Cosmos DB, and GCP Firestore.
    """

    def __init__(self, config: CloudConfig):
        """
        Initialize the database client.

        Args:
            config: Cloud configuration
        """
        self.config = config

    @abstractmethod
    def create_item(self, table_name: str, item: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Create an item in the database.

        Args:
            table_name: Name of the table/collection
            item: Item/document to create
            **kwargs: Additional provider-specific arguments

        Returns:
            Dictionary with creation result information
        """
        pass

    @abstractmethod
    def get_item(self, table_name: str, key: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Get an item from the database.

        Args:
            table_name: Name of the table/collection
            key: Key attributes to identify the item
            **kwargs: Additional provider-specific arguments

        Returns:
            Item data or empty dict if not found
        """
        pass

    @abstractmethod
    def update_item(
        self, table_name: str, key: Dict[str, Any], updates: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """
        Update an item in the database.

        Args:
            table_name: Name of the table/collection
            key: Key attributes to identify the item
            updates: Attributes to update
            **kwargs: Additional provider-specific arguments

        Returns:
            Dictionary with update result information
        """
        pass

    @abstractmethod
    def delete_item(self, table_name: str, key: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Delete an item from the database.

        Args:
            table_name: Name of the table/collection
            key: Key attributes to identify the item
            **kwargs: Additional provider-specific arguments

        Returns:
            Dictionary with deletion result information
        """
        pass

    @abstractmethod
    def query_items(self, table_name: str, query: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        """
        Query items from the database.

        Args:
            table_name: Name of the table/collection
            query: Query parameters
            **kwargs: Additional provider-specific arguments

        Returns:
            List of items matching the query
        """
        pass

    @abstractmethod
    def create_table(self, table_name: str, key_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Create a table in the database.

        Args:
            table_name: Name of the table/collection
            key_schema: Schema definition for the table
            **kwargs: Additional provider-specific arguments

        Returns:
            Dictionary with table creation result
        """
        pass

    @abstractmethod
    def delete_table(self, table_name: str, **kwargs) -> Dict[str, Any]:
        """
        Delete a table from the database.

        Args:
            table_name: Name of the table/collection
            **kwargs: Additional provider-specific arguments

        Returns:
            Dictionary with table deletion result
        """
        pass

    @abstractmethod
    def list_tables(self, **kwargs) -> List[str]:
        """
        List tables in the database.

        Args:
            **kwargs: Additional provider-specific arguments

        Returns:
            List of table names
        """
        pass

    @abstractmethod
    def table_exists(self, table_name: str, **kwargs) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name: Name of the table/collection
            **kwargs: Additional provider-specific arguments

        Returns:
            True if the table exists, False otherwise
        """
        pass

    @abstractmethod
    def get_table_info(self, table_name: str, **kwargs) -> Dict[str, Any]:
        """
        Get information about a table.

        Args:
            table_name: Name of the table/collection
            **kwargs: Additional provider-specific arguments

        Returns:
            Dictionary with table information
        """
        pass

    @abstractmethod
    def batch_write(self, table_name: str, items: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Write multiple items to the database in a batch.

        Args:
            table_name: Name of the table/collection
            items: List of items to write
            **kwargs: Additional provider-specific arguments

        Returns:
            Dictionary with batch write result
        """
        pass
