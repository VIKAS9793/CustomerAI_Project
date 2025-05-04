"""
Cloud database service clients for CustomerAI platform.

This package provides integrations with cloud database services
such as AWS DynamoDB, Azure Cosmos DB, and Google Cloud Firestore.
"""

from cloud.database.base import CloudDatabaseClient

__all__ = ['CloudDatabaseClient'] 