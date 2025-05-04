"""
AWS DynamoDB client implementation for CustomerAI platform.

This module provides integration with AWS DynamoDB service.
"""

import logging
import time
import boto3
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError
from typing import Any, Dict, List, Optional, Union, Iterator

from cloud.database.base import CloudDatabaseClient
from cloud.config import CloudConfig

logger = logging.getLogger(__name__)

class AWSDynamoDBClient(CloudDatabaseClient):
    """
    AWS DynamoDB client implementation.
    
    This class provides methods for interacting with AWS DynamoDB service.
    """
    
    def __init__(self, config: CloudConfig):
        """
        Initialize the AWS DynamoDB client.
        
        Args:
            config: Cloud configuration
        """
        super().__init__(config)
        
        # Get AWS configuration
        aws_config = config.get_config("aws")
        self.region = aws_config.get("region", "us-east-1")
        self.table_name = aws_config.get("dynamodb_table", "customerai-data")
        
        # Initialize DynamoDB resources
        try:
            self.dynamodb = boto3.resource(
                'dynamodb',
                region_name=self.region,
                aws_access_key_id=aws_config.get("access_key_id"),
                aws_secret_access_key=aws_config.get("secret_access_key")
            )
            
            self.dynamodb_client = boto3.client(
                'dynamodb',
                region_name=self.region,
                aws_access_key_id=aws_config.get("access_key_id"),
                aws_secret_access_key=aws_config.get("secret_access_key")
            )
            
            logger.info(f"AWS DynamoDB client initialized in {self.region}")
        except Exception as e:
            logger.error(f"Error initializing AWS DynamoDB client: {e}")
            self.dynamodb = None
            self.dynamodb_client = None
    
    def create_item(self, table_name: str, item: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Create an item in a DynamoDB table.
        
        Args:
            table_name: Name of the table
            item: Item to create
            **kwargs: Additional arguments
                condition_expression: Condition expression for the creation
                
        Returns:
            Dictionary with creation result information
        """
        if not self.dynamodb:
            return {"success": False, "error": "DynamoDB resource not initialized"}
        
        try:
            # Get table
            table = self.dynamodb.Table(table_name)
            
            # Prepare parameters
            params = {"Item": item}
            
            # Add conditional expression if provided
            condition_expression = kwargs.get("condition_expression")
            if condition_expression:
                params["ConditionExpression"] = condition_expression
            
            # Create item
            response = table.put_item(**params)
            
            return {
                "success": True,
                "table": table_name,
                "http_status": response["ResponseMetadata"]["HTTPStatusCode"]
            }
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            
            if error_code == "ConditionalCheckFailedException":
                logger.warning(f"Conditional check failed for item in DynamoDB table {table_name}")
                return {
                    "success": False,
                    "error": "Conditional check failed",
                    "table": table_name
                }
            else:
                logger.error(f"Error creating item in DynamoDB table {table_name}: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "table": table_name
                }
    
    def get_item(self, table_name: str, key: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Get an item from a DynamoDB table.
        
        Args:
            table_name: Name of the table
            key: Key attributes to identify the item
            **kwargs: Additional arguments
                consistent_read: Whether to use strong consistency
                projection_expression: Attributes to retrieve
                
        Returns:
            Item data or empty dict if not found
        """
        if not self.dynamodb:
            return {}
        
        try:
            # Get table
            table = self.dynamodb.Table(table_name)
            
            # Prepare parameters
            params = {"Key": key}
            
            # Add consistent read if requested
            consistent_read = kwargs.get("consistent_read")
            if consistent_read is not None:
                params["ConsistentRead"] = consistent_read
            
            # Add projection expression if provided
            projection_expression = kwargs.get("projection_expression")
            if projection_expression:
                params["ProjectionExpression"] = projection_expression
            
            # Get item
            response = table.get_item(**params)
            
            # Return item if found
            return response.get("Item", {})
            
        except ClientError as e:
            logger.error(f"Error getting item from DynamoDB table {table_name}: {e}")
            return {}
    
    def update_item(self, table_name: str, key: Dict[str, Any], updates: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Update an item in a DynamoDB table.
        
        Args:
            table_name: Name of the table
            key: Key attributes to identify the item
            updates: Attributes to update
            **kwargs: Additional arguments
                condition_expression: Condition expression for the update
                return_values: Values to return (NONE, ALL_OLD, UPDATED_OLD, ALL_NEW, UPDATED_NEW)
                
        Returns:
            Dictionary with update result information
        """
        if not self.dynamodb:
            return {"success": False, "error": "DynamoDB resource not initialized"}
        
        try:
            # Get table
            table = self.dynamodb.Table(table_name)
            
            # Build update expression and attribute values
            update_expression_parts = []
            expression_attribute_values = {}
            
            for attr_name, attr_value in updates.items():
                update_expression_parts.append(f"#{attr_name} = :{attr_name}")
                expression_attribute_values[f":{attr_name}"] = attr_value
            
            update_expression = "SET " + ", ".join(update_expression_parts)
            
            # Build attribute names (to handle reserved words)
            expression_attribute_names = {f"#{attr_name}": attr_name for attr_name in updates.keys()}
            
            # Prepare parameters
            params = {
                "Key": key,
                "UpdateExpression": update_expression,
                "ExpressionAttributeNames": expression_attribute_names,
                "ExpressionAttributeValues": expression_attribute_values
            }
            
            # Add condition expression if provided
            condition_expression = kwargs.get("condition_expression")
            if condition_expression:
                params["ConditionExpression"] = condition_expression
            
            # Add return values if provided
            return_values = kwargs.get("return_values")
            if return_values:
                params["ReturnValues"] = return_values
            
            # Update item
            response = table.update_item(**params)
            
            result = {
                "success": True,
                "table": table_name,
                "http_status": response["ResponseMetadata"]["HTTPStatusCode"]
            }
            
            # Add attributes if returned
            if "Attributes" in response:
                result["attributes"] = response["Attributes"]
            
            return result
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            
            if error_code == "ConditionalCheckFailedException":
                logger.warning(f"Conditional check failed for update in DynamoDB table {table_name}")
                return {
                    "success": False,
                    "error": "Conditional check failed",
                    "table": table_name
                }
            else:
                logger.error(f"Error updating item in DynamoDB table {table_name}: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "table": table_name
                }
    
    def delete_item(self, table_name: str, key: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Delete an item from a DynamoDB table.
        
        Args:
            table_name: Name of the table
            key: Key attributes to identify the item
            **kwargs: Additional arguments
                condition_expression: Condition expression for the deletion
                return_values: Values to return (NONE, ALL_OLD)
                
        Returns:
            Dictionary with deletion result information
        """
        if not self.dynamodb:
            return {"success": False, "error": "DynamoDB resource not initialized"}
        
        try:
            # Get table
            table = self.dynamodb.Table(table_name)
            
            # Prepare parameters
            params = {"Key": key}
            
            # Add condition expression if provided
            condition_expression = kwargs.get("condition_expression")
            if condition_expression:
                params["ConditionExpression"] = condition_expression
            
            # Add return values if provided
            return_values = kwargs.get("return_values")
            if return_values:
                params["ReturnValues"] = return_values
            
            # Delete item
            response = table.delete_item(**params)
            
            result = {
                "success": True,
                "table": table_name,
                "http_status": response["ResponseMetadata"]["HTTPStatusCode"]
            }
            
            # Add attributes if returned
            if "Attributes" in response:
                result["attributes"] = response["Attributes"]
            
            return result
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            
            if error_code == "ConditionalCheckFailedException":
                logger.warning(f"Conditional check failed for deletion in DynamoDB table {table_name}")
                return {
                    "success": False,
                    "error": "Conditional check failed",
                    "table": table_name
                }
            else:
                logger.error(f"Error deleting item from DynamoDB table {table_name}: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "table": table_name
                }
    
    def query_items(self, table_name: str, query: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        """
        Query items from a DynamoDB table.
        
        Args:
            table_name: Name of the table
            query: Query parameters
                key_condition_expression: KeyConditionExpression for the query
                filter_expression: FilterExpression for the query
                expression_attribute_values: ExpressionAttributeValues for the query
            **kwargs: Additional arguments
                index_name: Name of the index to query
                limit: Maximum number of items to return
                consistent_read: Whether to use strong consistency
                scan_index_forward: Whether to scan the index forward
                projection_expression: Attributes to retrieve
                exclusive_start_key: Key to start with for pagination
                
        Returns:
            List of items matching the query
        """
        if not self.dynamodb:
            return []
        
        try:
            # Get table
            table = self.dynamodb.Table(table_name)
            
            # Prepare parameters
            params = {}
            
            # Add key condition expression
            key_condition_expression = query.get("key_condition_expression")
            if key_condition_expression:
                params["KeyConditionExpression"] = key_condition_expression
            
            # Add filter expression
            filter_expression = query.get("filter_expression")
            if filter_expression:
                params["FilterExpression"] = filter_expression
            
            # Add expression attribute values
            expression_attribute_values = query.get("expression_attribute_values")
            if expression_attribute_values:
                params["ExpressionAttributeValues"] = expression_attribute_values
            
            # Add expression attribute names
            expression_attribute_names = query.get("expression_attribute_names")
            if expression_attribute_names:
                params["ExpressionAttributeNames"] = expression_attribute_names
            
            # Add index name if provided
            index_name = kwargs.get("index_name")
            if index_name:
                params["IndexName"] = index_name
            
            # Add limit if provided
            limit = kwargs.get("limit")
            if limit is not None:
                params["Limit"] = limit
            
            # Add consistent read if provided
            consistent_read = kwargs.get("consistent_read")
            if consistent_read is not None:
                params["ConsistentRead"] = consistent_read
            
            # Add scan index forward if provided
            scan_index_forward = kwargs.get("scan_index_forward")
            if scan_index_forward is not None:
                params["ScanIndexForward"] = scan_index_forward
            
            # Add projection expression if provided
            projection_expression = kwargs.get("projection_expression")
            if projection_expression:
                params["ProjectionExpression"] = projection_expression
            
            # Add exclusive start key if provided
            exclusive_start_key = kwargs.get("exclusive_start_key")
            if exclusive_start_key:
                params["ExclusiveStartKey"] = exclusive_start_key
            
            # Determine whether to use query or scan
            if key_condition_expression:
                # Use query
                response = table.query(**params)
            else:
                # Use scan
                response = table.scan(**params)
            
            # Extract items
            items = response.get("Items", [])
            
            # Handle pagination if necessary
            if "LastEvaluatedKey" in response and kwargs.get("paginate", False):
                last_key = response["LastEvaluatedKey"]
                
                while last_key and (not limit or len(items) < limit):
                    # Update exclusive start key
                    params["ExclusiveStartKey"] = last_key
                    
                    # Adjust limit if specified
                    if limit:
                        params["Limit"] = limit - len(items)
                    
                    # Execute next query/scan
                    if key_condition_expression:
                        next_response = table.query(**params)
                    else:
                        next_response = table.scan(**params)
                    
                    # Add new items
                    items.extend(next_response.get("Items", []))
                    
                    # Update last key
                    last_key = next_response.get("LastEvaluatedKey")
            
            return items
            
        except ClientError as e:
            logger.error(f"Error querying items from DynamoDB table {table_name}: {e}")
            return []
    
    def create_table(self, table_name: str, key_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Create a DynamoDB table.
        
        Args:
            table_name: Name of the table
            key_schema: Schema definition for the table
                key_schema: List of key definitions
                attribute_definitions: List of attribute definitions
            **kwargs: Additional arguments
                billing_mode: Billing mode (PROVISIONED or PAY_PER_REQUEST)
                provisioned_throughput: Provisioned throughput settings
                global_secondary_indexes: Global secondary indexes
                local_secondary_indexes: Local secondary indexes
                stream_specification: Stream specification
                
        Returns:
            Dictionary with table creation result
        """
        if not self.dynamodb_client:
            return {"success": False, "error": "DynamoDB client not initialized"}
        
        try:
            # Prepare parameters
            params = {
                "TableName": table_name,
                "KeySchema": key_schema.get("key_schema", []),
                "AttributeDefinitions": key_schema.get("attribute_definitions", [])
            }
            
            # Add billing mode and provisioned throughput
            billing_mode = kwargs.get("billing_mode", "PAY_PER_REQUEST")
            params["BillingMode"] = billing_mode
            
            if billing_mode == "PROVISIONED":
                provisioned_throughput = kwargs.get("provisioned_throughput", {
                    "ReadCapacityUnits": 5,
                    "WriteCapacityUnits": 5
                })
                params["ProvisionedThroughput"] = provisioned_throughput
            
            # Add global secondary indexes if provided
            global_secondary_indexes = kwargs.get("global_secondary_indexes")
            if global_secondary_indexes:
                params["GlobalSecondaryIndexes"] = global_secondary_indexes
            
            # Add local secondary indexes if provided
            local_secondary_indexes = kwargs.get("local_secondary_indexes")
            if local_secondary_indexes:
                params["LocalSecondaryIndexes"] = local_secondary_indexes
            
            # Add stream specification if provided
            stream_specification = kwargs.get("stream_specification")
            if stream_specification:
                params["StreamSpecification"] = stream_specification
            
            # Create table
            response = self.dynamodb_client.create_table(**params)
            
            # Wait for table to become active if requested
            if kwargs.get("wait_for_active", True):
                logger.info(f"Waiting for DynamoDB table {table_name} to become active...")
                waiter = self.dynamodb_client.get_waiter("table_exists")
                waiter.wait(TableName=table_name)
            
            return {
                "success": True,
                "table": table_name,
                "status": response.get("TableDescription", {}).get("TableStatus"),
                "arn": response.get("TableDescription", {}).get("TableArn")
            }
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            
            if error_code == "ResourceInUseException":
                logger.warning(f"DynamoDB table {table_name} already exists")
                return {
                    "success": False,
                    "error": "Table already exists",
                    "table": table_name
                }
            else:
                logger.error(f"Error creating DynamoDB table {table_name}: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "table": table_name
                }
    
    def delete_table(self, table_name: str, **kwargs) -> Dict[str, Any]:
        """
        Delete a DynamoDB table.
        
        Args:
            table_name: Name of the table
            **kwargs: Additional arguments
                wait_for_deletion: Whether to wait for the table to be deleted
                
        Returns:
            Dictionary with table deletion result
        """
        if not self.dynamodb_client:
            return {"success": False, "error": "DynamoDB client not initialized"}
        
        try:
            # Delete table
            response = self.dynamodb_client.delete_table(TableName=table_name)
            
            # Wait for table to be deleted if requested
            if kwargs.get("wait_for_deletion", True):
                logger.info(f"Waiting for DynamoDB table {table_name} to be deleted...")
                waiter = self.dynamodb_client.get_waiter("table_not_exists")
                waiter.wait(TableName=table_name)
            
            return {
                "success": True,
                "table": table_name,
                "status": response.get("TableDescription", {}).get("TableStatus")
            }
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            
            if error_code == "ResourceNotFoundException":
                logger.warning(f"DynamoDB table {table_name} does not exist")
                return {
                    "success": False,
                    "error": "Table does not exist",
                    "table": table_name
                }
            else:
                logger.error(f"Error deleting DynamoDB table {table_name}: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "table": table_name
                }
    
    def list_tables(self, **kwargs) -> List[str]:
        """
        List DynamoDB tables.
        
        Args:
            **kwargs: Additional arguments
                limit: Maximum number of tables to return
                exclusive_start_table_name: Table name to start with for pagination
                
        Returns:
            List of table names
        """
        if not self.dynamodb_client:
            return []
        
        try:
            # Prepare parameters
            params = {}
            
            # Add limit if provided
            limit = kwargs.get("limit")
            if limit is not None:
                params["Limit"] = limit
            
            # Add exclusive start table name if provided
            exclusive_start_table_name = kwargs.get("exclusive_start_table_name")
            if exclusive_start_table_name:
                params["ExclusiveStartTableName"] = exclusive_start_table_name
            
            # List tables
            response = self.dynamodb_client.list_tables(**params)
            
            # Extract table names
            tables = response.get("TableNames", [])
            
            # Handle pagination if necessary
            if "LastEvaluatedTableName" in response and kwargs.get("paginate", False):
                last_table = response["LastEvaluatedTableName"]
                
                while last_table and (not limit or len(tables) < limit):
                    # Update exclusive start table name
                    params["ExclusiveStartTableName"] = last_table
                    
                    # Adjust limit if specified
                    if limit:
                        params["Limit"] = limit - len(tables)
                    
                    # Execute next list
                    next_response = self.dynamodb_client.list_tables(**params)
                    
                    # Add new tables
                    tables.extend(next_response.get("TableNames", []))
                    
                    # Update last table
                    last_table = next_response.get("LastEvaluatedTableName")
            
            return tables
            
        except ClientError as e:
            logger.error(f"Error listing DynamoDB tables: {e}")
            return []
    
    def table_exists(self, table_name: str, **kwargs) -> bool:
        """
        Check if a DynamoDB table exists.
        
        Args:
            table_name: Name of the table
            **kwargs: Additional arguments
                
        Returns:
            True if the table exists, False otherwise
        """
        if not self.dynamodb_client:
            return False
        
        try:
            # Describe table
            self.dynamodb_client.describe_table(TableName=table_name)
            return True
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            
            if error_code == "ResourceNotFoundException":
                return False
            else:
                logger.error(f"Error checking if DynamoDB table {table_name} exists: {e}")
                return False
    
    def get_table_info(self, table_name: str, **kwargs) -> Dict[str, Any]:
        """
        Get information about a DynamoDB table.
        
        Args:
            table_name: Name of the table
            **kwargs: Additional arguments
                
        Returns:
            Dictionary with table information
        """
        if not self.dynamodb_client:
            return {}
        
        try:
            # Describe table
            response = self.dynamodb_client.describe_table(TableName=table_name)
            
            # Extract table description
            return response.get("Table", {})
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            
            if error_code == "ResourceNotFoundException":
                logger.warning(f"DynamoDB table {table_name} does not exist")
                return {}
            else:
                logger.error(f"Error getting information about DynamoDB table {table_name}: {e}")
                return {}
    
    def batch_write(self, table_name: str, items: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Write multiple items to a DynamoDB table in a batch.
        
        Args:
            table_name: Name of the table
            items: List of items to write
            **kwargs: Additional arguments
                
        Returns:
            Dictionary with batch write result
        """
        if not self.dynamodb:
            return {"success": False, "error": "DynamoDB resource not initialized"}
        
        try:
            # Get table
            table = self.dynamodb.Table(table_name)
            
            # Batch write
            with table.batch_writer() as batch:
                for item in items:
                    batch.put_item(Item=item)
            
            return {
                "success": True,
                "table": table_name,
                "count": len(items)
            }
            
        except ClientError as e:
            logger.error(f"Error batch writing to DynamoDB table {table_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "table": table_name
            } 