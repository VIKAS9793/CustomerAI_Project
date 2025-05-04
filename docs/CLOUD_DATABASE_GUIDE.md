# Cloud Database Integration Guide

This guide provides detailed instructions for working with cloud database services in the CustomerAI Insights Platform.

## Table of Contents
- [Overview](#overview)
- [Supported Database Services](#supported-database-services)
- [Configuration](#configuration)
- [Basic Operations](#basic-operations)
- [Integration Patterns](#integration-patterns)
- [Performance Considerations](#performance-considerations)
- [Security Best Practices](#security-best-practices)
- [Troubleshooting](#troubleshooting)
- [Migration Guide](#migration-guide)

## Overview

The CustomerAI platform offers a provider-agnostic approach to cloud database integrations, allowing seamless storage and retrieval of data across different cloud providers. The abstraction layer enables your application to work with any supported database service through a consistent interface, making it easy to switch providers or use multiple providers simultaneously.

### Key Benefits

- **Consistent API**: Use the same code regardless of underlying cloud provider
- **Runtime Provider Selection**: Switch providers without code changes
- **Multi-Provider Support**: Use different providers for different data types
- **Automatic Connection Management**: Efficient handling of database connections
- **Comprehensive Operations**: Full CRUD operations plus advanced querying
- **Optimized Batch Operations**: Efficiently handle bulk data operations

## Supported Database Services

The platform supports the following cloud database services:

### AWS DynamoDB

- **Type**: NoSQL, key-value and document database
- **Strengths**: 
  - Seamless scalability
  - Single-digit millisecond latency
  - Fully managed with auto-scaling
  - Point-in-time recovery
  - Flexible data modeling

### Azure Cosmos DB

- **Type**: Multi-model database (document, key-value, graph)
- **Strengths**:
  - Global distribution
  - Multi-master replication
  - Tunable consistency levels
  - Automatic indexing
  - Rich query capabilities

### Google Cloud Firestore

- **Type**: NoSQL document database
- **Strengths**:
  - Real-time updates
  - Offline support
  - Automatic multi-region replication
  - ACID transactions
  - Expressive querying

## Configuration

### Environment Settings

Configure your preferred database provider in the `.env` file:

```
# Select your cloud provider
CLOUD_PROVIDER=aws  # Options: aws, azure, gcp, none

# AWS DynamoDB settings
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DYNAMODB_TABLE=customerai-data

# Azure Cosmos DB settings
AZURE_TENANT_ID=your-tenant-id
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret
AZURE_COSMOS_DB_ACCOUNT=your-cosmos-account
AZURE_COSMOS_DB_DATABASE=customerai

# Google Cloud Firestore settings
GCP_PROJECT_ID=your-project-id
GCP_CREDENTIALS_FILE=path/to/credentials.json
GCP_FIRESTORE_COLLECTION=customerai-data
```

### Initialization

```python
from cloud.config import CloudConfig
from cloud.factory import CloudServiceFactory

# Initialize with default provider from environment
config = CloudConfig()
factory = CloudServiceFactory(config)

# Get the database client
db_client = factory.get_database_client()

# Or specify a provider explicitly
azure_config = CloudConfig("azure")
azure_factory = CloudServiceFactory(azure_config)
azure_db = azure_factory.get_database_client()
```

### Table/Collection Creation

Each provider has different requirements for creating tables or collections. The platform abstracts these differences:

```python
# For AWS DynamoDB
key_schema = {
    "key_schema": [
        {"AttributeName": "id", "KeyType": "HASH"},  # Partition key
        {"AttributeName": "created_at", "KeyType": "RANGE"}  # Sort key
    ],
    "attribute_definitions": [
        {"AttributeName": "id", "AttributeType": "S"},
        {"AttributeName": "created_at", "AttributeType": "S"}
    ]
}

# Create table
result = db_client.create_table("customer_interactions", key_schema)
```

## Basic Operations

### Creating Items

```python
# Create a single item
item = {
    "id": "customer-123",
    "created_at": "2023-10-15T14:30:00Z",
    "name": "John Doe",
    "email": "john.doe@example.com",
    "subscription_tier": "premium",
    "last_login": "2023-10-14T09:15:00Z"
}

result = db_client.create_item("customers", item)

# Batch creation
items = [
    {"id": "customer-124", "created_at": "2023-10-15T14:35:00Z", "name": "Jane Smith"},
    {"id": "customer-125", "created_at": "2023-10-15T14:40:00Z", "name": "Bob Johnson"}
]

batch_result = db_client.batch_write("customers", items)
```

### Reading Items

```python
# Get a single item by key
key = {"id": "customer-123", "created_at": "2023-10-15T14:30:00Z"}
customer = db_client.get_item("customers", key)

# Use projection to retrieve specific attributes
customer_name = db_client.get_item(
    "customers", 
    key, 
    projection_expression="name,email"
)
```

### Updating Items

```python
# Update an item
key = {"id": "customer-123", "created_at": "2023-10-15T14:30:00Z"}
updates = {
    "subscription_tier": "basic",
    "last_updated": "2023-10-15T16:45:00Z"
}

result = db_client.update_item("customers", key, updates)

# Conditional update (only if condition is met)
result = db_client.update_item(
    "customers", 
    key, 
    updates, 
    condition_expression="subscription_tier = :tier",
    expression_attribute_values={":tier": "premium"}
)
```

### Deleting Items

```python
# Delete an item
key = {"id": "customer-123", "created_at": "2023-10-15T14:30:00Z"}
result = db_client.delete_item("customers", key)

# Conditional delete
result = db_client.delete_item(
    "customers", 
    key, 
    condition_expression="subscription_tier = :tier",
    expression_attribute_values={":tier": "basic"}
)
```

### Querying Items

```python
# Simple query
query = {
    "key_condition_expression": "id = :id",
    "expression_attribute_values": {":id": "customer-123"}
}

items = db_client.query_items("customers", query)

# Query with filter
query = {
    "key_condition_expression": "id = :id",
    "filter_expression": "subscription_tier = :tier",
    "expression_attribute_values": {
        ":id": "customer-123",
        ":tier": "premium"
    }
}

items = db_client.query_items("customers", query)

# Query with limit and pagination
items = db_client.query_items(
    "customers", 
    query, 
    limit=10,
    paginate=True
)
```

## Integration Patterns

### Event-Driven Architecture

Store events in the database and process them asynchronously:

```python
# Create an event
event = {
    "id": str(uuid.uuid4()),
    "created_at": datetime.now().isoformat(),
    "type": "customer_login",
    "customer_id": "customer-123",
    "metadata": {
        "ip_address": "192.168.1.1",
        "device": "mobile",
        "location": "New York"
    },
    "processed": False
}

db_client.create_item("events", event)

# Process events
unprocessed_query = {
    "filter_expression": "processed = :processed",
    "expression_attribute_values": {":processed": False}
}

events = db_client.query_items("events", unprocessed_query, limit=100)

for event in events:
    # Process event
    process_event(event)
    
    # Mark as processed
    db_client.update_item(
        "events",
        {"id": event["id"], "created_at": event["created_at"]},
        {"processed": True, "processed_at": datetime.now().isoformat()}
    )
```

### Repository Pattern

Create a repository layer to abstract database operations:

```python
class CustomerRepository:
    def __init__(self, db_client):
        self.db_client = db_client
        self.table_name = "customers"
    
    def create(self, customer):
        # Add required fields
        if "id" not in customer:
            customer["id"] = str(uuid.uuid4())
        if "created_at" not in customer:
            customer["created_at"] = datetime.now().isoformat()
            
        return self.db_client.create_item(self.table_name, customer)
    
    def get_by_id(self, customer_id):
        # Find customer by ID across all created_at values
        query = {
            "key_condition_expression": "id = :id",
            "expression_attribute_values": {":id": customer_id}
        }
        
        items = self.db_client.query_items(self.table_name, query, limit=1)
        return items[0] if items else None
    
    def update(self, customer_id, updates):
        # Get customer to find the created_at value
        customer = self.get_by_id(customer_id)
        if not customer:
            return {"success": False, "error": "Customer not found"}
            
        # Update with timestamp
        updates["last_updated"] = datetime.now().isoformat()
        
        return self.db_client.update_item(
            self.table_name,
            {"id": customer_id, "created_at": customer["created_at"]},
            updates
        )
    
    def delete(self, customer_id):
        # Get customer to find the created_at value
        customer = self.get_by_id(customer_id)
        if not customer:
            return {"success": False, "error": "Customer not found"}
            
        return self.db_client.delete_item(
            self.table_name,
            {"id": customer_id, "created_at": customer["created_at"]}
        )
    
    def find_by_subscription_tier(self, tier):
        query = {
            "filter_expression": "subscription_tier = :tier",
            "expression_attribute_values": {":tier": tier}
        }
        
        return self.db_client.query_items(self.table_name, query)
```

### Data Access Objects (DAO)

Create DAOs for different entity types:

```python
class CustomerDAO:
    # Customer-specific operations
    pass

class TransactionDAO:
    # Transaction-specific operations
    pass

class FeedbackDAO:
    # Feedback-specific operations
    pass
```

## Performance Considerations

### Index Design

Each provider has different indexing capabilities. Here are some general guidelines:

1. **AWS DynamoDB**:
   - Design tables with access patterns in mind
   - Use sparse secondary indexes where possible
   - Consider Global Secondary Indexes for frequently queried attributes

2. **Azure Cosmos DB**:
   - Leverage automatic indexing for most cases
   - Exclude properties from indexing if not queried
   - Use composite indexes for complex queries

3. **Google Firestore**:
   - Create composite indexes for complex queries
   - Avoid using != and NOT IN operators when possible
   - Consider denormalization for frequently joined data

### Batch Operations

Use batch operations for better performance:

```python
# Batch write (create/update multiple items at once)
items = [
    {"id": "item1", "created_at": "2023-10-15T10:00:00Z", "value": "data1"},
    {"id": "item2", "created_at": "2023-10-15T10:05:00Z", "value": "data2"},
    # ... more items
]

result = db_client.batch_write("mytable", items)
```

### Connection Pooling

The database clients automatically handle connection pooling for better performance. Configuration can be adjusted in `.env`:

```
# Connection pooling settings
DB_MAX_CONNECTIONS=50
DB_CONNECTION_TIMEOUT=5
DB_KEEP_ALIVE=True
```

## Security Best Practices

### Encryption

1. **Data at Rest**: All supported cloud providers offer encryption at rest
   ```
   # In .env
   ENABLE_ENCRYPTION_AT_REST=true
   ```

2. **Data in Transit**: All API calls to cloud providers use HTTPS
   ```
   # In .env
   ENFORCE_HTTPS=true
   ```

### Access Control

Follow the principle of least privilege:

1. **IAM Roles/Policies**: Grant only required permissions
   ```
   # Example AWS IAM policy (minimal permissions)
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "dynamodb:GetItem",
           "dynamodb:PutItem",
           "dynamodb:UpdateItem",
           "dynamodb:DeleteItem",
           "dynamodb:Query",
           "dynamodb:Scan"
         ],
         "Resource": "arn:aws:dynamodb:*:*:table/customerai-*"
       }
     ]
   }
   ```

2. **Scoped Credentials**: Use separate credentials for different environments
   ```
   # In .env.production
   AWS_ACCESS_KEY_ID=production-key
   AWS_SECRET_ACCESS_KEY=production-secret
   
   # In .env.development
   AWS_ACCESS_KEY_ID=development-key
   AWS_SECRET_ACCESS_KEY=development-secret
   ```

### Sensitive Data Handling

1. **Anonymization**: Use the built-in anonymization for PII before storage
   ```python
   from privacy.anonymizer import DataAnonymizer
   
   anonymizer = DataAnonymizer()
   customer_data = {
       "name": "John Smith",
       "email": "john.smith@example.com",
       "ssn": "123-45-6789",
       "preferences": {
           "marketing_emails": True
       }
   }
   
   # Anonymize PII
   anonymized = anonymizer.anonymize_data(customer_data)
   
   # Store anonymized data
   db_client.create_item("customers", anonymized)
   ```

2. **Field-Level Encryption**: Encrypt sensitive fields before storage
   ```python
   from src.utils.security import encrypt_field
   
   customer_data["ssn"] = encrypt_field(customer_data["ssn"])
   db_client.create_item("customers", customer_data)
   ```

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   ```
   ERROR: Access denied when connecting to cloud provider
   ```
   
   **Solutions:**
   - Verify credentials in `.env` file
   - Check IAM permissions
   - Ensure proper region configuration
   - Verify network connectivity to the cloud provider

2. **Resource Not Found**
   ```
   ERROR: Table/collection not found
   ```
   
   **Solutions:**
   - Verify table/collection name
   - Check if table exists: `db_client.table_exists("table_name")`
   - Create table if needed: `db_client.create_table(...)`
   - Verify correct region/account configuration

3. **Query Performance Issues**
   ```
   WARNING: Slow query detected
   ```
   
   **Solutions:**
   - Review index strategy
   - Optimize query conditions
   - Consider caching for frequent queries
   - Check for full table scans

### Logging

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('cloud.database').setLevel(logging.DEBUG)
```

Or in `.env`:
```
LOG_LEVEL=DEBUG
DB_QUERY_LOGGING=true
```

### Diagnostics

Run diagnostics to test database connectivity and performance:

```bash
python scripts/db_diagnostics.py --provider aws --test-connectivity
python scripts/db_diagnostics.py --provider aws --test-performance
```

## Migration Guide

### Migrating Between Providers

To migrate from one cloud provider to another:

1. **Setup target provider**: Configure credentials for the new provider
2. **Create tables**: Ensure tables/collections exist in the target provider
3. **Migrate data**: Use the migration utility to transfer data

```python
from cloud.migration import DatabaseMigration

# Initialize migration tool with source and target providers
migration = DatabaseMigration(
    source_provider="aws",
    target_provider="azure"
)

# Migrate a specific table
migration.migrate_table(
    source_table="customers",
    target_table="customers",
    batch_size=100  # Process in batches of 100 items
)

# Or migrate all tables
migration.migrate_all_tables()
```

### Schema Evolution

As your data model evolves, update your database schema:

1. **AWS DynamoDB**: Update GSIs/LSIs as needed
2. **Azure Cosmos DB**: Schema-less, but update indexes if needed
3. **Google Firestore**: Schema-less, but update indexes if needed

### Backup and Restore

Always back up data before migrations:

```python
from cloud.database.backup import DatabaseBackup

# Initialize backup tool
backup = DatabaseBackup(provider="aws")

# Back up a table
backup.create_backup(
    table_name="customers",
    backup_name="customers-pre-migration"
)

# Restore from backup if needed
backup.restore_from_backup(
    backup_name="customers-pre-migration",
    target_table="customers-restored"
)
```

## Advanced Usage

### Transactions

Some operations can be performed in transactions (provider-dependent):

```python
# Example with AWS DynamoDB
from cloud.database.aws import AWSDynamoDBClient

aws_client = factory.get_database_client("aws")
if isinstance(aws_client, AWSDynamoDBClient):
    # Start transaction
    with aws_client.transaction():
        # Operations in this block are part of the transaction
        aws_client.update_item("accounts", {"id": "account1"}, {"balance": 500})
        aws_client.update_item("accounts", {"id": "account2"}, {"balance": 1500})
```

### Custom Queries

For provider-specific advanced queries:

```python
# Example with DynamoDB PartiQL
from cloud.database.aws import AWSDynamoDBClient

aws_client = factory.get_database_client("aws")
if isinstance(aws_client, AWSDynamoDBClient):
    result = aws_client.execute_statement(
        "SELECT * FROM customers WHERE region = 'east' AND status = 'active'"
    )
```

### Data Pagination

For large result sets:

```python
# Paginated query
query = {
    "filter_expression": "status = :status",
    "expression_attribute_values": {":status": "active"}
}

# Get first page
page1 = db_client.query_items("customers", query, limit=100)

# If there are more results, get the next page
if page1.get("last_evaluated_key"):
    page2 = db_client.query_items(
        "customers", 
        query, 
        limit=100,
        exclusive_start_key=page1["last_evaluated_key"]
    )
``` 