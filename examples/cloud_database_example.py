"""
Example script demonstrating how to use cloud database services in CustomerAI.

This example shows how to create, query, update, and delete items in cloud database services.
"""

import logging
import os
import sys
import time
import uuid

from dotenv import load_dotenv

from src.utils.date_provider import DateProvider

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cloud.config import CloudConfig
from cloud.factory import CloudServiceFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def create_table(db_client, table_name="customerai_example"):
    """Create a table/collection in the database."""

    logger.info(f"Creating table: {table_name}")

    # Define key schema (provider-specific)
    # For AWS DynamoDB:
    key_schema = {
        "key_schema": [
            {"AttributeName": "id", "KeyType": "HASH"},  # Partition key
            {"AttributeName": "created_at", "KeyType": "RANGE"},  # Sort key
        ],
        "attribute_definitions": [
            {"AttributeName": "id", "AttributeType": "S"},
            {"AttributeName": "created_at", "AttributeType": "S"},
        ],
    }

    # For other providers, different schema formats may be needed
    result = db_client.create_table(table_name, key_schema)

    if result.get("success", False):
        logger.info(f"Table created successfully: {result}")
    else:
        logger.warning(f"Table creation result: {result}")

    return result


def create_items(db_client, table_name="customerai_example", count=5):
    """Create sample items in the database."""

    logger.info(f"Creating {count} sample items in table: {table_name}")

    items = []
    for i in range(count):
        item_id = str(uuid.uuid4())
        created_at = DateProvider.get_instance().iso_format()

        item = {
            "id": item_id,
            "created_at": created_at,
            "name": f"Sample Item {i+1}",
            "value": i * 10.5,
            "is_active": i % 2 == 0,
            "tags": [f"tag{j}" for j in range(1, (i % 3) + 2)],
        }

        logger.info(f"Creating item: {item_id}")
        result = db_client.create_item(table_name, item)

        if result.get("success", False):
            logger.info("Item created successfully")
            items.append(item)
        else:
            logger.error(f"Error creating item: {result}")

    return items


def get_item(db_client, item_id, created_at, table_name="customerai_example"):
    """Get an item from the database."""

    logger.info(f"Getting item with ID: {item_id}")

    # Define key based on table schema
    key = {"id": item_id, "created_at": created_at}

    item = db_client.get_item(table_name, key)

    if item:
        logger.info(f"Item retrieved: {item}")
    else:
        logger.warning(f"Item not found with ID: {item_id}")

    return item


def update_item(db_client, item_id, created_at, updates, table_name="customerai_example"):
    """Update an item in the database."""

    logger.info(f"Updating item with ID: {item_id}")

    # Define key based on table schema
    key = {"id": item_id, "created_at": created_at}

    result = db_client.update_item(table_name, key, updates)

    if result.get("success", False):
        logger.info(f"Item updated successfully: {result}")
    else:
        logger.error(f"Error updating item: {result}")

    return result


def query_items(db_client, table_name="customerai_example"):
    """Query items from the database."""

    logger.info(f"Querying items from table: {table_name}")

    # Define query (provider-specific)
    # For AWS DynamoDB:
    query = {
        "filter_expression": "is_active = :active",
        "expression_attribute_values": {":active": True},
    }

    items = db_client.query_items(table_name, query)

    logger.info(f"Query returned {len(items)} items")
    for item in items:
        logger.info(f"  - {item.get('id')}: {item.get('name')}")

    return items


def delete_item(db_client, item_id, created_at, table_name="customerai_example"):
    """Delete an item from the database."""

    logger.info(f"Deleting item with ID: {item_id}")

    # Define key based on table schema
    key = {"id": item_id, "created_at": created_at}

    result = db_client.delete_item(table_name, key)

    if result.get("success", False):
        logger.info(f"Item deleted successfully: {result}")
    else:
        logger.error(f"Error deleting item: {result}")

    return result


def delete_table(db_client, table_name="customerai_example"):
    """Delete a table from the database."""

    logger.info(f"Deleting table: {table_name}")

    result = db_client.delete_table(table_name)

    if result.get("success", False):
        logger.info(f"Table deleted successfully: {result}")
    else:
        logger.warning(f"Table deletion result: {result}")

    return result


def run_example(provider=None):
    """Run the cloud database example with the specified provider."""

    # Create cloud configuration
    config = CloudConfig(provider)
    provider_name = config.get_provider_name()

    logger.info(f"Running cloud database example with provider: {provider_name}")

    # Check if provider is configured
    if not config.is_configured():
        logger.error(f"Provider {provider_name} is not properly configured.")
        logger.error("Please set the required environment variables.")
        return

    # Create cloud service factory and get database client
    factory = CloudServiceFactory(config)
    db_client = factory.get_database_client()

    if not db_client:
        logger.error(f"Failed to create database client for provider: {provider_name}")
        return

    # Define table name
    table_name = f"customerai_example_{int(time.time())}"

    # Run example operations
    try:
        # 1. List existing tables
        logger.info("Listing existing tables:")
        tables = db_client.list_tables()
        for table in tables:
            logger.info(f"  - {table}")

        # 2. Create table
        create_result = create_table(db_client, table_name)

        if not create_result.get("success", False):
            logger.error("Failed to create table, aborting example")
            return

        # Wait for table to be ready
        logger.info("Waiting for table to be fully active...")
        time.sleep(5)

        # 3. Create sample items
        items = create_items(db_client, table_name)

        if not items:
            logger.error("Failed to create any items, aborting example")
            return

        # 4. Get a specific item
        sample_item = items[0]
        item = get_item(db_client, sample_item["id"], sample_item["created_at"], table_name)

        # 5. Update an item
        if item:
            updates = {
                "name": f"{item['name']} (Updated)",
                "last_updated": DateProvider.get_instance().iso_format(),
                "counter": 42,
            }
            update_item(db_client, item["id"], item["created_at"], updates, table_name)

            # Verify update
            updated_item = get_item(db_client, item["id"], item["created_at"], table_name)
            logger.info(f"Updated item: {updated_item}")

        # 6. Query items
        query_items(db_client, table_name)

        # 7. Delete an item
        if len(items) > 1:
            delete_item_sample = items[1]
            delete_item(
                db_client,
                delete_item_sample["id"],
                delete_item_sample["created_at"],
                table_name,
            )

        # 8. Clean up - delete table
        delete_table(db_client, table_name)

        logger.info(f"Example completed successfully with provider: {provider_name}")

    except Exception as e:
        logger.exception(f"Error during example execution: {str(e)}")

        # Attempt to clean up
        try:
            delete_table(db_client, table_name)
        except Exception:
            pass


if __name__ == "__main__":
    # Determine provider from command line argument or environment
    import argparse

    parser = argparse.ArgumentParser(description="Cloud Database Example")
    parser.add_argument(
        "--provider",
        choices=["aws", "azure", "gcp"],
        help="Cloud provider to use (aws, azure, gcp)",
    )
    args = parser.parse_args()

    # Run example with specified provider
    run_example(args.provider)
