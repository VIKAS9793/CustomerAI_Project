"""
Integration tests for cloud database functionality in CustomerAI platform.

These tests verify the functionality of cloud database operations
across different providers.

Usage:
    pytest tests/integration/test_cloud_database.py --provider aws
    pytest tests/integration/test_cloud_database.py --provider azure
    pytest tests/integration/test_cloud_database.py --provider gcp
"""

import logging
import os
import sys
import time
import uuid

import pytest

from src.utils.date_provider import DateProvider

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cloud.config import CloudConfig
from cloud.factory import CloudServiceFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test table name - with timestamp to avoid conflicts
TEST_TABLE = f"customerai_test_{int(time.time())}"


# Test fixture for provider selection
def pytest_addoption(parser):
    parser.addoption(
        "--provider",
        default="aws",
        choices=["aws", "azure", "gcp"],
        help="Cloud provider to test: aws, azure, gcp",
    )


@pytest.fixture
def provider(request):
    return request.config.getoption("--provider")


@pytest.fixture
def cloud_config(provider):
    """Create cloud configuration for tests."""
    return CloudConfig(provider)


@pytest.fixture
def db_client(cloud_config):
    """Create database client for tests."""
    factory = CloudServiceFactory(cloud_config)
    client = factory.get_database_client()

    if not client:
        pytest.skip(
            f"Database client not available for provider: {cloud_config.get_provider_name()}"
        )

    if not cloud_config.is_configured():
        pytest.skip(f"Provider {cloud_config.get_provider_name()} is not properly configured")

    return client


@pytest.fixture
def test_table(db_client):
    """Create test table and clean up after tests."""
    # Define key schema based on provider
    provider_name = db_client.__class__.__name__

    if "AWS" in provider_name:
        # AWS DynamoDB schema
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
    else:
        # Simpler schema for other providers
        key_schema = {
            "key_schema": [{"AttributeName": "id", "KeyType": "HASH"}],
            "attribute_definitions": [{"AttributeName": "id", "AttributeType": "S"}],
        }

    # Create the table
    logger.info(f"Creating test table: {TEST_TABLE}")
    result = db_client.create_table(TEST_TABLE, key_schema)

    if not result.get("success", False):
        pytest.skip(f"Failed to create test table: {result}")

    # Wait for table to be ready
    logger.info("Waiting for table to be ready...")
    time.sleep(5)

    yield TEST_TABLE

    # Clean up - delete the table
    logger.info(f"Cleaning up - deleting test table: {TEST_TABLE}")
    db_client.delete_table(TEST_TABLE)


def test_table_exists(db_client, test_table):
    """Test checking if a table exists."""
    # Table should exist
    assert db_client.table_exists(test_table)

    # Non-existent table should not exist
    assert not db_client.table_exists(f"nonexistent_{uuid.uuid4()}")


def test_list_tables(db_client, test_table):
    """Test listing tables."""
    tables = db_client.list_tables()
    assert isinstance(tables, list)
    assert test_table in tables


def test_get_table_info(db_client, test_table):
    """Test getting table information."""
    info = db_client.get_table_info(test_table)
    assert isinstance(info, dict)
    assert len(info) > 0


def test_create_and_get_item(db_client, test_table):
    """Test creating and retrieving an item."""
    # Create a test item
    item_id = str(uuid.uuid4())
    created_at = DateProvider.get_instance().iso_format()

    item = {
        "id": item_id,
        "created_at": created_at,
        "name": "Test Item",
        "value": 42,
        "is_active": True,
        "tags": ["test", "integration"],
    }

    # Create the item
    result = db_client.create_item(test_table, item)
    assert result.get("success", False)

    # Get the item
    key = {"id": item_id, "created_at": created_at}
    retrieved_item = db_client.get_item(test_table, key)

    # Verify item contents
    assert retrieved_item
    assert retrieved_item["id"] == item_id
    assert retrieved_item["name"] == "Test Item"
    assert retrieved_item["value"] == 42
    assert retrieved_item["is_active"] is True
    assert "test" in retrieved_item["tags"]


def test_update_item(db_client, test_table):
    """Test updating an item."""
    # Create a test item
    item_id = str(uuid.uuid4())
    created_at = DateProvider.get_instance().iso_format()

    item = {
        "id": item_id,
        "created_at": created_at,
        "name": "Original Name",
        "value": 100,
    }

    # Create the item
    db_client.create_item(test_table, item)

    # Update the item
    key = {"id": item_id, "created_at": created_at}
    updates = {"name": "Updated Name", "value": 200, "new_field": "New Value"}

    result = db_client.update_item(test_table, key, updates)
    assert result.get("success", False)

    # Get the updated item
    updated_item = db_client.get_item(test_table, key)

    # Verify updates
    assert updated_item["name"] == "Updated Name"
    assert updated_item["value"] == 200
    assert updated_item["new_field"] == "New Value"


def test_delete_item(db_client, test_table):
    """Test deleting an item."""
    # Create a test item
    item_id = str(uuid.uuid4())
    created_at = DateProvider.get_instance().iso_format()

    item = {"id": item_id, "created_at": created_at, "name": "Item to Delete"}

    # Create the item
    db_client.create_item(test_table, item)

    # Verify item exists
    key = {"id": item_id, "created_at": created_at}
    assert db_client.get_item(test_table, key)

    # Delete the item
    result = db_client.delete_item(test_table, key)
    assert result.get("success", False)

    # Verify item no longer exists
    assert not db_client.get_item(test_table, key)


def test_batch_write(db_client, test_table):
    """Test batch writing items."""
    # Create multiple items
    items = []
    for i in range(5):
        items.append(
            {
                "id": f"batch-{uuid.uuid4()}",
                "created_at": DateProvider.get_instance().iso_format(),
                "index": i,
                "name": f"Batch Item {i}",
            }
        )

    # Batch write items
    result = db_client.batch_write(test_table, items)
    assert result.get("success", False)

    # Verify items were created
    for item in items:
        key = {"id": item["id"], "created_at": item["created_at"]}
        retrieved = db_client.get_item(test_table, key)
        assert retrieved
        assert retrieved["name"] == item["name"]
        assert retrieved["index"] == item["index"]


def test_query_items(db_client, test_table):
    """Test querying items."""
    # Create items with a common prefix
    query_id_prefix = f"query-{uuid.uuid4()}"

    # Create items with different statuses
    items = []
    for i in range(10):
        status = "active" if i % 2 == 0 else "inactive"
        items.append(
            {
                "id": f"{query_id_prefix}-{i}",
                "created_at": DateProvider.get_instance().iso_format(),
                "index": i,
                "status": status,
            }
        )

    # Batch write items
    db_client.batch_write(test_table, items)

    # Query for active items
    query = {
        "filter_expression": "status = :status",
        "expression_attribute_values": {":status": "active"},
    }

    active_items = db_client.query_items(test_table, query)

    # Verify filtered results
    active_count = sum(1 for item in items if item["status"] == "active")
    filtered_count = len([item for item in active_items if item["id"].startswith(query_id_prefix)])

    # We may get other active items from other tests, so check that we got at least ours
    assert filtered_count >= active_count


def test_conditional_operations(db_client, test_table):
    """Test conditional operations."""
    # Create an item
    item_id = str(uuid.uuid4())
    created_at = DateProvider.get_instance().iso_format()

    item = {"id": item_id, "created_at": created_at, "status": "new", "value": 50}

    db_client.create_item(test_table, item)
    key = {"id": item_id, "created_at": created_at}

    # Conditional update that should succeed
    updates_1 = {"status": "processing", "value": 100}
    result_1 = db_client.update_item(
        test_table,
        key,
        updates_1,
        condition_expression="status = :old_status",
        expression_attribute_values={":old_status": "new"},
    )

    assert result_1.get("success", False)

    # Verify update
    updated_item = db_client.get_item(test_table, key)
    assert updated_item["status"] == "processing"
    assert updated_item["value"] == 100

    # Conditional update that should fail
    updates_2 = {"status": "completed", "value": 200}
    result_2 = db_client.update_item(
        test_table,
        key,
        updates_2,
        condition_expression="status = :old_status",
        expression_attribute_values={":old_status": "new"},  # Status is now "processing"
    )

    # This should fail due to condition
    assert not result_2.get("success", True)

    # Verify no change
    unchanged_item = db_client.get_item(test_table, key)
    assert unchanged_item["status"] == "processing"  # Still "processing"
    assert unchanged_item["value"] == 100  # Still 100


def test_pagination(db_client, test_table):
    """Test pagination of query results."""
    # Create many items
    prefix = f"page-{uuid.uuid4()}"
    items = []

    # Create 25 items
    for i in range(25):
        items.append(
            {
                "id": f"{prefix}-{i:03d}",
                "created_at": DateProvider.get_instance().iso_format(),
                "value": i,
            }
        )

    # Batch write items
    db_client.batch_write(test_table, items)

    # Query with small page size
    query = {
        "filter_expression": "begins_with(id, :prefix)",
        "expression_attribute_values": {":prefix": prefix},
    }

    # First page
    page1 = db_client.query_items(test_table, query, limit=10)
    assert len(page1) <= 10

    # Count total items matching our prefix
    all_items = db_client.query_items(test_table, query)
    matching_count = len([item for item in all_items if item["id"].startswith(prefix)])

    # Should be the number we created
    assert matching_count == 25


def test_complex_data_types(db_client, test_table):
    """Test handling of complex data types."""
    item_id = str(uuid.uuid4())
    created_at = DateProvider.get_instance().iso_format()

    # Create item with nested structures
    item = {
        "id": item_id,
        "created_at": created_at,
        "name": "Complex Item",
        "nested_object": {
            "key1": "value1",
            "key2": 42,
            "key3": True,
            "nested_again": {"deep_key": "deep_value"},
        },
        "string_array": ["one", "two", "three"],
        "number_array": [1, 2, 3, 4, 5],
        "mixed_array": ["string", 42, True, {"object_in_array": "value"}],
    }

    # Create the item
    result = db_client.create_item(test_table, item)
    assert result.get("success", False)

    # Get the item
    key = {"id": item_id, "created_at": created_at}
    retrieved = db_client.get_item(test_table, key)

    # Verify complex types
    assert retrieved["nested_object"]["key1"] == "value1"
    assert retrieved["nested_object"]["nested_again"]["deep_key"] == "deep_value"
    assert "two" in retrieved["string_array"]
    assert 3 in retrieved["number_array"]
    assert "string" in retrieved["mixed_array"]

    # Test updating nested attributes
    updates = {
        "nested_object": {
            "key1": "updated_value",
            "new_key": "new_value",
            "nested_again": {
                "deep_key": "updated_deep_value",
                "another_deep_key": "another_value",
            },
        }
    }

    update_result = db_client.update_item(test_table, key, updates)
    assert update_result.get("success", False)

    # Get updated item
    updated = db_client.get_item(test_table, key)

    # Verify updates to nested structures
    assert updated["nested_object"]["key1"] == "updated_value"
    assert updated["nested_object"]["new_key"] == "new_value"
    assert updated["nested_object"]["nested_again"]["deep_key"] == "updated_deep_value"
    assert updated["nested_object"]["nested_again"]["another_deep_key"] == "another_value"


def test_error_handling(db_client, test_table):
    """Test error handling in database operations."""
    # Attempt to get non-existent item
    nonexistent_key = {
        "id": f"nonexistent-{uuid.uuid4()}",
        "created_at": DateProvider.get_instance().iso_format(),
    }
    non_existent_item = db_client.get_item(test_table, nonexistent_key)

    # Should return empty dict, not error
    assert isinstance(non_existent_item, dict)
    assert not non_existent_item

    # Attempt to update non-existent item
    update_result = db_client.update_item(test_table, nonexistent_key, {"status": "updated"})

    # Should indicate failure but not raise exception
    assert not update_result.get("success", True)

    # Attempt to delete non-existent item
    delete_result = db_client.delete_item(test_table, nonexistent_key)

    # Should indicate failure but not raise exception
    assert not delete_result.get("success", True)

    # Attempt to access non-existent table
    nonexistent_table = f"nonexistent_table_{uuid.uuid4()}"

    # These should handle the error gracefully
    assert not db_client.table_exists(nonexistent_table)
    assert not db_client.get_table_info(nonexistent_table)
    assert [] == db_client.query_items(nonexistent_table, {})


if __name__ == "__main__":
    # Manually run tests
    import sys

    provider_name = sys.argv[1] if len(sys.argv) > 1 else "aws"

    # Setup
    config = CloudConfig(provider_name)
    factory = CloudServiceFactory(config)
    db = factory.get_database_client()

    # Run a single test
    table_name = TEST_TABLE

    try:
        # Create test table
        key_schema = {
            "key_schema": [
                {"AttributeName": "id", "KeyType": "HASH"},
                {"AttributeName": "created_at", "KeyType": "RANGE"},
            ],
            "attribute_definitions": [
                {"AttributeName": "id", "AttributeType": "S"},
                {"AttributeName": "created_at", "AttributeType": "S"},
            ],
        }
        db.create_table(table_name, key_schema)

        # Run test_create_and_get_item
        test_create_and_get_item(db, table_name)
        print("Test passed: test_create_and_get_item")

    finally:
        # Clean up
        db.delete_table(table_name)
