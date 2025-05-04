"""
Example script demonstrating how to use cloud storage services in CustomerAI.

This example shows how to upload, download, and list files using different cloud providers.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cloud.config import CloudConfig, CloudProvider
from cloud.factory import CloudServiceFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def upload_test_file(storage_client, test_file_path="examples/test_data/sample.csv"):
    """Upload a test file to cloud storage."""
    
    # Create test file if it doesn't exist
    os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
    if not os.path.exists(test_file_path):
        logger.info(f"Creating test file: {test_file_path}")
        with open(test_file_path, "w") as f:
            f.write("id,name,value\n")
            f.write("1,sample1,10.5\n")
            f.write("2,sample2,20.3\n")
            f.write("3,sample3,15.7\n")
    
    # Upload file
    remote_path = "examples/sample.csv"
    logger.info(f"Uploading test file to: {remote_path}")
    result = storage_client.upload_file(test_file_path, remote_path)
    
    if result.get("success", False):
        logger.info(f"Upload successful: {result}")
    else:
        logger.error(f"Upload failed: {result}")
    
    return remote_path, result

def download_test_file(storage_client, remote_path, local_path="examples/test_data/downloaded.csv"):
    """Download a file from cloud storage."""
    
    # Download file
    logger.info(f"Downloading file from: {remote_path} to {local_path}")
    result = storage_client.download_file(remote_path, local_path)
    
    if result.get("success", False):
        logger.info(f"Download successful: {result}")
        
        # Verify file contents
        if os.path.exists(local_path):
            with open(local_path, "r") as f:
                content = f.read()
                logger.info(f"Downloaded file content:\n{content}")
    else:
        logger.error(f"Download failed: {result}")
    
    return result

def list_files(storage_client, prefix="examples/"):
    """List files in cloud storage."""
    
    # List files
    logger.info(f"Listing files with prefix: {prefix}")
    files = storage_client.list_files(prefix)
    
    if files:
        logger.info(f"Found {len(files)} files:")
        for file in files:
            logger.info(f"  - {file}")
    else:
        logger.info(f"No files found with prefix: {prefix}")
    
    return files

def delete_test_file(storage_client, remote_path):
    """Delete a file from cloud storage."""
    
    # Delete file
    logger.info(f"Deleting file: {remote_path}")
    result = storage_client.delete_file(remote_path)
    
    if result.get("success", False):
        logger.info(f"Deletion successful: {result}")
    else:
        logger.error(f"Deletion failed: {result}")
    
    return result

def run_example(provider=None):
    """Run the cloud storage example with the specified provider."""
    
    # Create cloud configuration
    config = CloudConfig(provider)
    provider_name = config.get_provider_name()
    
    logger.info(f"Running cloud storage example with provider: {provider_name}")
    
    # Check if provider is configured
    if not config.is_configured():
        logger.error(f"Provider {provider_name} is not properly configured.")
        logger.error("Please set the required environment variables.")
        return
    
    # Create cloud service factory and get storage client
    factory = CloudServiceFactory(config)
    storage_client = factory.get_storage_client()
    
    if not storage_client:
        logger.error(f"Failed to create storage client for provider: {provider_name}")
        return
    
    # Run example operations
    try:
        # 1. Upload test file
        remote_path, upload_result = upload_test_file(storage_client)
        
        if not upload_result.get("success", False):
            return
        
        # 2. Get file URL
        url = storage_client.get_file_url(remote_path)
        logger.info(f"File URL: {url}")
        
        # 3. Check if file exists
        exists = storage_client.file_exists(remote_path)
        logger.info(f"File exists: {exists}")
        
        # 4. Get file metadata
        metadata = storage_client.get_file_metadata(remote_path)
        logger.info(f"File metadata: {metadata}")
        
        # 5. List files
        list_files(storage_client)
        
        # 6. Download test file
        download_test_file(storage_client, remote_path)
        
        # 7. Delete test file
        delete_test_file(storage_client, remote_path)
        
        # 8. Verify deletion
        exists = storage_client.file_exists(remote_path)
        logger.info(f"File exists after deletion: {exists}")
        
        # 9. Create folder
        folder_path = "examples/test_folder"
        folder_result = storage_client.create_folder(folder_path)
        logger.info(f"Folder creation result: {folder_result}")
        
        logger.info(f"Example completed successfully with provider: {provider_name}")
        
    except Exception as e:
        logger.exception(f"Error during example execution: {str(e)}")

if __name__ == "__main__":
    # Determine provider from command line argument or environment
    import argparse
    parser = argparse.ArgumentParser(description="Cloud Storage Example")
    parser.add_argument("--provider", choices=["aws", "azure", "gcp"], 
                      help="Cloud provider to use (aws, azure, gcp)")
    args = parser.parse_args()
    
    # Run example with specified provider
    run_example(args.provider) 