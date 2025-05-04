"""
Cloud services API endpoints for CustomerAI.

This module provides API endpoints for interacting with cloud services
such as storage, database, and AI/ML.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from cloud.config import CloudConfig, CloudProvider
from cloud.factory import CloudServiceFactory

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/cloud", tags=["cloud"])

# Initialize cloud services
cloud_config = CloudConfig()
cloud_factory = CloudServiceFactory(cloud_config)

# Models for API requests/responses
class StorageItemInfo(BaseModel):
    key: str
    size: Optional[int] = None
    last_modified: Optional[str] = None
    content_type: Optional[str] = None
    url: Optional[str] = None

class TableItem(BaseModel):
    id: str
    attributes: Dict[str, Any]

class CreateItemRequest(BaseModel):
    table_name: str
    item: Dict[str, Any]

class QueryRequest(BaseModel):
    table_name: str
    query: Dict[str, Any]
    limit: Optional[int] = 50

class StorageListResponse(BaseModel):
    success: bool
    items: List[StorageItemInfo]
    count: int
    prefix: Optional[str] = None
    provider: str

class DatabaseListResponse(BaseModel):
    success: bool
    tables: List[str]
    count: int
    provider: str

class AIModelsResponse(BaseModel):
    success: bool
    models: List[Dict[str, Any]]
    count: int
    provider: str

class CloudProviderInfo(BaseModel):
    name: str
    configured: bool
    is_current: bool
    services: Dict[str, bool]

class CloudProvidersResponse(BaseModel):
    current_provider: str
    providers: List[CloudProviderInfo]

# Helper to get storage client
def get_storage_client(provider: Optional[str] = None):
    """Get storage client for the specified provider."""
    if provider:
        config = CloudConfig(provider)
        factory = CloudServiceFactory(config)
    else:
        config = cloud_config
        factory = cloud_factory

    client = factory.get_storage_client()
    if not client:
        raise HTTPException(
            status_code=503,
            detail=f"Storage service not available for provider: {config.get_provider_name()}"
        )
    return client, config.get_provider_name()

# Helper to get database client
def get_database_client(provider: Optional[str] = None):
    """Get database client for the specified provider."""
    if provider:
        config = CloudConfig(provider)
        factory = CloudServiceFactory(config)
    else:
        config = cloud_config
        factory = cloud_factory

    client = factory.get_database_client()
    if not client:
        raise HTTPException(
            status_code=503,
            detail=f"Database service not available for provider: {config.get_provider_name()}"
        )
    return client, config.get_provider_name()

# Helper to get AI client
def get_ai_client(provider: Optional[str] = None):
    """Get AI client for the specified provider."""
    if provider:
        config = CloudConfig(provider)
        factory = CloudServiceFactory(config)
    else:
        config = cloud_config
        factory = cloud_factory

    client = factory.get_ai_client()
    if not client:
        raise HTTPException(
            status_code=503,
            detail=f"AI service not available for provider: {config.get_provider_name()}"
        )
    return client, config.get_provider_name()

# Provider information endpoint
@router.get("/providers", response_model=CloudProvidersResponse)
async def get_providers_info():
    """Get information about available cloud providers."""
    current_provider = cloud_config.get_provider_name()

    providers = []
    for provider in CloudProvider:
        if provider != CloudProvider.NONE:
            # Check provider configuration
            config = CloudConfig(provider)
            
            # Check if services are available for this provider
            factory = CloudServiceFactory(config)
            storage_available = factory.get_storage_client() is not None
            database_available = factory.get_database_client() is not None
            ai_available = factory.get_ai_client() is not None
            
            providers.append(CloudProviderInfo(
                name=provider.value,
                configured=config.is_configured(),
                is_current=provider.value == current_provider,
                services={
                    "storage": storage_available,
                    "database": database_available,
                    "ai": ai_available
                }
            ))
    
    return CloudProvidersResponse(
        current_provider=current_provider,
        providers=providers
    )

# Storage endpoints
@router.get("/storage/list", response_model=StorageListResponse)
async def list_storage_files(prefix: str = "", provider: Optional[str] = None):
    """List files in cloud storage."""
    client, provider_name = get_storage_client(provider)
    
    items = client.list_files(prefix)
    
    # Convert to response format
    result_items = []
    for item in items:
        # Handle different provider response formats
        if "key" in item:
            # AWS S3 format
            key = item["key"]
        elif "name" in item:
            # Azure Blob and GCP Storage format
            key = item["name"]
        else:
            # Unknown format
            continue
        
        result_items.append(StorageItemInfo(
            key=key,
            size=item.get("size"),
            last_modified=str(item.get("last_modified", "")),
            content_type=item.get("content_type")
        ))
    
    return StorageListResponse(
        success=True,
        items=result_items,
        count=len(result_items),
        prefix=prefix,
        provider=provider_name
    )

@router.post("/storage/upload")
async def upload_file(
    file: UploadFile = File(...),
    path: str = Form(...),
    make_public: bool = Form(False),
    provider: Optional[str] = None
):
    """Upload a file to cloud storage."""
    client, provider_name = get_storage_client(provider)
    
    # Save file to temporary location
    temp_file_path = f"tmp/{file.filename}"
    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
    
    try:
        contents = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(contents)
        
        # Upload to cloud storage
        result = client.upload_file(
            temp_file_path,
            path,
            public=make_public,
            content_type=file.content_type
        )
        
        return result
    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.get("/storage/url/{path:path}")
async def get_file_url(
    path: str,
    expiration: int = 3600,
    provider: Optional[str] = None
):
    """Get a temporary URL for a file in cloud storage."""
    client, provider_name = get_storage_client(provider)
    
    url = client.get_file_url(path, expiration)
    
    return {
        "success": True,
        "url": url,
        "expiration": expiration,
        "provider": provider_name
    }

@router.delete("/storage/{path:path}")
async def delete_file(path: str, provider: Optional[str] = None):
    """Delete a file from cloud storage."""
    client, provider_name = get_storage_client(provider)
    
    result = client.delete_file(path)
    
    return result

# Database endpoints
@router.get("/database/tables", response_model=DatabaseListResponse)
async def list_tables(provider: Optional[str] = None):
    """List tables in the cloud database."""
    client, provider_name = get_database_client(provider)
    
    tables = client.list_tables()
    
    return DatabaseListResponse(
        success=True,
        tables=tables,
        count=len(tables),
        provider=provider_name
    )

@router.post("/database/tables/{table_name}")
async def create_table(
    table_name: str,
    key_schema: Dict[str, Any],
    provider: Optional[str] = None
):
    """Create a table in the cloud database."""
    client, provider_name = get_database_client(provider)
    
    result = client.create_table(table_name, key_schema)
    
    return result

@router.delete("/database/tables/{table_name}")
async def delete_table(table_name: str, provider: Optional[str] = None):
    """Delete a table from the cloud database."""
    client, provider_name = get_database_client(provider)
    
    result = client.delete_table(table_name)
    
    return result

@router.post("/database/item")
async def create_item(request: CreateItemRequest, provider: Optional[str] = None):
    """Create an item in the cloud database."""
    client, provider_name = get_database_client(provider)
    
    result = client.create_item(request.table_name, request.item)
    
    return result

@router.post("/database/query")
async def query_items(request: QueryRequest, provider: Optional[str] = None):
    """Query items from the cloud database."""
    client, provider_name = get_database_client(provider)
    
    items = client.query_items(
        request.table_name,
        request.query,
        limit=request.limit
    )
    
    return {
        "success": True,
        "items": items,
        "count": len(items),
        "provider": provider_name
    }

# AI/ML endpoints
@router.get("/ai/models", response_model=AIModelsResponse)
async def list_models(provider: Optional[str] = None):
    """List deployed AI models."""
    client, provider_name = get_ai_client(provider)
    
    models = client.list_models()
    
    return AIModelsResponse(
        success=True,
        models=models,
        count=len(models),
        provider=provider_name
    )

@router.post("/ai/predict/{model_name}")
async def predict(
    model_name: str,
    data: Dict[str, Any],
    provider: Optional[str] = None
):
    """Make a prediction using a deployed model."""
    client, provider_name = get_ai_client(provider)
    
    result = client.predict(data, model_name)
    
    return result 