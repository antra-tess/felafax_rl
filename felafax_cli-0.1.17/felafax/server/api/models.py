from fastapi import APIRouter, HTTPException, Depends, status
from typing import List
from ..common import get_storage_provider
from ...core.storage.base import StorageProvider
from ..models.model import ModelMetadata
from ..handlers.model import ModelHandler
from ..common import handle_error
import httpx
router = APIRouter(prefix="/models", tags=["models"])

@router.get("/{user_id}/list", response_model=List[ModelMetadata])
async def list_models(
    user_id: str,
    storage_provider: StorageProvider = Depends(get_storage_provider)
):
    """List all models for a user"""
    try:
        handler = ModelHandler(storage_provider, user_id)
        return await handler.list_models()
    except Exception as e:
        handle_error(e)

@router.get("/{user_id}/{model_id}/download")
async def download_model(
    user_id: str,
    model_id: str,
    storage_provider: StorageProvider = Depends(get_storage_provider)
):
    """Get download URL for model weights"""
    # Check if model exists
    handler = ModelHandler(storage_provider, user_id)
    model = await handler.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Generate download URL for weights directory
    try:
        # Here you would implement logic to generate a signed URL or download token
        download_url = await handler.get_download_url(model_id)
        return {"download_url": download_url}
    except Exception as e:
        handle_error(e)

@router.delete("/{user_id}/{model_id}")
async def delete_model(
    user_id: str,
    model_id: str,
    storage_provider: StorageProvider = Depends(get_storage_provider)
):
    try:    
        """Delete a model and its resources"""
        # Update catalog first
        handler = ModelHandler(storage_provider, user_id)
        await handler.delete_model(model_id)
        
        return {"status": "deleted"}
    except Exception as e:
        handle_error(e)

@router.get("/{user_id}/{model_id}/info")
async def get_model_info(
    user_id: str,
    model_id: str,
    storage_provider: StorageProvider = Depends(get_storage_provider)
):
    """Get detailed model information including metadata"""
    try:
        handler = ModelHandler(storage_provider, user_id)
        model = await handler.get_model_info(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        return model
    except Exception as e:
        handle_error(e)

@router.post("/{user_id}/{model_id}/init-chat")
async def initialize_chat(
    user_id: str,
    model_id: str,
    storage_provider: StorageProvider = Depends(get_storage_provider)
):
    """Initialize chat session with a model"""
    try:
        handler = ModelHandler(storage_provider, user_id)
        accelerator_id = await handler.init_chat(model_id)
        return {"accelerator_id": accelerator_id}
    except Exception as e:
        handle_error(e)

@router.post("/{user_id}/{model_id}/chat/{accelerator_id}")
async def chat_with_model(
    user_id: str,
    model_id: str,
    accelerator_id: str,
    messages: dict,
    storage_provider: StorageProvider = Depends(get_storage_provider)
):
    """Chat with a specific model using an existing accelerator"""
    try:
        handler = ModelHandler(storage_provider, user_id)
        response = await handler.chat(accelerator_id, messages["messages"])
        return {"response": response}
    except httpx.HTTPStatusError as e:
        if e.response and e.response.status_code == 503:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is still warming up. Please wait 5 mins and try again."
            )
        handle_error(e)
    except Exception as e:
        handle_error(e)
