from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict
from pydantic import BaseModel
from ..common import get_storage_provider, handle_error
from ...core.storage.base import StorageProvider
from ..handlers.accelerator import AcceleratorHandler
from ..models.accelerator import AcceleratorMetadata, AcceleratorRequest

router = APIRouter(prefix="/accelerators", tags=["accelerators"])

    
class AcceleratorUpdateRequest(BaseModel):
    """Request model for updating accelerator configuration"""
    docker_image: Optional[str] = None
    custom_config: Optional[dict] = None

@router.post("/{user_id}/start", response_model=AcceleratorMetadata)
async def start_accelerator(
    user_id: str,
    request: AcceleratorRequest,
    storage_provider: StorageProvider = Depends(get_storage_provider)
):
    """Start a new accelerator instance"""
    try:
        handler = AcceleratorHandler(storage_provider, user_id)
        # Create and start accelerator
        request.tags = ["allow-ssh-tpu-vm"]
        metadata = await handler.create_accelerator(
            request=request,
            wait_for_ready=request.wait
        )
        
        return metadata
    except Exception as e:
        handle_error(e)

@router.post("/{user_id}/{accelerator_id}/stop")
async def stop_accelerator(
    user_id: str,
    accelerator_id: str,
    storage_provider: StorageProvider = Depends(get_storage_provider)
):
    """Stop an accelerator instance"""
    try:
        handler = AcceleratorHandler(storage_provider, user_id)
        await handler.stop_accelerator(accelerator_id)
        return {"status": "stopping", "message": "Accelerator is being stopped"}
    except Exception as e:
        handle_error(e)

@router.get("/{user_id}/{accelerator_id}/status", response_model=AcceleratorMetadata)
async def get_accelerator_status(
    user_id: str,
    accelerator_id: str,
    storage_provider: StorageProvider = Depends(get_storage_provider)
):
    """Get current status of an accelerator"""
    try:
        handler = AcceleratorHandler(storage_provider, user_id)
        metadata = await handler._metadata_handler.get(accelerator_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Accelerator not found")
        return metadata
    except Exception as e:
        handle_error(e)

@router.get("/{user_id}/list", response_model=List[AcceleratorMetadata])
async def list_accelerators(
    user_id: str,
    storage_provider: StorageProvider = Depends(get_storage_provider)
):
    """List all accelerators for a user"""
    try:
        handler = AcceleratorHandler(storage_provider, user_id)
        return await handler._metadata_handler.list()
    except Exception as e:
        handle_error(e)

@router.post("/{user_id}/{accelerator_id}/update")
async def update_accelerator(
    user_id: str,
    accelerator_id: str,
    request: AcceleratorUpdateRequest,
    storage_provider: StorageProvider = Depends(get_storage_provider)
):
    """Update accelerator configuration"""
    try:
        handler = AcceleratorHandler(storage_provider, user_id)
        metadata = await handler._metadata_handler.get(accelerator_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Accelerator not found")
            
        # Update configuration
        if request.docker_image:
            metadata.config["docker_image"] = request.docker_image
        if request.custom_config:
            metadata.config.update(request.custom_config)
            
        # Save updated metadata
        await handler._metadata_handler.update(metadata)
        return metadata
    except Exception as e:
        handle_error(e)

@router.delete("/{user_id}/{accelerator_id}")
async def delete_accelerator(
    user_id: str,
    accelerator_id: str,
    storage_provider: StorageProvider = Depends(get_storage_provider)
):
    """Delete an accelerator and its resources"""
    try:
        handler = AcceleratorHandler(storage_provider, user_id)
        
        # Stop accelerator first
        await handler.stop_accelerator(accelerator_id)
        
        # Delete metadata
        await handler._metadata_handler.remove(accelerator_id)
        
        return {"status": "deleted", "message": "Accelerator deleted successfully"}
    except Exception as e:
        handle_error(e)

@router.post("/{user_id}/{accelerator_id}/run-command")
async def run_command(
    user_id: str,
    accelerator_id: str,
    command: List[str],
    storage_provider: StorageProvider = Depends(get_storage_provider)
):
    """Run a command on the accelerator"""
    try:
        handler = AcceleratorHandler(storage_provider, user_id)
        metadata = await handler._metadata_handler.get(accelerator_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Accelerator not found")
            
        # Initialize provider
        provider = handler._provider_map[metadata.provider]()
        await provider.initialize(metadata.config)
        
        # Run command
        result = await provider.run_command(command)
        return result
    except Exception as e:
        handle_error(e)
