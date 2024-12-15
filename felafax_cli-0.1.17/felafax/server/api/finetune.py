from fastapi import APIRouter, HTTPException, Depends, Header
from typing import List
from ...core.storage.base import StorageProvider
from ..common import get_storage_provider
from ..models.finetune import (
    FineTuneRequest, 
    FinetuneStatus, 
    FinetuneMetadata
)
from ..handlers.finetune import FineTuneHandler
from ..common import handle_error
from fastapi.responses import StreamingResponse
import google.api_core.exceptions

router = APIRouter(prefix="/fine-tune", tags=["fine-tune"])


@router.post("/{user_id}/start")
async def start_fine_tune(
    user_id: str, 
    request: FineTuneRequest,
    storage_provider: StorageProvider = Depends(get_storage_provider)
):
    try:
        handler = FineTuneHandler(storage_provider, user_id)
        
        # # Validate dataset exists
        if request.dataset_id is not None and not await handler.validate_dataset(request.dataset_id):
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Start fine-tuning
        tune_id, status, message = await handler.start_finetune(request)
        
        return {
            "tune_id": tune_id,
            "status": status,
            "message": message
        }

    except Exception as e:
        handle_error(e)

@router.get("/{user_id}/{tune_id}/status", response_model=FinetuneStatus)
async def get_tune_status(
    user_id: str, 
    tune_id: str,
    storage_provider: StorageProvider = Depends(get_storage_provider)
):
    try:
        handler = FineTuneHandler(storage_provider, user_id)
        return await handler.get_status(tune_id)
    except Exception as e:
        handle_error(e)

@router.post("/{user_id}/{tune_id}/stop")
async def stop_fine_tune(
    user_id: str,
    tune_id: str,
    storage_provider: StorageProvider = Depends(get_storage_provider)
):
    try:
        handler = FineTuneHandler(storage_provider, user_id)
        tune_id, status, message = await handler.stop_finetune(tune_id)
        
        return {
            "tune_id": tune_id,
            "status": status,
            "message": message
        }

    except Exception as e:
        handle_error(e)

@router.get("/{user_id}/list", response_model=List[FinetuneMetadata])
async def list_finetune_jobs(
    user_id: str,
    storage_provider: StorageProvider = Depends(get_storage_provider)
):
    handler = FineTuneHandler(storage_provider, user_id)
    return await handler.get_all_jobs()

@router.get("/{user_id}/{tune_id}/log")
async def stream_tune_logs(
    user_id: str,
    tune_id: str,
    range_header: str = Header(None, alias="Range"),
    storage_provider: StorageProvider = Depends(get_storage_provider)
):
    """Stream debug logs for a fine-tuning job"""
    try:
        handler = FineTuneHandler(storage_provider, user_id)
        
        # Validate tune exists
        if not await handler.check_job_exists(tune_id):
            raise HTTPException(status_code=404, detail="Tune not found")
        
        try:
            # Get file size first to validate range
            file_size = await handler.get_log_file_size(tune_id)
        except google.api_core.exceptions.NotFound:
            return StreamingResponse(
                iter(["No logs available yet"]),
                media_type="text/plain",
                headers={"Accept-Ranges": "bytes"}
            )
        
        # Parse range header if present
        start_byte = 0
        if range_header:
            try:
                range_str = range_header.split('=')[1]
                start_byte = int(range_str.split('-')[0])
            except (IndexError, ValueError):
                raise HTTPException(status_code=400, detail="Invalid Range header")
        
        # If range is beyond file size, return 416
        if start_byte >= file_size:
            raise HTTPException(
                status_code=416,
                detail="Requested range not satisfiable",
                headers={"Content-Range": f"bytes */{file_size}"}
            )
            
        return StreamingResponse(
            handler.stream_logs(tune_id, start_byte=start_byte),
            media_type="text/plain",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes {start_byte}-{file_size-1}/{file_size}"
            }
        )
        
    except FileNotFoundError:
        return StreamingResponse(
            iter(["No file yet"]),
            media_type="text/plain",
            headers={"Accept-Ranges": "bytes"}
        )
    except HTTPException as e:
        # Re-raise HTTP exceptions (including 416)
        raise
    except Exception as e:
        handle_error(e)