from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Form
from typing import List
from pydantic import BaseModel
from ..common import get_storage_provider
from ...core.storage.base import StorageProvider
from fastapi import Depends
from ..handlers.dataset import DatasetHandler
from ..models.dataset import DatasetMetadata

router = APIRouter(prefix="/datasets", tags=["datasets"])

@router.post("/{user_id}/upload")
async def upload_dataset(
    user_id: str, 
    file: UploadFile = File(...),
    storage_provider: StorageProvider = Depends(get_storage_provider)
):
    handler = DatasetHandler(storage_provider, user_id)
    content = await file.read()
    return await handler.upload_dataset(content, file.filename)

@router.post("/{user_id}/upload/chunked")
async def upload_chunked_dataset(
    user_id: str,
    upload_id: str = Form(...),
    chunk_number: int = Form(...),
    total_chunks: int = Form(...),
    file: UploadFile = File(...),
    storage_provider: StorageProvider = Depends(get_storage_provider)
):
    handler = DatasetHandler(storage_provider, user_id)
    content = await file.read()
    chunk_info = {
        "upload_id": upload_id,
        "chunk_number": chunk_number,
        "total_chunks": total_chunks
    }
    return await handler.upload_chunk(chunk_info, content, file.filename)

@router.get("/{user_id}/list", response_model=List[DatasetMetadata])
async def list_datasets(
    user_id: str,
    storage_provider: StorageProvider = Depends(get_storage_provider)
):
    handler = DatasetHandler(storage_provider, user_id)
    return await handler.list_datasets()

@router.delete("/{user_id}/{dataset_id}")
async def delete_dataset(
    user_id: str,
    dataset_id: str,
    storage_provider: StorageProvider = Depends(get_storage_provider)
):
    handler = DatasetHandler(storage_provider, user_id)
    try:
        return await handler.delete_dataset(dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")