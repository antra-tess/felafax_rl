from fastapi import APIRouter, HTTPException, Request
from typing import Dict, List
from datetime import datetime

from ..handlers.user import UserHandler
from ..common import get_storage_provider, handle_error
from ..models.model import ModelPaths
from ..models.dataset import DatasetStoragePaths
from ..models.finetune import FineTuneStoragePaths
from ..models.accelerator import AcceleratorStoragePaths
from ..models.user import UserPaths
from pydantic import BaseModel, EmailStr
from typing import Optional
import logging

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger(__name__)

# Models for request/response
class UserAuth(BaseModel):
    email: str
    name: Optional[str] = None
    user_id: Optional[str] = None

class TokenAuth(BaseModel):
    token: str

class UserResponse(BaseModel):
    user_id: str

class TokenResponse(BaseModel):
    token: str

class LoginResponse(BaseModel):
    user_id: str
    token: str

@router.post("/create_user", response_model=UserResponse)
async def create_user(auth: UserAuth):
    try:
        if auth.user_id:
            user_id = auth.user_id
        else:
            user_id = await UserHandler.create_user_id(auth.email)
        
        storage_client = await get_storage_provider()
        if await UserHandler(storage_client, user_id).user_exists():
            return {"user_id": user_id}

        # Create user in storage
        storage_client = await get_storage_provider()
        user_handler = UserHandler(storage_client, user_id)
        await user_handler.create_new_user(auth.email, auth.name)
        
        logger.info(f"Successfully created user with ID: {user_id}")
        return {"user_id": user_id}
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise handle_error(e)

@router.post("/create_token", response_model=TokenResponse)
async def create_token(user_id: str):
    try:
        storage_client = await get_storage_provider()
        user_handler = UserHandler(storage_client, user_id)
        
        # Verify user exists
        user_info = await user_handler.get_user_info()
        if not user_info:
            raise HTTPException(status_code=404, detail="User not found")
            
        token = await user_handler.create_token()
        return {"token": token}
    except Exception as e:
        raise handle_error(e)
    
@router.get("/get_tokens", response_model=List[str])
async def get_tokens(user_id: str):
    # curl -X GET "http://localhost:8000/auth/get_tokens?user_id=b4c9a289323b"
    try:
        storage_client = await get_storage_provider()
        user_handler = UserHandler(storage_client, user_id)
        return await user_handler.get_tokens()
    except Exception as e:
        raise handle_error(e)
    
@router.delete("/delete_token")
async def delete_token(user_id: str, token: str):
    try:
        storage_client = await get_storage_provider()
        user_handler = UserHandler(storage_client, user_id)
        await user_handler.remove_token(token)
    except Exception as e:
        raise handle_error(e)
 

    
@router.post("/login", response_model=LoginResponse)
async def login(auth: TokenAuth):
    try:
        storage_client = await get_storage_provider()
        user_id = await UserHandler.get_user_id_with_token(auth.token, storage_client)
        
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        return {"user_id": user_id, "token": auth.token}
    except HTTPException:
        raise
    except Exception as e:
        raise handle_error(e)

@router.post("/reset")
async def reset(user_id: str):
    """Reset all user data and metadata"""
    try:
        storage_client = await get_storage_provider()
        
        # Get handlers
        user_handler = UserHandler(storage_client, user_id)
        
        # Verify user exists
        if not await user_handler.user_exists():
            raise HTTPException(status_code=404, detail="User not found")
            
        # Delete all metadata files
        metadata_paths = [
            ModelPaths.metadata_path(user_id),
            DatasetStoragePaths.metadata_path(user_id),
            FineTuneStoragePaths.metadata_path(user_id),
            AcceleratorStoragePaths.metadata_path(user_id),
            # No need to delete user metadata
            # UserPaths.metadata_path(user_id)
        ]
        
        # Delete base directories
        base_paths = [
            ModelPaths.base_path(user_id),
            DatasetStoragePaths.base_path(user_id),
            FineTuneStoragePaths.base_path(user_id),
            AcceleratorStoragePaths.base_path(user_id)
        ]
        
        # Delete metadata files
        for path in metadata_paths:
            if await storage_client.file_exists(path):
                await storage_client.delete_file(path)
                logger.info(f"Deleted metadata file: {path}")
                
        # Delete base directories
        for path in base_paths:
            if await storage_client.directory_exists(path):
                await storage_client.delete_directory(path, convert_iterator=True)
                logger.info(f"Deleted directory: {path}")
                
        # Remove user's tokens from token index
        user_info = await user_handler.get_user_info()
        if user_info and user_info.tokens:
            for token in user_info.tokens:
                await user_handler.remove_token(token)
                
        return {"status": "success", "message": "All user data has been reset"}
        
    except Exception as e:
        logger.error(f"Error resetting user data: {str(e)}")
        raise handle_error(e)
