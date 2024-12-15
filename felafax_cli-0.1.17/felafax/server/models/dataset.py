from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, List
from felafax.config import Config

class DatasetMetadata(BaseModel):
    """Dataset metadata schema"""
    dataset_id: str
    name: str
    file_name: str
    file_path: str
    full_file_path: str
    created_at: datetime
    size_bytes: int
    format: str
    stats: Optional[Dict] = {}
    
class DatasetStoragePaths:
    """Dataset storage path generator"""
    
    @staticmethod
    def base_path(user_id: str) -> str:
        return f"users/{user_id}/data"
    
    @staticmethod
    def dataset_dir_path(user_id: str, dataset_id: str, include_bucket: bool = False) -> str:
        if include_bucket:
            return f"gs://{Config.GCS_BUCKET_NAME}/{DatasetStoragePaths.base_path(user_id)}/{dataset_id}"
        else:
            return f"{DatasetStoragePaths.base_path(user_id)}/{dataset_id}"
    
    @staticmethod
    def metadata_path(user_id: str) -> str:
        return f"users/{user_id}/metadata/datasets.json"
    