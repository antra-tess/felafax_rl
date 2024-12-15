from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, List
from felafax.config import Config

class ModelMetadata(BaseModel):
    """Model metadata schema"""
    model_id: str
    base_model: str
    created_at: datetime
    updated_at: datetime
    status: str  # e.g., "training", "ready", "failed"
    description: Optional[str] = None
    config: Dict = {}

class ModelPaths:
    """Model storage path generator"""
    
    @staticmethod
    def base_path(user_id: str) -> str:
        return f"users/{user_id}/models"
    
    @staticmethod
    def model_base_path(user_id: str, model_id: str) -> str:
        return f"{ModelPaths.base_path(user_id)}/{model_id}"
    
    @staticmethod
    def weights_path(user_id: str, model_id: str) -> str:
        return f"{ModelPaths.model_base_path(user_id, model_id)}/weights"
    
    @staticmethod
    def job_path(user_id: str, model_id: str, include_bucket: bool = False) -> str:
        if include_bucket:
            return f"gs://{Config.GCS_BUCKET_NAME}/{ModelPaths.model_base_path(user_id, model_id)}/job"
        else:
            return f"{ModelPaths.model_base_path(user_id, model_id)}/job"
    
    @staticmethod
    def config_path(user_id: str, model_id: str) -> str:
        return f"{ModelPaths.model_base_path(user_id, model_id)}/config.json"
    
    @staticmethod
    def status_path(user_id: str, model_id: str) -> str:
        return f"{ModelPaths.model_base_path(user_id, model_id)}/status.json"
    
    @staticmethod
    def log_path(user_id: str, model_id: str) -> str:
        return f"{ModelPaths.model_base_path(user_id, model_id)}/stdout.log"

    @staticmethod
    def metadata_path(user_id: str) -> str:
        return f"users/{user_id}/metadata/models.json"
    
    @staticmethod
    def model_path(user_id: str, model_id: str, include_bucket: bool = False) -> str:
        """Get the relative GCS path for a model"""
        if include_bucket:
            return f"gs://{Config.GCS_BUCKET_NAME}/{ModelPaths.model_base_path(user_id, model_id)}"
        else:
            return f"{ModelPaths.model_base_path(user_id, model_id)}"
    