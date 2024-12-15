from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, List, Union
from enum import Enum
from felafax.config import Config

class FinetuneStatus(BaseModel):
    status: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    tune_id: Optional[str] = None
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    progress: Optional[float] = None
    error: Optional[str] = None

class FinetuneMetadata(BaseModel):
    """Tune metadata schema"""
    tune_id: str
    model_id: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_path: Optional[str] = None
    base_model: str
    status: str
    accelerator_id: str
    created_at: datetime
    updated_at: datetime

class FineTuneRequest(BaseModel):
    model_name: str
    config: dict
    dataset_id: Optional[str] = None
    hf_dataset_path: Optional[str] = None

# Configuration for fine-tuning
class DataConfig(BaseModel):
    batch_size: int = Field(default=16, description="Batch size for training")
    max_seq_length: int = Field(default=2048, description="Maximum sequence length")
    data_source: str = Field(default="", description="Path to the dataset")
    dataset_input_field: str = Field(default="instruction", description="Input field in the dataset")
    dataset_output_field: str = Field(default="output", description="Output field in the dataset")


class TrainerConfig(BaseModel):
    model_name: str = Field(..., description="Name of the model to fine-tune")
    param_dtype: str = Field(default="bfloat16", description="Parameter data type")
    compute_dtype: str = Field(default="bfloat16", description="Compute data type")
    num_epochs: int = Field(default=1, description="Number of training epochs")
    num_steps: int = Field(default=5, description="Number of training steps")
    mesh_shape: Optional[List[int]] = Field(default=None, description="Mesh shape for parallel training")
    learning_rate: float = Field(default=1e-3, description="Learning rate")
    lora_rank: int = Field(default=16, description="LoRA rank")
    use_lora: bool = Field(default=True, description="Whether to use LoRA")
    log_interval: int = Field(default=5, description="Logging interval")
    eval_interval: int = Field(default=5, description="Evaluation interval")
    eval_steps: int = Field(default=10, description="Number of evaluation steps")

class CheckpointerConfig(BaseModel):
    save_interval_steps: int = Field(default=100, description="Steps between checkpoints")

class InternalConfig(BaseModel):
    """Internal configuration for fine-tuning paths and settings"""
    base_storage_path: str = Field(default="/mnt/persistent-disk", description="Base storage path for fine-tuning")
    trainer_dir: str = Field(default="/mnt/persistent-disk/train", description="Training directory path")
    export_dir: str = Field(default="/mnt/persistent-disk/hf_export", description="Export directory path")
    dataset_path: str = Field(default="/mnt/persistent-disk/dataset.jsonl", description="Path to the dataset")
    model_gcs_path: str = Field(default="", description="GCS path to the model weights")
    dataset_gcs_path: str = Field(default="", description="GCS path to the dataset")
    tune_id: str = Field(default="", description="Fine-tuning job ID")  
    user_id: str = Field(default="", description="User ID")

class FineTuneConfig(BaseModel):
    """Trainer configuration"""
    trainer_dir: str = Field(default="/mnt/persistent-disk/train", description="Training directory path")
    export_dir: str = Field(default="/mnt/persistent-disk/hf_export", description="Export directory path")
    hf_token: str = Field(default="", description="Hugging Face token")
    hf_repo: str = Field(default="", description="Hugging Face repository")
    hf_model_download_token: str = Field(default="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", description="Hugging Face model download token")
    test_mode: bool = Field(default=True, description="Whether to run in test mode")
    data_config: DataConfig
    trainer_config: TrainerConfig
    checkpointer_config: CheckpointerConfig

class FineTuneStoragePaths:
    """Fine-tune storage path generator"""
    
    @staticmethod
    def base_path(user_id: str) -> str:
        return f"users/{user_id}/finetunes"
    
    @staticmethod
    def tune_path(user_id: str, tune_id: str, include_bucket: bool = False) -> str:
        if include_bucket:
            return f"gs://{Config.GCS_BUCKET_NAME}/{FineTuneStoragePaths.base_path(user_id)}/{tune_id}"
        else:
            return f"{FineTuneStoragePaths.base_path(user_id)}/{tune_id}"
    
    @staticmethod
    def user_config_path(user_id: str, tune_id: str) -> str:
        return f"{FineTuneStoragePaths.tune_path(user_id, tune_id)}/config.yml"
    
    @staticmethod
    def trainer_config_path(user_id: str, tune_id: str) -> str:
        return f"{FineTuneStoragePaths.tune_path(user_id, tune_id)}/trainer_config.yml"

    @staticmethod
    def internal_config_path(user_id: str, tune_id: str) -> str:
        return f"{FineTuneStoragePaths.tune_path(user_id, tune_id)}/internal_config.yml"

    @staticmethod
    def log_path(user_id: str, tune_id: str) -> str:
        return f"{FineTuneStoragePaths.tune_path(user_id, tune_id)}/training.log"

    @staticmethod
    def stdout_path(user_id: str, tune_id: str) -> str:
        return f"{FineTuneStoragePaths.tune_path(user_id, tune_id)}/debug.log"
    
    @staticmethod
    def status_path(user_id: str, tune_id: str) -> str:
        return f"{FineTuneStoragePaths.tune_path(user_id, tune_id)}/status.json"

    @staticmethod
    def checkpoints_path(user_id: str, tune_id: str) -> str:
        return f"{FineTuneStoragePaths.tune_path(user_id, tune_id)}/checkpoints"

    @staticmethod
    def metadata_path(user_id: str) -> str:
        return f"users/{user_id}/metadata/finetunes.json"
    
