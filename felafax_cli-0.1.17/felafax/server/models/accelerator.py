from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any, List
from ...core.accelerators.base import AcceleratorStatus

class AcceleratorRequest(BaseModel):
    """Request model for creating accelerator"""
    type: str = "tpu"  # Accelerator type
    accelerator_type: str = "v3"  # v3 or v5p
    accelerator_core_count: int = 8  # 8, 16, 32
    framework: str = "jax"  # jax or pytorch-xla or vllm or None
    attach_disk: bool = False
    auto_shutdown: Optional[int] = 3  # 1-6 hours
    name: Optional[str] = None
    zone: Optional[str] = None
    disk_size_gb: Optional[int] = 1000
    custom_config: Optional[dict] = None
    docker_env: Optional[Dict[str, str]] = None
    docker_image: Optional[str] = None
    ssh_key: Optional[str] = None
    preemptible: Optional[bool] = True
    tags: Optional[List[str]] = None
    wait: bool = False

class AcceleratorMetadata(BaseModel):
    """Accelerator metadata schema"""

    accelerator_id: str
    name: str
    provider: str
    created_at: datetime
    updated_at: datetime
    ip_address: Optional[str] = None
    status: AcceleratorStatus
    config: Optional[Dict[str, Any]] = {}
    auto_shutdown_hrs: Optional[int] = 3
    tags: List[str] = []

    class Config:
        arbitrary_types_allowed = True


class AcceleratorCommandResult(BaseModel):
    """Command execution result schema"""

    returncode: int
    stdout: str
    stderr: str


class AcceleratorStoragePaths:
    """Accelerator storage path generator"""

    @staticmethod
    def base_path(user_id: str) -> str:
        return f"users/{user_id}/accelerators"

    @staticmethod
    def metadata_path(user_id: str) -> str:
        return f"users/{user_id}/metadata/accelerators.json"

