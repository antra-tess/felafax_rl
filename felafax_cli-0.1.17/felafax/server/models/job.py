from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CLEANUP = "cleanup"

class JobMetadata(BaseModel):
    job_id: str
    job_type: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    next_update_at: Optional[datetime]
    cleanup_at: Optional[datetime]
    data: Dict[str, Any]
    error: Optional[str] = None

class JobStoragePaths:
    @staticmethod
    def metadata_path() -> str:
        return f"generic/metadata/jobs.json" 
