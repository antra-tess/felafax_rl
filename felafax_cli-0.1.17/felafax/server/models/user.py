from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, List

class UserMetadata(BaseModel):
    """User metadata schema"""
    user_id: str
    email: str
    created_at: datetime
    status: str
    last_login: datetime
    tokens: List[str]
    settings: Optional[Dict] = {}
    name: Optional[str] = None


class UserPaths:
    """User storage path generator"""
    
    @staticmethod
    def base_path(user_id: str) -> str:
        return f"users/{user_id}"
    
    @staticmethod
    def metadata_path(user_id: str) -> str:
        return f"users/{user_id}/metadata/users.json"
