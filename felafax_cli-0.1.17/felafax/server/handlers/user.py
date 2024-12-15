from ...core.storage.base import StorageProvider
from typing import Dict
from ..models.user import UserPaths
from ..models.finetune import FineTuneStoragePaths
from ..models.dataset import DatasetStoragePaths
from ..models.model import ModelPaths
from ..models.accelerator import AcceleratorStoragePaths
from ..models.indexes import IndexPaths
from ..handlers.base import ListMetadataHandler
from ..models.user import UserMetadata
from ..handlers.indexes import TokenIndexHandler
from datetime import datetime
from typing import Optional, List
import hashlib
import secrets
class UserHandler:
    def __init__(self, storage_provider: StorageProvider, user_id: str):
        self.storage = storage_provider
        self.user_id = user_id
        self._metadata_handler = ListMetadataHandler(UserMetadata, UserPaths.metadata_path(user_id), "user_id", self.storage)
        self._token_index_handler = TokenIndexHandler(self.storage)

    async def get_user_info(self) -> Optional[UserMetadata]:
        """Get user metadata"""
        return await self._metadata_handler.get(self.user_id)

    async def user_exists(self) -> bool:
        """Check if a user exists"""
        return await self._metadata_handler.get(self.user_id) is not None

    async def update_user_info(self, info: Dict) -> None:
        """Update user metadata"""
        metadata = await self._metadata_handler.get(self.user_id)
        metadata.update(info)
        await self._metadata_handler.update(metadata)
    
    @staticmethod
    async def create_user_id(email: str) -> str:
        """Create a user ID from an email"""
        user_id = hashlib.sha256(email.encode()).hexdigest()[:12]
        return user_id
    
    @staticmethod
    async def get_user_id_with_token(token: str, storage_client: StorageProvider) -> Optional[str]:
        """Find a user ID from a token"""
        token_index_handler = TokenIndexHandler(storage_client)
        return await token_index_handler.get_user_id(token)
    
    async def create_new_user(self, email: str, name: Optional[str] = None) -> None:
        """Create a new user"""
        await self.ensure_user_paths()

        user_info = UserMetadata(
            user_id=self.user_id,
            email=email,
            created_at=datetime.utcnow().isoformat(),
            status="active",
            last_login=datetime.utcnow().isoformat(),
            tokens=[],
            name=name
        )
        await self._metadata_handler.add(user_info)

        return self.user_id
    async def create_token(self) -> str:
        """Create a new token for the user"""
        token = secrets.token_hex(8)
        await self.add_token(token)
        return token
    
    async def add_token(self, token: str) -> None:
        """Add a token to the user"""
        user_info = await self.get_user_info()
        if user_info is None:
            raise ValueError("User not found")
        user_info.tokens.append(token)
        await self._token_index_handler.add_token(token, self.user_id)
        await self._metadata_handler.update(user_info)
        
    async def remove_token(self, token: str) -> None:
        """Remove a token from the user"""
        user_info = await self.get_user_info()
        if user_info is None:
            raise Exception("User not found")
        user_info.tokens.remove(token)
        await self._token_index_handler.remove_token(token)
        await self.update_user_info(user_info)
    
    async def validate_token(self, token: str) -> bool:
        """Validate a token"""
        return await self._token_index_handler.get_user_id(token) == self.user_id

    async def ensure_user_paths(self) -> None:
        """Ensure all required user paths exist"""
        paths = [
            UserPaths.base_path(self.user_id),
            DatasetStoragePaths.base_path(self.user_id),
            FineTuneStoragePaths.base_path(self.user_id),
            ModelPaths.base_path(self.user_id),
            AcceleratorStoragePaths.base_path(self.user_id),
        ]
        for path in paths:
            await self.storage.create_directory(path)
        
    async def get_tokens(self) -> List[str]:
        """Get all tokens for the user"""
        user_info = await self.get_user_info()
        return user_info.tokens if user_info else []
    
