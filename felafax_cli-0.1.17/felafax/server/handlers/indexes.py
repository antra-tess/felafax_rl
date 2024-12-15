from typing import Optional
from ..models.indexes import TokenIndex, IndexPaths
from ..handlers.base import DictMetadataHandler
from ...core.storage.base import StorageProvider

class TokenIndexHandler:
    def __init__(self, storage_provider: StorageProvider):
        self.storage = storage_provider
        self._metadata_handler = DictMetadataHandler(
            TokenIndex,
            IndexPaths.token_index_path(),
            self.storage
        )
    
    async def get_all(self) -> TokenIndex:
        """Get or create token index"""
        return await self._metadata_handler.get_all()
    
    async def add_token(self, token: str, user_id: str) -> None:
        """Add token to user_id mapping"""
        await self._metadata_handler.set(token, user_id)
    
    async def remove_token(self, token: str) -> None:
        """Remove token mapping"""
        await self._metadata_handler.remove(token)
    
    async def get_user_id(self, token: str) -> Optional[str]:
        """Get user_id for token"""
        return await self._metadata_handler.get(token) 