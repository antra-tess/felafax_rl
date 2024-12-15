from typing import List, TypeVar, Generic, Type
from pydantic import BaseModel
from ..common import StorageProvider
from typing import Optional
import asyncio
from abc import ABC, abstractmethod
import os
import logging
from ..models.job import JobMetadata
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

class ListMetadataHandler(Generic[T]):
    """Handler for catalog operations with list-based storage"""
    def __init__(self, model_class: Type[T], catalog_path: str, catalog_id_field: str, storage_provider: StorageProvider):
        self.model_class = model_class
        self.catalog_path = catalog_path
        self.catalog_id_field = catalog_id_field
        self.storage = storage_provider
        
    async def initialize(self):
        # create directory if not exists
        directory = os.path.dirname(self.catalog_path)
        if not await self.storage.directory_exists(directory):
            await self.storage.create_directory(directory)

        # create file if not exists
        if not await self.storage.file_exists(self.catalog_path):
            await self.storage.write_json(self.catalog_path, {})
        

    async def add(self, item: T) -> None:
        catalog = {f"{self.catalog_id_field}": []}
        if await self.storage.file_exists(self.catalog_path):
            catalog = await self.storage.read_json(self.catalog_path)
        
        if self.catalog_id_field not in catalog:
            catalog[self.catalog_id_field] = []
        catalog[self.catalog_id_field].append(item.dict())
        await self.storage.write_json(self.catalog_path, catalog)
    
    async def remove(self, item_id: str) -> None:
        try:
            if not await self.storage.file_exists(self.catalog_path):
                return None
            
            catalog = await self.storage.read_json(self.catalog_path)
            if self.catalog_id_field not in catalog:
                return None
            
            catalog[self.catalog_id_field] = [item for item in catalog[self.catalog_id_field] if item[self.catalog_id_field] != item_id]
            await self.storage.write_json(self.catalog_path, catalog)
        except Exception as e:
            # Log the error if you have logging set up
            logger.error(f"Error in remove operation: {str(e)}")
            return None
    
    async def list(self) -> List[T]:
        if not await self.storage.file_exists(self.catalog_path):
            return []
        catalog = await self.storage.read_json(self.catalog_path)
        if self.catalog_id_field not in catalog:
            return []
        return [self.model_class(**item) for item in catalog[self.catalog_id_field]]
    
    async def get(self, item_id: str) -> Optional[T]:
        catalog = await self.list()
        for item in catalog:
            if getattr(item, self.catalog_id_field) == item_id:
                return item
        return None

    
    async def update(self, item: T) -> None:
        await self.remove(getattr(item, self.catalog_id_field))
        await self.add(item)

class DictModel(ABC):
    """Abstract base class for models that support dictionary-like operations"""
    @abstractmethod
    def get_value(self, key: str) -> Optional[str]:
        pass
    
    @abstractmethod
    def set_value(self, key: str, value: str) -> None:
        pass
    
    @abstractmethod
    def remove_value(self, key: str) -> None:
        pass

class DictBaseModel(BaseModel, DictModel):
    """Base class combining BaseModel and DictModel"""
    pass

M = TypeVar('M', bound=DictBaseModel)
class DictMetadataHandler(Generic[M]):
    """Handler for catalog operations with dictionary-based storage"""
    def __init__(self, model_class: Type[M], catalog_path: str, storage_provider: StorageProvider):
        self.model_class = model_class
        self.catalog_path = catalog_path
        self.storage = storage_provider

    async def set(self, key: str, value: str) -> None:
        data: M = await self.get_all()
        data.set_value(key, value)
        await self.storage.write_json(self.catalog_path, data.dict())

    async def remove(self, key: str) -> None:
        data: M = await self.get_all()
        data.remove_value(key)
        await self.storage.write_json(self.catalog_path, data.dict())

    async def get(self, key: str) -> Optional[str]:
        data: M = await self.get_all()
        return data.get_value(key)

    async def get_all(self) -> M:
        if not await self.storage.file_exists(self.catalog_path):
            return self.model_class()
        stored_data = await self.storage.read_json(self.catalog_path)
        return self.model_class(**stored_data)

class JobProcessor(ABC):
    @abstractmethod
    async def process_update(self, job: JobMetadata) -> JobMetadata:
        """Process job update"""
        pass
    
    @abstractmethod
    async def process_cleanup(self, job: JobMetadata) -> None:
        """Process job cleanup"""
        pass

    @abstractmethod
    async def handle_error(self, job: JobMetadata, error: Exception) -> JobMetadata:
        """Handle job error"""
        pass

class JobConfig:
    def __init__(
        self,
        update_interval: int,  # minutes
        cleanup_delay: int,    # minutes
        max_retries: int = 3
    ):
        self.update_interval = update_interval
        self.cleanup_delay = cleanup_delay
        self.max_retries = max_retries