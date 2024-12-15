from ...core.storage.base import StorageProvider
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import uuid
from ..models.dataset import DatasetMetadata, DatasetStoragePaths
from ..handlers.base import ListMetadataHandler
from ..common import generate_random_name


class DatasetHandler:
    def __init__(self, storage_provider: StorageProvider, user_id: str):
        self.storage = storage_provider
        self.user_id = user_id
        self._metadata_handler = ListMetadataHandler(DatasetMetadata, DatasetStoragePaths.metadata_path(user_id), "dataset_id", self.storage)
        
    async def validate_dataset(self, dataset_id: str) -> bool:
        """Validate dataset exists"""
        path = DatasetStoragePaths.dataset_dir_path(self.user_id, dataset_id)
        return await self.storage.directory_exists(path)

    async def upload_dataset(self, file_content: bytes, filename: str) -> DatasetMetadata:
        """Upload a complete dataset file"""
        # dataset_id = f"dataset_{uuid.uuid4().hex[:12]}"
        dataset_id = f"dataset-{generate_random_name()}"
        file_format = filename.split('.')[-1]
        
        # Create temporary file and upload
        temp_path = Path(f"/tmp/{dataset_id}_{filename}")
        try:
            with temp_path.open("wb") as f:
                f.write(file_content)
            
            raw_path = f"{DatasetStoragePaths.dataset_dir_path(self.user_id, dataset_id)}/{filename}"
            await self.storage.upload_file(temp_path, raw_path)
            
            # Create metadata
            metadata = await self._create_dataset_metadata(dataset_id, filename, temp_path)
            await self._metadata_handler.update(metadata)
            
            return metadata
        finally:
            temp_path.unlink(missing_ok=True)

    async def upload_chunk(self, chunk_info: Dict, file_content: bytes, filename: str) -> Dict:
        """Handle chunked upload of dataset"""
        # Create local temp directory for chunks if first chunk
        chunk_dir = Path(f"/tmp/upload_{chunk_info['upload_id']}")
        if chunk_info['chunk_number'] == 0:
            chunk_dir.mkdir(exist_ok=True)
        
        # Save chunk locally
        chunk_path = chunk_dir / f"chunk_{chunk_info['chunk_number']}"
        with chunk_path.open("wb") as f:
            f.write(file_content)
            
        if chunk_info['chunk_number'] == chunk_info['total_chunks'] - 1:
            # On final chunk, combine and upload
            # dataset_id = f"dataset_{uuid.uuid4().hex[:12]}"
            dataset_id = f"dataset-{generate_random_name()}"
            final_path = Path(f"/tmp/{dataset_id}_{filename}")
            
            try:
                # Combine chunks
                with final_path.open("wb") as outfile:
                    for i in range(chunk_info['total_chunks']):
                        chunk = chunk_dir / f"chunk_{i}"
                        with chunk.open("rb") as chunk_file:
                            outfile.write(chunk_file.read())
                
                # Upload final file
                raw_path = f"{DatasetStoragePaths.dataset_dir_path(self.user_id, dataset_id)}/{filename}"
                await self.storage.upload_file(final_path, raw_path)
                
                # Create metadata
                metadata = await self._create_dataset_metadata(dataset_id, filename, final_path)
                await self._metadata_handler.update(metadata)
                
                return metadata
            finally:
                # Cleanup
                final_path.unlink(missing_ok=True)
                for chunk in chunk_dir.glob("chunk_*"):
                    chunk.unlink(missing_ok=True)
                chunk_dir.rmdir()
                
        return {"status": "chunk uploaded"}

    async def list_datasets(self) -> List[DatasetMetadata]:
        """List all datasets for the user"""
        return await self._metadata_handler.list()
    
    async def get_dataset(self, dataset_id: str) -> DatasetMetadata:
        """Get a dataset"""
        return await self._metadata_handler.get(dataset_id)

    async def delete_dataset(self, dataset_id: str) -> Dict:
        """Delete a dataset"""
        # Update catalog
        await self._metadata_handler.remove(dataset_id)
        
        # Delete dataset files
        dataset_path = DatasetStoragePaths.dataset_dir_path(self.user_id, dataset_id)
        files = await self.storage.list_files(dataset_path)
        for file in files:
            await self.storage.delete_file(file["name"])
        
        return {"status": "deleted"}

    async def _create_dataset_metadata(self, dataset_id: str, filename: str, file_path: Path) -> DatasetMetadata:
        """Create and save dataset metadata"""
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            name=filename.rsplit('.', 1)[0],
            file_name=filename,
            file_path=f"{DatasetStoragePaths.dataset_dir_path(self.user_id, dataset_id)}/{filename}",
            full_file_path=f"{DatasetStoragePaths.dataset_dir_path(self.user_id, dataset_id, include_bucket=True)}/{filename}",
            created_at=datetime.utcnow(),
            size_bytes=file_path.stat().st_size,
            format=filename.split('.')[-1],
            stats={
                "samples": None,
                "preprocessing": {
                    "tokenizer": None,
                    "max_length": None
                }
            }
        )
        
        return metadata
        