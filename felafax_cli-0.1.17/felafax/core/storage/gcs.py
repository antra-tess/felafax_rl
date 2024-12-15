from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator
import asyncio
from datetime import datetime
from google.cloud import storage
import os
from google.oauth2 import service_account
from google.auth import default
from fastapi import HTTPException

from .base import StorageProvider
import logging

logger = logging.getLogger(__name__)

class GCSStorageProvider(StorageProvider):
    """Google Cloud Storage provider implementation"""
    
    # controls whether to use file locks for writes for more consistent writes
    USE_LOCKS = False
    
    def __init__(self):
        self.client = None
        self.bucket_name = None
        self._bucket = None

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize GCS client with config
        
        Args:
            config: Dict containing:
                - bucket_name: GCS bucket name
                - project_id: (optional) Google Cloud project ID
        """
        self.bucket_name = config["bucket_name"]
        project_id = config.get("project_id")
        
        def _init():
            try:
                # First try to use default credentials (including user credentials)
                credentials, project = default()
                self.client = storage.Client(
                    project=project_id or project,
                    credentials=credentials
                )
            except Exception as e:
                # Fallback to application default credentials
                self.client = storage.Client(project=project_id)
            
            self._bucket = self.client.bucket(self.bucket_name)
            
        await asyncio.get_event_loop().run_in_executor(None, _init)

    @property
    def bucket(self) -> storage.Bucket:
        """Lazy initialization of bucket"""
        if self._bucket is None:
            raise RuntimeError("Storage provider not initialized")
        return self._bucket

    async def upload_file(self, file_path: Path, storage_path: str) -> Dict[str, Any]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        def _upload():
            blob = self.bucket.blob(storage_path)
            blob.upload_from_filename(str(file_path))
            return {
                "path": storage_path,
                "size": blob.size,
                "created": blob.time_created.isoformat() if blob.time_created else None,
                "provider": "gcs", 
                "bucket": self.bucket_name
            }

        return await asyncio.get_event_loop().run_in_executor(None, _upload)

    async def download_file(self, storage_path: str, local_path: Path) -> None:
        def _download():
            blob = self.bucket.blob(storage_path)
            os.makedirs(local_path.parent, exist_ok=True)
            blob.download_to_filename(str(local_path))

        await asyncio.get_event_loop().run_in_executor(None, _download)

    async def delete_file(self, storage_path: str) -> None:
        def _delete():
            blob = self.bucket.blob(storage_path)
            blob.delete()

        await asyncio.get_event_loop().run_in_executor(None, _delete)
    
    async def delete_directory(self, prefix: str, convert_iterator: bool = True) -> None:
        def _delete():
            blobs = self.bucket.list_blobs(prefix=prefix)
            if convert_iterator:
                blobs = list(blobs)  # Convert iterator to list
            self.bucket.delete_blobs(blobs)
        
        await asyncio.get_event_loop().run_in_executor(None, _delete)

    async def list_files(self, prefix: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        def _list():
            blobs = self.client.list_blobs(
                self.bucket_name,
                prefix=prefix,
                max_results=limit
            )
            return [{
                "name": blob.name,
                "size": blob.size,
                "created": blob.time_created.isoformat() if blob.time_created else None,
                "updated": blob.updated.isoformat() if blob.updated else None,
                "provider": "gcs",
                "bucket": self.bucket_name
            } for blob in blobs]

        return await asyncio.get_event_loop().run_in_executor(None, _list)

    async def create_directory(self, path: str) -> None:
        path = path.rstrip('/') + '/'
        
        def _create_dir():
            blob = self.bucket.blob(path)
            blob.upload_from_string('')
            
        await asyncio.get_event_loop().run_in_executor(None, _create_dir)

    @staticmethod
    def datetime_handler(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

    async def write_json(self, path: str, data: Dict[str, Any]) -> None:
        import json
        
        if self.USE_LOCKS:
            lock_path = f"{path}.lock"
            
            async def wait_for_lock():
                while True:
                    def _check_lock():
                        return self.bucket.blob(lock_path).exists()
                    
                    if not await asyncio.get_event_loop().run_in_executor(None, _check_lock):
                        break
                    await asyncio.sleep(0.5)
            
            def _create_lock():
                blob = self.bucket.blob(lock_path)
                blob.upload_from_string('')
                
            def _delete_lock():
                self.bucket.blob(lock_path).delete()
            
            await wait_for_lock()
            await asyncio.get_event_loop().run_in_executor(None, _create_lock)
            try:
                def _write():
                    blob = self.bucket.blob(path)
                    blob.upload_from_string(json.dumps(data, default=self.datetime_handler))
                await asyncio.get_event_loop().run_in_executor(None, _write)
            finally:
                await asyncio.get_event_loop().run_in_executor(None, _delete_lock)
        else:
            # Simple write without locks
            def _write():
                blob = self.bucket.blob(path)
                blob.upload_from_string(json.dumps(data, default=self.datetime_handler))
            await asyncio.get_event_loop().run_in_executor(None, _write)

    async def read_json(self, path: str) -> Dict[str, Any]:
        import json
        
        def _read():
            blob = self.bucket.blob(path)
            content = blob.download_as_string()
            return json.loads(content)
            
        return await asyncio.get_event_loop().run_in_executor(None, _read)

    async def ensure_structure(self, base_path: str, structure: Dict[str, Any]) -> None:
        async def create_recursive(current_path: str, struct: Dict[str, Any]):
            for name, content in struct.items():
                path = f"{current_path}/{name}"
                if isinstance(content, dict):
                    await self.create_directory(path)
                    await create_recursive(path, content)
                    
        await create_recursive(base_path.rstrip('/'), structure)

    async def directory_exists(self, path: str) -> bool:
        path = path.rstrip('/') + '/'
        
        def _check():
            blobs = list(self.bucket.list_blobs(prefix=path, max_results=1))
            return len(blobs) > 0
            
        return await asyncio.get_event_loop().run_in_executor(None, _check)

    async def file_exists(self, path: str) -> bool:
        def _check():
            blob = self.bucket.blob(path)
            return blob.exists()
            
        return await asyncio.get_event_loop().run_in_executor(None, _check)

    async def stream_file(self, path: str, chunk_size: int = 8192, start_position: int = 0) -> AsyncGenerator[bytes, None]:
        def _read_chunk(blob: storage.Blob, start: int, size: int) -> bytes:
            try:
                end = min(start + size, blob.size or 0)
                if start >= end:
                    return b''
                return blob.download_as_bytes(start=start, end=end)
            except Exception as e:
                logger.error(f"Error reading chunk: {str(e)}")
                return b''

        try:
            blob = self.bucket.blob(path)
            if not blob.exists():
                logger.error(f"File not found: {path}")
                yield b"Log file not found or empty\n"
                return

            # Load blob metadata before accessing size
            await asyncio.get_event_loop().run_in_executor(None, blob.reload)
            size = blob.size or 0
            
            if size == 0:
                logger.info(f"File {path} is empty")
                yield b"Log file is empty\n"
                return
                
            # Start from the requested position
            position = start_position
            if position >= size:
                # Return 416 by raising an exception that will be caught by the API
                raise HTTPException(status_code=416, detail="Requested range not satisfiable")
                
            while position < size:
                chunk = await asyncio.get_event_loop().run_in_executor(
                    None,
                    _read_chunk,
                    blob,
                    position,
                    chunk_size
                )
                if not chunk:  # End of file or error
                    break
                    
                yield chunk
                position += len(chunk)
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error streaming file {path}: {str(e)}")
            yield str(f"Error reading logs: {str(e)}\n").encode('utf-8')

    async def stream_file_text(self, path: str, chunk_size: int = 8192, start_position: int = 0) -> AsyncGenerator[str, None]:
        """Stream a text file from storage in chunks, decoding as UTF-8"""
        async for chunk in self.stream_file(path, chunk_size, start_position):
            yield chunk.decode('utf-8')
 
    async def get_file_size(self, path: str) -> int:
        blob = self.bucket.blob(path)
        await asyncio.get_event_loop().run_in_executor(None, blob.reload)
        return blob.size or 0