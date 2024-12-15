from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Any, AsyncGenerator

class StorageProvider(ABC):
    """Abstract base class for storage providers"""

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the storage provider with config"""
        pass

    @abstractmethod
    async def upload_file(self, file_path: Path, storage_path: str) -> Dict[str, Any]:
        """Upload a file to storage
        
        Args:
            file_path: Local path to file
            storage_path: Destination path in storage where file will be uploaded
            
        Returns:
            Dict containing uploaded file info
        """
        pass

    @abstractmethod
    async def download_file(self, storage_path: str, local_path: Path) -> None:
        """Download a file from storage
        
        Args:
            storage_path: Path in storage
            local_path: Local path to save file
        """
        pass

    @abstractmethod
    async def delete_file(self, storage_path: str) -> None:
        """Delete a file from storage
        
        Args:
            storage_path: Path to file in storage
        """
        pass
    
    @abstractmethod
    async def delete_directory(self, storage_path: str) -> None:
        """Delete a directory from storage
        
        Args:
            storage_path: Path to directory in storage
        """
        pass

    @abstractmethod
    async def list_files(
        self, 
        prefix: Optional[str] = None, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List files in storage
        
        Args:
            prefix: Optional prefix to filter files
            limit: Maximum number of files to return
            
        Returns:
            List of file information dictionaries
        """
        pass

    @abstractmethod
    async def create_directory(self, path: str) -> None:
        """Create a directory in storage
        
        Args:
            path: Directory path to create
        """
        pass

    @abstractmethod
    async def write_json(self, path: str, data: Dict[str, Any]) -> None:
        """Write JSON data to a file in storage
        
        Args:
            path: Path where to write the JSON file
            data: Dictionary data to write as JSON
        """
        pass

    @abstractmethod
    async def read_json(self, path: str) -> Dict[str, Any]:
        """Read JSON data from a file in storage
        
        Args:
            path: Path to the JSON file
            
        Returns:
            Dictionary containing the JSON data
        """
        pass

    @abstractmethod
    async def ensure_structure(self, base_path: str, structure: Dict[str, Any]) -> None:
        """Ensure a directory structure exists, creating it if necessary
        
        Args:
            base_path: Base path where to create the structure
            structure: Dictionary describing the directory structure
        """
        pass

    @abstractmethod
    async def directory_exists(self, path: str) -> bool:
        """Check if a directory exists in storage
        
        Args:
            path: Directory path to check
            
        Returns:
            True if directory exists, False otherwise
        """
        pass

    @abstractmethod
    async def file_exists(self, path: str) -> bool:
        """Check if a file exists in storage
        
        Args:
            path: File path to check
            
        Returns:
            True if file exists, False otherwise
        """
        pass 
    
    @abstractmethod
    async def stream_file_text(self, path: str, chunk_size: int = 8192, start_position: int = 0) -> AsyncGenerator[str, None]:
        """Stream text from a file in storage"""
        pass
    
    @abstractmethod
    async def stream_file(self, path: str, chunk_size: int = 8192, start_position: int = 0) -> AsyncGenerator[bytes, None]:
        """Stream a file from storage in chunks"""
        pass

    @abstractmethod
    async def get_file_size(self, path: str) -> int:
        """Get the size of a file in storage"""
        pass
