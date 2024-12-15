from ...core.storage.base import StorageProvider
from typing import Dict, List, Optional
from ..models.model import ModelPaths
from ..handlers.base import ListMetadataHandler
from ..handlers.accelerator import AcceleratorHandler
from ..models.model import ModelMetadata
from ..models.accelerator import AcceleratorRequest, AcceleratorMetadata
from felafax.config import Config
import httpx
import logging
import asyncio
import json
from ..models.accelerator import AcceleratorStatus

logger = logging.getLogger(__name__)


class ModelHandler:
    def __init__(self, storage_provider: StorageProvider, user_id: str):
        self.storage = storage_provider
        self.user_id = user_id
        self._metadata_handler = ListMetadataHandler(ModelMetadata, ModelPaths.metadata_path(user_id), "model_id", self.storage)
        self._accelerator_handler = AcceleratorHandler(self.storage, self.user_id)
    

    async def init_chat(self, model_id: str) -> str:
        """Initialize chat by starting the accelerator"""
        logger.info(f"Initializing chat with model {model_id}")
        try:
            if not await self.check_model_exists(model_id):
                logger.error(f"Model {model_id} does not exist")
                raise ValueError(f"Model {model_id} does not exist")
            
            model_full_path = ModelPaths.model_path(self.user_id, model_id, include_bucket=True)
            model_job_full_path = ModelPaths.job_path(self.user_id, model_id, include_bucket=True)

            # Create job directory
            await self.storage.create_directory(model_job_full_path)

            logger.debug(f"Starting accelerator for model at path: {model_full_path}")
            accelerator = await self._start_vllm_inference_accelerator(model_full_path, model_job_full_path)
            
            # Wait for accelerator to be fully ready
            max_retries = 60  # 10 minutes with 10s sleep
            for attempt in range(max_retries):
                status = await self._accelerator_handler.get_and_update_status(accelerator.accelerator_id)
                if status == AcceleratorStatus.READY:
                    break
                if attempt == max_retries - 1:
                    raise TimeoutError("Accelerator failed to become ready within timeout period")
                await asyncio.sleep(10)
                
            logger.info(f"Successfully initialized chat with accelerator ID: {accelerator.accelerator_id}")
            return accelerator.accelerator_id
        except Exception as e:
            logger.error(f"Failed to initialize chat with model {model_id}: {str(e)}")
            raise

    async def chat(self, accelerator_id: str, messages: List[Dict]) -> str:
        """Chat with a model using an existing accelerator"""
        logger.debug(f"Starting chat with accelerator {accelerator_id}")
        try:
            ip_address = await self._accelerator_handler.get_ip_address(accelerator_id)
            logger.debug(f"Retrieved IP address: {ip_address}")
            response = await self._vllm_chat(messages, ip_address)
            logger.debug("Chat completed successfully")
            return response
        except Exception as e:
            logger.error(f"Chat failed with accelerator {accelerator_id}: {str(e)}")
            raise

    async def _start_vllm_inference_accelerator(self, model_path: str, model_job_path: str) -> AcceleratorMetadata:
        """Start inference accelerator with specific config"""
        accelerator_request = AcceleratorRequest(
            type="tpu",
            accelerator_type=Config.VLLM_DEFAULT_TPU_TYPE,
            accelerator_core_count=8,
            zone=Config.VLLM_DEFAULT_ZONE,
            attach_disk=True,
            disk_size_gb=Config.VLLM_DEFAULT_DISK_SIZE,
            docker_image=Config.VLLM_IMAGE_NAME,
            docker_env={
                "SCRIPT_NAME": "vllm_runner.py",
                "CONFIG": json.dumps({"model_gcs_path": model_path, "model_job_full_path": model_job_path, "vllm_args": {"--enforce-eager": True, "--max-num-seqs": 1}})
            },
            tags=Config.VLLM_TAGS,
        )
        return await self._accelerator_handler.create_accelerator(accelerator_request)
    
    async def _vllm_chat(self, messages: List[Dict], ip_address: str) -> str:
        """Chat with a model using VLLM"""
        url = f"http://{ip_address}:8000/v1/chat/completions"
        logger.debug(f"Making VLLM request to {url}")
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                try:
                    response = await client.post(url, json={
                        "model": "/mnt/persistent-disk/model", 
                        "messages": messages
                    })
                    response_json = response.json()
                    logger.debug("VLLM request completed successfully")
                    return response_json["choices"][0]["message"]["content"]
                except httpx.ConnectError:
                    logger.info("VLLM server not ready yet")
                    # Raise HTTPException with 503 status code
                    raise httpx.HTTPStatusError(
                        message="Model is still initializing. Please wait a moment and try again.",
                        request=None,
                        response=httpx.Response(503)
                    )
        except Exception as e:
            logger.error(f"VLLM request failed: {str(e)}")
            raise

    async def get_model_info(self, model_id: str) -> Dict:
        """Get model metadata"""
        return await self._metadata_handler.get(model_id)

    async def update_model_info(self, model_id: str, info: Dict) -> None:
        """Update model metadata"""
        metadata = await self._metadata_handler.get(model_id)
        metadata.update(info)
        await self._metadata_handler.update(metadata)
        
    async def get_download_url(self, model_id: str) -> str:
        """Get download URL for model weights"""
        return f"/download/{self.user_id}/models/{model_id}/weights"
    
    async def delete_model(self, model_id: str) -> None:
        """Delete a model"""
        logger.info(f"Deleting model {model_id}")
        try:
            await self._metadata_handler.remove(model_id)
            model_path = ModelPaths.model_path(self.user_id, model_id)
            await self.storage.delete_directory(model_path, convert_iterator=True)
            logger.info(f"Successfully deleted model {model_id}")
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {str(e)}")
            raise

    async def list_models(self) -> List[ModelMetadata]:
        """List all models"""
        return await self._metadata_handler.list()
    
    async def check_model_exists(self, model_id: str) -> bool:
        """Check if a model exists"""
        return await self._metadata_handler.get(model_id) is not None

