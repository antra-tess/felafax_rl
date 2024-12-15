import asyncio
import logging
from typing import Dict, Type, Any
from datetime import datetime, timedelta
from ..models.job import JobMetadata, JobStatus
from ..handlers.base import ListMetadataHandler
from .base import JobProcessor, JobConfig
from ...core.storage.base import StorageProvider
from ..models.job import JobStoragePaths
logger = logging.getLogger(__name__)

class JobHandler:
    _instance = None
    _initialized = False

    def __new__(cls, storage_provider=None):
        if cls._instance is None:
            if storage_provider is None:
                raise ValueError("Storage provider is required for initialization")
            cls._instance = super(JobHandler, cls).__new__(cls)
        return cls._instance

    def __init__(self, storage_provider=None):
        if not JobHandler._initialized and storage_provider is not None:
            self.storage = storage_provider
            self._processors: Dict[str, Type[JobProcessor]] = {}
            self._configs: Dict[str, JobConfig] = {}
            self._metadata_handler = ListMetadataHandler(
                JobMetadata,
                JobStoragePaths.metadata_path(),
                "job_id",
                storage_provider
            )
            asyncio.create_task(self._metadata_handler.initialize())
            self._running = False
            JobHandler._initialized = True

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise RuntimeError("JobHandler not initialized. Call JobHandler(storage_provider) first")
        return cls._instance

    def register_processor(
        self,
        job_type: str,
        processor: Type[JobProcessor],
        config: JobConfig
    ):
        """Register a job processor for a specific job type"""
        self._processors[job_type] = processor
        self._configs[job_type] = config

    async def add_job(
        self,
        job_type: str,
        data: Dict[str, Any],
    ) -> JobMetadata:
        """Add a new job to the queue"""
        if job_type not in self._processors:
            raise ValueError(f"No processor registered for job type: {job_type}")

        config = self._configs[job_type]
        now = datetime.utcnow()
        
        job = JobMetadata(
            job_id=f"{job_type}_{now.timestamp()}",
            job_type=job_type,
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now,
            next_update_at=now + timedelta(minutes=config.update_interval),
            cleanup_at=now + timedelta(minutes=config.cleanup_delay),
            data=data
        )

        await self._metadata_handler.add(job)
        return job

    async def process_jobs(self):
        """Process all pending jobs"""
        jobs = await self._metadata_handler.list()
        now = datetime.utcnow()

        for job in jobs:
            if job.status == JobStatus.CLEANUP and job.cleanup_at <= now:
                await self._process_cleanup(job)
                continue

            if job.next_update_at and job.next_update_at <= now:
                await self._process_update(job)

    async def _process_update(self, job: JobMetadata):
        """Process a single job update"""
        processor = self._processors[job.job_type](storage_provider=self.storage)
        config = self._configs[job.job_type]

        try:
            updated_job = await processor.process_update(job)
            updated_job.updated_at = datetime.utcnow()
            updated_job.next_update_at = updated_job.updated_at + timedelta(
                minutes=config.update_interval
            )
            await self._metadata_handler.update(updated_job)
        except Exception as e:
            logger.error(f"Error processing job {job.job_id}: {str(e)}")
            #TODO: handle error and maybe don't just remove the job
            updated_job = await processor.handle_error(job, e)
            await self._metadata_handler.remove(job.job_id)

    async def _process_cleanup(self, job: JobMetadata):
        """Process job cleanup"""
        processor = self._processors[job.job_type](storage_provider=self.storage)
        try:
            await processor.process_cleanup(job)
            await self._metadata_handler.remove(job.job_id)
        except Exception as e:
            # TODO: handle error and maybe don't just remove the job
            await self._metadata_handler.remove(job.job_id)
            logger.error(f"Error cleaning up job {job.job_id}: {str(e)}")

    async def start(self, interval: int = 60):
        """Start the job processor"""
        self._running = True
        while self._running:
            try:
                await self.process_jobs()
            except Exception as e:
                logger.error(f"Error in job processor: {str(e)}")
            await asyncio.sleep(interval)

    async def stop(self):
        """Stop the job processor"""
        self._running = False
