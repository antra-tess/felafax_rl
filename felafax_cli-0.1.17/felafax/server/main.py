from fastapi import FastAPI
from ..config import Config
import uvicorn
from .api.auth import router as auth_router
from .api.finetune import router as finetune_router
from .api.models import router as models_router
from .api.dataset import router as dataset_router
from .api.billing import router as billing_router
from .api.accelerators import router as accelerators_router
import logging
import colorlog
import asyncio
from .handlers.job import JobHandler
from .handlers.accelerator import AcceleratorJobProcessor
from .handlers.finetune import FinetuneJobProcessor
from .handlers.base import JobConfig
from .common import get_storage_provider

app = FastAPI(
    debug=False,  # Config.DEBUG,
    title="Felafax API",
    description="API for the Felafax server",
    version="0.1.0"
)
app.include_router(auth_router)
app.include_router(finetune_router)
app.include_router(models_router)
app.include_router(dataset_router)
app.include_router(billing_router)
app.include_router(accelerators_router)


# setup logger
if Config.DEBUG:
    # Setup colored logging for development
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)-8s%(reset)s %(log_color)s%(message)s',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        }
    ))
    
    logger = colorlog.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
else:
    # Standard logging for production
    logging.basicConfig(level=logging.INFO)


async def init_background_jobs():
    storage_provider = await get_storage_provider()
    job_handler = JobHandler(storage_provider)
    
    # Register processors with their configs
    job_handler.register_processor(
        "accelerator",
        AcceleratorJobProcessor,
        JobConfig(update_interval=1, cleanup_delay=10)
    )
    job_handler.register_processor(
        "finetune",
        FinetuneJobProcessor,
        JobConfig(update_interval=1, cleanup_delay=10)
    )
    asyncio.create_task(job_handler.start())

@app.on_event("startup")
async def startup_event():
    await init_background_jobs()

