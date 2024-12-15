from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os
from typing import List

load_dotenv()


class Settings(BaseSettings):
    # Server settings
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # CLI settings
    DEV: bool = False
    DEBUG: bool = DEV  # False
    PROD: bool = not DEV
    PROD_URL: str = "http://api.felafax.ai:8000"
    DEV_URL: str = f"http://{HOST}:{PORT}"
    CONFIG_DIR: str = os.path.expanduser("~/.felafax")
    CONFIG_FILE: str = os.path.join(CONFIG_DIR, "config.json")

    # inference settings
    VLLM_PORT: int = 8000
    # VLLM_IMAGE_NAME: str = "gcr.io/felafax-training/vllm:latest_v3"
    VLLM_IMAGE_NAME: str = "gcr.io/felafax-training/vllm:latest_v4"
    VLLM_DEFAULT_TPU_TYPE: str = "v5p"
    VLLM_DEFAULT_ZONE: str = "europe-west4-b"
    VLLM_DEFAULT_DISK_SIZE: int = 1000
    VLLM_TAGS: List[str] = ["vllm-server"]

    # Finetune settings
    FINETUNE_IMAGE_NAME: str = "gcr.io/felafax-training/trainer:latest"
    MAX_FINETUNE_RUNTIME_HRS: int = 3

    # GCS settings
    GCS_BUCKET_NAME: str = "felafax-storage-v2"
    GCS_PROJECT_ID: str = "felafax-training"

    # TPU settings
    TPU_ZONE: str = "us-central1-a"
    TPU_PROJECT: str = ""
    TPU_NAME: str = ""
    TPU_ACCELERATOR_TYPE: str = ""

    # AWS settings (for future)
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = True


Config = Settings()
