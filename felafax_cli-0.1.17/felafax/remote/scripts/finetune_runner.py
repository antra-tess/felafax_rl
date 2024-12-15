#!/usr/bin/env python3

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, IO
import subprocess
import shutil
from contextlib import nullcontext
import yaml
import aiohttp

from google.cloud import storage
from pydantic import BaseModel, Field

# Initial basic logging to stdout only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('training')

# TODO: add a way to capture the entire execution of this script to GCS
class TrainingStatus(BaseModel):
    status: str
    progress: float = 0.0
    error: Optional[str] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SubprocessError(Exception):
    """Custom exception for subprocess failures"""
    def __init__(self, command: str, returncode: int, stderr: str):
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"Command '{command}' failed with return code {returncode}: {stderr}")

class TrainingManager:
    def __init__(self):
        # Get environment variables and parse GCS path
        self.gcs_job_path = os.environ["GCS_JOB_PATH"].replace('gs://', '')
        parts = self.gcs_job_path.split("/")
        self.bucket_name = parts[0]
        self.object_prefix = '/'.join(parts[1:])
        
        # Setup mount paths
        # self.workspace = Path("/mnt/persistent-disk/")
        self.workspace = Path("/home/")
        self.gcs_mount = self.workspace / "gcs_mount"
        self.job_dir = self.gcs_mount
        self.repo_path = self.workspace / "felafax"
        
        # Ensure directories exist
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.gcs_mount.mkdir(parents=True, exist_ok=True)
        
        # Mount GCS bucket
        self._mount_gcs()
        
        # Add file handler after GCS is mounted
        debug_log_path = str(self.job_dir / "debug.log")
        file_handler = logging.FileHandler(debug_log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.info("File logging initialized")

    def _mount_gcs(self):
        """Mount GCS bucket using gcsfuse"""
        logger.info(f"Mounting GCS path {self.gcs_job_path} to {self.gcs_mount}")
        try:
            subprocess.run([
                'gcsfuse',
                '--implicit-dirs',
                '--key-file', '/home/.gcloud_key/key.json',
                '--only-dir', self.object_prefix,
                self.bucket_name,
                str(self.gcs_mount)
            ], check=True)
            logger.info("GCS job directory mounted successfully")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to mount GCS directory: {str(e)}")

    def __del__(self):
        """Cleanup: Unmount GCS when object is destroyed"""
        try:
            subprocess.run(['fusermount', '-u', str(self.gcs_mount)], check=True)
            logger.info("GCS bucket unmounted successfully")
        except Exception as e:
            logger.error(f"Failed to unmount GCS bucket: {str(e)}")

    async def update_status(self, status: str, progress: float = 0.0, error: Optional[str] = None):
        """Update status field in status.json while preserving other fields"""
        status_path = self.job_dir / "status.json"
        try:
            # Read existing status
            current_status = TrainingStatus.parse_file(status_path)
            
            # Update only the status, progress, and updated_at fields
            current_status.status = status
            current_status.progress = progress
            current_status.updated_at = datetime.now(timezone.utc)
            current_status.error = error
            
            # Write updated status
            status_path.write_text(current_status.json())
            logger.info(f"Updated status to: {status} (progress: {progress})")
            
        except Exception as e:
            logger.error(f"Failed to update status: {str(e)}")
            raise

    async def mark_done(self, failed: bool = False):
        """Write DONE or FAILED marker to mounted path and notify API"""
        marker_name = "FAILED" if failed else "DONE"
        marker_path = self.job_dir / marker_name
        marker_path.touch()
        logger.info(f"{marker_name} marker written at: {marker_path}")
        
        # Notify API that the job has stopped
        await self.stop_finetune_job()

    async def _run_subprocess(self, command: list[str], cwd: Optional[str] = None, 
                            stdout_file: Optional[IO] = None) -> tuple[str, str]:
        """
        Wrapper for subprocess execution with consistent error handling
        
        Args:
            command: Command and arguments as list
            cwd: Working directory for command execution
            stdout_file: Optional file object to write stdout to
        
        Returns:
            Tuple of (stdout, stderr) if capture_output is True
        """
        logger.info(f"Running command: {' '.join(command)}")
        
        # Always capture output with PIPE
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )
        
        debug_log_path = self.job_dir / "debug.log"
        
        # Read and process output incrementally
        async def read_stream(stream, is_stderr=False):
            output = []
            with open(debug_log_path, 'a') as debug_file, \
                 (open(stdout_file, 'a') if stdout_file else nullcontext()) as stdout_f:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    line_str = line.decode()
                    output.append(line_str)
                    prefix = "STDERR: " if is_stderr else "STDOUT: "
                    if stdout_f:
                        stdout_f.write(f"{prefix}{line_str}")
                        stdout_f.flush()
                    
                    # Write to debug log in real-time
                    debug_file.write(f"{prefix}{line_str}")
                    debug_file.flush()
            return ''.join(output)
        # Write command to debug log
        with open(debug_log_path, 'a') as debug_file:
            debug_file.write(f"\n{'='*50}\nCommand: {' '.join(command)}\n{'='*50}\n")
        
        # Gather both stdout and stderr
        stdout_str, stderr_str = await asyncio.gather(
            read_stream(process.stdout),
            read_stream(process.stderr, True)
        )
        
        await process.wait()
        
        if process.returncode != 0:
            raise SubprocessError(' '.join(command), process.returncode, stderr_str)
        return stdout_str, stderr_str

    async def clone_and_setup(self):
        """Clone felafax repo and install requirements"""
        logger.info("Cloning felafax repository...")
        try:
            if self.repo_path.exists():
                shutil.rmtree(self.repo_path)
            
            # Clone repository
            await self._run_subprocess([
                'git', 'clone', 'https://github.com/felafax/felafax.git',
                str(self.repo_path)
            ])
            
            # Install requirements
            await self._run_subprocess([
                'pip', 'install', '-r', str(self.repo_path / 'requirements.txt')
            ])
                
            logger.info("Repository cloned and requirements installed successfully")
            
        except SubprocessError as e:
            await self.update_status("failed", error=str(e))
            raise
    
    async def run_trainer(self):
        """Run the training pipeline"""
        try:
            await self.update_status("running", progress=0.1)
            
            await self._run_subprocess(
                ['python', '-m', 'trainers.cli_tuner.pipeline',
                 '--config_path', str(self.job_dir / "trainer_config.yml")],
                cwd=str(self.repo_path)
            )
            
            await self.update_status("completed", progress=1.0)
            
        except SubprocessError as e:
            await self.update_status("failed", error=str(e))
            raise

    async def just_copy_hf_model(self):
        """Test version of trainer that just downloads the model"""
        try:
            await self.update_status("running", progress=0.1)

            # Install huggingface_cli
            await self._run_subprocess([
                'pip', 'install', '-U', 'huggingface_hub[cli]'
            ])
            
            # Read configs
            with open(self.job_dir / "trainer_config.yml", 'r') as f:
                trainer_config = yaml.safe_load(f)
            with open(self.job_dir / "internal_config.yml", 'r') as f:
                internal_config = yaml.safe_load(f)
            
            # Create export directory
            export_dir = internal_config['export_dir']
            os.makedirs(export_dir, exist_ok=True)
            
            # Download model using Hugging Face CLI
            await self._run_subprocess(
                ['huggingface-cli', 'login', '--token', trainer_config['hf_token']],
                stdout_file=str(self.job_dir / "training.log")
            )
            
            await self._run_subprocess(
                ['python', '-c', f'''
import os
from huggingface_hub import snapshot_download
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
snapshot_download(
    repo_id="{trainer_config['trainer_config']['model_name']}", 
    local_dir="{export_dir}",
    token="{trainer_config['hf_token']}"
)
'''],
                stdout_file=str(self.job_dir / "training.log")
            )
            
            
        except SubprocessError as e:
            await self.update_status("failed", error=str(e))
            raise

    async def export_model_to_gcs(self):
        """Export the trained model to GCS using paths from config"""
        try:
            # Read the config to get paths
            with open(self.job_dir / "internal_config.yml", 'r') as f:
                config = yaml.safe_load(f)
                model_gcs_path = config.get('model_gcs_path')
                export_dir = config.get('export_dir')

            if model_gcs_path and export_dir:
                logger.info(f"Copying exported model from {export_dir} to {model_gcs_path}")
                await self._run_subprocess([
                    'gsutil', '-m', 'cp', '-r',
                    export_dir,
                    model_gcs_path
                ])
                logger.info("Model export completed successfully")
            else:
                logger.warning("Model export skipped: missing model_gcs_path or export_dir in config")
                
        except Exception as e:
            logger.error(f"Failed to export model: {str(e)}")
            raise

    async def copy_dataset_from_gcs(self):
        """Copy the dataset from GCS to local path using paths from config"""
        try:
            # Read the config to get paths
            with open(self.job_dir / "internal_config.yml", 'r') as f:
                config = yaml.safe_load(f)
                dataset_gcs_path = config.get('dataset_gcs_path')
                dataset_path = config.get('dataset_path')

            if dataset_gcs_path and dataset_path:
                logger.info(f"Copying dataset from {dataset_gcs_path} to {dataset_path}")
                os.makedirs(dataset_path, exist_ok=True)
                await self._run_subprocess([
                    'gsutil', '-m', 'cp', '-r',
                    dataset_gcs_path,
                    dataset_path
                ])
                logger.info("Dataset copy completed successfully")
            else:
                logger.warning("Dataset copy skipped: missing dataset_gcs_path or dataset_path in config")
                
        except Exception as e:
            logger.error(f"Failed to copy dataset: {str(e)}")
            raise

    async def stop_finetune_job(self):
        """Notify the API that the job has been stopped"""
        # TODO: you should have a better way to stop this outside of this script
        try:
            # Read the internal config to get user_id and tune_id
            with open(self.job_dir / "internal_config.yml", 'r') as f:
                config = yaml.safe_load(f)
                user_id = config.get('user_id')
                tune_id = config.get('tune_id')

            if not user_id or not tune_id:
                logger.warning("Missing user_id or tune_id in config, skipping API notification")
                return

            # Get the API URL from environment or use default
            api_url = os.getenv('PROD_URL', 'http://api.felafax.ai:8000')
            stop_url = f"{api_url}/fine-tune/{user_id}/{tune_id}/stop"

            async with aiohttp.ClientSession() as session:
                async with session.post(stop_url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to notify API of job stop: {await response.text()}")
                    else:
                        logger.info("Successfully notified API of job stop")

        except Exception as e:
            logger.error(f"Error notifying API of job stop: {str(e)}")

async def main():
    manager = TrainingManager()
    try:
        await manager.update_status("running", progress=0.2)
        await manager.copy_dataset_from_gcs()

        await manager.update_status("running", progress=0.1)
        await manager.clone_and_setup()

        try:
            await manager.update_status("running", progress=0.3)
            await manager.run_trainer()
        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            # continue as this could be some HF error
            pass
        
        await manager.update_status("running", progress=0.9)
        await manager.export_model_to_gcs()

        await manager.update_status("running", progress=0.95)
        await manager.mark_done()

        await manager.update_status("completed", progress=1.0)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        await manager.mark_done(failed=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
    logger.info("Training job completed successfully")

