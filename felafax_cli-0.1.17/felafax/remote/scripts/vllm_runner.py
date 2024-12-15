#!/usr/bin/env python3

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, IO
import subprocess
from contextlib import nullcontext
from pydantic import BaseModel, Field

# Initial basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('vllm-server')

class ServerStatus(BaseModel):
    status: str
    error: Optional[str] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SubprocessError(Exception):
    def __init__(self, command: str, returncode: int, stderr: str):
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"Command '{command}' failed with return code {returncode}: {stderr}")

class VLLMManager:
    def __init__(self):
        # Parse config from environment
        self.config = json.loads(os.environ.get('CONFIG', '{}'))
        self.model_job_full_path = self.config.get('model_job_full_path')
        self.model_gcs_full_path = self.config.get('model_gcs_path')
        
        # Setup paths
        self.workspace = Path("/home")
        self.persistent_disk = Path("/mnt/persistent-disk")

        self.gcs_mount = self.workspace / "gcs_mount"
        self.job_dir = self.gcs_mount  # The mounted directory will contain the job files directly
        self.model_path = self.persistent_disk / "model"  # Local path for model files
        
        # Extract GCS paths
        if self.model_job_full_path:
            self.model_job_full_path = self.model_job_full_path.replace('gs://', '')
            parts = self.model_job_full_path.split("/")
            self.bucket_name = parts[0]
            self.model_job_path_prefix = '/'.join(parts[1:])
        
        # Ensure directories exist
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.gcs_mount.mkdir(parents=True, exist_ok=True)
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Mount GCS if model_job_path is provided
        if self.model_job_full_path:
            self._mount_gcs()

            # Add file handler after GCS is mounted
            file_handler = logging.FileHandler(str(self.job_dir / "stdout.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
            logger.info("File logging initialized")

    def _mount_gcs(self):
        """Mount GCS bucket using gcsfuse"""
        logger.info(f"Mounting GCS path {self.model_job_full_path} to {self.gcs_mount}")
        try:
            subprocess.run([
                'gcsfuse',
                '--implicit-dirs',
                '--key-file', '/workspace/.gcloud_key/storage_key.json',
                '--only-dir', self.model_job_path_prefix,
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

    async def update_status(self, status: str, error: Optional[str] = None):
        """Update status in the model job path"""
        if not self.model_job_full_path:
            return

        status_path = self.job_dir / "status.json"
        try:
            server_status = ServerStatus(
                status=status,
                error=error,
                updated_at=datetime.now(timezone.utc)
            )
            status_path.write_text(server_status.json())
            logger.info(f"Updated status to: {status}")
        except Exception as e:
            logger.error(f"Failed to update status: {str(e)}")
            raise

    async def _run_subprocess(self, command: list[str], cwd: Optional[str] = None) -> tuple[str, str]:
        """Execute subprocess with logging"""
        logger.info(f"Running command: {' '.join(command)}")
        
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )

        async def read_stream(stream, is_stderr=False):
            output = []
            log_path = self.job_dir / "stdout.log" if self.model_job_full_path else None
            
            with (open(log_path, 'a') if log_path else nullcontext()) as log_file:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    line_str = line.decode()
                    output.append(line_str)
                    
                    if log_file:
                        prefix = "STDERR: " if is_stderr else "STDOUT: "
                        log_file.write(f"{prefix}{line_str}")
                        log_file.flush()
                    
                    # Also log to console
                    if is_stderr:
                        logger.error(line_str.strip())
                    else:
                        logger.info(line_str.strip())
                        
            return ''.join(output)

        stdout_str, stderr_str = await asyncio.gather(
            read_stream(process.stdout),
            read_stream(process.stderr, True)
        )
        
        await process.wait()
        
        if process.returncode != 0:
            raise SubprocessError(' '.join(command), process.returncode, stderr_str)
        return stdout_str, stderr_str

    async def copy_model(self):
        """Copy model from GCS to persistent disk"""
        try:
            if not self.model_gcs_full_path:
                logger.info("No GCS model path provided, skipping model copy")
                return

            logger.info(f"Copying model from {self.model_gcs_full_path} to {self.model_path}")
            await self._run_subprocess([
                'gsutil', '-m', 'cp', '-r',
                f"{self.model_gcs_full_path.rstrip('/')}/*",
                str(self.model_path)
            ])
            logger.info("Model copied successfully")
            
        except Exception as e:
            await self.update_status("failed", error=str(e))
            raise

    async def start_server(self):
        """Start the vLLM server"""
        try:
            model_path = self.config.get('hf_path', str(self.model_path))
            tokenizer_path = str(self.model_path) if not self.config.get('hf_path') else None
            
            # Base command with model path
            cmd = ['vllm', 'serve', model_path]
            
            # Add vllm_args from config if present
            vllm_args = self.config.get('vllm_args', {})
            for arg, value in vllm_args.items():
                if isinstance(value, bool) and value:
                    cmd.append(arg)
                elif not isinstance(value, bool):
                    cmd.extend([arg, str(value)])
            
            # Add tokenizer path if specified
            if tokenizer_path:
                cmd.extend(['--tokenizer', tokenizer_path])

            await self.update_status("running")
            await self._run_subprocess(cmd)
            
        except Exception as e:
            await self.update_status("failed", error=str(e))
            raise

async def main():
    manager = VLLMManager()
    try:
        await manager.update_status("starting")
        
        # Copy model if using GCS path
        if manager.model_gcs_full_path:
            await manager.copy_model()
        
        # Start the server
        await manager.start_server()
        
    except Exception as e:
        logger.error(f"Server startup failed: {str(e)}")
        await manager.update_status("failed", error=str(e))
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
