from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from pathlib import Path

from google.cloud import tpu_v2
import asyncio
from google.cloud import compute_v1
from google.api_core import retry
import google.api_core.exceptions
import random
from dataclasses import field
import logging

from .base import (
    AcceleratorConfig,
    AcceleratorProvider,
    AcceleratorStatus,
    AcceleratorMetrics,
)

logger = logging.getLogger(__name__)


class TPUConfig(AcceleratorConfig):
    """TPU-specific configuration"""
    name: str
    type: str = "tpu"
    project_id: str
    accelerator_type: str = "v3"
    accelerator_core_count: Optional[int] = 8
    runtime_version: str = "tpu-vm-tf-2.16.1-pod-pjrt"
    attach_disk: bool = False
    preemptible: bool = True
    zone: Optional[str] = None
    backup_zones: List[str] = []
    disk_size_gb: Optional[int] = 1000
    docker_image: Optional[str] = None
    docker_env: Optional[Dict[str, str]] = {}
    ssh_key: Optional[str] = None
    tags: List[str] = []
    custom_config: Optional[Dict[str, Any]] = {}

    @property
    def valid_accelerator_types(self) -> set:
        return {"v3", "v5e", "v5p"}

    def model_post_init(self, *args, **kwargs):
        """Validate config after initialization"""
        if self.accelerator_type not in self.valid_accelerator_types:
            raise ValueError(
                f"Invalid accelerator_type: {self.accelerator_type}. Must be one of {self.valid_accelerator_types}."
            )

        if self.accelerator_core_count % 8 != 0 or not (8 <= self.accelerator_core_count <= 1024):
            raise ValueError(
                "accelerator_core_count must be a multiple of 8 and between 8 and 1024."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary representation."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "TPUConfig":
        """Create config instance from dictionary data."""
        if not data:
            raise ValueError("Configuration data is required")
        return cls(**data)


class TPUProvider(AcceleratorProvider):
    """TPU accelerator provider implementation"""

    def __init__(self):
        self.client = tpu_v2.TpuAsyncClient()
        self.compute_client = compute_v1.DisksClient()
        self.config: Optional[TPUConfig] = None
        self.node_name: Optional[str] = None
    

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the TPU provider with config"""
        self.config = TPUConfig.from_dict(config)
        
        if self.config.zone is None:
            self.config.zone = self._get_region_for_accelerator(
                self.config.accelerator_type
            )

        # generate 2 backup zones
        self.config.backup_zones = []
        while len(self.config.backup_zones) < 2:
            backup_zone = self._get_region_for_accelerator(self.config.accelerator_type)
            # TODO: check that backup zone is not the same as the primary zone only when there are more than 2 regions
            self.config.backup_zones.append(backup_zone)

        self.node_name = f"projects/{self.config.project_id}/locations/{self.config.zone}/nodes/{self.config.name}"
        # TODO: validate that self.config.tags are valid GCP tags
        

    async def _get_tpu_state(self) -> Optional[str]:
        """Get the current state of a TPU node."""
        response = await self.client.get_node(name=self.node_name)
        return response.state.name

    async def _create_persistent_disk(self, disk_name: str) -> str:
        """Create a persistent disk using the Compute Engine API."""
        disk_path = f"projects/{self.config.project_id}/zones/{self.config.zone}/disks/{disk_name}"

        # If disk already exists, just return the path
        if await self._disk_exists(disk_name):
            return disk_path

        try:
            disk_config = compute_v1.Disk()
            disk_config.name = disk_name
            disk_config.size_gb = self.config.disk_size_gb
            disk_config.type_ = f"projects/{self.config.project_id}/zones/{self.config.zone}/diskTypes/pd-balanced"

            operation = self.compute_client.insert(
                project=self.config.project_id,
                zone=self.config.zone,
                disk_resource=disk_config,
            )
            operation.result()  # Wait for operation to complete

            return disk_path
        except Exception as e:
            # If error is not about disk already existing, raise it
            if "already exists" not in str(e):
                raise RuntimeError(f"Failed to create persistent disk: {str(e)}")
            return disk_path

    async def _attach_disk_to_tpu(
        self, disk_path: str, mode: str = "READ_WRITE"
    ) -> None:
        """Attach a persistent disk to TPU VM."""
        logger.info(f"Attempting to attach disk {disk_path} to TPU VM in {mode} mode")
        try:
            # Get current node configuration
            node = await self.client.get_node(name=self.node_name)

            # Check if disk is already attached
            for disk in node.data_disks:
                if disk.source_disk == disk_path:
                    logger.info(f"Disk {disk_path} is already attached to TPU VM")
                    return

            # Create AttachedDisk configuration with proper mode enum
            disk_mode = (
                tpu_v2.AttachedDisk.DiskMode.READ_WRITE
                if mode == "READ_WRITE"
                else tpu_v2.AttachedDisk.DiskMode.READ_ONLY
            )

            attached_disk = tpu_v2.AttachedDisk()
            attached_disk.source_disk = disk_path
            attached_disk.mode = disk_mode

            # Update node with new disk configuration
            logger.debug("Updating node with new disk configuration")
            node.data_disks = [attached_disk]

            # Create update request with proper field mask
            request = tpu_v2.UpdateNodeRequest()
            request.node = node
            request.update_mask = {"paths": ["dataDisks"]}

            # Add retry and timeout parameters
            operation = await self.client.update_node(
                request=request,
                retry=retry.Retry(
                    initial=1.0,
                    maximum=60.0,
                    multiplier=2.0,
                    predicate=retry.if_transient_error,
                ),
                timeout=300.0,
            )

            # Wait for operation to complete with timeout
            logger.debug("Waiting for disk attachment operation to complete")
            await asyncio.wait_for(operation.result(), timeout=300)
            logger.info(f"Successfully attached disk {disk_path} to TPU VM")

        except asyncio.TimeoutError:
            logger.error("Timeout exceeded while attaching disk to TPU")
            raise RuntimeError("Timeout while attaching disk to TPU")
        except Exception as e:
            logger.error(f"Failed to attach disk to TPU: {str(e)}")
            raise RuntimeError(f"Failed to attach disk to TPU: {str(e)}")
    
    async def get_ip_address(self) -> str:
        """Get the external IP address of a TPU VM."""
        node = await self.client.get_node(name=self.node_name)
        
        if node.network_endpoints:
            for endpoint in node.network_endpoints:
                # Check for external IP specifically
                if hasattr(endpoint, 'access_config') and endpoint.access_config.external_ip:
                    return endpoint.access_config.external_ip
                # Fallback to IP address if no explicit external IP is found
                if endpoint.ip_address:
                    return endpoint.ip_address
        
        raise RuntimeError("No IP address found for TPU VM")

    async def _mount_disk_in_container(self) -> None:
        """Mount the persistent disk inside the TPU VM."""
        logger.info("Starting disk mount process in TPU VM")
        try:
            mount_script = """
                DISK_PATH=$(readlink -f /dev/disk/by-id/google-persistent-disk-1)
                echo "Disk path: $DISK_PATH"
                sudo mkdir -p /mnt/persistent-disk
                
                # Check if the disk is already formatted as ext4
                if ! sudo blkid -p -n ext4 $DISK_PATH > /dev/null; then
                    echo "Disk is not formatted as ext4. Formatting..."
                    sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard $DISK_PATH
                else
                    echo "Disk is already formatted as ext4. Skipping format."
                fi
                
                # Mount the disk
                sudo mount -o discard,defaults $DISK_PATH /mnt/persistent-disk
                
                # Set permissions
                sudo chown -R $USER:$USER /mnt/persistent-disk
                sudo chmod 777 /mnt/persistent-disk
            """

            logger.debug("Executing mount script on TPU VM")
            result = await self.run_command([mount_script])
            if result.get("returncode", -1) != 0:
                logger.error(
                    f"Mount script failed with stderr: {result.get('stderr', '')}"
                )
                raise RuntimeError(f"Failed to mount disk: {result.get('stderr', '')}")

            logger.info("Successfully mounted disk in TPU VM")

        except Exception as e:
            logger.error(f"Error during disk mount process: {str(e)}")
            raise RuntimeError(f"Failed to mount disk: {str(e)}")

    async def _delete_disk(self, disk_name: str) -> None:
        """Delete a persistent disk."""
        try:
            operation = self.compute_client.delete(
                project=self.config.project_id, zone=self.config.zone, disk=disk_name
            )
            operation.result()  # Wait for operation to complete
        except Exception as e:
            raise RuntimeError(f"Failed to delete disk: {str(e)}")

    async def _disk_exists(self, disk_name: str) -> bool:
        """Check if a persistent disk exists."""
        try:
            self.compute_client.get(
                project=self.config.project_id, zone=self.config.zone, disk=disk_name
            )
            return True
        except Exception:
            return False

    async def _create_tpu_node(self) -> Any:
        """Create a new TPU node."""
        logger.info(
            f"Creating TPU node in {self.config.zone} with type {self.config.accelerator_type}-{self.config.accelerator_core_count}"
        )
        parent = f"projects/{self.config.project_id}/locations/{self.config.zone}"

        tpu_type = (
            f"{self.config.accelerator_type}-{self.config.accelerator_core_count}"
        )

        # Create TPU node configuration
        node = tpu_v2.Node(
            accelerator_type=tpu_type,
            runtime_version=self.config.runtime_version,
            network_config=tpu_v2.NetworkConfig(enable_external_ips=True),
            scheduling_config=tpu_v2.SchedulingConfig(
                preemptible=self.config.preemptible
            ),
            tags=self.config.tags,
        )

        # If disk attachment is needed, configure it during node creation
        if self.config.attach_disk:
            disk_name = f"{self.config.name}-disk"
            if not await self._disk_exists(disk_name):
                logger.info(f"Creating new persistent disk: {disk_name}")
                disk_path = await self._create_persistent_disk(disk_name)
            else:
                logger.info(f"Using existing disk: {disk_name}")
                disk_path = f"projects/{self.config.project_id}/zones/{self.config.zone}/disks/{disk_name}"

            # Add disk configuration directly to node creation
            # TODO: if cores is > 8, make it read-only else read-write works too
            node.data_disks = [
                tpu_v2.AttachedDisk(
                    source_disk=disk_path, mode=tpu_v2.AttachedDisk.DiskMode.READ_WRITE
                )
            ]

        # Create node with complete configuration
        request = tpu_v2.CreateNodeRequest(
            parent=parent, node_id=self.config.name, node=node
        )

        logger.info(f"Submitting TPU node creation request for {self.config.name}")
        operation = await self.client.create_node(request=request)
        tpu_node = await operation.result()

        # Wait for TPU to be ready
        logger.info("Waiting for TPU node to reach READY state")
        max_retries = 30
        for attempt in range(max_retries):
            state = await self._get_tpu_state()
            if state == "READY":
                logger.info("TPU node is now READY")
                break
            await asyncio.sleep(10)
        else:
            error_msg = "TPU failed to reach READY state after maximum retries"
            logger.error(error_msg)
            raise TimeoutError(error_msg)


        if self.config.attach_disk:
            logger.info("Mounting disk in container")
            await self._mount_disk_in_container()
            
        if self.config.docker_image:
            logger.info(f"Starting docker container with image {self.config.docker_image}")
            await self._run_docker_container()

        logger.info(f"Successfully created TPU node {self.config.name}")
        return tpu_node

    async def _run_docker_container(self) -> None:
        """Run a docker container on TPU VM."""
        if not self.config.docker_image:
            logger.debug("No docker image specified, skipping container setup")
            return

        container_name = self.config.docker_image.split("/")[-1].split(":")[0]
        logger.info(f"Setting up docker container '{container_name}' from image {self.config.docker_image}")

        # Stop and remove existing container if it exists
        await self.run_command([f"sudo docker stop {container_name} 2>/dev/null || true"])
        await self.run_command([f"sudo docker rm {container_name} 2>/dev/null || true"])

        # Pull the docker image
        logger.info(f"Pulling docker image: {self.config.docker_image}")
        result = await self.run_command([f"sudo docker pull {self.config.docker_image}"])
        if result["returncode"] != 0:
            raise RuntimeError(f"Failed to pull docker image: {result['stderr']}")

        # Construct docker run command with properly escaped environment variables
        docker_run_cmd = [
            "sudo docker run -d --rm --net=host --shm-size=16G",
            f"--name {container_name} --privileged"
        ]

        # Add environment variables if specified
        if self.config.docker_env:
            logger.debug(f"Adding environment variables: {self.config.docker_env}")
            for key, value in self.config.docker_env.items():
                # Properly escape the value using shell escaping
                escaped_value = value.replace('"', '\\"')
                docker_run_cmd.append(f'-e {key}="{escaped_value}"')

        if self.config.attach_disk:
            logger.debug("Mounting persistent disk volume in container")
            docker_run_cmd.append("-v /mnt/persistent-disk:/mnt/persistent-disk")

        docker_run_cmd.append(self.config.docker_image)

        # Join all parts with spaces
        final_cmd = " ".join(docker_run_cmd)

        # Run the container
        logger.info("Starting docker container")
        result = await self.run_command([final_cmd])

        if result["returncode"] != 0:
            raise RuntimeError(f"Failed to start docker container: {result['stderr']}")

        logger.info(f"Successfully started docker container '{container_name}'")

    async def start(self) -> Dict[str, Any]:
        """Start the TPU node"""
        if not self.config or not self.node_name:
            logger.error("Provider not initialized - config or node_name missing")
            raise RuntimeError("Provider not initialized")

        logger.info(f"Starting TPU node {self.node_name}")
        
        try:
            state = await self._get_tpu_state()
        except google.api_core.exceptions.NotFound:
            logger.info("No existing TPU found, creating new TPU node")
            state = None
        
        try:
            if state is None:
                # Try all zones in sequence until successful
                all_zones = [self.config.zone] + self.config.backup_zones
                last_error = None
                
                for zone in all_zones:
                    try:
                        self.config.zone = zone
                        self.node_name = f"projects/{self.config.project_id}/locations/{zone}/nodes/{self.config.name}"
                        logger.info(f"Attempting to create TPU in zone {zone}")
                        response = await self._create_tpu_node()
                        logger.info(f"Successfully created TPU in zone {zone}")
                        return {
                            "status": "started", 
                            "response": str(response), 
                            "zone": zone,
                            "ip_address": await self.get_ip_address()
                        }
                    except Exception as e:
                        last_error = e
                        logger.warning(f"Failed to create TPU in zone {zone}: {str(e)}")
                        continue
                
                # If we get here, all zones failed
                error_msg = f"Failed to create TPU in any available zone. Last error: {str(last_error)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            elif state == "READY":
                logger.info("TPU is already in READY state")
                response = await self.client.get_node(name=self.node_name)
                return {
                    "status": "started", 
                    "response": str(response), 
                    "zone": self.config.zone,
                    "ip_address": await self.get_ip_address()
                }
            else:
                logger.info(f"TPU exists but not ready (state={state}), starting node")
                request = tpu_v2.StartNodeRequest(name=self.node_name)
                operation = await self.client.start_node(request=request)
                response = await operation.result()
                return {
                    "status": "started", 
                    "response": str(response), 
                    "zone": self.config.zone,
                    "ip_address": await self.get_ip_address()
                }

        except Exception as e:
            error_msg = f"Failed to start TPU: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    async def stop(self) -> Dict[str, Any]:
        """Stop the TPU node"""
        logger.info("Attempting to stop TPU node")

        if not self.config or not self.node_name:
            logger.error("Provider not initialized")
            raise RuntimeError("Provider not initialized")

        # TODO: if v5p, can't stop only delete. Handle that

        # Wait for TPU to finish creating before attempting to stop
        max_retries = 30  # Maximum number of retries (5 minutes total with 10s sleep)
        logger.debug(
            f"Waiting for TPU to be in stoppable state, max {max_retries} retries"
        )

        for retry in range(max_retries):
            tpu_state = await self._get_tpu_state()
            logger.debug(
                f"Current TPU state: {tpu_state} (attempt {retry + 1}/{max_retries})"
            )

            if tpu_state == "CREATING":
                logger.debug("TPU still creating, waiting 30 seconds before next check")
                await asyncio.sleep(30)  # Wait 30 seconds before checking again
                continue

            if tpu_state in ["READY", "RUNNING"]:
                # TPU is in a state where it can be stopped
                logger.info(f"TPU is in stoppable state ({tpu_state}), initiating stop")
                request = tpu_v2.StopNodeRequest(name=self.node_name)
                operation = await self.client.stop_node(request=request)
                response = await operation.result()
                logger.info("TPU node stopped successfully")
                return {"status": "stopped", "response": str(response)}

            # If TPU is already stopped or in another state, return current status
            logger.info(f"TPU already in non-running state: {tpu_state}")
            return {"status": "already_stopped", "state": tpu_state}

        error_msg = "Timeout waiting for TPU to finish creating before stopping"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    async def get_status(self) -> AcceleratorStatus:
        """Get TPU node status"""
        if not self.config or not self.node_name:
            raise RuntimeError("Provider not initialized")

        tpu_state = await self._get_tpu_state()

        # Map TPU states to AcceleratorStatus
        status_mapping = {
            "CREATING": AcceleratorStatus.PROVISIONING,
            "READY": AcceleratorStatus.READY,
            "STOPPING": AcceleratorStatus.STOPPING,
            "STOPPED": AcceleratorStatus.TERMINATED,
            "DELETING": AcceleratorStatus.TERMINATED,
            "RESTARTING": AcceleratorStatus.PROVISIONING,
            "STARTING": AcceleratorStatus.PROVISIONING,
            None: AcceleratorStatus.UNKNOWN,
        }

        return status_mapping.get(tpu_state, AcceleratorStatus.UNKNOWN)

    async def get_metrics(self) -> AcceleratorMetrics:
        """Get TPU metrics - Note: Actual TPU metrics implementation needed"""
        # This is a placeholder - actual TPU metrics collection needed
        metrics = AcceleratorMetrics()
        metrics.cpu_usage = 0.0
        metrics.memory_usage = 0.0
        return metrics

    async def run_command(
        self,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run command on TPU VM"""
        if not self.config:
            raise RuntimeError("Provider not initialized")

        # Construct gcloud command to run on TPU VM
        gcloud_cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            self.config.name,
            f"--zone={self.config.zone}",
            "--command",
            " ".join(command),
        ]

        logger.debug(f"Running command on TPU VM: {' '.join(gcloud_cmd)}")
        process = await asyncio.create_subprocess_exec(
            *gcloud_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f"Command failed with return code {process.returncode}")
            logger.error(f"stderr: {stderr.decode() if stderr else ''}")
        else:
            logger.info(f"Command completed successfully")
            logger.debug(f"stdout: {stdout.decode() if stdout else ''}")

        return {
            "returncode": process.returncode,
            "stdout": stdout.decode() if stdout else "",
            "stderr": stderr.decode() if stderr else "",
        }

    async def upload_file(self, local_path: Path, remote_path: Path) -> None:
        """Upload file to TPU VM using gcloud scp"""
        if not self.config:
            raise RuntimeError("Provider not initialized")

        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "scp",
            str(local_path),
            f"{self.config.name}:{str(remote_path)}",
            f"--zone={self.config.zone}",
        ]

        logger.debug(f"Uploading file to TPU VM: {' '.join(cmd)}")
        process = await asyncio.create_subprocess_exec(*cmd)
        await process.wait()
        if process.returncode != 0:
            logger.error("Failed to upload file to TPU VM")
            raise RuntimeError(f"Failed to upload file to TPU VM")
        logger.info(f"Successfully uploaded {local_path} to {remote_path} on TPU VM")

    async def download_file(self, remote_path: Path, local_path: Path) -> None:
        """Download file from TPU VM using gcloud scp"""
        if not self.config:
            raise RuntimeError("Provider not initialized")

        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "scp",
            f"{self.config.name}:{str(remote_path)}",
            str(local_path),
            f"--zone={self.config.zone}",
        ]

        logger.debug(f"Downloading file from TPU VM: {' '.join(cmd)}")
        process = await asyncio.create_subprocess_exec(*cmd)
        await process.wait()
        if process.returncode != 0:
            logger.error("Failed to download file from TPU VM")
            raise RuntimeError(f"Failed to download file from TPU VM")
        logger.info(
            f"Successfully downloaded {remote_path} from TPU VM to {local_path}"
        )

    def _get_region_for_accelerator(self, accelerator_type: str) -> str:
        """Get a region based on accelerator type."""
        if accelerator_type.startswith("v3"):
            zones = [
                "europe-west4-a",
                "europe-west4-b",
                "europe-west4-c",
                "us-central1-a",
            ]
            return random.choice(zones)
        elif accelerator_type.startswith("v5"):
            zones = [
                "europe-west4-b",
                "us-east5-a"
            ]
            return random.choice(zones)

