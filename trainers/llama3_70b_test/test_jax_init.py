import os
import jax
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Worker %(process)d - %(message)s'
)
logger = logging.getLogger(__name__)

def get_worker_info():
    hostname = os.uname()[1]
    if 'w-' not in hostname:
        return 0, 1
    worker_id = int(hostname.split('w-')[1])
    return worker_id, 8

def initialize_jax(process_id: int, num_processes: int):
    logger.info(f"Initializing JAX process {process_id} of {num_processes}")
    coordinator = f"t1v-n-affc976c-w-0:1234"
    
    try:
        jax.distributed.initialize(
            coordinator_address=coordinator,
            num_processes=num_processes,
            process_id=process_id
        )
        logger.info(f"JAX initialization successful for process {process_id}")
    except Exception as e:
        logger.error(f"JAX initialization failed: {e}")
        raise

def main():
    process_id, num_processes = get_worker_info()
    logger.info(f"Starting process {process_id} of {num_processes}")
    
    try:
        # Step 1: Initialize JAX
        initialize_jax(process_id, num_processes)
        logger.info("JAX initialization complete")
        
        # Step 2: Check local devices
        local_devices = jax.local_devices()
        logger.info(f"Process {process_id} sees {len(local_devices)} local devices")
        for i, dev in enumerate(local_devices):
            logger.info(f"Local device {i}: {dev}")
        
        # Step 3: Try to create a simple array
        x = jax.numpy.ones((8, 8))
        logger.info(f"Created test array with shape {x.shape}")
        
        # Step 4: Try basic operation
        result = jax.numpy.sum(x)
        logger.info(f"Test computation result: {result}")
        
        logger.info("All tests passed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed on process {process_id}: {e}")
        raise

if __name__ == "__main__":
    main()
