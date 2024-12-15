import os
import jax
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting JAX device test")
    
    # Print hostname
    hostname = os.uname()[1]
    logger.info(f"Running on host: {hostname}")
    
    # Get backend info
    backend = jax.lib.xla_bridge.get_backend()
    logger.info(f"JAX backend: {backend}")
    
    # Check devices
    devices = jax.devices()
    logger.info(f"Number of devices: {len(devices)}")
    
    # Print device info
    for i, dev in enumerate(devices):
        logger.info(f"Device {i}: {dev}")
    
    # Try simple computation
    logger.info("Attempting simple computation...")
    x = jax.numpy.ones((2, 2))
    result = jax.numpy.sum(x)
    logger.info(f"Test computation result: {result}")
    
    logger.info("Test completed successfully!")

if __name__ == "__main__":
    main()
