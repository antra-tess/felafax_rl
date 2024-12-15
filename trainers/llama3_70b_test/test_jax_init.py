import os
import jax
import time
import numpy as np

print("Starting TPU environment check...")
print(f"Python process ID: {os.getpid()}")

# Print environment information
print("\nEnvironment variables:")
for k, v in sorted(os.environ.items()):
    if any(x in k for x in ['TPU', 'JAX', 'XLA']):
        print(f"{k}={v}")

print("\nSystem information:")
print("Hostname:", os.uname()[1])
print("Process ID:", os.getpid())
print("JAX version:", jax.__version__)

print("\nChecking TPU runtime...")
try:
    # Check if libtpu is loaded
    import ctypes
    tpu_lib = os.environ.get('TPU_LIBRARY_PATH')
    if tpu_lib:
        print(f"Loading TPU library from: {tpu_lib}")
        ctypes.CDLL(tpu_lib)
        print("TPU library loaded successfully")
    else:
        print("TPU_LIBRARY_PATH not set")
except Exception as e:
    print("Error loading TPU library:", str(e))

print("\nInitializing JAX for TPU...")
try:
    # Get worker information from hostname
    hostname = os.uname()[1]
    if 'w-' not in hostname:
        worker_id = 0
    else:
        worker_id = int(hostname.split('w-')[1])
    print(f"Worker ID: {worker_id}")

    # Set TPU-specific JAX flags
    jax.config.update('jax_platform_name', 'tpu')
    os.environ['JAX_PLATFORMS'] = 'tpu'
    print("Set JAX platform to TPU")

    # Initialize JAX runtime
    print("Initializing JAX runtime...")
    jax.distributed.initialize()
    print("JAX runtime initialized")

    # Now try to get device information
    print("\nGetting device information...")
    device_count = jax.device_count()
    print(f"Device count: {device_count}")
    
    local_device_count = jax.local_device_count()
    print(f"Local device count: {local_device_count}")
    
except Exception as e:
    print("Error in JAX setup:", str(e))
    import traceback
    traceback.print_exc()

print("\nTrying to get devices...")
try:
    # Try both local and all devices
    local_devices = jax.local_devices()
    all_devices = jax.devices()
    print(f"Number of local devices: {len(local_devices)}")
    print(f"Number of total devices: {len(all_devices)}")
    
    # Print detailed device info
    for i, dev in enumerate(local_devices):
        print(f"\nLocal device {i}:")
        print(f"  Platform: {dev.platform}")
        print(f"  Device kind: {dev.device_kind}")
        print(f"  Process index: {dev.process_index}")
except Exception as e:
    print("Error getting devices:", str(e))

print("\nTest complete")
