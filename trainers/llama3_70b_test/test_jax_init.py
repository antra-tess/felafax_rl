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

print("\nInitializing JAX...")
try:
    # Try to get backend info directly
    print("\nGetting backend information...")
    print("JAX version:", jax.__version__)
    
    # Try to get device count first
    device_count = jax.device_count()
    print(f"Device count: {device_count}")
    
    # Try to get local device count
    local_device_count = jax.local_device_count()
    print(f"Local device count: {local_device_count}")
    
    # Try to get process information
    try:
        process_count = jax.distributed.process_count()
        process_index = jax.distributed.process_index()
        print(f"\nProcess information:")
        print(f"Process index: {process_index}")
        print(f"Total processes: {process_count}")
    except Exception as e:
        print("Note: Could not get process information:", str(e))
    
    # Try to get platform information
    try:
        from jax.lib import xla_client
        platform = xla_client.get_local_backend().platform
        print(f"\nPlatform information:")
        print(f"Platform: {platform}")
    except Exception as e:
        print("Note: Could not get platform information:", str(e))
    
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
