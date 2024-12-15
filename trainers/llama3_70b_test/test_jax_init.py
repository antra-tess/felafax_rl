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

print("\nSetting up JAX environment...")
try:
    # Set JAX environment variables
    os.environ['JAX_PLATFORMS'] = 'tpu'
    os.environ['JAX_BACKEND_TARGET'] = 'libtpu'
    os.environ['PJRT_DEVICE'] = 'TPU'
    print("Set JAX environment variables")
    
    # Try direct TPU initialization
    print("\nTrying TPU initialization...")
    from jax.config import config
    config.update('jax_platform_name', 'tpu')
    config.update('jax_xla_backend', 'tpu')
    print("JAX config updated for TPU")
    
    # Try to initialize backend
    print("\nGetting backend information...")
    backend = jax.extend.backend.get_backend()
    print("Backend:", backend)
    print("JAX version:", jax.__version__)
    
    # Try to get platform info
    platform = backend.platform
    print("Platform:", platform)
    
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
