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

print("\nTrying to initialize JAX...")
try:
    # Set initialization timeout
    os.environ['JAX_PLATFORMS'] = 'tpu'  # Force TPU platform
    print("Set platform to TPU")
    
    # Try to get backend info with timeout
    start_time = time.time()
    timeout = 30  # 30 seconds
    
    while time.time() - start_time < timeout:
        try:
            backend = jax.extend.backend.get_backend()
            print("Backend:", backend)
            print("JAX version:", jax.__version__)
            
            # Try to get platform info
            platform = backend.platform
            print("Platform:", platform)
            break
        except Exception as e:
            print(".", end="", flush=True)
            time.sleep(1)
    else:
        print("\nTimeout waiting for JAX initialization")
        
except Exception as e:
    print("Error initializing JAX:", str(e))

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
