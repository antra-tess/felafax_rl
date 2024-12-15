import os
import jax
import time

print("Starting TPU environment check...")

# Print environment information
print("\nEnvironment variables:")
for k, v in sorted(os.environ.items()):
    if 'TPU' in k or 'JAX' in k:
        print(f"{k}={v}")

print("\nHostname:", os.uname()[1])
print("Process ID:", os.getpid())

print("\nTrying to initialize JAX...")
try:
    # Try to get TPU platform
    platforms = jax.lib.xla_bridge.get_backend().platform
    print("Available platforms:", platforms)
except Exception as e:
    print("Error getting platform:", str(e))

print("\nTrying to get local devices...")
try:
    local_devices = jax.local_devices()
    print("Local devices:", local_devices)
except Exception as e:
    print("Error getting local devices:", str(e))

print("\nTest complete")
