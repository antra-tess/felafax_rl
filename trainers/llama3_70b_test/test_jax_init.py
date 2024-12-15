import jax
import time
print("Starting JAX TPU test...")
print("Waiting for TPU initialization...")

# Try to get TPU devices with a timeout
start_time = time.time()
timeout = 30  # 30 seconds timeout

while time.time() - start_time < timeout:
    try:
        devices = jax.devices()
        if devices:
            print(f"\nFound {len(devices)} devices:")
            for i, d in enumerate(devices):
                print(f"Device {i}: {d}")
            break
    except:
        print(".", end="", flush=True)
        time.sleep(1)
else:
    print("\nTimeout waiting for TPU devices")

print("\nTest complete")
