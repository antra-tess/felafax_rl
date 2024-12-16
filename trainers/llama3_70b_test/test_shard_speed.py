import os
import time
import sys

def main():
    model_path = "/mnt/gcs-bucket/llama-70b-files/llama-70b-files"
    # Get all safetensors files
    shard_files = [f for f in os.listdir(model_path) if f.startswith("model-") and f.endswith(".safetensors")]
    shard_files.sort()  # Ensure consistent order
    
    num_iterations = 5
    for i in range(num_iterations):
        # Pick a different shard each time
        shard_file = shard_files[i % len(shard_files)]
        shard_path = os.path.join(model_path, shard_file)
        
        # Get file size
        size = os.path.getsize(shard_path)
        size_gb = size/1024/1024/1024
        
        print(f"\nIteration {i+1}/{num_iterations}")
        print(f"Testing read speed for: {shard_file}")
        print(f"File size: {size_gb:.2f} GB")
        
        # Read file and measure speed
        start = time.time()
        with open(shard_path, 'rb') as f:
            data = f.read()
        read_time = time.time() - start
        
        speed = size_gb / read_time
        print(f"Read {size_gb:.2f} GB in {read_time:.2f} seconds")
        print(f"Speed: {speed:.2f} GB/s")
        
        # Force clear any OS cache
        os.system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
        time.sleep(1)  # Give system time to clear caches

if __name__ == "__main__":
    main()
