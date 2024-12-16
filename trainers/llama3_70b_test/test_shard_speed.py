import os
import time
import sys

def main():
    model_path = "/mnt/gcs-bucket/llama-70b-files/llama-70b-files"
    # Get first safetensors file
    shard_file = next(f for f in os.listdir(model_path) if f.startswith("model-") and f.endswith(".safetensors"))
    shard_path = os.path.join(model_path, shard_file)
    
    # Get file size
    size = os.path.getsize(shard_path)
    size_gb = size/1024/1024/1024
    
    print(f"Testing read speed for: {shard_file}")
    print(f"File size: {size_gb:.2f} GB")
    
    # Read file and measure speed
    start = time.time()
    with open(shard_path, 'rb') as f:
        data = f.read()
    read_time = time.time() - start
    
    speed = size_gb / read_time
    print(f"\nRead {size_gb:.2f} GB in {read_time:.2f} seconds")
    print(f"Average read speed: {speed:.2f} GB/s")

if __name__ == "__main__":
    main()
