import os
import time
import jax
import numpy as np
from transformers import AutoTokenizer, LlamaConfig

def main():
    model_path = "/mnt/gcs-bucket/llama-70b-files/llama-70b-files"
    print(f"Testing read speed from: {model_path}")
    
    # Time tokenizer loading
    start = time.time()
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer_time = time.time() - start
    print(f"Tokenizer loaded in {tokenizer_time:.2f} seconds")
    
    # Time config loading
    start = time.time()
    print("\nLoading config...")
    config = LlamaConfig.from_pretrained(model_path)
    config_time = time.time() - start
    print(f"Config loaded in {config_time:.2f} seconds")
    
    # Time model file reading
    start = time.time()
    print("\nReading model files...")
    total_size = 0
    for filename in os.listdir(model_path):
        if filename.startswith("model-") and filename.endswith(".safetensors"):
            filepath = os.path.join(model_path, filename)
            size = os.path.getsize(filepath)
            total_size += size
            print(f"File: {filename}, Size: {size/1024/1024/1024:.2f} GB")
            
            # Read file to actually test I/O
            with open(filepath, 'rb') as f:
                _ = f.read()
                
    read_time = time.time() - start
    total_size_gb = total_size/1024/1024/1024
    speed = total_size_gb / read_time
    
    print(f"\nRead {total_size_gb:.2f} GB in {read_time:.2f} seconds")
    print(f"Average read speed: {speed:.2f} GB/s")

if __name__ == "__main__":
    main()
