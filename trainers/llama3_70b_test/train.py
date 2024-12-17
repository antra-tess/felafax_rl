import os
import sys
import json
import numpy as np
import safetensors.torch
from datetime import datetime
from transformers import AutoTokenizer
from jax_smi import initialise_tracking

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    print(f"[{timestamp}] {msg}", flush=True)

log("Starting script execution")

# Initialize JAX SMI tracking
log("Initializing JAX SMI tracking...")
initialise_tracking()
log("JAX SMI tracking initialized")

# Check for HF_TOKEN at startup
if not os.environ.get('HF_TOKEN'):
    log("Error: HF_TOKEN environment variable is not set")
    sys.exit(1)

def get_worker_info():
    """Get TPU worker information from hostname."""
    hostname = os.uname()[1]
    log(f"Hostname: {hostname}")
    worker_id = int(hostname.split('w-')[1]) if 'w-' in hostname else 0
    log(f"Identified as worker {worker_id}")
    return worker_id, 8

# Initialize JAX first
log("Getting worker info...")
process_id, num_processes = get_worker_info()

log(f"Setting JAX environment (process {process_id} of {num_processes})")
os.environ['JAX_PROCESS_COUNT'] = str(num_processes)
os.environ['JAX_PROCESS_INDEX'] = str(process_id)

log("Importing JAX...")
import jax
log("Initializing JAX distributed...")
jax.distributed.initialize()
log("JAX distributed initialization complete")

# Now load model files
log("Loading model files into memory...")
local_path = "/tmp/model-shards"

# Load config first
log("Loading config.json...")
with open(os.path.join(local_path, "config.json")) as f:
    config_data = json.load(f)
    
# Add 'type' to rope_scaling if needed
if 'rope_scaling' in config_data:
    config_data['rope_scaling']['type'] = config_data['rope_scaling']['rope_type']
    log("Added 'type' field to rope_scaling configuration")

log("Config loaded and updated successfully")

# Get shard range for this worker
def get_worker_shards(worker_id):
    shard_ranges = {
        0: (1, 4),    # Worker 0: shards 1-4
        1: (4, 8),    # Worker 1: shards 4-8
        2: (8, 12),   # Worker 2: shards 8-12
        3: (12, 15),  # Worker 3: shards 12-15
        4: (16, 19),  # Worker 4: shards 16-19
        5: (19, 23),  # Worker 5: shards 19-23
        6: (23, 27),  # Worker 6: shards 23-27
        7: (27, 30),  # Worker 7: shards 27-30
    }
    return shard_ranges[worker_id]

start_shard, end_shard = get_worker_shards(process_id)
log(f"Worker {process_id} loading shards {start_shard}-{end_shard}")

# Load model shards into memory
loaded_shards = {}
for shard_idx in range(start_shard, end_shard + 1):
    shard_file = f"model-{shard_idx:05d}-of-00030.safetensors"
    shard_path = os.path.join(local_path, shard_file)
    log(f"Loading shard: {shard_path}")
    shard_data = safetensors.torch.load_file(shard_path)
    loaded_shards[shard_idx] = shard_data
    log(f"Loaded shard {shard_idx} successfully")

log("All shards loaded into memory")

# Import remaining dependencies
log("Importing core dependencies...")
from datasets import load_dataset
from transformers import LlamaConfig
from felafax.trainer_engine.data.data import SFTDataset, create_dataloader, DatasetConfig
from felafax.trainer_engine.trainer import Trainer, TrainerConfig
from felafax.trainer_engine.models.llama3.jax.model import LlamaForCausalLM
log("Core dependencies imported")

# Small delay to ensure all workers are synchronized
import time
time.sleep(1)

# Load tokenizer
log("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(local_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
log("Tokenizer loaded successfully")

class AlpacaDataset(SFTDataset):
    """Alpaca dataset for supervised fine-tuning."""
    
    def apply_format(self, example):
        instruction = example["instruction"]
        input_text = example["input"]
        output = example["output"]
        
        prompt = f"Instruction: {instruction}\n"
        if input_text:
            prompt += f"Input: {input_text}\n"
        prompt += "Output: "
        return prompt, output

def main():
    log("Starting main function")
    
    # Set up TPU mesh
    log("Getting JAX devices...")
    devices = jax.devices()
    log(f"Found {len(devices)} devices")
    log("Creating device mesh...")
    devices = np.array(devices).reshape(1, 8, 4)  # 8 workers × 4 devices per worker
    log("Creating JAX mesh...")
    mesh = jax.sharding.Mesh(devices, ("batch", "fsdp", "mp"))
    log("Mesh creation complete")
    
    # Load from local shards based on worker ID
    local_path = "/tmp/model-shards"
    log(f"Loading model from local shards at: {local_path}")
    log(f"Available files in {local_path}:")
    if os.path.exists(local_path):
        files = os.listdir(local_path)
        log(f"Found {len(files)} files:")
        for f in files:
            log(f"  {f}")
    else:
        log(f"Directory {local_path} does not exist!")

    # Configure training
    trainer_config = TrainerConfig(
        model_name=local_path,
        num_tpus=32,
        mesh_shape=(1, 8, 4),
        learning_rate=1e-5,
        num_steps=20,
        base_dir=f"/dev/shm/checkpoints/worker_{process_id}",
        use_lora=True,
        lora_rank=8,
        param_dtype="bfloat16",
        compute_dtype="bfloat16",
    )
    
    # Load tokenizer from local path
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    
    # Configure and load dataset
    dataset_config = DatasetConfig(
        data_source="yahma/alpaca-cleaned",
        max_seq_length=1024,
        batch_size=8,
        num_workers=4
    )
    
    # Load dataset with HF token
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set")
    
    print(f"Using HF token: {hf_token[:4]}...{hf_token[-4:]}")
    
    # Load dataset
    dataset = load_dataset(
        dataset_config.data_source,
        split="train",
        token=hf_token
    )
    train_dataset = AlpacaDataset(
        config=dataset_config,
        data=[ex for ex in dataset],
        tokenizer=tokenizer
    )
    
    train_dataloader = create_dataloader(
        dataset_config,
        train_dataset,
        shuffle=True
    )
    
    # Initialize and run trainer
    trainer = Trainer(
        trainer_config,
        train_dataloader=train_dataloader,
        val_dataloader=None
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
