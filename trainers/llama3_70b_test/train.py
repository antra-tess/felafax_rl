import os
import sys
from datetime import datetime

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    print(f"[{timestamp}] {msg}", flush=True)

log("Starting script execution")

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

# Initialize JAX before importing
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

log("Starting imports...")
log("Importing numpy...")
import numpy as np
log("Importing datasets...")
from datasets import load_dataset
log("Importing transformers...")
from transformers import AutoTokenizer, LlamaConfig
log("Importing felafax data modules...")
from felafax.trainer_engine.data.data import SFTDataset, create_dataloader, DatasetConfig
log("Importing felafax trainer modules...")
from felafax.trainer_engine.trainer import Trainer, TrainerConfig
log("Importing felafax model modules...")
from felafax.trainer_engine.models.llama3.jax.model import LlamaForCausalLM
log("All imports complete")

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
