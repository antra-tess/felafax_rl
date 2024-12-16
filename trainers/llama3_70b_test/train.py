import os
import sys

# Check for HF_TOKEN at startup
if not os.environ.get('HF_TOKEN'):
    print("Error: HF_TOKEN environment variable is not set")
    sys.exit(1)

def get_worker_info():
    """Get TPU worker information from hostname."""
    hostname = os.uname()[1]
    worker_id = int(hostname.split('w-')[1]) if 'w-' in hostname else 0
    return worker_id, 8

# Initialize JAX before importing
process_id, num_processes = get_worker_info()
os.environ['JAX_PROCESS_COUNT'] = str(num_processes)
os.environ['JAX_PROCESS_INDEX'] = str(process_id)

import jax
jax.distributed.initialize()

# Now import the rest
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaConfig
from felafax.trainer_engine.data.data import SFTDataset, create_dataloader, DatasetConfig
from felafax.trainer_engine.trainer import Trainer, TrainerConfig
from felafax.trainer_engine.models.llama3.jax.model import LlamaForCausalLM

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
    
    # Set up TPU mesh
    devices = np.array(jax.devices()).reshape(1, 8, 4)  # 8 workers × 4 devices per worker
    mesh = jax.sharding.Mesh(devices, ("batch", "fsdp", "mp"))
    
    # Load from local shards based on worker ID
    local_path = "/tmp/model-shards"
    print(f"Loading model from local shards at: {local_path}")

    # Configure training
    trainer_config = TrainerConfig(
        model_name=local_path,
        num_tpus=32,
        mesh_shape=(1, 8, 4),
        learning_rate=1e-5,
        num_steps=20,
        base_dir=f"/tmp/model-shards/checkpoints/worker_{process_id}",
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
