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
from felafax.trainer_engine.data.data import SFTDataset, create_dataloader
from felafax.trainer_engine.trainer import Trainer, TrainerConfig
from felafax.trainer_engine.models.llama3.jax.model import LlamaForCausalLM

from dataclasses import dataclass
from src.felafax.trainer_engine.data.data import DefaultDatasetLoader, DatasetConfig

@dataclass
class AlpacaDatasetConfig(DatasetConfig):
    """Configuration for Alpaca dataset."""
    data_source: str = "yahma/alpaca-cleaned"
    max_seq_length: int = 1024

class AlpacaDataset(DefaultDatasetLoader):
    """Alpaca dataset for supervised fine-tuning."""
    
    def __init__(self, config: AlpacaDatasetConfig):
        super().__init__(config)
        
    def setup(self, tokenizer=None):
        self.tokenizer = tokenizer or self.tokenizer
        dataset = load_dataset(self.config.data_source, split="train")
        train_data = [sample for sample in dataset]
        
        self.train_dataset = SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length
        )
        self.val_dataset = None

def main():
    
    # Set up TPU mesh
    devices = np.array(jax.devices()).reshape(1, 8, 4)  # 8 workers × 4 devices per worker
    mesh = jax.sharding.Mesh(devices, ("batch", "fsdp", "mp"))
    
    # Configure training
    trainer_config = TrainerConfig(
        model_name="meta-llama/Llama-3.1-70B",
        num_tpus=32,
        mesh_shape=(1, 8, 4),
        learning_rate=1e-5,
        num_steps=20,
        base_dir=f"/tmp/llama_test/worker_{process_id}",
        use_lora=True,
        lora_rank=8,
    )
    
    # Load tokenizer with HF token
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set")
    
    print(f"Using HF token: {hf_token[:4]}...{hf_token[-4:]}")  # Print first and last 4 chars
    
    # Update trainer config with token
    trainer_config.hf_token = hf_token
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        trainer_config.model_name,
        token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    
    # Configure and load dataset
    dataset_config = AlpacaDatasetConfig(
        data_source="yahma/alpaca-cleaned",
        max_seq_length=1024,
        batch_size=8,
        num_workers=4
    )
    alpaca_dataset = AlpacaDataset(config=dataset_config)
    alpaca_dataset.setup(tokenizer=tokenizer)
    
    train_dataloader = alpaca_dataset.train_dataloader()
    
    # Initialize and run trainer
    trainer = Trainer(
        trainer_config,
        train_dataloader=train_dataloader,
        val_dataloader=None
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
