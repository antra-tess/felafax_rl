import os
import jax
import numpy as np
from typing import Dict, Any, Tuple
import multiprocessing as mp
from functools import partial

from felafax.trainer_engine.data.data import SFTDataset, create_dataloader
from felafax.trainer_engine.trainer import Trainer, TrainerConfig
from felafax.trainer_engine.automodel_lib import AutoJAXModelForCausalLM

def get_worker_info():
    """Get TPU worker information from hostname."""
    hostname = os.uname()[1]
    if 'w-' not in hostname:
        return 0, 1
    worker_id = int(hostname.split('w-')[1])
    # For v4-64, we have 8 workers
    return worker_id, 8

def run_training(process_id: int, num_processes: int):
    """Run training on a single TPU worker."""
    # Configure local devices
    local_devices = jax.local_devices()
    print(f"Process {process_id}/{num_processes} sees {len(local_devices)} local devices")
    
    # Configure TPU mesh for 70B model - adjusted for multi-worker
    devices = np.array(jax.devices()).reshape(1, 8, 8)  # Global mesh shape stays the same
    mesh = jax.sharding.Mesh(devices, ("batch", "fsdp", "mp"))
    
    # Training configuration
    trainer_config = TrainerConfig(
        model_name="meta-llama/Llama-3.1-70B",
        num_tpus=64,
        mesh_shape=(1, 8, 8),
        learning_rate=1e-5,
        num_steps=20,  # Small number for testing
        base_dir="/tmp/llama_test",  # Adjust as needed
        use_lora=True,  # Use LoRA to reduce memory requirements
        lora_rank=8,
        lora_alpha=16,
    )
    
    if process_id == 0:
        print("Loading model...")
    model, config, tokenizer = AutoJAXModelForCausalLM.from_pretrained(
        trainer_config.model_name,
        mesh=mesh,
        dtype=jax.numpy.bfloat16,  # Use bfloat16 for TPU
        param_dtype=jax.numpy.bfloat16,
        use_lora=trainer_config.use_lora,
        lora_rank=trainer_config.lora_rank,
        lora_alpha=trainer_config.lora_alpha,
    )
    
    if process_id == 0:
        print("Loading dataset...")
    train_dataset = AlpacaDataset(
        "yahma/alpaca-cleaned",
        tokenizer=tokenizer,
        max_length=1024,
        max_examples=100
    )
    
    train_dataloader = create_dataloader(
        {"batch_size": 8, "num_workers": 4},
        train_dataset,
        shuffle=True
    )
    
    # Initialize trainer
    trainer = Trainer(
        trainer_config,
        train_dataloader=train_dataloader,
        val_dataloader=None,  # No validation for this test
        model=model,
        model_config=config,
    )
    
    # Start training
    if process_id == 0:
        print("Starting training...")
    trainer.train()

class AlpacaDataset(SFTDataset):
    """Simple Alpaca dataset for testing."""
    
    def apply_format(self, example: Dict[str, Any]) -> Tuple[str, str]:
        instruction = example["instruction"]
        input_text = example["input"]
        output = example["output"]
        
        if input_text:
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput: "
        else:
            prompt = f"Instruction: {instruction}\nOutput: "
            
        return prompt, output

def main():
    # Get worker information
    worker_id, num_workers = get_worker_info()
    print(f"Starting process {worker_id} of {num_workers}")
    
    # Create and start processes
    if worker_id == 0:
        # Only worker 0 spawns processes
        processes = []
        for i in range(num_workers):
            p = mp.Process(target=run_training, args=(i, num_workers))
            p.start()
            processes.append(p)
            
        # Wait for all processes
        for p in processes:
            p.join()
    else:
        # Other workers just run their part
        run_training(worker_id, num_workers)

if __name__ == "__main__":
    # Initialize JAX distributed system
    jax.distributed.initialize()
    main()
