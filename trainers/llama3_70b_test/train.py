import os
import jax
import numpy as np
from typing import Dict, Any, Tuple

from felafax.trainer_engine.data.data import SFTDataset, create_dataloader
from felafax.trainer_engine.trainer import Trainer, TrainerConfig
from felafax.trainer_engine.automodel_lib import AutoJAXModelForCausalLM

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
    # Configure TPU mesh for 70B model
    mesh_shape = (1, 8, 8)  # (batch, fsdp, mp) - optimized for 70B
    devices = np.array(jax.devices()).reshape(*mesh_shape)
    mesh = jax.sharding.Mesh(devices, ("batch", "fsdp", "mp"))
    
    # Training configuration
    trainer_config = TrainerConfig(
        model_name="meta-llama/Llama-3.1-70B",
        num_tpus=64,
        mesh_shape=mesh_shape,
        learning_rate=1e-5,
        num_steps=20,  # Small number for testing
        base_dir="/tmp/llama_test",  # Adjust as needed
        use_lora=True,  # Use LoRA to reduce memory requirements
        lora_rank=8,
        lora_alpha=16,
    )
    
    # Dataset configuration
    dataset_config = {
        "data_source": "yahma/alpaca-cleaned",
        "max_seq_length": 1024,
        "batch_size": 8,
        "num_workers": 4,
        "max_examples": 100  # Limit for testing
    }
    
    # Load model
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
    
    # Create datasets
    print("Loading dataset...")
    train_dataset = AlpacaDataset(
        dataset_config["data_source"],
        tokenizer=tokenizer,
        max_length=dataset_config["max_seq_length"],
        max_examples=dataset_config["max_examples"]
    )
    
    train_dataloader = create_dataloader(
        dataset_config,
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
    print("Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()
