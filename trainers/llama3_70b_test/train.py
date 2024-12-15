import os
import jax
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from felafax.trainer_engine.data.data import SFTDataset, create_dataloader
from felafax.trainer_engine.trainer import Trainer, TrainerConfig
from felafax.trainer_engine.automodel_lib import AutoJAXModelForCausalLM

def get_worker_info():
    """Get TPU worker information from hostname."""
    hostname = os.uname()[1]
    worker_id = int(hostname.split('w-')[1]) if 'w-' in hostname else 0
    return worker_id, 8

class AlpacaDataset(SFTDataset):
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
    # Initialize JAX
    process_id, num_processes = get_worker_info()
    jax.distributed.initialize()
    
    # Set up TPU mesh
    devices = np.array(jax.devices()).reshape(1, 8, 8)
    mesh = jax.sharding.Mesh(devices, ("batch", "fsdp", "mp"))
    
    # Configure training
    trainer_config = TrainerConfig(
        model_name="meta-llama/Llama-3.1-70B",
        num_tpus=64,
        mesh_shape=(1, 8, 8),
        learning_rate=1e-5,
        num_steps=20,
        base_dir=f"/tmp/llama_test/worker_{process_id}",
        use_lora=True,
        lora_rank=8,
        lora_alpha=16,
    )
    
    # Load model
    model_path, model, config, tokenizer = AutoJAXModelForCausalLM.from_pretrained(
        trainer_config.model_name,
        dtype=jax.numpy.bfloat16,
        param_dtype=jax.numpy.bfloat16,
        lora_rank=trainer_config.lora_rank,
        lora_alpha=trainer_config.lora_alpha,
    )
    
    # Load dataset
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    train_dataset = AlpacaDataset(
        data=[ex for ex in dataset],
        tokenizer=tokenizer,
        max_seq_length=1024
    )
    
    train_dataloader = create_dataloader(
        {"batch_size": 8, "num_workers": 4},
        train_dataset,
        shuffle=True
    )
    
    # Initialize and run trainer
    trainer = Trainer(
        trainer_config,
        train_dataloader=train_dataloader,
        val_dataloader=None,
        model=model,
        model_config=config,
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
