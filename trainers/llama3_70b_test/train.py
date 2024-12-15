import os
import jax
import numpy as np
from typing import Dict, Any, Tuple
import logging
from functools import partial

from felafax.trainer_engine.data.data import SFTDataset, create_dataloader
from felafax.trainer_engine.trainer import Trainer, TrainerConfig
from felafax.trainer_engine.automodel_lib import AutoJAXModelForCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Worker %(process)d - %(message)s'
)
logger = logging.getLogger(__name__)

def get_worker_info():
    """Get TPU worker information from hostname."""
    hostname = os.uname()[1]
    if 'w-' not in hostname:
        return 0, 1
    worker_id = int(hostname.split('w-')[1])
    # For v4-64, we have 8 workers
    return worker_id, 8

def initialize_jax(process_id: int, num_processes: int):
    """Initialize JAX's distributed runtime."""
    logger.info(f"Initializing JAX process {process_id} of {num_processes}")
    
    # Get coordinator address (worker-0)
    coordinator = f"t1v-n-affc976c-w-0:1234"
    
    try:
        jax.distributed.initialize(
            coordinator_address=coordinator,
            num_processes=num_processes,
            process_id=process_id
        )
        logger.info(f"JAX initialization successful for process {process_id}")
    except Exception as e:
        logger.error(f"JAX initialization failed: {e}")
        raise

def run_training():
    """Run training on a single TPU worker."""
    """Run training on TPU worker."""
    # Get worker information
    process_id, num_processes = get_worker_info()
    logger.info(f"Starting training on process {process_id} of {num_processes}")
    
    try:
        # Configure local devices
        local_devices = jax.local_devices()
        logger.info(f"Process {process_id} sees {len(local_devices)} local devices")
        
        # Wait for all processes to see their devices
        jax.distributed.barrier()
        
        # Configure TPU mesh for 70B model
        global_devices = jax.devices()
        logger.info(f"Total available devices: {len(global_devices)}")
        
        # Create mesh with error handling
        try:
            devices = np.array(global_devices).reshape(1, 8, 8)
            mesh = jax.sharding.Mesh(devices, ("batch", "fsdp", "mp"))
            logger.info("Mesh creation successful")
        except Exception as e:
            logger.error(f"Mesh creation failed: {e}")
            raise
        
        # Training configuration
        trainer_config = TrainerConfig(
            model_name="meta-llama/Llama-3.1-70B",
            num_tpus=64,
            mesh_shape=(1, 8, 8),
            learning_rate=1e-5,
            num_steps=20,  # Small number for testing
            base_dir=f"/tmp/llama_test/worker_{process_id}",  # Separate dirs per worker
            use_lora=True,
            lora_rank=8,
            lora_alpha=16,
        )
        
        logger.info("Loading model...")
        # Load model and tokenizer
        model, config, tokenizer = AutoJAXModelForCausalLM.from_pretrained(
            trainer_config.model_name,
            mesh=mesh,
            dtype=jax.numpy.bfloat16,  # Use bfloat16 for TPU
            param_dtype=jax.numpy.bfloat16,
            use_lora=trainer_config.use_lora,
            lora_rank=trainer_config.lora_rank,
            lora_alpha=trainer_config.lora_alpha,
        )
        
        logger.info("Loading dataset...")
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
        logger.info("Starting training...")
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training step failed: {e}")
        raise

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
    process_id, num_processes = get_worker_info()
    
    try:
        # Initialize JAX distributed system
        initialize_jax(process_id, num_processes)
        
        # Run training
        run_training()
        
    except Exception as e:
        logger.error(f"Training failed on process {process_id}: {e}")
        raise

if __name__ == "__main__":
    main()
