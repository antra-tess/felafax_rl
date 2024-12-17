"""Loads and saves checkpoint."""

import os
import json
from typing import Optional, Tuple, Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import torch
import equinox as eqx
import orbax.checkpoint as ocp
import safetensors.torch
from transformers import (
    LlamaForCausalLM as HFLlamaForCausalLM,
    LlamaConfig as HFLlamaConfig,
    AutoTokenizer,
)

from typing import Optional, Tuple, Any
from jaxtyping import PyTree
from jax.sharding import NamedSharding, PartitionSpec as PS
from jax.experimental import mesh_utils

from .models.llama3.jax.model import (
    LlamaConfig,
    LlamaForCausalLM,
)
from .utils import named_tree_map


@dataclass
class CheckpointerConfig:
    """Configuration for checkpointing"""

    checkpoint_dir: str
    max_to_keep: int = 3
    save_interval_steps: int = 10
    create: bool = True  # Create the checkpoint directory if it doesn't exist
    enable_async_checkpointing: bool = True
    erase_existing_checkpoints: bool = False


class Checkpointer:
    def __init__(self, config: CheckpointerConfig):
        if not config.checkpoint_dir:
            raise ValueError("Checkpoint directory cannot be empty")
        self.config = config
        self.checkpoint_dir = config.checkpoint_dir
        if config.erase_existing_checkpoints:
            ocp.test_utils.erase_and_create_empty(self.checkpoint_dir)

        self.options = ocp.CheckpointManagerOptions(
            max_to_keep=config.max_to_keep,
            save_interval_steps=config.save_interval_steps,
            create=config.create,
            enable_async_checkpointing=config.enable_async_checkpointing,
        )

        self.checkpoint_mgr = ocp.CheckpointManager(
            directory=self.checkpoint_dir,
            options=self.options,
            item_names=["model_pytree", "model_config"],
        )

    @classmethod
    def get_abstract_pytree(cls, tree):
        return jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, tree)

    def save_checkpoint(
        self, model: eqx.Module, model_config: LlamaConfig, step: int = 0
    ):
        """Saves model checkpoint."""
        model_pytree, _ = eqx.partition(model, eqx.is_inexact_array)
        self.checkpoint_mgr.save(
            step=step,
            args=ocp.args.Composite(
                model_pytree=ocp.args.StandardSave(model_pytree),
                model_config=ocp.args.JsonSave(model_config.to_dict()),
            ),
            force=True,
        )

    def restore_checkpoint(self) -> Tuple[eqx.Module, LlamaConfig]:
        """Restores model checkpoint."""
        # Step 1: Restore the model_config first
        restored_config = self.checkpoint_mgr.restore(
            step=self.checkpoint_mgr.latest_step(),
            items=["model_config"],
            args=ocp.args.Composite(
                model_config=ocp.args.JsonRestore(),
            ),
        )
        model_config = LlamaConfig(**restored_config["model_config"])

        # Step 2: Construct the model and create the abstract pytree
        model = LlamaForCausalLM(model_config)
        model_params, model_static = eqx.partition(model, eqx.is_inexact_array)
        model_abstract_pytree = self.get_abstract_pytree(model_params)

        # Step 3: Restore the model parameters using the abstract pytree
        restored_params = self.checkpoint_mgr.restore(
            step=self.checkpoint_mgr.latest_step(),
            items=["model_pytree"],
            args=ocp.args.Composite(
                model_pytree=ocp.args.StandardRestore(model_abstract_pytree),
            ),
        )

        # Combine restored model parameters with model static
        model_params = restored_params["model_pytree"]
        model = eqx.combine(model_params, model_static)
        return model, model_config

    def wait_until_finished(self):
        """Wait for any async operations to complete."""
        self.checkpoint_mgr.wait_until_finished()

    @property
    def directory(self):
        return self.checkpoint_mgr.directory


def load_model(
    model_name: str, mesh: jax.sharding.Mesh, token: Optional[str] = None
):
    """Loads a model from a checkpoint or Hugging Face.

    Args:
        model_name: Name or path of the model to load
        token: HuggingFace token for accessing gated models
    """
    return load_llama_from_hf(model_name, mesh=mesh, token=token)


def load_checkpoint_or_model(
    model_name: str,
    mesh: jax.sharding.Mesh,
    checkpointer: Checkpointer,
    param_dtype=jnp.float32,
    compute_dtype=jnp.float32,
) -> LlamaForCausalLM:
    """Loads checkpoint from local storage using Orbax or downloads from HF with specified dtypes.

    Args:
        model_name: Name of HF model (e.g. 'meta-llama/Llama-2-7b') or path to local checkpoint
        checkpointer: An instance of Checkpointer to manage loading
        param_dtype: The dtype in which parameters are stored and loaded
        compute_dtype: The dtype in which computations are performed and outputs are returned

    Returns:
        tuple: (model, model_config)
    """
    has_checkpoints = len(checkpointer.checkpoint_mgr.all_steps()) > 0
    if has_checkpoints:
        # Restores the model in whatever dtypes are stored in the checkpoint.
        model, model_config = checkpointer.restore_checkpoint()
        print(
            f"Restored checkpoint from step {checkpointer.checkpoint_mgr.latest_step()}"
        )
        return model, model_config

    model, model_config = load_llama_from_hf(
        model_name,
        mesh=mesh,
        param_dtype=param_dtype,
        compute_dtype=compute_dtype,
    )
    return model, model_config


def create_llama_config_from_hf_model(hf_model) -> LlamaConfig:
    """Creates Equinox config from Hugging Face model config."""
    return LlamaConfig(
        vocab_size=hf_model.config.vocab_size,
        hidden_size=hf_model.config.hidden_size,
        intermediate_size=hf_model.config.intermediate_size,
        num_hidden_layers=hf_model.config.num_hidden_layers,
        num_attention_heads=hf_model.config.num_attention_heads,
        num_key_value_heads=hf_model.config.num_key_value_heads,
        max_position_embeddings=hf_model.config.max_position_embeddings,
        rms_norm_eps=hf_model.config.rms_norm_eps,
        rope_theta=hf_model.config.rope_theta,
        attention_bias=hf_model.config.attention_bias,
    )


def _make_torch_to_jax(dtype, mesh):
    """Creates a closure that converts PyTorch tensors to JAX arrays with sharding annotations."""

    def _torch_to_jax(tensor, sharding_spec):
        jax_array = jnp.array(tensor.detach().numpy(), dtype=dtype)
        sharding = NamedSharding(mesh, sharding_spec)
        return jax.device_put(jax_array, sharding)

    return _torch_to_jax


# TODO(refactor): Move load model into models/llama.
def get_worker_shards(worker_id: int) -> Tuple[int, int]:
    """Get the shard range for a given worker."""
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

def load_llama_from_hf(
    model_name: str,
    mesh: jax.sharding.Mesh,
    token: Optional[str] = None,
    lora_rank: int = 0,
    param_dtype: Any = jnp.float32,
    compute_dtype: Any = jnp.float32,
    use_optimized_decoder: bool = True,
) -> LlamaForCausalLM:
    """Downloads and converts Hugging Face model to Equinox model with specified dtypes.

    Args:
        model_name: Name of the Hugging Face model to load
        mesh: JAX sharding mesh
        token: HuggingFace token for accessing gated models
        lora_rank: Rank for LoRA parameters (set to 0 for no LoRA)
        param_dtype: The dtype in which parameters are stored
        compute_dtype: The dtype in which computations are performed

    Returns:
        eqx_model: LlamaForCausalLM model with specified dtypes
        model_config: Configuration of the model
    """
    if not use_optimized_decoder:
        return load_llama_from_hf_unoptimized(
            model_name, mesh, token, lora_rank, param_dtype, compute_dtype
        )

    # Get worker ID from hostname
    hostname = os.uname()[1]
    worker_id = int(hostname.split('w-')[1]) if 'w-' in hostname else 0
    print(f"Loading model for worker {worker_id}")

    # Get worker ID and shard range
    start_shard, end_shard = get_worker_shards(worker_id)
    print(f"Worker {worker_id} loading shards {start_shard}-{end_shard}")

    # Load config and create model config
    print("Loading config from file...")
    with open(os.path.join(model_name, "config.json")) as f:
        config_data = json.load(f)
    print("Config loaded successfully")

    print("Creating model config...")
    model_config = LlamaConfig(
        vocab_size=config_data["vocab_size"],
        hidden_size=config_data["hidden_size"],
        intermediate_size=config_data["intermediate_size"],
        num_hidden_layers=config_data["num_hidden_layers"],
        num_attention_heads=config_data["num_attention_heads"],
        num_key_value_heads=config_data["num_key_value_heads"],
        max_position_embeddings=config_data["max_position_embeddings"],
        rms_norm_eps=config_data["rms_norm_eps"],
        rope_theta=config_data.get("rope_theta", 10000.0),
        attention_bias=config_data.get("attention_bias", False),
        lora_rank=lora_rank,
        param_dtype=param_dtype,
        compute_dtype=compute_dtype
    )
    print("Model config created")

    # Load all shards into CPU memory using pure Python/NumPy
    print("Loading weights into CPU memory...")
    accumulated_state_dict = {}
    for shard_idx in range(start_shard, end_shard + 1):
        shard_file = f"model-{shard_idx:05d}-of-00030.safetensors"
        shard_path = os.path.join(model_name, shard_file)
        print(f"Loading shard: {shard_path}")
        shard_dict = safetensors.torch.load_file(shard_path)
        # Convert tensors to NumPy arrays
        shard_dict = {k: np.array(v) for k, v in shard_dict.items()}
        accumulated_state_dict.update(shard_dict)
        del shard_dict  # Free memory immediately
        print(f"Loaded shard {shard_idx} into CPU memory")

    # Initialize model on TPU
    print("Initializing model structure on TPU...")
    key = jax.random.PRNGKey(42)
    model = LlamaForCausalLM(
        model_config,
        param_dtype=param_dtype,
        compute_dtype=compute_dtype,
        key=key,
    )
    print("Model structure initialized on TPU")

    # Set up JAX conversion functions
    print("Setting up TPU transfer functions...")
    torch_to_jax_float32 = _make_torch_to_jax(dtype=jnp.float32, mesh=mesh)
    torch_to_jax = _make_torch_to_jax(dtype=param_dtype, mesh=mesh)

    # Transfer weights to TPU with proper sharding
    print("Transferring weights to TPU with proper sharding...")
    
    # Group weights by layer for more efficient transfer
    layer_weights = {}
    global_weights = {}
    
    for key, value in accumulated_state_dict.items():
        if any(x in key for x in ["self_attn", "mlp", "layernorm"]):
            layer_idx = int(key.split('.')[2])
            if layer_idx not in layer_weights:
                layer_weights[layer_idx] = {}
            layer_weights[layer_idx][key] = value
        else:
            global_weights[key] = value
            
    # First transfer global weights (embeddings, norms, etc.)
    print("Transferring global weights...")
    for key, value in global_weights.items():
        if "embed_tokens" in key:
            sharding = jax.sharding.NamedSharding(mesh, PS(("mp", "fsdp")))
            weight_jax = jax.device_put(value, sharding)
            model = eqx.tree_at(
                lambda m: m.model.embed_tokens.weight,
                model,
                weight_jax.astype(jnp.bfloat16)
            )
            print(f"Transferred {key}")
        elif "norm" in key:
            sharding = jax.sharding.NamedSharding(mesh, PS())
            weight_jax = jax.device_put(value, sharding)
            model = eqx.tree_at(
                lambda m: m.model.norm.weight,
                model,
                weight_jax.astype(jnp.bfloat16)
            )
            print(f"Transferred {key}")
        elif "lm_head" in key:
            sharding = jax.sharding.NamedSharding(mesh, PS(("fsdp", "mp")))
            weight_jax = jax.device_put(value, sharding)
            model = eqx.tree_at(
                lambda m: m.lm_head.weight,
                model,
                weight_jax.astype(param_dtype)
            )
            print(f"Transferred {key}")
            
    # Then transfer layer weights one layer at a time
    print("Transferring layer weights...")
    for layer_idx in sorted(layer_weights.keys()):
        print(f"Processing layer {layer_idx}...")
        layer_dict = layer_weights[layer_idx]
        
        # Process attention weights
        for key, value in layer_dict.items():
            if "self_attn.q_proj" in key:
                sharding = jax.sharding.NamedSharding(mesh, PS("fsdp", "mp"))
                weight_jax = jax.device_put(value, sharding)
                model = eqx.tree_at(
                    lambda m: m.model.layers.self_attn.q_proj.weight[layer_idx],
                    model,
                    weight_jax.astype(param_dtype)
                )
            elif "self_attn.k_proj" in key:
                sharding = jax.sharding.NamedSharding(mesh, PS("fsdp", "mp"))
                weight_jax = jax.device_put(value, sharding)
                model = eqx.tree_at(
                    lambda m: m.model.layers.self_attn.k_proj.weight[layer_idx],
                    model,
                    weight_jax.astype(param_dtype)
                )
            elif "self_attn.v_proj" in key:
                sharding = jax.sharding.NamedSharding(mesh, PS("fsdp", "mp"))
                weight_jax = jax.device_put(value, sharding)
                model = eqx.tree_at(
                    lambda m: m.model.layers.self_attn.v_proj.weight[layer_idx],
                    model,
                    weight_jax.astype(param_dtype)
                )
            elif "self_attn.o_proj" in key:
                sharding = jax.sharding.NamedSharding(mesh, PS("mp", "fsdp"))
                weight_jax = jax.device_put(value, sharding)
                model = eqx.tree_at(
                    lambda m: m.model.layers.self_attn.o_proj.weight[layer_idx],
                    model,
                    weight_jax.astype(param_dtype)
                )
            elif "mlp.gate_proj" in key:
                sharding = jax.sharding.NamedSharding(mesh, PS("fsdp", "mp"))
                weight_jax = jax.device_put(value, sharding)
                model = eqx.tree_at(
                    lambda m: m.model.layers.mlp.gate_proj.weight[layer_idx],
                    model,
                    weight_jax.astype(param_dtype)
                )
            elif "mlp.up_proj" in key:
                sharding = jax.sharding.NamedSharding(mesh, PS("fsdp", "mp"))
                weight_jax = jax.device_put(value, sharding)
                model = eqx.tree_at(
                    lambda m: m.model.layers.mlp.up_proj.weight[layer_idx],
                    model,
                    weight_jax.astype(param_dtype)
                )
            elif "mlp.down_proj" in key:
                sharding = jax.sharding.NamedSharding(mesh, PS("mp", "fsdp"))
                weight_jax = jax.device_put(value, sharding)
                model = eqx.tree_at(
                    lambda m: m.model.layers.mlp.down_proj.weight[layer_idx],
                    model,
                    weight_jax.astype(param_dtype)
                )
            elif "input_layernorm" in key:
                sharding = jax.sharding.NamedSharding(mesh, PS())
                weight_jax = jax.device_put(value, sharding)
                model = eqx.tree_at(
                    lambda m: m.model.layers.input_layernorm.weight[layer_idx],
                    model,
                    weight_jax.astype(jnp.bfloat16)
                )
            elif "post_attention_layernorm" in key:
                sharding = jax.sharding.NamedSharding(mesh, PS())
                weight_jax = jax.device_put(value, sharding)
                model = eqx.tree_at(
                    lambda m: m.model.layers.post_attention_layernorm.weight[layer_idx],
                    model,
                    weight_jax.astype(jnp.bfloat16)
                )
        
        # Clean up layer weights after transfer
        del layer_dict
        print(f"Layer {layer_idx} transferred")

    return model, model_config


def save_model_to_hf(
    model: eqx.Module,
    model_config: LlamaConfig,
    output_dir: str,
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
    use_optimized_decoder: bool = True,
):
    """Converts and saves an Equinox model to Hugging Face format.

    Args:
        model: Equinox LlamaForCausalLM model instance.
        model_config: Corresponding model configuration.
        output_dir: Directory to save the Hugging Face model.
        tokenizer_name: Name of the tokenizer to save alongside the model.
    """
    if not use_optimized_decoder:
        return save_model_to_hf_unoptimized(
            model, model_config, output_dir, tokenizer_name
        )

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a Hugging Face config from Equinox config
    hf_config = HFLlamaConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        intermediate_size=model_config.intermediate_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        num_key_value_heads=model_config.num_key_value_heads,
        max_position_embeddings=model_config.max_position_embeddings,
        rms_norm_eps=model_config.rms_norm_eps,
        rope_theta=model_config.rope_theta,
        attention_bias=model_config.attention_bias,
    )

    # Initialize a Hugging Face model with the same configuration
    hf_model = HFLlamaForCausalLM(config=hf_config)

    # Remove sharding and convert JAX arrays to NumPy arrays
    model_params, _ = eqx.partition(model, eqx.is_array)
    model_params = jax.tree_util.tree_map(
        lambda x: np.array(x).astype(np.float32),
        model_params,
    )

    def jax_to_torch(x):
        """Convert JAX array to PyTorch tensor."""
        return torch.tensor(jax.device_get(x), dtype=torch.float32)

    # Copy embedding and output layer weights
    hf_model.model.embed_tokens.weight.data = jax_to_torch(
        model_params.model.embed_tokens.weight
    )
    hf_model.lm_head.weight.data = jax_to_torch(model_params.lm_head.weight)
    hf_model.model.norm.weight.data = jax_to_torch(
        model_params.model.norm.weight
    )

    hf_layers = hf_model.model.layers

    def _copy_weights(from_eqx_layer, to_hf_layer_name):
        """Copies weights from vmapped Equinox layers to Hugging Face layers."""
        for i in range(len(hf_layers)):
            # Navigate through nested attributes to get the target layer (e.g. "self_attn.q_proj" -> layer.self_attn.q_proj)
            hf_submodule = hf_layers[i]
            for attr in to_hf_layer_name.split("."):
                hf_submodule = getattr(hf_submodule, attr)

            # Copy the weights from the eqx layer to hf submodule
            hf_submodule.weight.data = jax_to_torch(from_eqx_layer.weight[i])

    # Copy transformer layer weights
    _copy_weights(
        model_params.model.layers.self_attn.q_proj,
        "self_attn.q_proj",
    )
    _copy_weights(
        model_params.model.layers.self_attn.k_proj,
        "self_attn.k_proj",
    )
    _copy_weights(
        model_params.model.layers.self_attn.v_proj,
        "self_attn.v_proj",
    )
    _copy_weights(
        model_params.model.layers.self_attn.o_proj,
        "self_attn.o_proj",
    )

    # Copy MLP weights
    _copy_weights(
        model_params.model.layers.mlp.gate_proj,
        "mlp.gate_proj",
    )
    _copy_weights(
        model_params.model.layers.mlp.up_proj,
        "mlp.up_proj",
    )
    _copy_weights(
        model_params.model.layers.mlp.down_proj,
        "mlp.down_proj",
    )

    # Copy layer norm weights
    _copy_weights(
        model_params.model.layers.input_layernorm,
        "input_layernorm",
    )
    _copy_weights(
        model_params.model.layers.post_attention_layernorm,
        "post_attention_layernorm",
    )

    # Save model and tokenizer
    hf_model.save_pretrained(output_dir)

    # Save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained(output_dir)

    print(f"Model and tokenizer saved to {output_dir}")


# TODO(refactor): Move load model into models/llama.
def load_llama_from_hf_unoptimized(
    model_name: str,
    mesh: jax.sharding.Mesh,
    token: Optional[str] = None,
    lora_rank: int = 0,
    param_dtype: Any = jnp.float32,
    compute_dtype: Any = jnp.float32,
) -> LlamaForCausalLM:
    """Downloads and converts Hugging Face model to Equinox model with specified dtypes.

    Args:
        model_name: Name of the Hugging Face model to load
        token: HuggingFace token for accessing gated models
        lora_rank: Rank for LoRA parameters (set to 0 for no LoRA)
        param_dtype: The dtype in which parameters are stored
        output_dtype: The dtype in which computations are performed

    Returns:
        eqx_model: LlamaForCausalLM model with specified dtypes
        model_config: Configuration of the model
    """
    # Load HF model
    hf_model = HFLlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, token=token
    )

    # Create config and initialize Equinox model
    model_config = create_llama_config_from_hf_model(hf_model)
    model_config.lora_rank = lora_rank
    model_config.use_optimized_decoder = False

    key = jax.random.PRNGKey(99)
    eqx_model = LlamaForCausalLM(
        model_config,
        param_dtype=param_dtype,
        compute_dtype=compute_dtype,
        key=key,
        use_optimized_decoder=False,
    )
    torch_to_jax_float32 = _make_torch_to_jax(dtype=jnp.float32, mesh=mesh)
    torch_to_jax = _make_torch_to_jax(dtype=param_dtype, mesh=mesh)

    # Copy weights from HF model to Equinox model
    eqx_model = eqx.tree_at(
        lambda t: t.model.embed_tokens.weight,
        eqx_model,
        # Copy embedding weights at float32 precision.
        torch_to_jax_float32(
            hf_model.model.embed_tokens.weight, PS(("mp", "fsdp"))
        ),
    )
    eqx_model = eqx.tree_at(
        lambda t: t.model.norm.weight,
        eqx_model,
        torch_to_jax(hf_model.model.norm.weight, PS()),
    )
    eqx_model = eqx.tree_at(
        lambda t: t.lm_head.weight,
        eqx_model,
        torch_to_jax(hf_model.lm_head.weight, PS(("fsdp", "mp"))),
    )

    # Copy layer weights with appropriate sharding
    for i in range(len(eqx_model.model.layers)):
        hf_layer = hf_model.model.layers[i]

        # Self-attention weights
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.q_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.q_proj.weight, PS(("fsdp", "mp"))),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.k_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.k_proj.weight, PS(("fsdp", "mp"))),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.v_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.v_proj.weight, PS(("fsdp", "mp"))),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.o_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.o_proj.weight, PS(("mp", "fsdp"))),
        )

        # MLP weights
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].mlp.gate_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.mlp.gate_proj.weight, PS(("fsdp", "mp"))),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].mlp.up_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.mlp.up_proj.weight, PS(("fsdp", "mp"))),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].mlp.down_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.mlp.down_proj.weight, PS(("mp", "fsdp"))),
        )

        # Layer norms
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].input_layernorm.weight,
            eqx_model,
            torch_to_jax(hf_layer.input_layernorm.weight, PS()),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].post_attention_layernorm.weight,
            eqx_model,
            torch_to_jax(hf_layer.post_attention_layernorm.weight, PS()),
        )

    return eqx_model, model_config


def save_model_to_hf_unoptimized(
    model: eqx.Module,
    model_config: LlamaConfig,
    output_dir: str,
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
):
    """Converts Equinox model back to Hugging Face format and saves it.

    Args:
        model: Equinox LlamaForCausalLM model
        model_config: LlamaConfig used for the model
        output_dir: Directory where to save the Hugging Face model
        tokenizer_name: Name of the tokenizer to download and save
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a Hugging Face config from Equinox config
    hf_config = HFLlamaConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        intermediate_size=model_config.intermediate_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        num_key_value_heads=model_config.num_key_value_heads,
        max_position_embeddings=model_config.max_position_embeddings,
        rms_norm_eps=model_config.rms_norm_eps,
        rope_theta=model_config.rope_theta,
        attention_bias=model_config.attention_bias,
    )

    # Initialize a Hugging Face model with the same configuration
    hf_model = HFLlamaForCausalLM(config=hf_config)

    # Remove sharding and convert JAX arrays to NumPy arrays
    model_params, _ = eqx.partition(model, eqx.is_array)
    model_params = jax.tree_util.tree_map(
        lambda x: np.array(x).astype(np.float32),
        model_params,
    )

    # Copy weights from Equinox model to Hugging Face model
    # Embedding weights
    hf_model.model.embed_tokens.weight.data = torch.tensor(
        model_params.model.embed_tokens.weight, dtype=torch.float32
    )
    hf_model.lm_head.weight.data = torch.tensor(
        model_params.lm_head.weight, dtype=torch.float32
    )
    hf_model.model.norm.weight.data = torch.tensor(
        model_params.model.norm.weight, dtype=torch.float32
    )

    # Layer-wise weights
    for i in range(len(hf_model.model.layers)):
        eqx_layer = model_params.model.layers[i]
        hf_layer = hf_model.model.layers[i]

        # Self-attention weights
        hf_layer.self_attn.q_proj.weight.data = torch.tensor(
            eqx_layer.self_attn.q_proj.weight, dtype=torch.float32
        )
        hf_layer.self_attn.k_proj.weight.data = torch.tensor(
            eqx_layer.self_attn.k_proj.weight, dtype=torch.float32
        )
        hf_layer.self_attn.v_proj.weight.data = torch.tensor(
            eqx_layer.self_attn.v_proj.weight, dtype=torch.float32
        )
        hf_layer.self_attn.o_proj.weight.data = torch.tensor(
            eqx_layer.self_attn.o_proj.weight, dtype=torch.float32
        )

        # MLP weights
        hf_layer.mlp.gate_proj.weight.data = torch.tensor(
            eqx_layer.mlp.gate_proj.weight, dtype=torch.float32
        )
        hf_layer.mlp.up_proj.weight.data = torch.tensor(
            eqx_layer.mlp.up_proj.weight, dtype=torch.float32
        )
        hf_layer.mlp.down_proj.weight.data = torch.tensor(
            eqx_layer.mlp.down_proj.weight, dtype=torch.float32
        )

        # Layer norms
        hf_layer.input_layernorm.weight.data = torch.tensor(
            eqx_layer.input_layernorm.weight, dtype=torch.float32
        )
        hf_layer.post_attention_layernorm.weight.data = torch.tensor(
            eqx_layer.post_attention_layernorm.weight, dtype=torch.float32
        )

    # Save the Hugging Face model
    hf_model.save_pretrained(output_dir)

    # Save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained(output_dir)

    print(f"Model and tokenizer saved to {output_dir}")
