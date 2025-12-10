"""
Model utilities using Unsloth for optimized Qwen3-4B loading.
Uses Unsloth's FastLanguageModel for dramatically lower VRAM usage (~3GB vs 8GB).
"""

import torch
from typing import Tuple, Optional

# Unsloth model identifier - use instruct version for proper config
UNSLOTH_MODEL_ID = "unsloth/Qwen3-4B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_qwen_model_unsloth(
    model_id: str = UNSLOTH_MODEL_ID,
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    use_gradient_checkpointing: str = "unsloth"  # 30% less VRAM
) -> Tuple:
    """
    Initialize Qwen3-4B using Unsloth's FastLanguageModel.
    
    Unsloth provides:
    - 60% less VRAM usage
    - 2x faster training
    - Optimized RoPE and MLP kernels
    - Automatic padding-free batching
    
    Args:
        model_id: Unsloth model identifier
        max_seq_length: Maximum sequence length
        lora_r: LoRA rank parameter
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout rate
        use_gradient_checkpointing: "unsloth" for 30% less VRAM
    
    Returns:
        Tuple of (model, tokenizer)
    """
    from unsloth import FastLanguageModel
    
    print(f"Loading model with Unsloth: {model_id}")
    print(f"Max seq length: {max_seq_length}, LoRA r: {lora_r}")
    
    # Workaround: Directly patch the config.json in HuggingFace cache
    # Unsloth's loader expects torch_dtype in config, but Qwen3 models don't have it
    
    from huggingface_hub import hf_hub_download, HfFileSystem
    import json
    import os
    
    print("  Patching HuggingFace cached config for Qwen3...")
    
    try:
        # Download config first (this will cache it)
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        
        # Read, patch, and write back
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        if 'torch_dtype' not in config_data or config_data.get('torch_dtype') is None:
            config_data['torch_dtype'] = 'bfloat16'
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            print(f"  Patched {config_path} with torch_dtype=bfloat16")
        else:
            print(f"  Config already has torch_dtype={config_data.get('torch_dtype')}")
            
    except Exception as e:
        print(f"  Warning: Could not patch config: {e}")
    
    # Now load with Unsloth - should work with patched config
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Apply LoRA adapters with Unsloth's optimized implementation
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"  # Also target MLP layers
        ],
        bias="none",
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=42,
    )
    
    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer


def get_trainable_param_count(model) -> Tuple[int, int]:
    """
    Count trainable and total parameters.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def prepare_for_training(model):
    """
    Prepare the Unsloth model for training mode.
    """
    from unsloth import FastLanguageModel
    FastLanguageModel.for_training(model)
    return model


def prepare_for_inference(model):
    """
    Prepare the Unsloth model for inference mode (2x faster).
    """
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    return model
