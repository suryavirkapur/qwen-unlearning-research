"""
Model utilities for Qwen3-4B with 4-bit quantization and LoRA.
Handles model loading, adapter configuration, and memory-efficient setup.
"""

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from typing import Tuple, Optional
import copy


# Model Configuration
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_quantization_config() -> BitsAndBytesConfig:
    """
    Create 4-bit quantization config for memory efficiency.
    Optimized for RTX 3080 (10GB VRAM).
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # NormalFloat4 for better quality
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # Nested quantization for extra memory savings
    )


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None
) -> LoraConfig:
    """
    Create LoRA configuration targeting Qwen3's GQA attention layers.
    
    The LoRA adapters serve as the 'plastic' medium for unlearning -
    we modify these adapters while keeping the base model frozen.
    """
    if target_modules is None:
        # Target all attention projections in Qwen3's GQA
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )


def setup_qwen_model(
    model_id: str = MODEL_ID,
    use_quantization: bool = True,
    use_lora: bool = True,
    lora_r: int = 16
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Initialize Qwen3-4B with optional 4-bit quantization and LoRA.
    
    Args:
        model_id: HuggingFace model identifier
        use_quantization: Whether to use 4-bit quantization
        use_lora: Whether to apply LoRA adapters
        lora_r: LoRA rank parameter
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_id}")
    print(f"Quantization: {use_quantization}, LoRA: {use_lora}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True
    )
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Model loading kwargs
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "use_cache": False,  # Incompatible with gradient checkpointing
        "torch_dtype": torch.bfloat16,
    }
    
    if use_quantization:
        model_kwargs["quantization_config"] = get_quantization_config()
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    
    if use_quantization:
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True
        )
    
    if use_lora:
        lora_config = get_lora_config(r=lora_r)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    return model, tokenizer


def create_reference_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_id: str = MODEL_ID
) -> AutoModelForCausalLM:
    """
    Create a frozen reference model for KL-divergence constraint.
    Used in DSU to prevent excessive drift from original behavior.
    
    For memory efficiency, we load a separate instance with inference mode.
    """
    print("Creating reference model for KL constraint...")
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=get_quantization_config(),
        torch_dtype=torch.bfloat16,
    )
    
    # Freeze all parameters
    for param in ref_model.parameters():
        param.requires_grad = False
    
    ref_model.eval()
    
    return ref_model


def get_trainable_param_count(model: AutoModelForCausalLM) -> Tuple[int, int]:
    """
    Count trainable and total parameters.
    
    Returns:
        Tuple of (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_checkpoint(
    model: AutoModelForCausalLM,
    output_dir: str,
    checkpoint_name: str = "unlearned_model"
) -> str:
    """
    Save model checkpoint (LoRA adapters only if using PEFT).
    """
    save_path = f"{output_dir}/{checkpoint_name}"
    
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    else:
        torch.save(model.state_dict(), f"{save_path}/pytorch_model.bin")
        print(f"State dict saved to {save_path}")
    
    return save_path


def load_checkpoint(
    checkpoint_path: str,
    base_model_id: str = MODEL_ID
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a saved checkpoint (base model + LoRA adapters).
    """
    from peft import PeftModel
    
    # Load base model
    model, tokenizer = setup_qwen_model(
        model_id=base_model_id,
        use_quantization=True,
        use_lora=False  # We'll load LoRA separately
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(model, checkpoint_path)
    
    return model, tokenizer
