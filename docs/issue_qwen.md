# Unsloth + Qwen3 Compatibility Issue

## Summary

When attempting to use Unsloth's `FastLanguageModel` with Qwen3-4B models, the loader fails with a `KeyError: 'torch_dtype'` error.

## Affected Models

- `unsloth/Qwen3-4B`
- `unsloth/Qwen3-4B-unsloth-bnb-4bit`
- `Qwen/Qwen3-4B`

## Error

```python
File ".../unsloth/models/loader.py", line 433, in from_pretrained
    "bnb_4bit_compute_dtype": model.config.to_dict()["torch_dtype"],
KeyError: 'torch_dtype'
```

## Root Cause

Unsloth's loader at line 433 expects `torch_dtype` to be present in the model's config dictionary:

```python
"bnb_4bit_compute_dtype": model.config.to_dict()["torch_dtype"],
```

However, Qwen3 model configs do **not** include `torch_dtype` in their `config.json`, causing the KeyError.

## Attempted Workarounds

### 1. Pre-quantized Model ❌
Used `unsloth/Qwen3-4B-unsloth-bnb-4bit` - same error.

### 2. HuggingFace Cache File Patching ❌
Patched `config.json` in HF cache to add `torch_dtype: "bfloat16"`. The file was updated, but `model.config.to_dict()` still doesn't include it because the PretrainedConfig class doesn't expose custom fields added to the JSON.

### 3. Monkey-patching FastLanguageModel ❌
Attempted to patch `from_pretrained` to modify config before access. Failed because Unsloth loads a fresh model instance internally.

### 4. Explicit dtype parameter ❌
Passing `dtype=torch.bfloat16` doesn't help - the error occurs after model load when setting up quantization config.

## Environment

- Unsloth: 2025.7.2
- Transformers: 4.57.3
- PyTorch: 2.9.1+cu128
- Python: 3.13

## Suggested Fix for Unsloth

In `unsloth/models/loader.py` line 433, add a fallback:

```python
# Current (broken for Qwen3):
"bnb_4bit_compute_dtype": model.config.to_dict()["torch_dtype"],

# Suggested fix:
"bnb_4bit_compute_dtype": model.config.to_dict().get("torch_dtype", "bfloat16"),
```

## Workaround for Users

Until Unsloth fixes this, Qwen3 models cannot be used with `FastLanguageModel.from_pretrained()` with 4-bit quantization. Use standard transformers + bitsandbytes instead:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B",
    quantization_config=quantization_config,
    device_map="auto",
)
```

## Related Links

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Qwen3 on HuggingFace](https://huggingface.co/Qwen/Qwen3-4B)
