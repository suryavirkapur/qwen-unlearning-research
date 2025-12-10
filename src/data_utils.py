"""
Data utilities for TOFU benchmark and unlearning experiments.
Handles dataset loading, preprocessing, and custom collation for dual-batch sampling.
"""

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass


def load_tofu_dataset(subset: str = "forget01") -> Dict[str, Any]:
    """
    Load the TOFU (Task of Fictitious Unlearning) benchmark dataset.
    
    TOFU uses separate configs for different subsets:
    - forget01, forget05, forget10: 1%, 5%, 10% of fictitious authors to forget
    - retain99, retain95, retain90: corresponding retain sets
    - retain_perturbed: perturbed retain set for evaluation
    
    Args:
        subset: Which forget subset to use. Options: 'forget01', 'forget05', 'forget10'
    
    Returns:
        Dictionary with 'forget' and 'retain' splits.
    """
    # Map forget subset to corresponding retain config
    retain_map = {
        "forget01": "retain99",
        "forget05": "retain95", 
        "forget10": "retain90"
    }
    retain_config = retain_map.get(subset, "retain99")
    
    # Load forget and retain as separate configs
    print(f"  Loading forget config: {subset}")
    forget_data = load_dataset("locuslab/TOFU", subset, split="train")
    
    print(f"  Loading retain config: {retain_config}")
    retain_data = load_dataset("locuslab/TOFU", retain_config, split="train")
    
    print(f"  Forget samples: {len(forget_data)}, Retain samples: {len(retain_data)}")
    
    return {
        "forget": forget_data,
        "retain": retain_data,
    }



class UnlearningDataset(Dataset):
    """Wrapper for unlearning datasets with tokenization."""
    
    def __init__(
        self, 
        data: Any,
        tokenizer: Any,
        max_length: int = 512,
        is_forget: bool = False
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_forget = is_forget
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # TOFU format: question + answer pairs
        question = item.get("question", item.get("prompt", ""))
        answer = item.get("answer", item.get("response", ""))
        
        # Format for Qwen3 chat template
        text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone(),
            "is_forget": torch.tensor(self.is_forget, dtype=torch.bool)
        }


@dataclass
class DualBatchCollator:
    """
    Custom collator that creates paired forget/retain batches for DSU and TROU.
    Returns a dictionary with separate forget and retain tensors.
    """
    tokenizer: Any
    max_length: int = 512
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate forget and retain samples
        forget_features = [f for f in features if f.get("is_forget", False)]
        retain_features = [f for f in features if not f.get("is_forget", True)]
        
        result = {}
        
        # Collate forget batch
        if forget_features:
            result["forget_input_ids"] = torch.stack([f["input_ids"] for f in forget_features])
            result["forget_attention_mask"] = torch.stack([f["attention_mask"] for f in forget_features])
            result["forget_labels"] = torch.stack([f["labels"] for f in forget_features])
        
        # Collate retain batch  
        if retain_features:
            result["retain_input_ids"] = torch.stack([f["input_ids"] for f in retain_features])
            result["retain_attention_mask"] = torch.stack([f["attention_mask"] for f in retain_features])
            result["retain_labels"] = torch.stack([f["labels"] for f in retain_features])
            
        return result


class InterleavedDataLoader:
    """
    Interleaved dataloader that yields paired forget/retain batches.
    Ensures each training step has both forget and retain samples.
    """
    
    def __init__(
        self,
        forget_dataset: Dataset,
        retain_dataset: Dataset,
        batch_size: int = 2,
        shuffle: bool = True
    ):
        self.forget_loader = DataLoader(
            forget_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle
        )
        self.retain_loader = DataLoader(
            retain_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
    def __iter__(self):
        forget_iter = iter(self.forget_loader)
        retain_iter = iter(self.retain_loader)
        
        while True:
            try:
                forget_batch = next(forget_iter)
            except StopIteration:
                break
                
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                # Restart retain iterator if it's shorter
                retain_iter = iter(self.retain_loader)
                retain_batch = next(retain_iter)
            
            yield {
                "forget": forget_batch,
                "retain": retain_batch
            }
    
    def __len__(self) -> int:
        return len(self.forget_loader)


def create_dataloaders(
    tokenizer: Any,
    forget_data: Any,
    retain_data: Any,
    batch_size: int = 2,
    max_length: int = 512
) -> Tuple[InterleavedDataLoader, DataLoader]:
    """
    Create training and evaluation dataloaders.
    
    Returns:
        Tuple of (train_loader, eval_loader)
    """
    forget_dataset = UnlearningDataset(forget_data, tokenizer, max_length, is_forget=True)
    retain_dataset = UnlearningDataset(retain_data, tokenizer, max_length, is_forget=False)
    
    train_loader = InterleavedDataLoader(
        forget_dataset,
        retain_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Evaluation loader for retain set only
    eval_loader = DataLoader(
        retain_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, eval_loader
