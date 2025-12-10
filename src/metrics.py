"""
Evaluation metrics for machine unlearning benchmarks.

Implements:
- Forget Efficacy: How well the model forgets targeted data
- Retain Performance: Preservation of general capabilities
- MIA Resistance: Membership Inference Attack resistance
- Computational Cost: Training time relative to retraining
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm
import numpy as np
from scipy import stats
import time


def compute_perplexity(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    max_batches: Optional[int] = None
) -> float:
    """
    Compute perplexity on a dataset.
    
    Lower perplexity = model is more confident about predictions.
    For forget set: higher is better (model is uncertain).
    For retain set: lower is better (model retains knowledge).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity", leave=False):
            # Extract inputs
            if isinstance(batch, dict):
                if "input_ids" in batch:
                    inputs = batch
                elif "forget_input_ids" in batch:
                    inputs = {
                        "input_ids": batch["forget_input_ids"],
                        "attention_mask": batch["forget_attention_mask"],
                        "labels": batch["forget_labels"]
                    }
                elif "retain_input_ids" in batch:
                    inputs = {
                        "input_ids": batch["retain_input_ids"],
                        "attention_mask": batch["retain_attention_mask"],
                        "labels": batch["retain_labels"]
                    }
                else:
                    continue
            else:
                continue
            
            inputs = {
                k: v.to(device) for k, v in inputs.items()
                if isinstance(v, torch.Tensor)
            }
            
            outputs = model(**inputs)
            
            # Count non-padding tokens
            mask = inputs["attention_mask"]
            num_tokens = mask.sum().item()
            
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
            
            batch_count += 1
            if max_batches and batch_count >= max_batches:
                break
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity


def compute_qa_accuracy(
    model: torch.nn.Module,
    tokenizer: Any,
    qa_pairs: List[Dict[str, str]],
    device: str = "cuda",
    max_samples: Optional[int] = None
) -> Tuple[float, List[bool]]:
    """
    Compute QA accuracy for the model.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        qa_pairs: List of {"question": ..., "answer": ...} dicts
        device: Compute device
        max_samples: Limit number of samples
    
    Returns:
        Tuple of (accuracy, list of per-sample correctness)
    """
    model.eval()
    correct = []
    
    samples = qa_pairs[:max_samples] if max_samples else qa_pairs
    
    with torch.no_grad():
        for item in tqdm(samples, desc="Evaluating QA", leave=False):
            question = item.get("question", item.get("prompt", ""))
            expected = item.get("answer", item.get("response", "")).strip().lower()
            
            # Format prompt for Qwen3
            prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(device)
            
            # Generate response
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip().lower()
            
            # Check if expected answer is in response
            is_correct = expected in response or response in expected
            correct.append(is_correct)
    
    accuracy = sum(correct) / len(correct) if correct else 0.0
    return accuracy, correct


def compute_forget_efficacy(
    model: torch.nn.Module,
    forget_loader: DataLoader,
    baseline_perplexity: float,
    device: str = "cuda",
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute forget efficacy metrics.
    
    Forget efficacy measures how well the model has "forgotten" the target data.
    
    Metrics:
    - perplexity_ratio: Current perplexity / baseline perplexity
    - efficacy_score: Normalized score (0-100%)
    
    A successfully unlearned model should have HIGH perplexity on forget data
    (i.e., be uncertain/confused about it).
    """
    current_ppl = compute_perplexity(model, forget_loader, device, max_batches)
    
    # Ratio > 1 means model is more uncertain (good for unlearning)
    ppl_ratio = current_ppl / baseline_perplexity if baseline_perplexity > 0 else 1.0
    
    # Convert to efficacy score (capped at 100%)
    # Higher perplexity = higher efficacy
    efficacy = min(100.0, max(0.0, (ppl_ratio - 1.0) * 100))
    
    return {
        "forget_perplexity": current_ppl,
        "baseline_perplexity": baseline_perplexity,
        "perplexity_ratio": ppl_ratio,
        "forget_efficacy": efficacy
    }


def compute_retain_performance(
    model: torch.nn.Module,
    retain_loader: DataLoader,
    baseline_perplexity: float,
    device: str = "cuda",
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute retain performance metrics.
    
    Retain performance measures how well the model preserves its general capabilities.
    
    A successfully unlearned model should have SIMILAR perplexity on retain data
    compared to the baseline (no degradation).
    """
    current_ppl = compute_perplexity(model, retain_loader, device, max_batches)
    
    # Ratio close to 1 is good (no change)
    # Ratio > 1 means degradation
    ppl_ratio = current_ppl / baseline_perplexity if baseline_perplexity > 0 else 1.0
    
    # Performance score: 100% if no change, decreases with degradation
    # Using exponential decay for smoother gradient
    performance = 100.0 * np.exp(-(ppl_ratio - 1.0))
    performance = max(0.0, min(100.0, performance))
    
    return {
        "retain_perplexity": current_ppl,
        "baseline_perplexity": baseline_perplexity,
        "perplexity_ratio": ppl_ratio,
        "retain_performance": performance
    }


def compute_mia_resistance(
    model: torch.nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    device: str = "cuda",
    max_samples: int = 100
) -> Dict[str, float]:
    """
    Compute Membership Inference Attack (MIA) resistance.
    
    MIA tries to determine if a specific data point was used to train the model.
    The attack exploits the fact that models have lower loss on training data.
    
    A resistant model should have similar loss distributions on forget vs retain,
    making it hard to distinguish members from non-members.
    
    Returns:
        Dictionary with MIA metrics:
        - attack_accuracy: How well the attack can distinguish (lower = better)
        - mia_resistance: 1 - attack_accuracy (higher = better)
        - auc_roc: Area under ROC curve (0.5 = random, 1.0 = perfect attack)
    """
    model.eval()
    
    forget_losses = []
    retain_losses = []
    
    # Collect losses on forget set
    with torch.no_grad():
        for i, batch in enumerate(forget_loader):
            if i >= max_samples:
                break
            
            if "forget_input_ids" in batch:
                inputs = {
                    "input_ids": batch["forget_input_ids"],
                    "attention_mask": batch["forget_attention_mask"],
                    "labels": batch["forget_labels"]
                }
            elif "input_ids" in batch:
                inputs = batch
            else:
                continue
            
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            
            outputs = model(**inputs)
            forget_losses.append(outputs.loss.item())
    
    # Collect losses on retain set  
    with torch.no_grad():
        for i, batch in enumerate(retain_loader):
            if i >= max_samples:
                break
            
            if "retain_input_ids" in batch:
                inputs = {
                    "input_ids": batch["retain_input_ids"],
                    "attention_mask": batch["retain_attention_mask"],
                    "labels": batch["retain_labels"]
                }
            elif "input_ids" in batch:
                inputs = batch
            else:
                continue
            
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            
            outputs = model(**inputs)
            retain_losses.append(outputs.loss.item())
    
    if not forget_losses or not retain_losses:
        return {
            "attack_accuracy": 0.5,
            "mia_resistance": 50.0,
            "forget_mean_loss": 0.0,
            "retain_mean_loss": 0.0
        }
    
    forget_losses = np.array(forget_losses)
    retain_losses = np.array(retain_losses)
    
    # Simple threshold-based attack
    # If forget loss < threshold, predict "member" (was in training)
    # Optimal threshold is between the means
    threshold = (np.mean(forget_losses) + np.mean(retain_losses)) / 2
    
    # Attack accuracy: correctly classifying members/non-members
    # For forget set (should be "members"): loss < threshold
    # For retain set (should be "non-members"): loss >= threshold
    
    # After unlearning, forget set should have HIGH loss (look like non-members)
    # So attack should FAIL on forget set
    
    forget_as_member = (forget_losses < threshold).sum()
    retain_as_nonmember = (retain_losses >= threshold).sum()
    
    correct = forget_as_member + retain_as_nonmember
    total = len(forget_losses) + len(retain_losses)
    
    attack_accuracy = correct / total if total > 0 else 0.5
    
    # For unlearned model, attack accuracy should be ~0.5 (random)
    # Resistance = how well we resist the attack (higher = better)
    # Perfect unlearning: attack_accuracy = 0.5, resistance = 50%
    # We want resistance close to 50% (indistinguishable)
    
    # Alternative metric: how far from 50%? Lower distance = better
    mia_resistance = 100.0 * (1.0 - abs(attack_accuracy - 0.5) * 2)
    
    return {
        "attack_accuracy": attack_accuracy,
        "mia_resistance": mia_resistance,
        "forget_mean_loss": float(np.mean(forget_losses)),
        "retain_mean_loss": float(np.mean(retain_losses)),
        "threshold": threshold
    }


class ComputationalCostTracker:
    """Track computational cost relative to full retraining."""
    
    def __init__(self, estimated_retrain_time: float = 3600.0):
        """
        Args:
            estimated_retrain_time: Estimated time for full retraining in seconds.
                                    Default 1 hour as placeholder.
        """
        self.estimated_retrain_time = estimated_retrain_time
        self.start_time = None
        self.total_time = 0.0
    
    def start(self):
        """Start timing."""
        self.start_time = time.time()
    
    def stop(self):
        """Stop timing and accumulate."""
        if self.start_time is not None:
            self.total_time += time.time() - self.start_time
            self.start_time = None
    
    def reset(self):
        """Reset tracker."""
        self.start_time = None
        self.total_time = 0.0
    
    def get_cost(self) -> Dict[str, float]:
        """
        Get computational cost metrics.
        
        Returns:
            Dictionary with:
            - elapsed_seconds: Total elapsed time
            - relative_cost: Percentage relative to full retraining
        """
        return {
            "elapsed_seconds": self.total_time,
            "relative_cost": 100.0 * self.total_time / self.estimated_retrain_time
        }


def run_full_evaluation(
    model: torch.nn.Module,
    tokenizer: Any,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    baseline_forget_ppl: float,
    baseline_retain_ppl: float,
    device: str = "cuda",
    computational_cost: Optional[float] = None,
    max_batches: int = 100  # Limit evaluation for speed
) -> Dict[str, float]:
    """
    Run complete evaluation suite.
    
    Returns dictionary with all metrics for LaTeX table.
    """
    print("Running full evaluation...")
    
    # Forget efficacy
    print("  Computing forget efficacy...")
    forget_metrics = compute_forget_efficacy(
        model, forget_loader, baseline_forget_ppl, device, max_batches
    )
    
    # Retain performance
    print("  Computing retain performance...")
    retain_metrics = compute_retain_performance(
        model, retain_loader, baseline_retain_ppl, device, max_batches
    )
    
    # MIA resistance
    print("  Computing MIA resistance...")
    mia_metrics = compute_mia_resistance(
        model, forget_loader, retain_loader, device, max_samples=max_batches
    )
    
    # Combine all metrics
    results = {
        "forget_efficacy": forget_metrics["forget_efficacy"],
        "retain_performance": retain_metrics["retain_performance"],
        "mia_resistance": mia_metrics["mia_resistance"],
        "computational_cost": computational_cost if computational_cost else 0.0,
        
        # Detailed metrics (for analysis)
        "forget_perplexity": forget_metrics["forget_perplexity"],
        "retain_perplexity": retain_metrics["retain_perplexity"],
        "attack_accuracy": mia_metrics["attack_accuracy"]
    }
    
    return results
