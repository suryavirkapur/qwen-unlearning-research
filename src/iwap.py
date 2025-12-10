"""
Influence Guided Weight Atom Pruning (IWAP) implementation.

IWAP uses influence functions to identify weights causally responsible
for memorizing the forget set, then surgically prunes them.

Key insight: Weights with high (|gradient| × |weight|) on forget data
are the "atoms" storing unwanted knowledge.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm


def compute_saliency_scores(
    model: torch.nn.Module,
    forget_loader: DataLoader,
    device: str = "cuda",
    accumulate_batches: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute influence-based saliency scores for all trainable parameters.
    
    Saliency = |weight| × |gradient on forget set|
    
    This approximates the influence of each weight on the forget loss
    using a first-order Taylor expansion.
    
    Args:
        model: Model to analyze
        forget_loader: DataLoader for forget set
        device: Compute device
        accumulate_batches: Max batches to process (None = all)
    
    Returns:
        Dict mapping parameter names to saliency tensors
    """
    model.eval()  # Important: no dropout during analysis
    
    # Initialize saliency score accumulators
    saliency_scores = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            saliency_scores[name] = torch.zeros_like(param, device=device)
    
    # Accumulate gradients across forget set
    batch_count = 0
    for batch in tqdm(forget_loader, desc="Computing influence scores"):
        model.zero_grad()
        
        # Prepare inputs
        if "forget" in batch:
            inputs = batch["forget"]
        else:
            inputs = {
                "input_ids": batch.get("input_ids", batch.get("forget_input_ids")),
                "attention_mask": batch.get("attention_mask", batch.get("forget_attention_mask")),
                "labels": batch.get("labels", batch.get("forget_labels"))
            }
        
        inputs = {
            k: v.to(device) for k, v in inputs.items() 
            if isinstance(v, torch.Tensor)
        }
        
        # Forward + backward pass
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        
        # Accumulate saliency: |param| × |grad|
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Influence proxy: magnitude × gradient magnitude
                    score = torch.abs(param.data * param.grad)
                    saliency_scores[name] += score
        
        batch_count += 1
        if accumulate_batches and batch_count >= accumulate_batches:
            break
    
    # Normalize by batch count
    for name in saliency_scores:
        saliency_scores[name] /= max(batch_count, 1)
    
    model.zero_grad()
    return saliency_scores


def compute_pruning_threshold(
    saliency_scores: Dict[str, torch.Tensor],
    prune_ratio: float = 0.02
) -> float:
    """
    Compute global threshold for top-k% most influential weights.
    
    Args:
        saliency_scores: Dict of saliency tensors
        prune_ratio: Fraction of weights to prune (0.02 = top 2%)
    
    Returns:
        Threshold value - weights with score >= threshold will be pruned
    """
    # Flatten all scores
    all_scores = torch.cat([s.view(-1) for s in saliency_scores.values()])
    
    # Find the percentile threshold
    # We want the top prune_ratio%, so we find the (1 - prune_ratio) quantile
    threshold = torch.quantile(all_scores, 1 - prune_ratio)
    
    return threshold.item()


def apply_pruning(
    model: torch.nn.Module,
    saliency_scores: Dict[str, torch.Tensor],
    threshold: float,
    device: str = "cuda"
) -> Tuple[int, int]:
    """
    Prune weights with saliency score >= threshold.
    
    Args:
        model: Model to prune
        saliency_scores: Pre-computed saliency scores
        threshold: Pruning threshold
        device: Compute device
    
    Returns:
        Tuple of (pruned_count, total_trainable_count)
    """
    pruned_count = 0
    total_count = 0
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in saliency_scores and param.requires_grad:
                score = saliency_scores[name]
                
                # Create mask: 0 where score >= threshold (prune), 1 elsewhere (keep)
                mask = (score < threshold).float().to(device)
                
                # Count pruned weights
                pruned = (mask == 0).sum().item()
                pruned_count += int(pruned)
                total_count += param.numel()
                
                # Apply mask (zero out pruned weights)
                param.data.mul_(mask)
    
    return pruned_count, total_count


def iwap_pruning(
    model: torch.nn.Module,
    forget_loader: DataLoader,
    prune_ratio: float = 0.02,
    device: str = "cuda",
    max_batches: Optional[int] = None
) -> Dict[str, any]:
    """
    Complete IWAP pruning pipeline.
    
    1. Compute saliency scores across forget set
    2. Determine global pruning threshold
    3. Prune top-k% most influential weights
    
    Args:
        model: Model to prune
        forget_loader: DataLoader for forget set
        prune_ratio: Fraction of weights to prune (default: 2%)
        device: Compute device
        max_batches: Limit computation to N batches (for speed)
    
    Returns:
        Dict with pruning statistics
    """
    print(f"IWAP: Pruning top {prune_ratio*100:.1f}% of weights...")
    
    # Step 1: Compute saliency scores
    print("Step 1/3: Computing saliency scores...")
    saliency_scores = compute_saliency_scores(
        model, 
        forget_loader, 
        device,
        accumulate_batches=max_batches
    )
    
    # Step 2: Compute threshold
    print("Step 2/3: Computing pruning threshold...")
    threshold = compute_pruning_threshold(saliency_scores, prune_ratio)
    print(f"  Threshold: {threshold:.6f}")
    
    # Step 3: Apply pruning
    print("Step 3/3: Applying surgical pruning...")
    pruned, total = apply_pruning(model, saliency_scores, threshold, device)
    
    print(f"  Pruned {pruned:,} / {total:,} weights ({100*pruned/total:.2f}%)")
    
    return {
        "pruned_count": pruned,
        "total_count": total,
        "prune_ratio_actual": pruned / total,
        "threshold": threshold
    }


class IWAPWithRepair:
    """
    IWAP with post-pruning repair phase.
    
    After pruning, the network may have degraded general capability.
    This class adds a brief fine-tuning phase on the retain set to
    "heal" the network while preserving the unlearning effect.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        repair_lr: float = 1e-5,
        repair_steps: int = 100
    ):
        self.model = model
        self.device = device
        self.repair_lr = repair_lr
        self.repair_steps = repair_steps
    
    def prune(
        self,
        forget_loader: DataLoader,
        prune_ratio: float = 0.02,
        max_batches: Optional[int] = None
    ) -> Dict[str, any]:
        """Execute IWAP pruning."""
        return iwap_pruning(
            self.model,
            forget_loader,
            prune_ratio,
            self.device,
            max_batches
        )
    
    def repair(
        self,
        retain_loader: DataLoader,
        max_steps: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Fine-tune on retain set to heal pruning damage.
        """
        from torch.optim import AdamW
        
        print("IWAP Repair: Fine-tuning on retain set...")
        
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.repair_lr
        )
        
        self.model.train()
        steps = max_steps or self.repair_steps
        step = 0
        total_loss = 0.0
        
        pbar = tqdm(total=steps, desc="Repair phase")
        
        for batch in retain_loader:
            if step >= steps:
                break
            
            optimizer.zero_grad()
            
            # Prepare inputs
            if "retain" in batch:
                inputs = batch["retain"]
            else:
                inputs = {
                    "input_ids": batch.get("input_ids", batch.get("retain_input_ids")),
                    "attention_mask": batch.get("attention_mask", batch.get("retain_attention_mask")),
                    "labels": batch.get("labels", batch.get("retain_labels"))
                }
            
            inputs = {
                k: v.to(self.device) for k, v in inputs.items()
                if isinstance(v, torch.Tensor)
            }
            
            outputs = self.model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": loss.item()})
        
        pbar.close()
        
        return {
            "repair_steps": step,
            "avg_loss": total_loss / max(step, 1)
        }
    
    def run(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        prune_ratio: float = 0.02,
        max_batches: Optional[int] = None,
        repair_steps: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Full IWAP pipeline: prune + repair.
        """
        # Prune
        prune_stats = self.prune(forget_loader, prune_ratio, max_batches)
        
        # Repair
        repair_stats = self.repair(
            retain_loader, 
            max_steps=repair_steps or self.repair_steps
        )
        
        return {**prune_stats, **repair_stats}
