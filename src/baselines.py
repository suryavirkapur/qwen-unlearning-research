"""
Baseline unlearning methods for comparison.
Implements Naive Fine-tuning and Gradient Ascent baselines.
"""

import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import Dict, Any, Optional
from tqdm import tqdm


class BaseTrainer:
    """Base class for all unlearning trainers."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        learning_rate: float = 2e-5,
        device: str = "cuda"
    ):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        
        if optimizer is None:
            self.optimizer = AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=learning_rate
            )
        else:
            self.optimizer = optimizer
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step. Override in subclasses."""
        raise NotImplementedError
    
    def train(
        self,
        dataloader,
        num_epochs: int = 1,
        max_steps: Optional[int] = None,
        log_interval: int = 10
    ) -> Dict[str, float]:
        """Training loop."""
        self.model.train()
        total_loss = 0.0
        step_count = 0
        
        for epoch in range(num_epochs):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in pbar:
                loss = self.train_step(batch)
                total_loss += loss
                step_count += 1
                
                if step_count % log_interval == 0:
                    pbar.set_postfix({"loss": loss})
                
                if max_steps and step_count >= max_steps:
                    break
            
            if max_steps and step_count >= max_steps:
                break
        
        return {
            "avg_loss": total_loss / max(step_count, 1),
            "steps": step_count
        }


class NaiveFineTuningTrainer(BaseTrainer):
    """
    Naive Fine-tuning baseline.
    Simply fine-tunes on the retain set, hoping the model "forgets" the
    undesired knowledge through parameter drift.
    
    This is a weak baseline - it does NOT actively unlearn.
    """
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.optimizer.zero_grad()
        
        # Only use retain data (ignore forget if present)
        if "retain" in batch:
            inputs = batch["retain"]
        elif "retain_input_ids" in batch:
            inputs = {
                "input_ids": batch["retain_input_ids"],
                "attention_mask": batch["retain_attention_mask"],
                "labels": batch["retain_labels"]
            }
        else:
            inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"]
            }
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        # Forward pass
        outputs = self.model(**inputs)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class GradientAscentTrainer(BaseTrainer):
    """
    Gradient Ascent (GA) baseline.
    Maximizes the loss on the forget set to "unlearn" the data.
    
    WARNING: This is known to cause catastrophic forgetting!
    The model often collapses to gibberish without retain data regularization.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        learning_rate: float = 1e-5,  # Lower LR for stability
        device: str = "cuda",
        max_grad_norm: float = 1.0
    ):
        super().__init__(model, optimizer, learning_rate, device)
        self.max_grad_norm = max_grad_norm
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.optimizer.zero_grad()
        
        # Use forget data
        if "forget" in batch:
            inputs = batch["forget"]
        elif "forget_input_ids" in batch:
            inputs = {
                "input_ids": batch["forget_input_ids"],
                "attention_mask": batch["forget_attention_mask"],
                "labels": batch["forget_labels"]
            }
        else:
            inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"]
            }
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        # Forward pass
        outputs = self.model(**inputs)
        
        # GRADIENT ASCENT: Negate loss to maximize
        # This pushes the model AWAY from correct predictions
        loss = -outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.max_grad_norm
        )
        
        self.optimizer.step()
        
        # Return the original (positive) loss for logging
        return -loss.item()


class GradientAscentWithRetainTrainer(BaseTrainer):
    """
    Gradient Ascent with Retain regularization.
    An improved baseline that combines gradient ascent on forget data
    with standard descent on retain data to prevent collapse.
    
    This is essentially a simplified version of DSU.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        learning_rate: float = 2e-5,
        device: str = "cuda",
        forget_weight: float = 1.0,
        retain_weight: float = 1.0,
        max_grad_norm: float = 1.0
    ):
        super().__init__(model, optimizer, learning_rate, device)
        self.forget_weight = forget_weight
        self.retain_weight = retain_weight
        self.max_grad_norm = max_grad_norm
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        
        # Process forget data (gradient ascent)
        if "forget" in batch or "forget_input_ids" in batch:
            if "forget" in batch:
                forget_inputs = batch["forget"]
            else:
                forget_inputs = {
                    "input_ids": batch["forget_input_ids"],
                    "attention_mask": batch["forget_attention_mask"],
                    "labels": batch["forget_labels"]
                }
            
            forget_inputs = {k: v.to(self.device) for k, v in forget_inputs.items() if isinstance(v, torch.Tensor)}
            forget_outputs = self.model(**forget_inputs)
            forget_loss = -self.forget_weight * forget_outputs.loss
            total_loss += forget_loss
        
        # Process retain data (gradient descent)
        if "retain" in batch or "retain_input_ids" in batch:
            if "retain" in batch:
                retain_inputs = batch["retain"]
            else:
                retain_inputs = {
                    "input_ids": batch["retain_input_ids"],
                    "attention_mask": batch["retain_attention_mask"],
                    "labels": batch["retain_labels"]
                }
            
            retain_inputs = {k: v.to(self.device) for k, v in retain_inputs.items() if isinstance(v, torch.Tensor)}
            retain_outputs = self.model(**retain_inputs)
            retain_loss = self.retain_weight * retain_outputs.loss
            total_loss += retain_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.max_grad_norm
        )
        
        self.optimizer.step()
        
        return total_loss.item()
