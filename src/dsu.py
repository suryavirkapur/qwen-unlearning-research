"""
Dual Signed Adaptation Unlearning (DSU) implementation.

DSU extends gradient ascent by applying a regularized push-pull dynamic:
- NEGATIVE gradient on forget set (ascent/unlearning)
- POSITIVE gradient on retain set (descent/preservation)
- KL-divergence constraint to prevent drift from original behavior
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, Any, Optional
from tqdm import tqdm


class DSUTrainer:
    """
    Dual Signed Adaptation Unlearning Trainer.
    
    Loss = -α * forget_loss + β * retain_loss + γ * kl_divergence
    
    Where:
        - forget_loss: Cross-entropy on forget set (negated for ascent)
        - retain_loss: Cross-entropy on retain set (standard descent)
        - kl_divergence: KL between current and reference model outputs
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        ref_model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        learning_rate: float = 2e-5,
        device: str = "cuda",
        alpha: float = 1.0,      # Forget weight (ascent strength)
        beta: float = 5.0,       # Retain weight (anchor strength) 
        gamma: float = 0.1,      # KL constraint weight
        max_grad_norm: float = 1.0,
        use_npo: bool = True     # Use NPO-style bounded loss
    ):
        """
        Args:
            model: Model to unlearn from
            ref_model: Frozen reference model for KL constraint (optional)
            alpha: Weight for forget loss (higher = more aggressive unlearning)
            beta: Weight for retain loss (higher = more stability)
            gamma: Weight for KL divergence constraint
            use_npo: Use Negative Preference Optimization style loss (bounded)
        """
        self.model = model
        self.ref_model = ref_model
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.use_npo = use_npo
        
        if optimizer is None:
            self.optimizer = AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=learning_rate
            )
        else:
            self.optimizer = optimizer
        
        # Freeze reference model if provided
        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
    
    def compute_forget_loss(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the forget loss (to be maximized).
        
        Uses NPO-style formulation: -log(sigmoid(-logits))
        which is numerically stable and bounded.
        """
        outputs = self.model(**inputs)
        
        if self.use_npo:
            # NPO formulation: minimize -log(1 - P(y|x))
            # Equivalent to: minimize log(sigmoid(-log_probs))
            # This is bounded compared to pure gradient ascent
            loss = -F.logsigmoid(-outputs.loss)
        else:
            # Simple negated CE (can be unstable)
            loss = -outputs.loss
        
        return loss
    
    def compute_retain_loss(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute standard cross-entropy loss on retain data."""
        outputs = self.model(**inputs)
        return outputs.loss
    
    def compute_kl_divergence(
        self, 
        inputs: Dict[str, torch.Tensor],
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute KL divergence between current and reference model.
        Acts as a regularization to prevent catastrophic drift.
        """
        if self.ref_model is None:
            return torch.tensor(0.0, device=self.device)
        
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)
            ref_logits = ref_outputs.logits / temperature
            ref_probs = F.softmax(ref_logits, dim=-1)
        
        curr_outputs = self.model(**inputs)
        curr_logits = curr_outputs.logits / temperature
        curr_log_probs = F.log_softmax(curr_logits, dim=-1)
        
        # KL(ref || curr) - we want current to stay close to reference
        kl_div = F.kl_div(
            curr_log_probs,
            ref_probs,
            reduction="batchmean"
        )
        
        return kl_div
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single DSU training step.
        
        Returns dict with individual loss components for logging.
        """
        self.optimizer.zero_grad()
        
        losses = {}
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Extract forget inputs
        if "forget" in batch:
            forget_inputs = batch["forget"]
        elif "forget_input_ids" in batch:
            forget_inputs = {
                "input_ids": batch["forget_input_ids"],
                "attention_mask": batch["forget_attention_mask"],
                "labels": batch["forget_labels"]
            }
        else:
            forget_inputs = None
        
        # Extract retain inputs
        if "retain" in batch:
            retain_inputs = batch["retain"]
        elif "retain_input_ids" in batch:
            retain_inputs = {
                "input_ids": batch["retain_input_ids"],
                "attention_mask": batch["retain_attention_mask"],
                "labels": batch["retain_labels"]
            }
        else:
            retain_inputs = None
        
        # Compute forget loss (with gradient ascent)
        if forget_inputs is not None:
            forget_inputs = {
                k: v.to(self.device) for k, v in forget_inputs.items() 
                if isinstance(v, torch.Tensor)
            }
            forget_loss = self.compute_forget_loss(forget_inputs)
            losses["forget_loss"] = forget_loss.item()
            total_loss = total_loss + self.alpha * forget_loss
        
        # Compute retain loss (standard descent)
        if retain_inputs is not None:
            retain_inputs = {
                k: v.to(self.device) for k, v in retain_inputs.items()
                if isinstance(v, torch.Tensor)
            }
            retain_loss = self.compute_retain_loss(retain_inputs)
            losses["retain_loss"] = retain_loss.item()
            total_loss = total_loss + self.beta * retain_loss
            
            # KL divergence on retain inputs
            if self.ref_model is not None and self.gamma > 0:
                kl_loss = self.compute_kl_divergence(retain_inputs)
                losses["kl_loss"] = kl_loss.item()
                total_loss = total_loss + self.gamma * kl_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.max_grad_norm
        )
        
        self.optimizer.step()
        
        losses["total_loss"] = total_loss.item()
        return losses
    
    def train(
        self,
        dataloader,
        num_epochs: int = 1,
        max_steps: Optional[int] = None,
        log_interval: int = 10
    ) -> Dict[str, float]:
        """Full training loop."""
        self.model.train()
        
        cumulative_losses = {}
        step_count = 0
        
        for epoch in range(num_epochs):
            pbar = tqdm(dataloader, desc=f"DSU Epoch {epoch+1}/{num_epochs}")
            
            for batch in pbar:
                losses = self.train_step(batch)
                step_count += 1
                
                # Accumulate losses
                for k, v in losses.items():
                    cumulative_losses[k] = cumulative_losses.get(k, 0.0) + v
                
                if step_count % log_interval == 0:
                    pbar.set_postfix({
                        k: f"{v:.4f}" for k, v in losses.items()
                    })
                
                if max_steps and step_count >= max_steps:
                    break
            
            if max_steps and step_count >= max_steps:
                break
        
        # Compute averages
        avg_losses = {
            f"avg_{k}": v / max(step_count, 1) 
            for k, v in cumulative_losses.items()
        }
        avg_losses["steps"] = step_count
        
        return avg_losses
