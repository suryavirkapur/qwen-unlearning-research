"""
Trust-Region Orthogonal Unlearning (TROU) implementation.

TROU addresses gradient interference by projecting the unlearning update
onto the null space of the retain gradient. This ensures (to first order)
that the retain loss remains unchanged while maximizing forgetting.

Key insight: g_orthogonal = g_forget - proj(g_forget onto g_retain)
"""

import torch
from torch.optim import AdamW
from typing import Dict, Any, Optional, List
from tqdm import tqdm


class TROUTrainer:
    """
    Trust-Region Orthogonal Unlearning Trainer.
    
    Algorithm:
    1. Compute gradient g_f on forget set (for ascent)
    2. Compute gradient g_r on retain set (constraint)
    3. Project: g_orth = g_f - (g_f · g_r / ||g_r||²) * g_r
    4. Trust region: clip ||g_orth|| to max_norm
    5. Update: θ = θ - lr * g_orth
    
    The orthogonal projection ensures the update has zero correlation
    with the retain gradient, theoretically preserving retention performance.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        learning_rate: float = 2e-5,
        device: str = "cuda",
        max_grad_norm: float = 1.0,  # Trust region bound
        projection_eps: float = 1e-8  # Numerical stability
    ):
        self.model = model
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.projection_eps = projection_eps
        
        if optimizer is None:
            self.optimizer = AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=learning_rate
            )
        else:
            self.optimizer = optimizer
    
    def _compute_gradients(
        self, 
        inputs: Dict[str, torch.Tensor],
        negate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gradients for all trainable parameters.
        
        Args:
            inputs: Model inputs
            negate: If True, compute gradient for ascent (forget)
        
        Returns:
            Dict mapping param names to gradient tensors
        """
        self.model.zero_grad()
        
        outputs = self.model(**inputs)
        loss = -outputs.loss if negate else outputs.loss
        loss.backward()
        
        grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone()
        
        return grads, loss.item()
    
    def _project_to_orthogonal(
        self,
        g_forget: Dict[str, torch.Tensor],
        g_retain: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Project forget gradient onto null space of retain gradient.
        
        g_orth = g_forget - proj_{g_retain}(g_forget)
               = g_forget - ((g_f · g_r) / ||g_r||²) * g_r
        
        This removes the component of g_forget that aligns with g_retain,
        ensuring the update doesn't affect retain performance (to first order).
        """
        g_orthogonal = {}
        
        for name in g_forget:
            if name not in g_retain:
                # No retain gradient for this param, use full forget gradient
                g_orthogonal[name] = g_forget[name]
                continue
            
            g_f = g_forget[name]
            g_r = g_retain[name]
            
            # Compute dot product: g_f · g_r
            dot_product = torch.sum(g_f * g_r)
            
            # Compute ||g_r||²
            norm_sq_retain = torch.sum(g_r * g_r)
            
            if norm_sq_retain > self.projection_eps:
                # Compute projection coefficient
                proj_coef = dot_product / norm_sq_retain
                
                # Projection of g_f onto g_r
                projection = proj_coef * g_r
                
                # Orthogonal component
                g_orth = g_f - projection
            else:
                # g_retain is essentially zero, use full g_forget
                g_orth = g_f
            
            g_orthogonal[name] = g_orth
        
        return g_orthogonal
    
    def _apply_trust_region(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply trust region constraint by clipping gradient magnitude.
        
        If ||g|| > max_norm, rescale: g = g * (max_norm / ||g||)
        """
        # Compute global gradient norm
        total_norm_sq = sum(
            torch.sum(g ** 2).item() for g in gradients.values()
        )
        total_norm = total_norm_sq ** 0.5
        
        if total_norm > self.max_grad_norm:
            scale = self.max_grad_norm / total_norm
            gradients = {
                name: g * scale for name, g in gradients.items()
            }
        
        return gradients
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single TROU training step.
        
        Returns dict with loss components and projection stats.
        """
        losses = {}
        
        # Extract inputs
        if "forget" in batch:
            forget_inputs = batch["forget"]
            retain_inputs = batch["retain"]
        else:
            forget_inputs = {
                "input_ids": batch["forget_input_ids"],
                "attention_mask": batch["forget_attention_mask"],
                "labels": batch["forget_labels"]
            }
            retain_inputs = {
                "input_ids": batch["retain_input_ids"],
                "attention_mask": batch["retain_attention_mask"],
                "labels": batch["retain_labels"]
            }
        
        # Move to device
        forget_inputs = {
            k: v.to(self.device) for k, v in forget_inputs.items()
            if isinstance(v, torch.Tensor)
        }
        retain_inputs = {
            k: v.to(self.device) for k, v in retain_inputs.items()
            if isinstance(v, torch.Tensor)
        }
        
        # Step 1: Compute retain gradients (the constraint)
        g_retain, retain_loss = self._compute_gradients(retain_inputs, negate=False)
        losses["retain_loss"] = retain_loss
        
        # Step 2: Compute forget gradients (for ascent, so negate)
        g_forget, forget_loss = self._compute_gradients(forget_inputs, negate=True)
        losses["forget_loss"] = -forget_loss  # Log the original positive loss
        
        # Step 3: Project forget gradient to orthogonal subspace
        g_orthogonal = self._project_to_orthogonal(g_forget, g_retain)
        
        # Compute projection statistics (for debugging)
        total_forget_norm = sum(
            torch.sum(g ** 2).item() for g in g_forget.values()
        ) ** 0.5
        total_orth_norm = sum(
            torch.sum(g ** 2).item() for g in g_orthogonal.values()
        ) ** 0.5
        
        if total_forget_norm > 0:
            losses["projection_ratio"] = total_orth_norm / total_forget_norm
        else:
            losses["projection_ratio"] = 1.0
        
        # Step 4: Apply trust region constraint
        g_orthogonal = self._apply_trust_region(g_orthogonal)
        
        # Step 5: Set gradients and update
        self.optimizer.zero_grad()
        
        for name, param in self.model.named_parameters():
            if name in g_orthogonal:
                param.grad = g_orthogonal[name]
        
        self.optimizer.step()
        
        return losses
    
    def train(
        self,
        dataloader,
        num_epochs: int = 1,
        max_steps: Optional[int] = None,
        log_interval: int = 10
    ) -> Dict[str, float]:
        """Full training loop with orthogonal projection."""
        self.model.train()
        
        cumulative = {}
        step_count = 0
        
        for epoch in range(num_epochs):
            pbar = tqdm(dataloader, desc=f"TROU Epoch {epoch+1}/{num_epochs}")
            
            for batch in pbar:
                losses = self.train_step(batch)
                step_count += 1
                
                for k, v in losses.items():
                    cumulative[k] = cumulative.get(k, 0.0) + v
                
                if step_count % log_interval == 0:
                    pbar.set_postfix({
                        k: f"{v:.4f}" for k, v in losses.items()
                    })
                
                if max_steps and step_count >= max_steps:
                    break
            
            if max_steps and step_count >= max_steps:
                break
        
        # Compute averages
        result = {
            f"avg_{k}": v / max(step_count, 1)
            for k, v in cumulative.items()
        }
        result["steps"] = step_count
        
        return result


class TROUWithSubspaceBatching(TROUTrainer):
    """
    Extended TROU that uses multiple retain batches to define a richer
    constraint subspace.
    
    Instead of projecting onto the null space of a single retain gradient,
    we accumulate multiple retain gradients and project onto the null space
    of their span. This provides stronger protection for retain performance.
    """
    
    def __init__(
        self,
        *args,
        subspace_batches: int = 4,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.subspace_batches = subspace_batches
        self.retain_gradient_buffer: List[Dict[str, torch.Tensor]] = []
    
    def accumulate_retain_gradient(self, inputs: Dict[str, torch.Tensor]):
        """Add a retain gradient to the subspace buffer."""
        g_retain, _ = self._compute_gradients(inputs, negate=False)
        
        self.retain_gradient_buffer.append(g_retain)
        
        # Keep only the most recent gradients
        if len(self.retain_gradient_buffer) > self.subspace_batches:
            self.retain_gradient_buffer.pop(0)
    
    def project_to_subspace_null(
        self,
        g_forget: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Project onto null space of the accumulated retain gradient subspace.
        
        Uses Gram-Schmidt orthogonalization against each retained gradient.
        """
        g_orthogonal = {name: g.clone() for name, g in g_forget.items()}
        
        # Iteratively project out each retain gradient direction
        for g_retain in self.retain_gradient_buffer:
            g_orthogonal = self._project_to_orthogonal(g_orthogonal, g_retain)
        
        return g_orthogonal
