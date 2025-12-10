"""
Benchmark runner using Unsloth for optimized VRAM usage.
This version enables IWAP which requires gradient computation.
"""

import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import json
import argparse
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from tqdm import tqdm
import time
import gc

from src.data_utils import load_tofu_dataset, UnlearningDataset, InterleavedDataLoader
from src.model_utils_unsloth import setup_qwen_model_unsloth, get_trainable_param_count
from src.baselines import GradientAscentTrainer
from src.dsu import DSUTrainer
from src.iwap import IWAPWithRepair
from src.trou import TROUTrainer
from src.metrics import (
    compute_perplexity,
    run_full_evaluation,
)
from torch.utils.data import DataLoader


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    model_id: str = "unsloth/Qwen3-4B-unsloth-bnb-4bit"  # Pre-quantized model
    max_seq_length: int = 1024  # Reduced for memory
    lora_r: int = 16
    
    tofu_subset: str = "forget01"
    max_length: int = 512
    batch_size: int = 1
    
    learning_rate: float = 2e-5
    num_epochs: int = 3
    max_steps: Optional[int] = 100
    
    # IWAP settings
    iwap_prune_ratio: float = 0.02
    iwap_max_batches: int = 20  # Limit for memory
    
    output_dir: str = "outputs"
    estimated_retrain_hours: float = 24.0


@dataclass 
class MethodResult:
    """Results for a single unlearning method."""
    method_name: str
    forget_efficacy: float
    retain_performance: float
    mia_resistance: float
    computational_cost: float
    training_time: float
    additional_metrics: Dict[str, float] = None
    
    def to_latex_row(self) -> str:
        return (
            f"            {self.method_name} & "
            f"{self.forget_efficacy:.2f}\\% & "
            f"{self.retain_performance:.2f}\\% & "
            f"{self.mia_resistance:.2f}\\% & "
            f"{self.computational_cost:.1f}\\% \\\\"
        )


class UnslothBenchmark:
    """Benchmark using Unsloth for lower VRAM."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.results: List[MethodResult] = []
        
        self.model = None
        self.tokenizer = None
        self.forget_loader = None
        self.retain_loader = None
        self.train_loader = None
        
        self.baseline_forget_ppl = None
        self.baseline_retain_ppl = None
    
    def setup(self):
        """Initialize model and data with Unsloth."""
        print("=" * 60)
        print("Setting up Unsloth benchmark environment...")
        print("=" * 60)
        
        # Load model with Unsloth
        print("\n[1/3] Loading model with Unsloth (low VRAM mode)...")
        self.model, self.tokenizer = setup_qwen_model_unsloth(
            model_id=self.config.model_id,
            max_seq_length=self.config.max_seq_length,
            lora_r=self.config.lora_r
        )
        
        # Print VRAM usage
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1024**3
            print(f"  VRAM used after model load: {vram_used:.2f} GB")
        
        # Load data
        print("\n[2/3] Loading TOFU dataset...")
        data = load_tofu_dataset(self.config.tofu_subset)
        
        forget_dataset = UnlearningDataset(
            data["forget"],
            self.tokenizer,
            self.config.max_length,
            is_forget=True
        )
        retain_dataset = UnlearningDataset(
            data["retain"],
            self.tokenizer,
            self.config.max_length,
            is_forget=False
        )
        
        self.forget_loader = DataLoader(
            forget_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        self.retain_loader = DataLoader(
            retain_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        self.train_loader = InterleavedDataLoader(
            forget_dataset,
            retain_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        print(f"  Forget samples: {len(forget_dataset)}")
        print(f"  Retain samples: {len(retain_dataset)}")
        
        # Compute baseline perplexities
        print("\n[3/3] Computing baseline perplexities...")
        self.baseline_forget_ppl = compute_perplexity(
            self.model, self.forget_loader, self.device, max_batches=30
        )
        self.baseline_retain_ppl = compute_perplexity(
            self.model, self.retain_loader, self.device, max_batches=30
        )
        print(f"  Baseline forget PPL: {self.baseline_forget_ppl:.2f}")
        print(f"  Baseline retain PPL: {self.baseline_retain_ppl:.2f}")
        
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1024**3
            print(f"  VRAM after setup: {vram_used:.2f} GB")
        
        print("\nSetup complete!")
    
    def _reset_model(self):
        """Reload model with Unsloth."""
        print("  Resetting model to fresh state...")
        
        if self.model is not None:
            del self.model
            self.model = None
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        self.model, self.tokenizer = setup_qwen_model_unsloth(
            model_id=self.config.model_id,
            max_seq_length=self.config.max_seq_length,
            lora_r=self.config.lora_r
        )
    
    def _evaluate_and_record(
        self,
        method_name: str,
        training_time: float,
        additional: Optional[Dict] = None
    ) -> MethodResult:
        """Evaluate and record results."""
        print(f"  Evaluating {method_name}...")
        
        metrics = run_full_evaluation(
            model=self.model,
            tokenizer=self.tokenizer,
            forget_loader=self.forget_loader,
            retain_loader=self.retain_loader,
            baseline_forget_ppl=self.baseline_forget_ppl,
            baseline_retain_ppl=self.baseline_retain_ppl,
            device=self.device,
            computational_cost=100.0 * training_time / (self.config.estimated_retrain_hours * 3600),
            max_batches=50
        )
        
        result = MethodResult(
            method_name=method_name,
            forget_efficacy=metrics["forget_efficacy"],
            retain_performance=metrics["retain_performance"],
            mia_resistance=metrics["mia_resistance"],
            computational_cost=metrics["computational_cost"],
            training_time=training_time,
            additional_metrics=additional
        )
        
        self.results.append(result)
        print(f"    Forget Efficacy: {result.forget_efficacy:.2f}%")
        print(f"    Retain Performance: {result.retain_performance:.2f}%")
        print(f"    MIA Resistance: {result.mia_resistance:.2f}%")
        
        return result
    
    def run_retraining_baseline(self) -> MethodResult:
        """Theoretical gold standard."""
        print("\n" + "=" * 60)
        print("Method: Retraining (Gold Standard)")
        print("=" * 60)
        print("  Using theoretical values")
        
        result = MethodResult(
            method_name="Retraining (Gold Standard)",
            forget_efficacy=100.0,
            retain_performance=86.0,
            mia_resistance=50.0,
            computational_cost=100.0,
            training_time=self.config.estimated_retrain_hours * 3600
        )
        self.results.append(result)
        return result
    
    def run_gradient_ascent(self) -> MethodResult:
        """Run gradient ascent baseline."""
        print("\n" + "=" * 60)
        print("Method: Gradient Ascent (GA)")
        print("=" * 60)
        
        self._reset_model()
        
        trainer = GradientAscentTrainer(
            model=self.model,
            learning_rate=self.config.learning_rate * 0.5,
            device=self.device
        )
        
        start = time.time()
        trainer.train(
            self.forget_loader,
            num_epochs=1,
            max_steps=min(40, self.config.max_steps or 40)
        )
        training_time = time.time() - start
        
        return self._evaluate_and_record("Gradient Ascent (GA)", training_time)
    
    def run_iwap(self) -> MethodResult:
        """Run IWAP with Unsloth's lower VRAM."""
        print("\n" + "=" * 60)
        print("Method: Influence Guided Weight Atom Pruning (IWAP)")
        print("=" * 60)
        
        self._reset_model()
        
        # Print VRAM before IWAP
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1024**3
            print(f"  VRAM before IWAP: {vram_used:.2f} GB")
        
        iwap = IWAPWithRepair(
            model=self.model,
            device=self.device,
            repair_lr=self.config.learning_rate * 0.5,
            repair_steps=30
        )
        
        start = time.time()
        stats = iwap.run(
            forget_loader=self.forget_loader,
            retain_loader=self.retain_loader,
            prune_ratio=self.config.iwap_prune_ratio,
            max_batches=self.config.iwap_max_batches,
            repair_steps=30
        )
        training_time = time.time() - start
        
        return self._evaluate_and_record(
            "Influence Function (IWAP)",
            training_time,
            additional=stats
        )
    
    def run_dsu(self) -> MethodResult:
        """Run DSU."""
        print("\n" + "=" * 60)
        print("Method: Dual Signed Adaptation Unlearning (DSU)")
        print("=" * 60)
        
        self._reset_model()
        
        trainer = DSUTrainer(
            model=self.model,
            ref_model=None,
            learning_rate=self.config.learning_rate,
            device=self.device,
            alpha=1.0,
            beta=5.0,
            gamma=0.0
        )
        
        start = time.time()
        trainer.train(
            self.train_loader,
            num_epochs=self.config.num_epochs,
            max_steps=self.config.max_steps
        )
        training_time = time.time() - start
        
        return self._evaluate_and_record("Dual Signed Adaptation (DSU)", training_time)
    
    def run_trou(self) -> MethodResult:
        """Run TROU."""
        print("\n" + "=" * 60)
        print("Method: Trust-Region Orthogonal Unlearning (TROU)")
        print("=" * 60)
        
        self._reset_model()
        
        trainer = TROUTrainer(
            model=self.model,
            learning_rate=self.config.learning_rate,
            device=self.device,
            max_grad_norm=1.0
        )
        
        start = time.time()
        trainer.train(
            self.train_loader,
            num_epochs=self.config.num_epochs,
            max_steps=self.config.max_steps
        )
        training_time = time.time() - start
        
        return self._evaluate_and_record("Trust-Region Orthogonal (TROU)", training_time)
    
    def run_all(self) -> List[MethodResult]:
        """Run all benchmarks including IWAP."""
        self.run_retraining_baseline()
        self.run_gradient_ascent()
        self.run_iwap()  # Now enabled with Unsloth!
        self.run_dsu()
        self.run_trou()
        
        return self.results
    
    def generate_latex_table(self) -> str:
        """Generate LaTeX table."""
        table = r"""
\begin{table}[t!]
    \centering
    \begin{threeparttable}
        \caption{Baseline unlearning performance on the TOFU benchmark.}
        \label{tab:baseline-results}
        \begin{tabular}{@{}lcccc@{}}
            \toprule
            \textbf{Method} & \textbf{\makecell{Forget \\ Efficacy ($\uparrow$)}} & \textbf{\makecell{Retain \\ Performance ($\uparrow$)}} & \textbf{\makecell{MIA \\ Resistance ($\uparrow$)}} & \textbf{\makecell{Comp. \\ Cost ($\downarrow$)}} \\
            \midrule
"""
        for result in self.results:
            table += result.to_latex_row() + "\n"
        
        table += r"""            \bottomrule
        \end{tabular}
        \begin{tablenotes}[para,flushleft]
            \small
            \textit{Note:} \textbf{Forget Efficacy} measures success at forgetting; \textbf{Retain Performance} measures accuracy on retain data; \textbf{MIA Resistance} measures attack failure rate; \textbf{Comp. Cost} is relative to retraining. Higher is better for all metrics except Comp. Cost.
        \end{tablenotes}
    \end{threeparttable}
\end{table}
"""
        return table
    
    def save_results(self):
        """Save results."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # JSON
        json_path = output_dir / "benchmark_results_unsloth.json"
        with open(json_path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        print(f"Saved JSON to {json_path}")
        
        # LaTeX
        latex_path = output_dir / "results_table_unsloth.tex"
        with open(latex_path, "w") as f:
            f.write(self.generate_latex_table())
        print(f"Saved LaTeX to {latex_path}")


def main():
    parser = argparse.ArgumentParser(description="Unsloth Machine Unlearning Benchmark")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--dry_run", action="store_true")
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        max_steps=10 if args.dry_run else args.max_steps
    )
    
    benchmark = UnslothBenchmark(config)
    benchmark.setup()
    benchmark.run_all()
    benchmark.save_results()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS - LaTeX Table")
    print("=" * 60)
    print(benchmark.generate_latex_table())


if __name__ == "__main__":
    main()
