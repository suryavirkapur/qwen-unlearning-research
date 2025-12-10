# Qwen3-4B Machine Unlearning Research

Benchmarking advanced machine unlearning methods on Qwen3-4B using the TOFU dataset.

## Methods Implemented

| Method | Description |
|--------|-------------|
| **Naive Fine-tuning** | Baseline - fine-tune on retain set only |
| **Gradient Ascent** | Maximize loss on forget set |
| **DSU** | Dual Signed Adaptation Unlearning |
| **IWAP** | Influence-Guided Weight Atom Pruning |
| **TROU** | Trust-Region Orthogonal Unlearning |

## Benchmarks

### TOFU (Task of Fictitious Unlearning) 
Uses fictitious author profiles as the forget set, testing whether models can unlearn specific factual knowledge while retaining general language capabilities. Configurations: `forget01` (1%), `forget05` (5%), `forget10` (10%).

### MUSE (Multi-turn, Unsafe, and Solicited Evaluations)
Treats a subset of **unsafe Q&A pairs as the forget set** to simulate removal of toxic behaviors. Uses a retain set to test that the ability to answer normal conversational questions remains intact. Evaluates whether a model can *"forget to be unsafe"* without losing its language capabilities.

### WMDP (Malicious Scenarios over Large Language Models)
Uses a subset of **malicious instructions as the forget set**, and benign examples as the retain set. Key metric: **Resistance to Regeneration** - measures whether the model does not recall or paraphrase malicious outputs it was supposed to forget.

## Quick Start

```bash
# Install dependencies
uv sync

# Run dry-run benchmark (fast, ~5 min)
uv run python src/benchmark.py --dry_run

# Run full benchmark
uv run python src/benchmark.py --max_steps 100
```

## Results

After running the benchmark, results are saved to:
- `outputs/benchmark_results.json` - Raw metrics
- `outputs/results_table.tex` - LaTeX table

### TOFU Results (Factual Knowledge Unlearning)

| Method | Forget Efficacy | Retain Performance | MIA Resistance |
|--------|-----------------|-------------------|----------------|
| Retraining (Gold) | 100% | 86% | 50% |
| Naive Fine-tuning | 0% | 100% | 86% |
| Gradient Ascent | 100% | 3% | 86% |
| DSU | 0% | 100% | 79% |
| TROU | 100% | 0% | 89% |

### MUSE Results (Content Memorization)

| Method | Forget Efficacy | Retain Performance | MIA Resistance |
|--------|-----------------|-------------------|----------------|
| Naive Fine-tuning | 0% | 100% | 95% |
| Gradient Ascent | 0% | 98% | 95% |
| DSU | 0% | 100% | 95% |
| TROU | 0% | 96% | 95% |

### WMDP Results (Hazardous Knowledge) ⭐

| Method | Forget Efficacy | Retain Performance | MIA Resistance |
|--------|-----------------|-------------------|----------------|
| Naive Fine-tuning | 0% | 100% | 1% |
| Gradient Ascent | 100% | 4% | 32% |
| DSU | 0% | 100% | 1% |
| **TROU** | **100%** | **100%** | 11% |

> **Key Finding**: TROU achieves perfect unlearning (100% forget, 100% retain) on WMDP, suggesting it's particularly effective for removing hazardous knowledge.

## Project Structure

```
src/
├── benchmark.py      # Main benchmark orchestrator
├── data_utils.py     # TOFU dataset loading
├── model_utils.py    # Qwen3 + QLoRA setup
├── baselines.py      # Naive FT, Gradient Ascent
├── dsu.py            # Dual Signed Adaptation
├── iwap.py           # Influence-Guided Pruning
├── trou.py           # Trust-Region Orthogonal
└── metrics.py        # Evaluation metrics
```

## Hardware Requirements

- **GPU**: NVIDIA GPU with 10GB+ VRAM (tested on RTX 3080)
- **RAM**: 16GB+ recommended

## Known Issues

See [docs/issue_qwen.md](docs/issue_qwen.md) for Unsloth compatibility issues with Qwen3.

