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

### Example Results (dry-run)

| Method | Forget Efficacy | Retain Performance | MIA Resistance |
|--------|-----------------|-------------------|----------------|
| Retraining (Gold) | 100% | 86% | 50% |
| Gradient Ascent | 100% | ~0% | 100% |
| DSU | 0% | 100% | 65% |
| TROU | 100% | ~0% | 51% |

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

## License

MIT
