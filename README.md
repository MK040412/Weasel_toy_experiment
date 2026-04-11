# Weasel Toy Experiment

Toy experiments on TPU v4-8: Qwen VLM inference/training, OGBench, CALVIN.

## Setup

```bash
git clone https://github.com/MK040412/Weasel_toy_experiment.git
cd Weasel_toy_experiment
uv sync
```

## Structure

| Directory | Role |
|-----------|------|
| `src/qwen/` | Qwen3-VL 2B / Qwen3.5-0.8B JAX model code |
| `bench/ogbench/` | Offline GCRL benchmark scripts |
| `bench/calvin/` | Robot manipulation benchmark scripts |
| `result/` | Benchmark output storage |
| `compare/` | HF vs JAX numerical validation |

## Usage

```bash
python src/qwen/inference.py --model qwen3vl    # VL inference
python src/qwen/train.py --mode both             # training benchmark
bash bench/ogbench/run.sh                        # OGBench
bash bench/calvin/run.sh                         # CALVIN
```

## Requirements

- TPU v4-8 (4 chips, single host)
- JAX >= 0.6.0, Python >= 3.10, uv
