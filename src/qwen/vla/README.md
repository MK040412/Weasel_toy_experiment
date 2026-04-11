# VLA: Vision-Language-Action (JAX/Flax on TPU v4-8)

Qwen3-VL 2B (frozen) + GemmaActionExpert (~311M) with flow matching.

## Architecture

```
Images(top+wrist) + Language
  -> Qwen3-VL 2B (frozen, JAX, 4-chip) -> hidden (B, seq, 2048)
  -> obs_proj (2048 -> 1536)
  -> GemmaActionExpert (12L transformer, GQA 12/4, SwiGLU)
    -> continuous: (B, 50, 6) via flow matching denoising
    -> gripper: (B, 50, 1) via BCE classification
```

## Gripper Handling

Gripper (dim 6) is **discrete** (open/close), not continuous:
- Separate `gripper_head` with BCE loss (not flow matching)
- Binarized at threshold 0: `gripper_gt = (raw_gripper > 0).float()`
- At inference: `sigmoid(logits) > 0.5`

## Training (2-stage)

```bash
# Standard (no RTC)
python src/qwen/vla/train.py

# With RTC (Recurrent Time Chunking, delay=15)
python src/qwen/vla/train.py --simulated-delay 15

# Quick test
python src/qwen/vla/train.py --stage1-epochs 3 --stage2-epochs 2
```

### Stages
1. **Cache VLM embeddings** (one-time Qwen3-VL forward)
2. **Stage 1**: Train obs_proj + action_expert (30 epochs, lr=5e-5)
3. **Stage 2**: Fine-tune with lower LR (20 epochs, lr=5e-6)

## Inference + Visualization

```bash
# Evaluate on validation set
python src/qwen/vla/inference.py --split val --visualize

# Memorization check (evaluate on training data)
python src/qwen/vla/inference.py --memorization-check --visualize
```

Output: `result/vla/rollout_ep*.mp4` (video + language annotation + gripper status)

## RTC Ablation

Train two models and compare:
```bash
python src/qwen/vla/train.py --simulated-delay 0   # baseline
python src/qwen/vla/train.py --simulated-delay 15   # RTC
```

## Flow Matching (openpi0.5 convention)

- t=1: noise, t=0: clean
- `x_t = t * noise + (1-t) * actions`
- Velocity target: `noise - actions`
- Time sampling: `t ~ Beta(1.5, 1.0)`, mapped to [0.001, 1.0]
- Denoising: 10-step Euler from t=1 -> t=0

## Dataset

`fywang/calvin-debug-lerobot` (HuggingFace, LeRobot v2.1):
- ~600 episodes, 34 tasks
- Actions: (7,) = [x, y, z, rx, ry, rz, gripper]
- Cameras: top, wrist (336x336)
- Quantile normalization (q01/q99) -> [-1, 1]
