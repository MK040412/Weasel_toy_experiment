# VLA: Vision-Language-Action (JAX/Flax NNX on TPU v4-8)

Qwen3-VL 2B (frozen) + GemmaActionExpert (~311M) with flow matching.

## 아키텍처

```
Images(top) + Language
  → Qwen3-VL 2B (frozen, JAX) → hidden (B, seq, 2048)
  → obs_proj (2048 → 1536)
  → GemmaActionExpert (12L, GQA 12/4, SwiGLU, ~311M)
    → continuous: (B, 50, 6) — flow matching denoising (pos + orn delta)
    → gripper:    (B, 50, 1) — BCE classification (discrete open/close)
```

## 2-Stage 학습 파이프라인

```
Stage 1: VLM Embedding Cache (학습 없음)
  Qwen3-VL forward → obs embeddings → PyArrow parquet 저장
  경로: {output_dir}/vlm_cache/embeddings.parquet
  이후 실행 시 VLM 로드 스킵, parquet에서 ~0.2s 로드

Stage 2: Action Expert Training
  캐시된 obs + actions → HBM-resident 배치 학습
  AdamW + warmup cosine decay, pure JAX indexing
```

## Gripper 처리

Gripper (dim 6)는 이산값 (open/close):
- `gripper_head` (별도 Linear) + BCE loss (flow matching 아님)
- GT: `(raw_gripper > 0).float()` → {0, 1}
- 추론: `sigmoid(logits) > 0.5`

## 실행

```bash
# 학습 + 평가 + 시각화 (debug dataset)
PYTHONPATH=src python src/qwen/vla/eval_and_viz.py

# CLI 학습
PYTHONPATH=src python src/qwen/vla/train.py                          # baseline
PYTHONPATH=src python src/qwen/vla/train.py --simulated-delay 15     # RTC
PYTHONPATH=src python src/qwen/vla/train.py --epochs 50 --lr 2e-4   # custom

# 추론
PYTHONPATH=src python src/qwen/vla/inference.py --checkpoint result/vla/checkpoint_train_final.npz
```

## CLI 인자

```
--epochs N              학습 에폭 수 (default: 100)
--lr FLOAT              학습률 (default: 5e-5)
--batch-size N          배치 크기 (default: 32)
--chunk-size N          action chunk 길이 (default: 50)
--simulated-delay N     RTC delay, 0=off (default: 0)
--output-dir PATH       checkpoint/cache 저장 경로
--seed N                랜덤 시드 (default: 42)
```

## RTC (Recurrent Time Chunking)

arXiv 2512.05964. Training-time RTC로 시간축 일관성 확보:

```
chunk:    [a₁, ..., a_d, a_{d+1}, ..., a₅₀]
timestep: [0,  ...,  0,   t,      ...,  t  ]
          ├─ prefix (GT) ─┤ ├─ postfix (noisy) ─┤
```

- `--simulated-delay d`: prefix 길이 [1, d] 랜덤 샘플링
- chunk_size=50 기준 `d=15` 권장

## Flow Matching (openpi0.5)

- t=1: noise, t=0: clean
- `x_t = t * noise + (1-t) * actions`
- velocity target: `noise - actions`
- timestep 분포: `Beta(1.5, 1.0)` → [0.001, 1.0]
- 추론: 10-step Euler denoising (t=1 → t=0)

## 데이터셋

| 데이터셋 | repo_id | chunks | 크기 |
|---------|---------|--------|------|
| Debug | `fywang/calvin-debug-lerobot` | 10 | 37 MB |
| ABCD→D | `fywang/calvin-task-ABCD-D-lerobot` | 19,037 | 67 GB |

- Action: (7,) delta = [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper] @ 30Hz
- 정규화: quantile (q01, q99) → [-1, 1], gripper 이진화 {0, 1}

## 디렉토리

```
config.py           설정 dataclass 모음
train.py            학습 CLI
inference.py        추론 CLI
eval_and_viz.py     학습 + 평가 + 3D 궤적 시각화
models/             ActionExpert, VLAPolicy, transformer layers
training/           VLATrainer, flow matching, RTC
data/               CalvinDataset (PyArrow parquet 직접 로드)
_pytorch_ref/       PyTorch 참고 구현 (아카이브, 미사용)
```
