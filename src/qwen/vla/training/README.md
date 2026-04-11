# vla/training/ — 학습 파이프라인

## 파일

### `trainer.py` — VLATrainer

2-stage 학습 + VLM 캐시 관리.

**Stage 1: VLM Embedding Cache**
- `cache_vlm_embeddings()`: parquet 캐시 존재 시 로드 (~0.2s), 없으면 VLM forward 후 저장
- 캐시 형식: PyArrow parquet (obs/actions/gripper를 numpy bytes로 직렬화)
- 캐시 경로: `{output_dir}/vlm_cache/embeddings.parquet` + `meta.json`
- `free_vlm()`: 캐시 후 VLM을 HBM에서 해제

**Stage 2: Action Expert Training**
- 배치 학습: `batch_size` (default 32), pure JAX array indexing (no `int()` sync)
- 마지막 배치 padding: static shape 유지 → JIT recompile 방지
- Optimizer: AdamW + warmup cosine decay schedule
- Loss: `compute_loss(vel_pred, vel_target, mask) + 0.1 * gripper_loss(logits, gt, mask)`

**VLM 캐시 parquet 구조:**
```
embeddings.parquet:
  obs     — binary column, per-sample (seq_len, 1536) float32 bytes
  actions — binary column, per-sample (50, 6) float32 bytes
  gripper — binary column, per-sample (50, 1) float32 bytes

meta.json:
  n_samples, max_seq_len, d_model, chunk_size
```

### `flow_matching.py` — Flow Matching Scheduler

openpi0.5 convention: t=1 noise, t=0 clean.

**함수:**

| 함수 | 역할 |
|------|------|
| `sample_timesteps(rng, B, T)` | Beta(1.5, 1.0) → [0.001, 1.0], 배치 uniform |
| `sample_timesteps_rtc(rng, B, T, delay)` | RTC: prefix t=0, postfix t=sampled |
| `make_noisy(actions, noise, t)` | `x_t = t*noise + (1-t)*actions` |
| `velocity_target(actions, noise)` | `noise - actions` |
| `compute_loss(pred, target, mask)` | MSE, masked, normalized |
| `gripper_loss(logits, gt, mask)` | BCE, masked, normalized |

**RTC 알고리즘 (`sample_timesteps_rtc`):**
```python
# 각 batch item 독립:
delay_len = randint(1, simulated_delay+1)  # prefix 길이
t_scalar  = Beta(1.5, 1.0)                 # postfix timestep

# position < delay_len → t=0 (clean GT)
# position >= delay_len → t=t_scalar (noisy)
# loss_mask: prefix=0, postfix=1
```
