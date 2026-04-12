# scripts/ — Standalone Python Scripts

`commands/*.sh` 가 실제로 호출하는 Python entry points.
직접 실행도 가능하지만 `commands/` 사용 권장 (환경변수 + 경로 자동 설정).

## 파일 목록

| 파일 | 용도 | 호출 래퍼 |
|------|------|----------|
| `preprocess_vlm_cache.py` | VLM embedding cache 생성 | `commands/preprocess.sh` |
| `eval_offline.py` | Offline eval (no sim, pos_err/grip_acc) | `commands/eval.sh` |
| `benchmark_calvin_mp.py` | **CALVIN benchmark (multiprocess 권장)** | `commands/benchmark.sh` |
| `benchmark_calvin.py` | CALVIN benchmark (single-process, 느림) | — (legacy) |

## `preprocess_vlm_cache.py`

VLM (Qwen3-VL 2B) 를 frozen으로 사용하여 각 샘플의 obs embedding을 계산 → parquet 저장.

**특징:**
- Queue-based CPU↔TPU pipeline (idle zero)
- pmap 4-device vision encoder
- batched language model (128 per batch)
- Numpy cache (host RAM, not HBM)

```bash
PYTHONPATH=src python scripts/preprocess_vlm_cache.py \
    --env calvin-abcd-flower \
    --output-dir result/vla_abcd_flower \
    --workers 180
```

**출력:** `{output-dir}/vlm_cache/embeddings.parquet` + `meta.json`

## `benchmark_calvin_mp.py` (권장)

**Multiprocessing** CALVIN sim benchmark.
- Main process: JAX policy on TPU (batched inference)
- N worker processes: pybullet CALVIN envs (multiprocessing Queue IPC)

공식 `evaluate_policy.py` 와 동일 metrics (1~5/5 success rates, avg chain length).

```bash
PYTHONPATH="src:$CALVIN_DIR/calvin_env:$CALVIN_DIR/calvin_models" \
.venv/bin/python scripts/benchmark_calvin_mp.py \
    --checkpoint result/vla_abcd_flower/checkpoint_train_final.npz \
    --num-sequences 100 \
    --num-workers 16 \
    --chunk-size 10 \
    --proprio-dim 8 \
    --save-videos 3
```

**출력:**
- `{output-dir}/results.json` — metrics
- `{output-dir}/success_*.mp4` — 성공 케이스 영상
- `{output-dir}/failure_*.mp4` — 실패 케이스 영상

## `eval_offline.py`

Sim 없이 val/test split 에서 offline prediction 비교.
**주의**: action MSE가 낮다고 sim success rate가 높진 않음 (compounding error).

```bash
bash commands/eval.sh calvin-abcd-flower val
```

**Metrics:** pos_error, orn_error, gripper_acc, action_mse

## 공통 의존성

- JAX (`.venv/bin/python` — TPU)
- `src/qwen/vla/` (우리 코드)
- `CALVIN_DIR` (benchmark only)
