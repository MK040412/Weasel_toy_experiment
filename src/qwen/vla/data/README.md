# vla/data/ — 데이터셋

## CalvinDataset (`lerobot_calvin.py`)

LeRobot v2.1 CALVIN format. PyArrow parquet 직접 로드 (HF `datasets` 미사용, 299x 빠른 init).

### 로드 전략 (3-phase)

1. **Phase 1 (metadata, ~10ms)**: action/episode/task/frame index만 PyArrow로 병렬 로드
2. **Phase 2 (lazy cache)**: episode별 action numpy cache + image parquet table cache
3. **Phase 3 (on-demand)**: PNG image decode는 `__getitem__` 호출 시에만

### `__getitem__` 반환값

```python
{
    "images":             np.ndarray  # (n_cameras, H, W, 3) float32 [0, 1]
    "actions_continuous": np.ndarray  # (T, 6) normalized [-1, 1]
    "gripper":            np.ndarray  # (T, 1) binary {0, 1}
    "raw_actions":        np.ndarray  # (T, 7) original action space
    "language":           str         # task description
    "episode":            int         # episode ID
}
```

### Action Space

```
dim 0-2: Δx, Δy, Δz        (position delta)
dim 3-5: Δrx, Δry, Δrz     (orientation delta)
dim 6:   gripper             (-1=open, +1=close → binarized to {0, 1})
```

- 정규화: quantile (q01, q99) → [-1, 1] (continuous 6-dim만)
- Gripper: `(raw > 0).float()` → {0, 1} (정규화 안 함)
- Chunk: stride = chunk_size // 2 (50% overlap)

### 데이터셋

| 이름 | repo_id | frames | episodes | chunks | 크기 |
|------|---------|--------|----------|--------|------|
| Debug | `fywang/calvin-debug-lerobot` | 24k | 12 | 10 | 37 MB |
| ABCD→D | `fywang/calvin-task-ABCD-D-lerobot` | 1.4M | 24k | 19k | 67 GB |

ABCD→D는 디스크 66 GB로 부족 → `/dev/shm` (RAM) 다운로드 사용.
`data/download/fywang/calvin-task-ABCD-D-lerobot.py` 참조.
