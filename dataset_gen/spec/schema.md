# Dataset Schema

## 출력 레이아웃

```
<output_dir>/<maze>/
├── frames.zarr/                  ← 이미지 + 상태 (zarr v2 DirectoryStore)
│   └── episodes/
│       ├── 0/
│       │   ├── obs_topdown       ← (T, 224, 224, 3) uint8
│       │   ├── obs_third         ← (T, 224, 224, 3) uint8
│       │   ├── actions           ← (T, 8) float32
│       │   ├── qpos              ← (T, 15) float32
│       │   ├── qvel              ← (T, 14) float32
│       │   └── ep_goal_xy        ← (1, 2) float32
│       ├── 1/
│       └── ...
├── samples.parquet               ← hindsight 재라벨링 훈련 샘플
├── meta.json                     ← 데이터셋 메타
└── validation/                   ← validate_antmaze_dataset.py 출력
    ├── verify_ep{i}.mp4
    ├── stats_report.json
    ├── goal_xy_heatmap.png
    └── subgoal_heatmap.png
```

## zarr 배열 상세

모든 배열은 **Zstd(level=1)** 압축, chunk는 에피소드별로 다름:

| 이름 | shape | dtype | chunks | 의미 |
|---|---|---|---|---|
| `obs_topdown` | `(T, 224, 224, 3)` | uint8 | `(min(50,T), 224, 224, 3)` | 미로 전체를 위에서 내려다본 RGB |
| `obs_third` | `(T, 224, 224, 3)` | uint8 | `(min(50,T), 224, 224, 3)` | ant를 따라다니는 3인칭 45° RGB |
| `actions` | `(T, 8)` | float32 | `(min(200,T), 8)` | 8-DOF 관절 토크 (OGBench 원본) |
| `qpos` | `(T, 15)` | float32 | `(min(200,T), 15)` | MuJoCo 일반좌표 (앞 2개가 xy) |
| `qvel` | `(T, 14)` | float32 | `(min(200,T), 14)` | 일반속도 |
| `ep_goal_xy` | `(1, 2)` | float32 | `(1, 2)` | 에피소드 목표 xy (에피소드 마지막 qpos[:2]) |

- `T`: 에피소드 길이 (AntMaze는 대부분 1001 스텝 고정)
- chunk 단위는 TPU 학습 시 I/O 단위와 정렬 (chunk_size=50 권장)

## `samples.parquet` — 학습용 hindsight 샘플

각 행이 `(start_frame, end_frame, goal)` 트리플:

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `ep_idx` | int | 에피소드 번호 |
| `step_i` | int | 시작 프레임 인덱스 (에피소드 내) |
| `step_j` | int | 목표 프레임 인덱스 (`i < j ≤ ep_len-1`) |
| `goal_xy_x` | float | `qpos[ep_start+j, 0]` |
| `goal_xy_y` | float | `qpos[ep_start+j, 1]` |
| `language` | str | `"Go to (x.xx, y.yy)."` — LLM 입력용 |
| `maze` | str | `medium`/`large`/`giant` |
| `zarr_path` | str | 해당 에피소드가 저장된 zarr 절대경로 |
| `chunk_size` | int | 학습 시 읽을 윈도우 길이 (기본 16) |
| `image_size` | int | 224 |

### Hindsight 재라벨링 규칙

1. 각 에피소드에서 `i ∈ [0, ep_len - chunk_size)` 순회
2. `TERMINAL_GOAL_FRACTION = 0.25` 확률로 `j = ep_len - 1` (실제 목표)
3. 나머지 75%는 `j = randint(i+1, ep_len)` (중간 상태를 fake goal로)
4. 이렇게 만든 `(i, j)` 쌍이 학습 샘플 1개

샘플 수 ≈ 각 에피소드당 `(ep_len - chunk_size)` × 1000 에피소드 = 미로당 ~985,000 샘플.

## `meta.json`

```json
{
  "maze": "medium",
  "n_episodes": 1000,
  "n_steps": 1001000,
  "image_size": 224,
  "chunk_size": 16,
  "ep_length_mean": 1001.0,
  "n_train_samples": 985000
}
```

## 좌표계 주의

- `qpos[:, 0]` = x, `qpos[:, 1]` = y (미터, MuJoCo world frame)
- Ant는 `qpos[:, 2]` 위의 높이 + `qpos[:, 3:7]` quaternion + `qpos[:, 7:15]` 관절 8개
- topdown 카메라 파라미터는 [`camera_params.md`](camera_params.md) 참조
