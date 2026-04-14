# Camera Parameters

두 종류의 카메라로 같은 시뮬 상태를 렌더링:

## 1. Topdown Camera (`obs_topdown`)

미로 전체를 한 장에 담기 위한 고정 위치 직교 근사 카메라. 미로 크기별 다름:

| maze | lookat_x | lookat_y | distance | elevation | azimuth |
|---|---|---|---|---|---|
| medium | 10.0 | 10.0 | 30.0 | -90° | 0° |
| large  | 18.0 | 10.0 | 50.0 | -90° | 0° |
| giant  | 26.0 | 18.0 | 70.0 | -90° | 0° |

- `elevation = -90°` → 완전 수직 아래 방향
- `distance`가 클수록 넓게 보임 (giant는 미로가 커서 70까지 끌어올림)
- `lookat`는 각 미로의 중심 좌표 (대략)

## Topdown 픽셀 ↔ 월드 좌표 변환

MuJoCo 렌더러는 FOV 기본 45°(half-angle 22.5°). 카메라 높이가 `distance`이므로 한 프레임이 커버하는 월드 반경:

```python
half_range = distance * tan(radians(22.5))
```

월드 `(x, y)` → 픽셀 `(col, row)`:

```python
col = int(w/2 - (y - lookat_y) / half_range * w/2)
row = int(h/2 - (x - lookat_x) / half_range * h/2)
```

주의:
- 이미지 좌표계: `(row, col)` = `(y, x)` (numpy 관례)
- 월드 x 축 → 이미지 **row** (위아래)
- 월드 y 축 → 이미지 **col** (좌우)
- 부호가 뒤집혀 있음 (`w/2 - ...`) — topdown이 위에서 아래를 보는 방향 때문

역변환 (픽셀 → 월드):
```python
y = lookat_y + (w/2 - col) * 2 * half_range / w
x = lookat_x + (h/2 - row) * 2 * half_range / h
```

## 2. Third-person Camera (`obs_third`)

Ant를 따라다니는 45° 사선 트래킹 카메라:

```python
track_cam.lookat = [qpos[t, 0], qpos[t, 1], 0.5]  # 매 프레임 ant 위치로 업데이트
track_cam.distance  = 8.0
track_cam.elevation = -45.0
track_cam.azimuth   = 45.0
```

- 매 스텝 ant의 xy 위치를 추적 (`lookat[0:2] = qpos[t, :2]`)
- 미로 크기에 무관하게 고정 파라미터
- VLA 학습에서 로봇의 로컬 맥락을 주는 용도

## 렌더링 구현

`make_paired_dataset_fast.py`의 `render_episode_chunk()`:

```python
# 워커마다 독립적 env + renderer
env = gymnasium.make(ENV_NAME_MAP[maze], render_mode="rgb_array",
                     width=image_size, height=image_size)
renderer = mujoco.Renderer(raw.model, height=image_size, width=image_size)

td_cam = make_topdown_camera(maze)
track_cam = mujoco.MjvCamera()
track_cam.lookat[2] = 0.5
track_cam.distance  = 8.0
track_cam.elevation = -45.0
track_cam.azimuth   = 45.0

# 매 스텝:
data.qpos[:] = qpos[t_global]         # 원본 npz에서 복원
data.qvel[:] = qvel[t_global]
mujoco.mj_forward(raw.model, data)    # physics 진행 없이 forward만

renderer.update_scene(data, camera=td_cam)
obs_td[t_local] = renderer.render()

track_cam.lookat[:2] = qpos[t_global, :2]
renderer.update_scene(data, camera=track_cam)
obs_3rd[t_local] = renderer.render()
```

핵심: **`mj_step` 대신 `mj_forward`** — 물리 재시뮬 없이 상태만 복원해 렌더링. 결정론적이며 원본 OGBench trajectory와 정확히 일치.
