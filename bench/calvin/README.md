# CALVIN Benchmark

Robot manipulation simulator wrapper. [mees/calvin](https://github.com/mees/calvin) 사용.

**주의**: 공식 benchmark는 `scripts/benchmark_calvin_mp.py` (multiprocess, TPU-integrated)로 실행.
여기 `run.sh`는 random rollout 검증용입니다.

## 설치

CLAUDE.md의 ["CALVIN Benchmark Dependencies"](../../CLAUDE.md#2-calvin-benchmark-dependencies) 섹션 참조.
핵심:
1. `uv pip install pybullet hydra-core==1.1.1 gym omegaconf ...`
2. `git clone --recurse-submodules https://github.com/mees/calvin.git ~/calvin`
3. (옵션) pyhash 패치
4. `export CALVIN_DIR=~/calvin`

## 실행

### Random rollout 검증

```bash
bash bench/calvin/run.sh
# → result/calvin/random_rollout.mp4
```

### 실제 benchmark (policy rollout + success rate)

```bash
bash commands/benchmark.sh calvin-abcd-flower --num-sequences 100 --num-workers 16
# → result/vla_abcd_flower/benchmark/results.json + MP4s
```

## 참고

- CALVIN task oracle: `calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml`
- Language annotations: `calvin_models/conf/annotations/new_playtable_validation.yaml`
- Multistep sequences: 1000 deterministic chains (`multistep_sequences.get_sequences(1000)` with `temp_seed(0)`)
- Observation: `rgb_static` (200×200), `rgb_gripper` (84×84), `robot_obs` (15)
- Action: (7,) delta `[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]`, gripper {-1, +1}
- Max steps per subtask: 360 (`EP_LEN`)
