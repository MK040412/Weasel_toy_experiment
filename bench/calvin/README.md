# CALVIN Benchmark

Robot manipulation simulator for VLA ablation. Uses [mees/calvin](https://github.com/mees/calvin).

## Setup

```bash
# calvin 별도 설치 (이 repo 외부)
git clone --recurse-submodules https://github.com/mees/calvin.git /path/to/calvin
cd /path/to/calvin
uv venv --python 3.10 && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# pyhash 패치 필요 (CLAUDE.md 참조)
cd calvin_env && pip install -e . && cd ..
cd calvin_models && pip install -e . && cd ..
```

## Run

```bash
# 랜덤 rollout 검증
bash bench/calvin/run.sh

# 학습
source /path/to/calvin/.venv/bin/activate
export PYOPENGL_PLATFORM=osmesa MESA_GL_VERSION_OVERRIDE=3.3
cd /path/to/calvin/calvin_models
python calvin_agent/training.py
```

## Results

Output: `result/calvin/` (rollout videos, metrics)
