# CALVIN Benchmark

Robot manipulation simulator. [mees/calvin](https://github.com/mees/calvin) 래핑.

## 설치 (이 repo 외부, 별도 venv)

```bash
git clone --recurse-submodules https://github.com/mees/calvin.git /path/to/calvin
cd /path/to/calvin
uv venv --python 3.10 && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### pyhash 패치

`pyhash` 빌드 실패 시 아래 함수로 대체:

```python
# calvin_models/calvin_agent/datasets/base_dataset.py
# calvin_models/calvin_agent/evaluation/utils.py
# 두 파일에서 import pyhash 제거, 아래 함수 추가:

def _fnv1_32(data: bytes) -> int:
    h = 0x811c9dc5
    for b in data:
        h = ((h * 0x01000193) ^ b) & 0xFFFFFFFF
    return h
```

`calvin_models/requirements.txt`에서 `pyhash` 줄 제거.

```bash
pip install wheel cmake==3.18.4
cd calvin_env && pip install -e . && cd ..
cd calvin_models && pip install -e . && cd ..
```

## 실행

```bash
source /path/to/calvin/.venv/bin/activate
export PYOPENGL_PLATFORM=osmesa MESA_GL_VERSION_OVERRIDE=3.3
unset DISPLAY

# 랜덤 rollout 검증
bash bench/calvin/run.sh
```

## 참고

- `hydra-core==1.1.1` — `version_base` 파라미터 사용 금지
- CPU 렌더링 텍스처 ≠ EGL (pretrained 평가 시 domain gap 주의)
- 결과: `result/calvin/`
