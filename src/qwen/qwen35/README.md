# Qwen3.5-0.8B — JAX 구현

Qwen3.5-0.8B의 JAX 포팅. GDN (Gated Delta Net) linear attention + full attention 혼합 아키텍처.

## 파일

| 파일 | 역할 |
|------|------|
| `modeling.py` | 전체 모델 구현 |
| `gated_delta_net.py` | GDN (Gated Delta Net) linear attention 블록 |
| `params.py` | safetensors → JAX parameter 변환 |

## 가중치

```bash
export HF_TOKEN=<token>
huggingface-cli download Qwen/Qwen3.5-0.8B --include "*.safetensors" --local-dir ../models/qwen35-0.8b
```

환경변수: `QWEN35_MODEL_PATH` (default: `../models/qwen35-0.8b`)
