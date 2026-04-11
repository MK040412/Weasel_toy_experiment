# Qwen3-VL 2B — JAX 구현

Qwen3-VL-2B-Instruct의 JAX/Flax NNX 포팅. jax-ml/bonsai 기반, JAX 0.6.2 호환.

## 모델 사양

| 항목 | 값 |
|------|---|
| Vision Encoder | 24L, 1024H, 16 heads |
| Text Decoder | 28L, 2048H, 16 heads / 8 kv-heads (GQA) |
| Total params | ~2B |
| Input | images + text tokens |
| Output | hidden states (B, seq, 2048) |

## 파일

| 파일 | 역할 |
|------|------|
| `modeling.py` | 전체 모델: `Qwen3VLForConditionalGeneration`, config, vision/text blocks |
| `params.py` | safetensors → JAX parameter 변환 |

## 사용

```python
from qwen.qwen3vl import modeling as qwen3vl

config = qwen3vl.ModelConfig.qwen3vl_2b()
model = qwen3vl.Qwen3VLForConditionalGeneration.from_pretrained(
    "/path/to/qwen3-vl-2b", config=config
)

# Hidden states 추출 (VLA에서 사용)
hidden = model.get_hidden_states(input_ids, pixel_values, image_grid_thw, token_type_ids)
# → (B, seq, 2048)
```

## 가중치

```bash
export HF_TOKEN=<token>
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --include "*.safetensors" --local-dir ../models/qwen3-vl-2b
```

환경변수: `QWEN3VL_MODEL_PATH` (default: `../models/qwen3-vl-2b`)
