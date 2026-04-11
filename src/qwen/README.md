# src/qwen/ — 소스 패키지

`from qwen.*` 으로 import. `PYTHONPATH=src` 필요.

## 모듈 구조

```
qwen/
  qwen3vl/          Qwen3-VL 2B (vision + text, 28L, GQA) [JAX]
  qwen35/           Qwen3.5-0.8B (GDN linear attn + full attn) [JAX]
  vla/              VLA pipeline (models, training, data, inference) [JAX/Flax NNX]
  inference.py      통합 추론 (--model qwen3vl | qwen35)
  train.py          JAX 학습 벤치마크 (single/multi-device)
```

## 의존 관계

```
vla/models/vla.py  →  qwen3vl/modeling.py   (VLM encoder, frozen)
vla/training/      →  vla/models/           (action expert forward/backward)
vla/data/          →  (독립, PyArrow only)
inference.py       →  qwen3vl/, qwen35/
train.py           →  qwen3vl/, qwen35/
```

## 빠른 실행

```bash
# VLM 추론
python src/qwen/inference.py --model qwen3vl
python src/qwen/inference.py --model qwen35

# 학습 벤치마크
python src/qwen/train.py --mode both

# VLA (PYTHONPATH 필요)
PYTHONPATH=src python src/qwen/vla/train.py
```
