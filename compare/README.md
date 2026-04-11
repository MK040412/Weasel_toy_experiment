# compare/ — 수치 검증 스크립트

HuggingFace 참조 구현 vs JAX 구현의 수치 정확도를 검증.

## 스크립트

| 파일 | 용도 | 비교 대상 |
|------|------|----------|
| `compare_blocks.py` | Transformer block 단위 비교 | HF PyTorch ↔ JAX |
| `compare_e2e.py` | End-to-end 모델 출력 비교 | HF full model ↔ JAX full model |
| `compare_gdn_exact.py` | Gated Delta Net 정확도 | HF GDN ↔ JAX GDN |
| `compare_rtc.py` | RTC ablation | baseline (d=0) ↔ RTC (d=15) |

## 실행

```bash
PYTHONPATH=src python compare/compare_blocks.py
PYTHONPATH=src python compare/compare_e2e.py
PYTHONPATH=src python compare/compare_gdn_exact.py
PYTHONPATH=src python compare/compare_rtc.py
```

모든 스크립트는 max absolute error, relative error를 출력하며, 허용 범위 내이면 PASS.
