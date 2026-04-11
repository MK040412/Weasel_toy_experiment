# bench/vla/ — VLA 벤치마크 래퍼

VLA 학습/추론/RTC ablation을 `run.sh`로 래핑.

## 사용법

```bash
# 학습 (baseline)
bash bench/vla/run.sh train

# 학습 (RTC)
bash bench/vla/run.sh train --simulated-delay 15

# 추론
bash bench/vla/run.sh inference result/vla/checkpoint_train_final.npz

# RTC ablation (baseline vs RTC 비교)
bash bench/vla/run.sh compare result/vla/ckpt_baseline.npz result/vla/ckpt_rtc.npz
```

## 결과

`result/vla/` 에 저장:
- `checkpoint_*.npz` — 모델 가중치
- `debug*.mp4` — 시각화 영상
- `vlm_cache/` — VLM embedding parquet 캐시
