# result/ — 실험 결과물

학습/추론/벤치마크 출력 저장. `.gitignore`에서 대부분 제외됨.

## 구조

```
result/
  vla/                         VLA 실험 결과
    vlm_cache/                   VLM embedding 캐시
      embeddings.parquet           obs + actions + gripper (binary columns)
      meta.json                    shape metadata
    checkpoint_train_final.npz   action expert weights + quantile 정보
    debug_with_RTC.mp4           RTC 학습 후 시각화
    debug_overfit_RTC.mp4        RTC 과적합 3D 궤적 비교
    debug.mp4                    baseline 시각화

  calvin/                      CALVIN 벤치마크
    debug.mp4                    rollout video

  ogbench/                     OGBench 벤치마크
    <run_name>/
      train.csv                  학습 metrics
      eval.csv                   평가 metrics
      flags.json                 실험 설정
```

## VLM 캐시

`vla/vlm_cache/`가 존재하면 학습 시 VLM 모델 로드를 자동 스킵.
삭제하면 다음 실행 시 VLM forward를 다시 수행하여 재생성.
