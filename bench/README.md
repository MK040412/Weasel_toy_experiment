# bench/ — 벤치마크 래퍼

외부 환경(OGBench, CALVIN)과 내부 VLA를 래핑하는 실행 스크립트.

## 구조

```
bench/
  ogbench/     Offline GCRL 벤치마크 (seohongpark/ogbench 래핑)
  calvin/      Robot manipulation 시뮬레이터 (mees/calvin 래핑)
  vla/         VLA 학습/추론/ablation 래퍼
```

## 빠른 실행

```bash
# VLA
bash bench/vla/run.sh train
bash bench/vla/run.sh train --simulated-delay 15

# OGBench (별도 설치 필요)
bash bench/ogbench/run.sh

# CALVIN (별도 설치 필요)
bash bench/calvin/run.sh
```

각 벤치마크 디렉토리의 README.md 참조.
