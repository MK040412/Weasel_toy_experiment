# data/ — 데이터셋 다운로드 스크립트

## 구조

```
data/download/fywang/
  calvin-task-ABCD-D-lerobot.py   ABCD→D 대규모 데이터셋 RAM 다운로드
```

## 전략: RAM 다운로드 (`/dev/shm`)

디스크 66 GB, ABCD→D 67 GB → 디스크에 안 들어감.
`/dev/shm` (tmpfs, 201 GB)에 다운로드 후 VLM 캐시만 디스크에 저장.

```bash
# 다운로드 + VLM 캐시 + RAM 정리 (디스크엔 캐시 ~12 GB만)
PYTHONPATH=src python data/download/fywang/calvin-task-ABCD-D-lerobot.py --cache-vlm --cleanup

# 다운로드만 (RAM에 유지, 후속 작업용)
PYTHONPATH=src python data/download/fywang/calvin-task-ABCD-D-lerobot.py

# RAM 정리
PYTHONPATH=src python data/download/fywang/calvin-task-ABCD-D-lerobot.py --cleanup
```

## Debug 데이터셋

`fywang/calvin-debug-lerobot` (37 MB)은 `CalvinDataset()` 생성 시 자동 다운로드 (~/.cache/huggingface/).
별도 다운로드 스크립트 불필요.
