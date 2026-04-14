# AntMaze Paired Dataset Generation

OGBench AntMaze navigation 데이터셋에서 MuJoCo를 재실행해 **픽셀 관측 (topdown + 3rd-person)** 을 추가 렌더링하고, VLA/VLM 학습용 `(state, image, goal)` 페어 데이터셋을 생성하는 파이프라인.

## 개요

- 원본: OGBench `antmaze-{medium,large,giant}-navigate-v0.npz` (상태 기반 1M 스텝)
- 산출: `frames.zarr` (이미지 포함) + `samples.parquet` (hindsight 재라벨링 샘플) + `meta.json`
- 미로 3종 × 1000 에피소드 × 1001 스텝 × 2 카메라 × 224×224×3 uint8

## 폴더 구조

```
dataset_gen/
├── README.md                      ← 이 파일
├── spec/
│   ├── schema.md                  ← 출력 zarr 구조 상세
│   ├── camera_params.md           ← 탑다운 카메라 파라미터 + 좌표 변환
│   └── episode_stats.md           ← 실측 크기/스루풋
├── scripts/
│   ├── make_paired_dataset_fast.py    ← 병렬 렌더러 (128 worker 스케일)
│   └── validate_antmaze_dataset.py    ← 검증 + mp4/히트맵 출력
└── run/
    ├── render_medium.sh
    ├── render_large.sh
    └── render_giant.sh
```

## Quick Start (단일 VM)

```bash
# 1. 환경
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=1

# 2. 렌더 (128 worker)
python dataset_gen/scripts/make_paired_dataset_fast.py \
    --maze medium \
    --output_dir /mnt/disks/data/antmaze \
    --n_workers 128

# 3. 검증
PYTHONPATH=src python dataset_gen/scripts/validate_antmaze_dataset.py \
    --zarr_dir /mnt/disks/data/antmaze/medium \
    --output_dir /mnt/disks/data/antmaze/medium/validation \
    --maze medium \
    --n_verify_eps 3
```

## 3-VM 병렬 배포

| pod | 미로 | 실행 스크립트 |
|---|---|---|
| pod-0 | medium | `bash dataset_gen/run/render_medium.sh` |
| pod-1 | large  | `bash dataset_gen/run/render_large.sh` |
| pod-2 | giant  | `bash dataset_gen/run/render_giant.sh` |

각 스크립트는 내부에서 128 worker로 렌더 + 검증까지 수행. 완료 후 GCS 업로드는 별도.

## 실측 성능 (2026-04-14, AMD EPYC 9B14 × 128 vCPU)

- 스루풋: **~463 frames/s** (단일 워커 warm-up 포함)
- 미로당: **~72 분** (medium), giant는 topdown 카메라 거리가 커서 ~80분 예상
- 산출물: **~35 GB / 미로**, 전체 ~104 GB

상세는 [`spec/episode_stats.md`](spec/episode_stats.md) 참조.

## 중요: 출력 경로 규칙

**`/dev/shm`, `/tmp`, tmpfs 마운트에 절대 쓰지 말 것.**

과거에 tmpfs에 저장 후 디스크 원본을 지웠다가 재부팅으로 데이터셋 전체를 잃은 사고가 있음. 반드시:

- 영속 디스크 (`/mnt/disks/data` 등) 또는
- 렌더 직후 GCS 업로드 (`gsutil -m rsync`)

`--output_dir`가 tmpfs인지 확인:
```bash
readlink -f $OUTPUT_DIR    # /dev/shm이 나오면 중단
df -T $OUTPUT_DIR          # Type이 tmpfs면 중단
```

## 참고

- OGBench: https://github.com/seohongpark/ogbench
- MuJoCo renderer: osmesa 백엔드 (headless VM에서 GPU 불필요)
- zarr v2 DirectoryStore + Zstd level 1 (압축률 > 속도 우선)
