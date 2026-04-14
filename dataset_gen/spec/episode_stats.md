# Episode Stats & Benchmark

## 원본 OGBench 데이터셋

| maze | 파일 | 스텝 | 에피소드 | 평균 길이 | npz 크기 |
|---|---|---|---|---|---|
| medium | `antmaze-medium-navigate-v0.npz` | 1,001,000 | 1,000 | 1,001 | 241 MB |
| large  | `antmaze-large-navigate-v0.npz`  | 1,001,000 | 1,000 | 1,001 | 242 MB |
| giant  | `antmaze-giant-navigate-v0.npz`  | 1,000,500 | 1,000 | 1,000~1,001 | 241 MB |
| val    | `*-val.npz` (각 미로당)           | ~100,100 | 100 | 1,001 | ~24 MB |

- 모든 에피소드 길이 고정 (≈1001 스텝)
- `terminals` 배열에서 에피소드 경계 추출 (`starts[i]`, `ends[i]`)
- 원본 스키마: `observations (N,29)`, `actions (N,8)`, `qpos (N,15)`, `qvel (N,14)`, `terminals (N,)`

## 렌더링 실측 (128 worker)

**환경**: AMD EPYC 9B14, 180 vCPU, 742 GB RAM, Ubuntu, MuJoCo osmesa, `OMP_NUM_THREADS=1`

벤치마크 (medium, 128 에피소드를 128 워커에 1개씩 분배, warm-up 포함):

| 항목 | 값 |
|---|---|
| 에피소드 | 128 |
| 프레임 | 256,256 (1001 × 2 cam × 128 ep) |
| Wall time | **553.6 s (9.23 min)** |
| 스루풋 | **463 frames/s** |
| eps/min | **13.9** |
| 출력 | 4.4 GB |

## 전체 생성 예상 시간

워커당 ~8 에피소드를 처리하면 warm-up 비중이 줄어 실측보다 빠름. 보수 추정:

| maze | 에피소드 | 예상 시간 (128 worker) | 출력 크기 |
|---|---|---|---|
| medium | 1,000 | **60~72 분** | ~35 GB |
| large  | 1,000 | 60~72 분 | ~35 GB |
| giant  | 1,000 | 65~80 분 (카메라 거리 70) | ~35 GB |
| **합계** | 3,000 | **3.0~3.8 시간** | ~105 GB |

3-VM 병렬로 돌리면 **wall-clock 60~80분에 완료**.

## Hindsight 샘플 수

각 에피소드에서 `(ep_len - chunk_size)`개 샘플 → `chunk_size=16` 기준:

| maze | 샘플 수 | parquet 크기 (대략) |
|---|---|---|
| medium | ~985,000 | ~80 MB |
| large  | ~985,000 | ~80 MB |
| giant  | ~984,500 | ~80 MB |

## 디스크 여유 체크

각 pod에서 **최소 45 GB** 여유 필요. 시작 전:
```bash
df -h $OUTPUT_DIR
```

VM 루트 디스크가 100 GB 미만이면 별도 데이터 디스크 (`gcloud compute disks create`) 마운트 권장.
