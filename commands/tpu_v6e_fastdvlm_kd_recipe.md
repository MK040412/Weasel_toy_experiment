# TPU v6e Fast-dVLM KD Training Recipe

Date: 2026-06-05

This note records the current practical TPU v6e-4 recipe for Fast-dVLM / GUI-Owl
2B continuation with noisy-branch KD.

## Recommended Setting

Use TPU v6e-4 with all 4 chips and data-parallel batch 32:

```bash
cd ~/Weasel_toy_experiment

QWEN_TPU_DPA_ATTENTION=1 .venv/bin/python scripts/train_fastdvlm_tpu.py \
  --model-dir ~/models/ckpt-bard-bd32-gmail-adb-vitlora-e1-final \
  --data ~/data/aitw_general/standard/train-00000-of-00256.parquet \
  --out ~/tpu_fastdvlm_runs/kd_gmail_general_v6e_bs32_pad512 \
  --max-samples 0 \
  --epochs 1 \
  --batch-size 32 \
  --data-parallel \
  --bd 32 \
  --bd-schedule "4:0.25,8:0.25,16:0.25,32:0.25" \
  --ctx-cap 512 \
  --pad-to 512 \
  --noisy-pad-to 512 \
  --vision-pad-to 84 \
  --loss-token-cap 128 \
  --dtype bf16 \
  --optim adamw_bf16 \
  --lr 1e-6 \
  --ce-noisy-weight 1.0 \
  --ce-clean-weight 0.75 \
  --kd-noisy-weight 0.25 \
  --kd-temp 2.0 \
  --prefetch-prep \
  --log-every 20 \
  --monitor-every 60 \
  --save-final
```

## Recommended Stable Streaming Run

The TPU VM has large RAM, but the safest long-run path is still a bounded
RAM-resident parquet window. This keeps enough samples in memory for stable
throughput without requiring a full-dataset vision cache.

Set `HF_TOKEN` before running if Hugging Face upload is required. Do not put the
token directly in the command line.

```bash
cd ~/Weasel_toy_experiment

export HF_TOKEN=...  # set securely in the shell/session

QWEN_TPU_DPA_ATTENTION=1 .venv/bin/python scripts/train_fastdvlm_tpu.py \
  --model-dir ~/models/ckpt-bard-bd32-gmail-adb-vitlora-e1-final \
  --data ~/data/aitw_general/standard \
  --data-pattern "train-*.parquet" \
  --out ~/tpu_fastdvlm_runs/kd_gmail_general_v6e_bs32_pad512_streaming \
  --max-samples 0 \
  --samples-per-window 8192 \
  --epochs 1 \
  --batch-size 32 \
  --data-parallel \
  --bd 32 \
  --bd-schedule "4:0.25,8:0.25,16:0.25,32:0.25" \
  --ctx-cap 512 \
  --pad-to 512 \
  --noisy-pad-to 512 \
  --vision-pad-to 84 \
  --loss-token-cap 128 \
  --dtype bf16 \
  --optim adamw_bf16 \
  --lr 1e-6 \
  --ce-noisy-weight 1.0 \
  --ce-clean-weight 0.75 \
  --kd-noisy-weight 0.25 \
  --kd-temp 2.0 \
  --prefetch-prep \
  --log-every 20 \
  --monitor-every 60 \
  --hf-upload-repo YOUR_HF_MODEL_REPO \
  --hf-upload-prefix fast-dvlm-kd-tpu/gmail-general \
  --hf-upload-every-steps 3000 \
  --hf-upload-final \
  --delete-local-uploaded-checkpoints
```

Important:

- `kd_noisy_weight=0.25` means noisy-branch KD is active.
- The teacher is the same model's clean/AR branch with stop-gradient.
- `train_log.jsonl` contains `loss`, `ce_noisy`, `ce_clean`, `kd_noisy`,
  throughput, host overhead, window index, and KD weights every logged step.
- `data_windows.jsonl` records each RAM-resident parquet window, row skips,
  sequence lengths, RAM snapshot, and window preparation time.
- Every 3000 steps, the script exports an HF-style `model.safetensors` bundle
  plus current logs and uploads it to the requested HF repo.

For short ETA probes, set `--max-samples 256 --max-steps 12 --log-every 1`.

## Measured Throughput

All measurements below use the same real AiTW shard and exact mRoPE + DeepStack.

| setting | global batch | dual length | mean compute sec/step | mean wall sec/step | wall rows/sec | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| DPA, pad768 | 32 | 1536 | 1.288 | n/a | n/a | previous baseline |
| DPA, pad512 | 32 | 1024 | 0.837 | 0.933 | 34.29 | host overhead 10.4% |
| DPA, pad512, prefetch | 32 | 1024 | 0.840 | 0.881 | 36.32 | host overhead 4.7% |

The important win is reducing `pad-to/noisy-pad-to` from 768 to 512. It improves
sample throughput by roughly 1.5x versus the pad768 compute baseline. CPU prep
prefetch gives an additional roughly 6% wall-time improvement by overlapping
mask/noising preparation with the TPU step.

## Why Not Increase Batch Size

Global batch 64, equivalent to per-chip batch 16, failed on v6e-4:

```text
RESOURCE_EXHAUSTED: Attempting to reserve 9.37G ... There are 7.43G free
```

Global batch 48 was stopped. Even if it compiles, the expected gain is small
relative to compile time and memory risk. The practical stable point is:

```text
global batch 32 = per-chip batch 8
```

## Current Bottlenecks

1. TPU compute is now the main steady-state bottleneck.
2. Host prep wait is mostly hidden by `--prefetch-prep`.
3. Device put remains around 0.034s/step and is now most of the non-compute overhead.
4. Vision embeddings are still precomputed sample-by-sample before training; for full
   dataset training, this startup/preprocessing path needs a separate streaming/cache
   strategy.

## Attention Backends

`QWEN_TPU_DPA_ATTENTION=1` is the current recommended backend. It uses
`jax.nn.dot_product_attention(..., implementation="xla")`.

An optional Splash Attention path exists behind:

```bash
QWEN_TPU_SPLASH_ATTENTION=1
```

Synthetic single-device/vmap tests work, but full data-parallel training currently
fails because Mosaic/Pallas kernels cannot be automatically partitioned under the
current pjit-style data-parallel path:

```text
NotImplementedError: Mosaic kernels cannot be automatically partitioned.
Please wrap the call in a shard_map.
```

Therefore Splash is not the short-term training path. The next implementation step
would be an explicit `shard_map` wrapper for Splash attention, or keeping Splash for
single-device inference/eval only while DPA remains the multi-chip training backend.

## Current Training Objective

The trainer uses the same-model clean branch as a stop-gradient teacher:

```text
loss = ce_noisy_weight * CE(noisy masked response tokens)
     + ce_clean_weight * CE(clean AR response tokens)
     + kd_noisy_weight * KL(noisy student || clean teacher)
```

Current measured/probed values:

```text
ce_noisy_weight = 1.0
ce_clean_weight = 0.75
kd_noisy_weight = 0.25
kd_temp = 2.0
bd_schedule = 4/8/16/32 uniformly mixed
```

This targets the known failure mode: AR/clean branch works, but large-block dVLM
strict JSON and action-type sampling collapse. KD is applied to the noisy/diffusion
branch rather than only the clean branch.
