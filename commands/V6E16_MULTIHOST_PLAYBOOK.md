# v6e-16 MULTI-HOST launch playbook — Fast-dVLM 1-epoch (episode-packing + kd_fewstep)

**Runs on ANY multi-host TPU pod** — the trainer reads `jax.process_count()/local_device_count()/
device_count()` dynamically (no hardcoded topology), so v6e-16 (4 hosts×4) and **v5p-16 (2 hosts; ~95GB
HBM/chip → roomier, less OOM)** both work with the SAME training flags. Requires `--multihost`
(`jax.distributed.initialize()` + per-process file shard + global-array feeding + process-0 IO).
The `--batch-size` must be a multiple of `jax.device_count()` (the pre-flight confirms the chip count;
32 is safe for 8 or 16 chips). Provision/zone/account differ per pod — set them to YOURS.

- **Account / Zone are per-pod — set them to your CURRENT pod** (they have changed run to run:
  e.g. `mkkang0412@gmail` / `asia-south1-c`). `--accelerator-type=v6e-16`, software `v2-alpha-tpuv6e`, **SPOT VM**.
- The trainer code (`--multihost`) is on branch `aw-blockdiffusion-eval-repro`.
- **Start weights = CONTINUE from `boltzmann-final`** (user choice). It is overfit/collapsed on the live
  emulator (known); the broad packed-hybrid + kd_fewstep is meant to correct it. **No steerable/selector
  module is added** — this is pure block-diffusion SFT (ce_noisy + ce_clean + kd_noisy + kd_fewstep only).

> ⚠️ **COST RULE (hard):** do nothing that adds TPU cost. One pod only, no parallel/extra pods, no
> redundant runs. Smoke = 20 steps (≤2 min). **Delete the pod the moment training/checkpoint finishes.**
> Spot preemption is expected — the resume flow (below) re-does only the *lost* steps, not the whole epoch.

## Topology cheat-sheet (v6e-16 = 4×4)
`jax.process_count()=4`, `jax.process_index()∈{0,1,2,3}`, `jax.local_device_count()=4`, `jax.device_count()=16`.
`per_process_batch = global_batch // 4`. With `--batch-size 32` → per_process 8 → 2/chip.
(Code reads these dynamically — works regardless of the host split.)

---

## 0. Provision (you run this; spot, the existing account/zone)

```bash
# v6e-16:  ZONE=asia-northeast1-b  ACC=v6e-16  VER=v2-alpha-tpuv6e
# v5p-16:  ZONE=europe-west4-b     ACC=v5p-16  VER=v2-alpha-tpuv5      <-- current pod (aweasel134, ses040515)
ZONE=europe-west4-b ; ACC=v5p-16 ; VER=v2-alpha-tpuv5
POD=aweasel134       # already created; skip create if it exists
gcloud compute tpus tpu-vm create $POD --zone=$ZONE --accelerator-type=$ACC --version=$VER --spot
gcloud compute tpus tpu-vm describe $POD --zone=$ZONE --format='value(state)'   # READY?
```
Auth as the pod's account first: `gcloud auth login` (e.g. ses040515@gmail.com) — the VM service account
lacks TPU scopes. Meter starts at READY → everything below is scripted to minimize wall-clock.

## 1. One-shot setup on ALL 4 workers (parallel)

```bash
gcloud compute tpus tpu-vm ssh $POD --zone=$ZONE --worker=all --command='
set -e
export HF_TOKEN='"$HF_TOKEN"'
cd ~ && [ -d Weasel_toy_experiment ] || git clone https://github.com/MK040412/Weasel_toy_experiment.git
cd Weasel_toy_experiment && git fetch origin && git checkout aw-blockdiffusion-eval-repro && git pull
( uv sync ) &
# CONTINUE from boltzmann-final (user choice). Pull the base for a COMPLETE processor, then overlay the
# trained weights so --model-dir has both a working AutoProcessor AND the final block-diffusion weights.
( huggingface-cli download mPLUG/GUI-Owl-1.5-2B-Instruct --local-dir ~/models/boltzmann-final >/tmp/dl_base.log 2>&1 &&
  huggingface-cli download KMK040412/fast-dvlm-guiowl-kd-tpu \
    --include "fast-dvlm-kd-tpu/aw-overfit-boltzmann/final/model.safetensors" "fast-dvlm-kd-tpu/aw-overfit-boltzmann/final/config.json" \
    --local-dir /tmp/bolt >/tmp/dl_ckpt.log 2>&1 &&
  cp /tmp/bolt/fast-dvlm-kd-tpu/aw-overfit-boltzmann/final/model.safetensors ~/models/boltzmann-final/model.safetensors &&
  rm -f ~/models/boltzmann-final/model-*-of-*.safetensors ~/models/boltzmann-final/model.safetensors.index.json
) &
( huggingface-cli download KMK040412/guiowl-aw-mix-hybrid-packed --repo-type dataset --local-dir ~/data/aw_mix_hybrid_packed >/tmp/dl_data.log 2>&1 ) &
wait; echo "SETUP_DONE $(hostname)"
'
```
> Every worker needs its OWN local copy of model + dataset (each process then reads only its file shard).
> **All 4 workers MUST end up with byte-identical `~/models/boltzmann-final/model.safetensors`** — the
> trainer asserts a cross-host weight checksum at startup and aborts if they differ.
> If the base ships sharded weights, the `rm` line drops the stale shards/index so only the overlaid
> single-file `model.safetensors` (the final weights) is loaded. (Verify AutoProcessor loads; if it
> errors, the base's `preprocessor_config.json`/tokenizer are already in the same dir.)

## 2. Pre-flight (cheap; verify 16 chips form BEFORE any compile)

```bash
gcloud compute tpus tpu-vm ssh $POD --zone=$ZONE --worker=all --command='
cd ~/Weasel_toy_experiment && export PYTHONPATH=$PWD/src PJRT_DEVICE=TPU
python -c "import jax; jax.distributed.initialize(); print(\"proc\",jax.process_index(),\"/\",jax.process_count(),\"global\",jax.device_count())"
'
# EXPECT: each of 4 workers prints proc i/4, global 16. If global!=16 → stop; topology/runtime wrong.
```

## 3. SMOKE GATE (≤20 steps, ~1–2 min) — fail-fast + measure step-time

```bash
gcloud compute tpus tpu-vm ssh $POD --zone=$ZONE --worker=all --command='
cd ~/Weasel_toy_experiment && export PYTHONPATH=$PWD/src PJRT_DEVICE=TPU PYTHONUNBUFFERED=1 JAX_COMPILATION_CACHE_DIR=~/jax_ccache
uv run python scripts/train_fastdvlm_tpu.py --multihost --data-parallel \
  --model-dir ~/models/boltzmann-final --data ~/data/aw_mix_hybrid_packed --data-pattern "packed-*.parquet" \
  --out ~/runs/v6e16_smoke --data-mode episode --max-turns 12 \
  --batch-size 32 --max-steps 20 --epochs 100 --samples-per-window 64 \
  --bd-curriculum degree2 --bd-values "1,2,4,8,16,32" --bd-lambda1 1.0 --bd-lambda2 0.3 --bd-lambda1-end -0.5 --bd-anneal-steps 7000 \
  --kd-fewstep-weight 0.25 --kd-fewstep-bd-cap 4.0 --kd-fewstep-warmup-steps 500 \
  --ctx-cap 4096 --pad-to 4096 --noisy-pad-to 1024 --vision-pad-to 1152 \
  --vision-precompute-batch-size 16 --loss-token-cap 256 --dtype bf16 --optim adamw_bf16 --lr 1e-6 \
  --ce-noisy-weight 1.0 --ce-clean-weight 0.75 --kd-noisy-weight 0.25 --kd-temp 2.0 \
  --prefetch-prep --prefetch-windows 2 --log-every 1 --monitor-every 5
'
```
**GO/NO-GO (worker-0 `~/runs/v6e16_smoke/train_log.jsonl`):**
1. `multihost_init`: proc_count=4, global_devices=16, per_process_batch=8.
2. every step: loss/ce_noisy/ce_clean/kd_noisy/kd_fewstep FINITE; `kd_fewstep_lambda` = b16 cap (1.0 @bd≥16 post-warmup).
3. no OOM; steady `compute_sec` after step ~3 (steps 0–2 include compile).
4. all 4 workers progress in lockstep (no hang at step 0 = no collective deadlock).

If OOM → `--batch-size 32→16` (per_process 4, 1/chip) and/or `--pad-to 4096→3072 --noisy-pad-to 768`.

## 4. Extrapolate exact cost, THEN decide (no blind full run)

`S` = median steady `compute_sec`/step. Dataset ≈ **57,669 episodes**.
`steps_per_epoch = ceil(57669 / batch_size)` (32→~1803; 16→~3605).
`python commands/v6e16_cost.py --step-sec S --batch 32 --price <USD_per_chip_hr>` prints wall-time + $.
Proceed to step 5 ONLY if the cost is acceptable.

## 5. Full 1-epoch — SPOT-RESILIENT (frequent HF checkpoints)

Same flags, but `--max-steps <steps_per_epoch>`, real out dir, and **HF upload every 300 steps**
(so a preemption loses ≤300 steps). Run under `nohup`; worker-0 uploads.

```bash
gcloud compute tpus tpu-vm ssh $POD --zone=$ZONE --worker=all --command='
cd ~/Weasel_toy_experiment && export PYTHONPATH=$PWD/src PJRT_DEVICE=TPU PYTHONUNBUFFERED=1 HF_TOKEN='"$HF_TOKEN"' JAX_COMPILATION_CACHE_DIR=~/jax_ccache
nohup uv run python scripts/train_fastdvlm_tpu.py --multihost --data-parallel \
  --model-dir ~/models/boltzmann-final --data ~/data/aw_mix_hybrid_packed --data-pattern "packed-*.parquet" \
  --out ~/runs/v6e16_episode_kdfs_e1 --data-mode episode --max-turns 12 \
  --batch-size 32 --max-steps <STEPS_PER_EPOCH> --epochs 100 --samples-per-window 512 \
  --bd-curriculum degree2 --bd-values "1,2,4,8,16,32" --bd-lambda1 1.0 --bd-lambda2 0.3 --bd-lambda1-end -0.5 --bd-anneal-steps 7000 \
  --kd-fewstep-weight 0.25 --kd-fewstep-bd-cap 4.0 --kd-fewstep-warmup-steps 500 \
  --ctx-cap 4096 --pad-to 4096 --noisy-pad-to 1024 --vision-pad-to 1152 \
  --vision-precompute-batch-size 16 --loss-token-cap 256 --dtype bf16 --optim adamw_bf16 --lr 1e-6 --peak-lr 5e-6 --warmup-steps 100 \
  --ce-noisy-weight 1.0 --ce-clean-weight 0.75 --kd-noisy-weight 0.25 --kd-temp 2.0 \
  --prefetch-prep --prefetch-windows 2 --log-every 20 --monitor-every 120 \
  --hf-upload-repo KMK040412/fast-dvlm-aw-episode-kdfs --hf-upload-prefix v6e16-episode-kdfs-degree2 \
  --hf-upload-every-steps 300 --hf-upload-final --save-final \
  > ~/runs/v6e16_episode_kdfs_e1/stdout.log 2>&1 & echo "LAUNCHED $(hostname)"
'
```
> `--epochs 100` is a re-iteration wrapper (a process that exhausts its shard re-reads, not exits) —
> NOT 100 passes. `--max-steps` is the real bound; all 16 chips stop together (deadlock-free).
> Checkpoints/HF upload on worker-0 only, every 300 steps.

## 6. SPOT PREEMPTION → RESUME (re-do only lost steps; no extra waste)

When the pod is preempted mid-run:
1. Re-acquire the pod (step 0) — same name/zone. Re-run step 1 setup (git/uv/downloads cached if disk survived; spot usually gives a fresh disk → re-download).
2. Find the **last uploaded checkpoint** under HF `KMK040412/fast-dvlm-aw-episode-kdfs/v6e16-episode-kdfs-degree2/` (highest `checkpoint-step*`). Download it.
3. Relaunch step 5 with `--model-dir <that_checkpoint>` (the resumed weights), keep `--max-steps
   <STEPS_PER_EPOCH>` UNCHANGED, and add **`--start-step <last_step>`** (the step the checkpoint was
   saved at). `--start-step` makes the kd_fewstep warmup ramp and the `--max-steps` budget CONTINUE
   instead of restarting from 0. It MUST be identical on all 4 workers (it is pure args → lockstep safe).
4. This re-does ≤300 steps (one upload interval), never the whole epoch.
   ⚠️ Adam moment state is NOT in the HF safetensors → on resume the optimizer moments restart from zero
   (one mildly-noisy step after each resume; acceptable for a short 1-epoch SFT). Full optimizer-state
   checkpointing is a deferred enhancement (noted in the trainer).

## 7. Teardown — STOP THE METER IMMEDIATELY

```bash
gcloud compute tpus tpu-vm delete $POD --zone=$ZONE --quiet
```
Final checkpoint is on HF. Then register in `aw_eval/config.py` + run the bd-sweep (b16/b32 strict-JSON
vs kd_fewstep-OFF baseline). **Do not leave the pod up idle — that is pure wasted cost.**

## Failure → action
| symptom | action |
|---|---|
| pre-flight global≠16 | topology/runtime wrong; recreate pod; check `--accelerator-type`/`--version` |
| hang at step 0 | collective deadlock → confirm ALL 4 workers launched; `--max-steps`>0; kill, report |
| OOM | batch 32→16; pad 4096→3072 / noisy 1024→768 |
| kd_fewstep NaN | check warmup; `--kd-fewstep-weight 0` to isolate |
| preempted | §6 resume from last HF checkpoint |
| one worker exits early | shard exhausted before max-steps → raise `--epochs` (re-iterates) |
