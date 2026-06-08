# DECOUPLED checkpointing for the multihost Fast-dVLM run (READ THIS before touching checkpoint code)

> If you are a Claude/engineer seeing this run for the first time and need to save or recover a checkpoint,
> this is the canonical procedure. It exists because the obvious in-process save **does not work** here and
> cost three ~14-min recompile restarts before we got it right.

## The problem (why a naive save fails)

- The run is **multihost** (v6e-16 = 4 hosts x 4 chips = 16 devices, `dp` mesh) with **ZeRO-1**.
- The model's **vocab embedding param ends up dp-SHARDED across all 16 chips** (151936 / 16 = 9496 rows/chip).
  (Params start replicated at load, but the ZeRO-1 optimizer update drifts the embedding to sharded.)
- The HF checkpoint upload runs on the **primary host only** (`jax.process_index()==0`, which on weasel16 is
  **gcloud worker 1**, NOT worker 0). That host alone holds only 4/16 of the embedding.
- So: `np.asarray(jax.device_get(x))` raises *"Fetching value for jax.Array that spans non-addressable
  devices"*; `x.addressable_shards[0].data` is only a partial shard; and a collective `process_allgather`
  called on the primary **alone deadlocks the whole pod**. Every in-process save AND the final upload fail
  → the run produces **zero** model. (Worker root disk is also ~97% full → even a correct save to `~/runs`
  ENOSPCs; we save to `/dev/shm`, a 355G RAM tmpfs.)

## The fix: decouple the save into a DUMB in-process part + a FIXABLE external part

**Part 1 — in-process, deliberately dumb (`dump_local_shards` in `scripts/train_fastdvlm_tpu.py`):**
Every host dumps ONLY its own local addressable shards — pure local reads, **NO collective, NO gather** →
cannot deadlock, cannot fail on sharded params. Wrapped in try/except so a save error can **never kill
training**. Writes `/dev/shm/v6e16_guiowlvit/shards-step{STEP:06d}.proc{P}.pkl` on each host (P = process
index). Fires every `--hf-upload-every-steps` steps and once at the end. The in-loop HF upload is removed.

**Part 2 — external, fixable WITHOUT restarting training (`scripts/stitch_and_ship_checkpoint.py`):**
Collect the per-host pkls, reassemble the full weights (place each shard at its global index — replicated
leaves cover the full slice, sharded leaves are disjoint row-blocks), map to HF safetensors, ship to HF/Vultr.
**If this has a bug, fix it and re-run on the already-dumped shards — training keeps running, no recompile.**

### Why this is correct (verified)
- Per-shard `index` is the shard's GLOBAL slice; `full[index] = shard.data` over all shards across all hosts
  reconstructs the original bit-identically. CPU-verified for both replicated and dp-sharded arrays
  (`/tmp/stitch_test.py`: 8 forced devices, sharded & replicated both reassemble == original).
- Part 1 does only local device reads → no collective → impossible to deadlock; try/except → cannot kill the run.

## How to produce + ship a checkpoint (copy-paste)

```bash
Z=asia-northeast1-b; P=mobile-computing-new; STEP=843
# 1) gather every host's shard pkl onto the primary (w1) via fast internal rsync, OR scp all to one box.
#    each host wrote /dev/shm/v6e16_guiowlvit/shards-step000843.proc{0,1,2,3}.pkl (one per host).
for w in 0 2 3; do gcloud compute tpus tpu-vm ssh weasel16 --zone $Z --project $P --worker $w \
  --command 'rsync -a /dev/shm/v6e16_guiowlvit/shards-step'"$STEP"'*.pkl -e "ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no" perelman@<w1-internal-ip>:/dev/shm/v6e16_guiowlvit/'; done
# 2) stitch + ship from a box that can reach HF (the primary w1 has the venv; or pull all to Vultr which
#    can do HF Xet). Run the EXTERNAL stitcher (only needs numpy + safetensors + huggingface_hub):
python scripts/stitch_and_ship_checkpoint.py --shards-dir /dev/shm/v6e16_guiowlvit --step $STEP \
  --source-model-dir ~/models/boltzmann-final --out-dir /dev/shm/ckpt-step$STEP \
  --hf-repo KMK040412/fastdvlm-aw-guiowlvit --hf-path-prefix fast-dvlm-kd-tpu
# (workers had HF Xet *download* trouble; if upload also stalls, rsync the produced safetensors w1->Vultr
#  and run the same upload from Vultr.)
```

## Recovery / resume after preemption or crash
- The shipped HF checkpoint (`KMK040412/fastdvlm-aw-guiowlvit/fast-dvlm-kd-tpu/checkpoint-stepN`) is the
  durable model. To resume training from it: relaunch `commands/launch_fastdvlm_v6e16.sh` with
  `--model-dir <ckpt>` + `--start-step N`. On a SAME pod, keep `~/jax_ccache` (do NOT wipe, do NOT copy
  across hosts) so the train_step is a cache-hit (no recompile). On a NEW pod, a recompile (~14 min) is
  unavoidable (cache is on the dead pod).

## Hard rules (each learned the expensive way)
1. NEVER do a single-host device_get/allgather of a possibly-sharded param. Dump local shards, stitch externally.
2. Keep the in-process save DUMB so a save bug is fixed externally, never by restarting training.
3. Save to `/dev/shm` (worker root disk is ~97% full).
4. The primary is `jax.process_index()==0` = gcloud **worker 1**, not worker 0; upload events log to
   `<out>/train_log.jsonl`, not stdout.
5. Do NOT wipe `~/jax_ccache` on a same-pod resume; do NOT copy it across hosts (deadlocks).
