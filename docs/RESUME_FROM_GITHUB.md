# RESUME FROM GITHUB — continue Fast-dVLM after the action SFT finishes

> **You are a fresh Claude / engineer on a new machine (account `mkkang0412@gmail.com`).** The ACTION
> continued-SFT run is done (or nearly done) and its checkpoints are on HF. Your job is the NEXT two
> phases: **(A) the REASONING-SFT** (`<think>…</think>` then act) and **(B) the AndroidWorld benchmark
> eval** to EVAL-SELECT the best reasoning checkpoint. Follow this file top to bottom; it is the wiring
> diagram. The deep *why* lives in [`docs/METHODOLOGY_AND_DECISIONS.md`](METHODOLOGY_AND_DECISIONS.md);
> dataset facts in [`docs/DATASETS.md`](DATASETS.md); the exact reasoning recipe in
> [`commands/REASONING_SFT_RECIPE.md`](../commands/REASONING_SFT_RECIPE.md) +
> [`commands/launch_fastdvlm_reasoning.sh`](../commands/launch_fastdvlm_reasoning.sh); checkpointing in
> [`commands/CHECKPOINT_DECOUPLED.md`](../commands/CHECKPOINT_DECOUPLED.md); the action-run engineering
> baseline in [`commands/tpu_v6e16_fastdvlm_zero1_recipe.md`](../commands/tpu_v6e16_fastdvlm_zero1_recipe.md).
> **Do not duplicate those — cross-reference them.**

**Project in one line:** convert GUI-Owl-1.5-2B (Qwen3-VL-2B arch) from autoregressive to a
**block-diffusion dVLM** via continued-SFT, vision tower FROZEN/precomputed, target = maximize
**AndroidWorld** (GUI agent). Trained on a **v6e-16 spot TPU** (4 hosts × 4 chips), multihost
data-parallel, ZeRO-1 (`--shard-opt-state`).

**Secrets rule (every step):** all tokens live in `~/.fastdvlm_secrets.env` on each TPU worker
(exports `HF_TOKEN`, `GH_TOKEN`). `source` it and reference `$HF_TOKEN` by name — **never print or
hardcode the value**. For any git push use `https://$GH_TOKEN@github.com/...` then reset the remote to
the token-free URL.

---

## Phase map

| Phase | What | Where it runs |
|---|---|---|
| 0 | Prereqs (clone, branch, CPU torch, HF token, jax[tpu]) | every TPU worker |
| 1 | Get + complete the BEST action checkpoint → `--model-dir` | one box w/ HF + every worker |
| 2 | Download reasoning data to every worker | every TPU worker |
| 3 | Launch reasoning SFT (checkpoint every 0.5 epoch; 2-ep CEILING) | per-worker parallel |
| 4 | Decoupled collect + stitch + ship checkpoints | external (primary host / Vultr) |
| 5 | AndroidWorld bd-sweep → EVAL-SELECT best (checkpoint × bd) | aw_eval 3-machine harness |
| 6 | Optional upgrades (mixed-noise, token-revision, bd4-anchor KD) | future |

---

## 0. Prereqs (on EVERY TPU worker — all 4 hosts, identical)

> Same baseline as the action run — see
> [`commands/tpu_v6e16_fastdvlm_zero1_recipe.md`](../commands/tpu_v6e16_fastdvlm_zero1_recipe.md)
> §"Per-worker prerequisites". Condensed here for copy-paste.

```bash
# 0.1 — clone the repo and switch to the reasoning branch
git clone https://github.com/MK040412/Weasel_toy_experiment.git ~/Weasel_toy_experiment
cd ~/Weasel_toy_experiment
git checkout reasoning-sft          # the think-then-act loader patch lives on this branch

# 0.2 — base env (uv + jax[tpu]); uv sync gets the project deps + jax[tpu] from pyproject.toml
~/.local/bin/uv sync               # installs the locked env incl. jax[tpu] (PJRT_DEVICE=TPU)

# 0.3 — CPU torch + torchvision IN THE VENV (REQUIRED, easy to miss).
#   transformers' Qwen3-VL AutoProcessor eagerly instantiates Qwen3VLVideoProcessor, which
#   hard-requires torch+torchvision even though this TPU trainer never processes video. A plain
#   `uv sync` on a fresh TPU VM does NOT install a usable build (default index serves CUDA). Without
#   this the run dies instantly at AutoProcessor.from_pretrained with:
#   "ImportError: Qwen3VLVideoProcessor requires the Torchvision/PyTorch library". CPU-only => no HBM.
~/.local/bin/uv pip install --python ~/Weasel_toy_experiment/.venv/bin/python \
  "torch>=2.7.1" "torchvision>=0.22.1" --index-url https://download.pytorch.org/whl/cpu

# 0.4 — secrets on every worker (HF_TOKEN, GH_TOKEN). Fresh pods LACK this; scp it in, never commit it.
#   (If you only have it on one machine, scp ~/.fastdvlm_secrets.env to each worker's $HOME.)
source ~/.fastdvlm_secrets.env      # exports HF_TOKEN / GH_TOKEN — reference by name, never print

# 0.5 — jax compilation cache (a SAME-host preemption restart becomes a cache hit; ~14 min saved).
export JAX_COMPILATION_CACHE_DIR=$HOME/jax_ccache
mkdir -p $HOME/jax_ccache $HOME/runs
```

**Verify env (each worker):**
```bash
cd ~/Weasel_toy_experiment && ~/.local/bin/uv run --no-sync python - <<'PY'
import jax; print("jax devices:", jax.device_count(), jax.devices()[0].platform)   # expect 4 TPU/host
import torch, torchvision; print("torch", torch.__version__, "tv", torchvision.__version__)  # CPU build
PY
```

Hard rules carried over from the action run: **never wipe `~/jax_ccache` on a same-pod resume, never
copy it across hosts** (deadlocks the first multihost collective). Worker root disk is ~97% full →
checkpoints go to `/dev/shm` (RAM tmpfs), never `~/runs`.

---

## 1. Get the BEST action checkpoint and COMPLETE its processor → set `--model-dir`

The reasoning run's **start checkpoint** is the best ACTION-SFT checkpoint, shipped to HF repo
**`KMK040412/fastdvlm-aw-guiowlvit`** under `fast-dvlm-kd-tpu/checkpoint-step{N}` (~4.89 GB each).
Documented fallback if action SFT is unavailable: `~/models/boltzmann-final`.

### 1.1 — List the shipped checkpoints, pick the best
```bash
source ~/.fastdvlm_secrets.env
HF_HUB_DISABLE_XET=1 python3 - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files("KMK040412/fastdvlm-aw-guiowlvit")
steps = sorted({int(f.split("checkpoint-step")[1].split("/")[0])
                for f in files if "checkpoint-step" in f})
print("shipped action checkpoint steps:", steps)
PY
```
**Which to pick.** The action run's epoch boundaries (= MAJOR checkpoints) are steps **3374 / 6748 /
10122**; intermediate saves land every ~1687 (½ epoch). **If you have AndroidWorld eval-select results
for the action checkpoints, use the winner.** Otherwise default to the **latest fully-shipped step**
(end of training, post bd16 consolidation). Set `STEP` below accordingly.

### 1.2 — Download that checkpoint
```bash
STEP=10122          # <-- set to your chosen step (eval-select winner, else latest)
HF_HUB_DISABLE_XET=1 python3 - "$STEP" <<'PY'
import sys
from huggingface_hub import snapshot_download
step = int(sys.argv[1])
p = snapshot_download(
    repo_id="KMK040412/fastdvlm-aw-guiowlvit",
    allow_patterns=[f"fast-dvlm-kd-tpu/checkpoint-step{step}/*"],
    local_dir="/root/models/action-ckpt-dl",
)
print("downloaded to:", p)
PY
# Canonical model dir:
DEST=~/models/action-sft-step${STEP}
mkdir -p "$DEST"
cp -a ~/models/action-ckpt-dl/fast-dvlm-kd-tpu/checkpoint-step${STEP}/. "$DEST"/
ls "$DEST"
```

### 1.3 — COMPLETE the processor (the missing-files GOTCHA — do this before launching)
Shipped action checkpoints **may LACK the image-processor files**
(`preprocessor_config.json`, `special_tokens_map.json`, `video_preprocessor_config.json`). A reasoning
run **processes images**, so `AutoProcessor.from_pretrained(<model-dir>)` will crash without them. Fix
by copying the missing files from the base **`Qwen/Qwen3-VL-2B-Instruct`** (or from
`~/models/boltzmann-final` if you have it locally).

```bash
DEST=~/models/action-sft-step${STEP}
# Pull the 3 processor files from the base model (config only — tiny):
HF_HUB_DISABLE_XET=1 python3 - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os
DEST = os.path.expanduser(os.environ.get("DEST", "~/models/action-sft"))
for fn in ["preprocessor_config.json", "special_tokens_map.json", "video_preprocessor_config.json"]:
    if os.path.exists(os.path.join(DEST, fn)):
        print("present, skip:", fn); continue
    try:
        src = hf_hub_download("Qwen/Qwen3-VL-2B-Instruct", fn)
        shutil.copy(src, os.path.join(DEST, fn)); print("copied:", fn)
    except Exception as e:
        print("MISSING upstream (ok if not required):", fn, e)
PY
# Verify the processor loads cleanly:
cd ~/Weasel_toy_experiment && DEST=$DEST ~/.local/bin/uv run --no-sync python - <<'PY'
import os
from transformers import AutoProcessor
p = AutoProcessor.from_pretrained(os.path.expanduser(os.environ["DEST"]))
print("OK:", type(p).__name__)   # expect Qwen3VLProcessor
PY
```
Distribute `$DEST` to **every worker** (each host reads the model from its own disk): internal GCP
rsync worker0 → 1/2/3 is fastest. Then set `--model-dir ~/models/action-sft-step${STEP}` in the launch
script (replacing the `best-action-sft-ckpt-PLACEHOLDER`).

---

## 2. Download the REASONING data to EVERY worker

Dataset **`KMK040412/gui-libra-reasoning-phone`** (HF, 8.93 GB, **11 NESTED parquet**:
`aitw/(2) amex/(5) gui_odyssey/(4)`, glob `*/*.parquet`; **23,948 episodes / 41,876 steps**;
`reasoning` CoT column 100% non-empty ~847 chars median; raw JPEG screenshots; coords 0–1000;
phone/portrait). Full facts: [`docs/DATASETS.md`](DATASETS.md) §2.

```bash
source ~/.fastdvlm_secrets.env       # HF_TOKEN (by name)
HF_HUB_DISABLE_XET=1 python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="KMK040412/gui-libra-reasoning-phone",
    repo_type="dataset",
    local_dir="/root/data/gui-libra-reasoning-phone",   # = ~/data/gui-libra-reasoning-phone
    allow_patterns=["*/*.parquet", "README.md", "balance_report.json"],
    max_workers=16,
)
PY
# Verify the NESTED layout (must be 11 files under aitw/ amex/ gui_odyssey/):
find ~/data/gui-libra-reasoning-phone -name '*.parquet' | sort | sed 's#.*/data/##'
find ~/data/gui-libra-reasoning-phone -name '*.parquet' | wc -l   # expect 11
```
**Gotchas (carry these into the launch):**
- **`HF_HUB_DISABLE_XET=1` is mandatory** — Xet-stored large files HANG on TPU workers. If even this
  stalls, rsync the data from a box that completed it (e.g. Vultr) → worker0 → internal rsync to 1/2/3.
- **NESTED layout → `--data-pattern "*/*.parquet"`** (the flat `packed-*.parquet` action pattern matches
  **zero** files here).
- **`--data-mode episode` is MANDATORY** (row mode `KeyError`s on `screenshot`).
- The data must be present on **every host** (each host reads its own slice).

---

## 3. Launch the REASONING SFT

The exact, verified launch is [`commands/launch_fastdvlm_reasoning.sh`](../commands/launch_fastdvlm_reasoning.sh)
(rationale in [`commands/REASONING_SFT_RECIPE.md`](../commands/REASONING_SFT_RECIPE.md) and
[`docs/METHODOLOGY_AND_DECISIONS.md`](METHODOLOGY_AND_DECISIONS.md) §6). Key design decisions you must
NOT change without the recipe's blessing:
- **Distillation OFF** (`--kd-noisy-weight 0 --kd-fewstep-weight 0`): the KD teacher is the model's own
  clean/AR branch, which has **no reasoning**, so it cannot teach CoT and its KL would fight the CoT
  cross-entropy. Reasoning is learned ONLY from the `reasoning` column via CE (ce_noisy + ce_clean).
  This is also **BARD-safe by construction** (no misaligned AR→diffusion KD; see §6 below).
- **Loader patch (already on `reasoning-sft`)** injects `<think>\n{reasoning}\n</think>\n\n{action}`
  into the supervised assistant turn so the CoT gets CE loss. Native Qwen3-VL tokens `<think>`=151667
  `</think>`=151668 `<tool_call>`=151657 `</tool_call>`=151658 (all single). **NEVER `<answer>`** (it
  splits into `[27,9217,29]`).
- **Memory:** batch **16** (1/chip), pair-batch **1**, ctx **4096**, noisy-pad **1536** (CoT lengthens
  the noisy branch; 12.5% of episodes exceed 1536 → drop-oldest-turn handles it, 0% skipped, 0% lose
  CoT; recipe pre-authorizes **2048** if there is OOM headroom). Turning KD off frees NO memory (cost is
  the dual-stream attention buffer, not KD) — do NOT raise batch/pairs to compensate.
- **Curriculum:** degree-2 Gaussian, eval-centered bd16 (`--bd-lambda2 1.04 --bd-lambda1 0.0
  --bd-lambda1-end -5.77 --bd-anneal-steps 1500` = 1 epoch), `--bd-values "1,2,4,8,16,32"`.

### 3.1 — Set the model-dir in the launch script
Edit [`commands/launch_fastdvlm_reasoning.sh`](../commands/launch_fastdvlm_reasoning.sh): replace
`--model-dir ~/models/best-action-sft-ckpt-PLACEHOLDER` with your completed checkpoint from Phase 1,
e.g. `--model-dir ~/models/action-sft-step10122`. (Fallback `~/models/boltzmann-final`.) Confirm
`--noisy-pad-to 1536` (bump to 2048 only if you have headroom).

### 3.2 — Deploy + launch PER-WORKER IN PARALLEL (idempotent guard)
**Never `gcloud ... --worker=all`** — it 255-retry-storms. Stage the script as `~/launch_fastdvlm.sh`
on every worker, then launch each worker with the idempotency guard (so a gcloud retry can't
double-launch). Adjust the TPU name/zone/project to your pod (the action run used
`weasel16` / `asia-northeast1-b` / `mobile-computing-new`):
```bash
# (on each worker, $HOME): cp ~/Weasel_toy_experiment/commands/launch_fastdvlm_reasoning.sh ~/launch_fastdvlm.sh
GUARD='if pgrep -f "[u]v run --no-sync python scripts/train_fastdvlm" >/dev/null; then echo ALREADY; \
  else : > ~/train.log; setsid bash -lc "~/launch_fastdvlm.sh >> ~/train.log 2>&1" </dev/null >/dev/null 2>&1 & echo LAUNCHED; fi'
for w in 0 1 2 3; do gcloud compute tpus tpu-vm ssh <TPU_NAME> --zone <ZONE> \
  --project <PROJECT> --worker $w --command "$GUARD" & done; wait
```
Tail progress: `tail -f ~/train.log` (per-step JSON: loss + ce_noisy/ce_clean + bd + input_len +
tokens/s); checkpoint/upload events log to `<out>/train_log.jsonl`, **not** stdout.

### 3.3 — VERIFY the CoT is supervised (do this once, early)
After the first step, decode the `labels != -100` span and confirm it contains the `<think>` CoT (not
just the action). If it is action-only, the loader patch did not take — re-check you are on branch
`reasoning-sft` and that `_build_episode_messages` injects the `<think>…</think>` content.

### 3.4 — EPOCH DECISION (read carefully — affects what you ship/eval)
**2 epochs is a CEILING, not a proven optimum.** The reasoning set is small AND already-distilled, so
overfitting the CoT templates is likely past ~1–1.5 epochs, while <1 epoch under-trains.
**Recommendation:** the launch already checkpoints every **0.5 epoch** (`--hf-upload-every-steps 750`,
~1497 steps/epoch → saves at ~750/1500/2250/3000) to repo **`KMK040412/fastdvlm-aw-reasoning`**. **EVAL-
SELECT** the best of these on AndroidWorld (Phase 5). **1 epoch (≈step 1500) may already be best — do
not assume 2.** ETA ≈ 2 h (smaller dataset, 2 epochs) + ~14–28 min compile.

---

## 4. Decoupled checkpoint: COLLECT + STITCH + SHIP

> Full canonical procedure (and *why* a naive save deadlocks): **READ**
> [`commands/CHECKPOINT_DECOUPLED.md`](../commands/CHECKPOINT_DECOUPLED.md). Do not invent a new save
> path. Summary only here.

Saving is decoupled because ZeRO-1 leaves the vocab embedding **dp-sharded across all 16 chips**, so a
single-host `device_get`/`allgather` either errors ("spans non-addressable devices") or deadlocks the
pod. The fix:
- **Part 1 (in-process, deliberately DUMB, already in `scripts/train_fastdvlm_tpu.py`):** every host
  dumps ONLY its own local addressable shards (no collective, no gather, wrapped in try/except so it can
  never kill training) to `/dev/shm/<out>/shards-step{STEP:06d}.proc{P}.pkl` (one per host). Fires every
  `--hf-upload-every-steps` and once at the end.
- **Part 2 (EXTERNAL, fixable WITHOUT restarting training):** `scripts/stitch_and_ship_checkpoint.py`
  collects the per-host pkls, reassembles full weights (place each shard at its global slice), writes HF
  safetensors, ships to HF. A bug here is fixed and re-run on the already-dumped shards — **no restart,
  no recompile.** Only deps: numpy + safetensors + huggingface_hub.

**The reasoning run already automates the ship** (`--hf-upload-every-steps 750 --hf-upload-final
--delete-local-uploaded-checkpoints` → repo `KMK040412/fastdvlm-aw-reasoning`). If the auto-ship stalls
(workers had HF Xet upload trouble in the action run), do the manual collect+stitch (note: `--out` for
the reasoning run is `/dev/shm/v6e16_reasoning`, and the **primary host is `jax.process_index()==0` =
gcloud worker 1, NOT worker 0**):
```bash
Z=<ZONE>; P=<PROJECT>; TPU=<TPU_NAME>; STEP=1500; OUT=v6e16_reasoning
# 1) gather every host's shard pkl onto the primary host (w1) via internal rsync:
for w in 0 2 3; do gcloud compute tpus tpu-vm ssh $TPU --zone $Z --project $P --worker $w \
  --command 'rsync -a /dev/shm/'"$OUT"'/shards-step'"$(printf %06d $STEP)"'*.pkl \
    -e "ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no" perelman@<w1-internal-ip>:/dev/shm/'"$OUT"'/'; done
# 2) stitch + ship (run on the primary, or rsync the pkls to Vultr which can do HF Xet upload):
cd ~/Weasel_toy_experiment && source ~/.fastdvlm_secrets.env
~/.local/bin/uv run --no-sync python scripts/stitch_and_ship_checkpoint.py \
  --shards-dir /dev/shm/$OUT --step $STEP \
  --source-model-dir ~/models/action-sft-step10122 \
  --out-dir /dev/shm/reasoning-ckpt-step$STEP \
  --hf-repo KMK040412/fastdvlm-aw-reasoning --hf-path-prefix fast-dvlm-reasoning
```
**Resume after preemption:** the shipped HF checkpoint is the durable model — relaunch with
`--model-dir <that ckpt> --start-step N`. On a SAME pod keep `~/jax_ccache` (cache hit); on a NEW pod a
~14 min recompile is unavoidable.

---

## 5. Benchmark eval — AndroidWorld bd-sweep → EVAL-SELECT the best reasoning checkpoint

The reproducible harness is **`aw_eval/`** (see its self-contained
[`aw_eval/CLAUDE.md`](../aw_eval/CLAUDE.md) and `aw_eval/PLAYBOOK_v6e16.md`). It orchestrates a
3-machine pipeline: **local orchestrator (`aw_eval/bd_sweep.py`) → TPU v6e-4 JAX policy server
(`androidworld_tpu_jax_server.py` + `dual_stream_decode_jax.py`, bd-parameterized) → Vultr emulator
farm (8 lanes)**, kept connected by a reverse-tunnel chain. It records **STRICT (raw model output) vs
REPAIRED (deployment) metrics separately**, is idempotent/resumable (a `(checkpoint, bd, repair)` cell
already in `runs/results.jsonl` is skipped), and reports **strict-JSON + success rate (= task progress)**
per cell. Config (hosts/ports/checkpoints/decode map) is centralized in **`aw_eval/config.py`** — edit
there, not the driver.

### 5.1 — Register each reasoning checkpoint in `aw_eval/config.py`
Add the reasoning checkpoints you shipped (Phase 4) to the `CHECKPOINTS` registry (each entry is either
a TPU-local path or an HF `(repo_id, path_in_repo)` the driver fetches on demand):
```python
# aw_eval/config.py  — CHECKPOINTS = {...}
CHECKPOINTS.update({
    "reasoning-step1500": {"hf": ("KMK040412/fastdvlm-aw-reasoning", "fast-dvlm-reasoning/checkpoint-step1500")},
    "reasoning-step2250": {"hf": ("KMK040412/fastdvlm-aw-reasoning", "fast-dvlm-reasoning/checkpoint-step2250")},
    "reasoning-step3000": {"hf": ("KMK040412/fastdvlm-aw-reasoning", "fast-dvlm-reasoning/checkpoint-step3000")},
})
```
(If a fetched checkpoint lacks the processor files, the same Phase-1.3 completion applies on the serving
host before it can process images.)

### 5.2 — Run the bd-sweep over the reasoning checkpoints × block sizes
```bash
cd /home/perelman/aw_eval          # the local orchestrator dir (this repo's aw_eval/ is its source)
export HF_TOKEN=...                # only to auto-fetch HF checkpoints; never hardcode

# quick smoke first (4 tasks, one bd) to confirm the pipeline is up:
python bd_sweep.py --checkpoints reasoning-step1500 --task-set smoke --bds 4 --repair both

# full eval-select sweep: every shipped reasoning checkpoint x all block sizes, strict + repaired:
python bd_sweep.py \
  --checkpoints reasoning-step1500,reasoning-step2250,reasoning-step3000 \
  --bds 1,2,4,8,16,32 --repair both --task-set standard_full
# progress: tail -f runs/driver.log ; results: runs/results.jsonl
```

### 5.3 — Pick the winner
For each `(checkpoint, bd)` cell, read **strict-JSON** (raw decode validity) and **success rate /
task-progress** from `runs/results.jsonl`. **EVAL-SELECT the (checkpoint, bd) with the best AndroidWorld
success.** Expectations from prior sweeps (action phase) you can sanity-check against:
- block-diffusion decode is **near-lossless up to bd4 (= AR equivalent)**; **bd16 is the usable frontier**
  (PAPER_BIB strict-JSON 0.945@bd16 vs 0.569@bd32 — bd32 collapses), which is exactly why training is
  centered on bd16.
- The reasoning curriculum is eval-centered at **bd16**, so bd16 is the primary candidate; still sweep
  bd1/2/4/8 (cheaper, may match) and bd32 (expect degradation) to confirm.
- **The 0.5-epoch checkpoints exist so you can pick the best — 1 epoch may beat 2.** Report the chosen
  checkpoint + bd and its strict + repaired AndroidWorld success.

---

## 6. Optional upgrades to try next (all detailed in METHODOLOGY §5–§6)

These are **future** experiments, not required to complete Phases 1–5. The reasoning rationale and the
BARD analysis behind each are in [`docs/METHODOLOGY_AND_DECISIONS.md`](METHODOLOGY_AND_DECISIONS.md)
§5 (BARD critique) and §6 (reasoning-SFT). In priority order:

1. **kd-on vs kd-off ablation at bd16 (action run).** Our `kd_fewstep` benefit is **UNVERIFIED** and is
   structurally BARD's "poorly-aligned AR→diffusion KD" applied heaviest at large blocks. Run the action
   recipe once with `--kd-fewstep-weight 0.25` and once with `0` and compare strict-JSON / AndroidWorld
   **at bd16**. Single most important missing data point.
2. **Frozen bd4 diffusion-anchor teacher** (BARD's *proven* fix): replace the AR teacher for the
   large-block capability term with a frozen diffusion model at the lossless bd4 anchor; keep the light
   clean-teacher `kd_noisy` as the alignment anchor.
3. **Token-revision decode** + **4. mixed-noise scheduler** — a **PAIR**: mixed-noise (uniform-vocab
   corruption + supervise the corrupted positions) *trains* the fix-wrong-token skill that token-revision
   (overwrite low-confidence committed tokens; our current decode is monotonic commit-only) *exploits*.
   For the **kd-off reasoning run**, adding **mixed-noise** is a reasonable standalone upgrade (denser
   supervision helps the small dataset), but its full benefit needs token-revision at decode.

> Reasoning-SFT is intentionally run **kd-OFF**, which is **BARD-safe by construction**. Do not turn KD
> back on for reasoning — see §6 for why the self-distillation AR teacher cannot teach CoT.

---

## Quick sanity checklist before you call it done
- [ ] On branch `reasoning-sft`; loader patch verified (CoT in `labels != -100` span — §3.3).
- [ ] `--model-dir` points at a completed action checkpoint **with the 3 processor files** (§1.3).
- [ ] Reasoning data on every worker, 11 files under `aitw/ amex/ gui_odyssey/`, `*/*.parquet` (§2).
- [ ] KD OFF (`--kd-noisy-weight 0 --kd-fewstep-weight 0`); batch 16 / pair 1 / ctx 4096 / noisy 1536.
- [ ] Checkpoints shipping to `KMK040412/fastdvlm-aw-reasoning` every 750 steps (§3.4, §4).
- [ ] AndroidWorld bd-sweep run over all shipped reasoning checkpoints × bds; best (ckpt, bd) selected
      on strict + repaired success (§5).
