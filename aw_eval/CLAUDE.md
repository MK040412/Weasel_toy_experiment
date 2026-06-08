# AndroidWorld bd-sweep / ablation eval — reproducible pipeline (set up from scratch)

Fast-dVLM (GUI-Owl-1.5-2B block-diffusion VLA) evaluated on the **AndroidWorld**
benchmark (116 tasks). This dir is the **canonical, re-runnable** harness for the
bd-sweep and checkpoint ablations.

**Why it matters:** the bd-sweep showed block-diffusion decode is near-lossless up to
bd4 (= AR) and that result must be reproducible + extendable to ablations (baseline vs
bd-curric vs Boltzmann; repair on/off).

> **This file is self-contained.** A future Claude with no prior context can stand up
> the entire pipeline — Vultr emulator farm + AndroidWorld install + model-serving
> endpoint + the bd-sweep driver + result collection — by following it top to bottom.
> Every host/port/credential needed is listed; secrets are sourced from env, never
> hardcoded here.

---

## 0. The pipeline at a glance (3 machines)

```
 local orchestrator (THIS dir, /home/perelman/aw_eval)
   bd_sweep.py ── ssh ──> TPU v6e-4  (JAX policy server, 127.0.0.1:8124)
        │                   androidworld_tpu_jax_server.py  --decode --bd
        │                   dual_stream_decode_jax.py  (bd-parameterized decode)
        │
        └── vultr3_ssh ──> Vultr KVM host  (emulator farm aw_pixel33_0..7)
                            vultr/scripts/run_eval_massive_autoopen.sh  (8 lanes, 1 shared server)
                            vultr/aw_auto_open.py        (GUIOWL_AUTO_OPEN -> launch task app)
                            vultr/guiowl_androidworld_agent.py  (calls /predict, coord norm)
                            vultr/scripts/summarize_androidworld_run.py  (.pkl.gz -> summary.json)

 tunnel chain (kept alive by tunnel_supervisor.sh):
   Vultr:18124 ──reverse(aw_reverse_tunnel.py)──> local:18124 ──fwd(ssh -L)──> TPU:8124
```

The single TPU server processes requests **serially** (LOCK + batch=1); the 8 emulator
lanes overlap *env* overhead, not model inference. Throughput upgrade = multi-instance
(see "Known issues").

**Two serving generations exist** (both use the SAME Vultr emulator install):
- **NEW gen (current, this doc):** model served from a **TPU v6e-4** JAX server; driven
  by `bd_sweep.py` → `vultr/scripts/run_eval_massive_autoopen.sh`. Single shared server,
  8 lanes, auto-open workaround, `COORD_MODE=normalized`.
- **OLD gen (historical):** 8× **RunPod GPU** servers on ports 8123–8130, `run_eval_*_8lane.sh`.
  Documented in the backup `RUNBOOK.md` (see §1 reference). Only needed if reviving the GPU path.

---

## TL;DR — reproduce (after one-time setup in §1–§4)

```bash
cd /home/perelman/aw_eval
export HF_TOKEN=...    # needed only to auto-fetch HF checkpoints; see §3.4

# main bd-sweep (Boltzmann-final checkpoint, all block sizes, repair on):
python bd_sweep.py --checkpoints boltzmann-final --bds 1,2,4,8,16,32 --repair on

# curriculum ablation (does Boltzmann give the bd-robustness?):
python bd_sweep.py --checkpoints boltzmann-final,bd-curric-6000 --bds 1,4,16 --repair both

# quick smoke (4 tasks, both repair modes):
python bd_sweep.py --task-set smoke --bds 4 --repair both

# results: runs/results.jsonl     progress: tail -f runs/driver.log
```

The driver is **idempotent**: a `(checkpoint, bd, repair)` cell already in
`runs/results.jsonl` is skipped, so it is safe to re-run after a crash/preemption.

---

## Repo layout (this dir)

| File | Role | Runs on |
|---|---|---|
| `bd_sweep.py` | Orchestrator: sweep `(checkpoint × bd × repair)`, idempotent, strict/repaired | local |
| `config.py` | Single source of truth: hosts/ports/checkpoints/decode-map | local |
| `summarize_cell.py` | Per-run metric extractor (reads `summary.json` → JSON) | local→Vultr |
| `tunnel_supervisor.sh` | Keep-alive for the 2-hop tunnel chain | local |
| `launch_aw_server.sh` | Generic TPU server launcher `<decode> <bd>` | local→TPU `~/` |
| `verify_repro.sh` | Reproducibility gate (checks everything is backed up off-TPU) | local |
| `vultr/scripts/run_eval_massive_autoopen.sh` | 8-lane single-server eval (NEW gen) | →Vultr |
| `vultr/scripts/run_one_eval.sh` | Per-lane runner invoked by the massive script | →Vultr |
| `vultr/scripts/summarize_androidworld_run.py` | `.pkl.gz` per-task → `summary.json` | →Vultr |
| `vultr/aw_auto_open.py` | Auto-launch task app at init (GUIOWL_AUTO_OPEN) | →Vultr |
| `vultr/guiowl_androidworld_agent.py` | AW agent: calls `/predict`, coord norm, repair | →Vultr |
| `vultr/setup_env_once.py` | Per-AVD AndroidWorld app installer (`--perform_emulator_setup`) | →Vultr |
| `runs/results.jsonl` | Accumulated results (one JSON line per cell) | local |

> The `vultr/` subdir is the **local backup of the Vultr-side NEW-gen scripts**
> (`/data2/androidworld_eval/...`). It exists so the eval is recoverable if the Vultr
> box is lost. Deploy them back with the rsync in §1.6.

**Host-level helpers (live in `$HOME`, NOT in this dir — see §3.5 for secrets):**
`~/vultr3_ssh.py` (paramiko SSH to Vultr; **contains a plaintext root password — never commit**),
`~/aw_reverse_tunnel.py` (reverse tunnel), `~/tunnel_supervisor.sh` (mirror of the one here),
`~/launch_aw_server.sh` (deployed to the TPU).

---

## 1. Vultr KVM host — provision the AndroidWorld emulator farm

This is the Android side. It is the SAME install for both serving generations.
A fully backed-up copy of the provisioning scripts + a pinned `python_freeze.txt` lives in
`/home/perelman/fast-dvlm-guiowl-runpod-backup/vultr_androidworld_backup/` (repo
`github.com/MK040412/fast-dvlm-guiowl-runpod-backup`, dir `vultr_androidworld_backup/`,
file `RUNBOOK.md`). The steps below reproduce it; cross-check exact versions there.

### 1.1 Order the host
- **Vultr** instance with **KVM/nested-virt enabled** (bare-metal or a plan that exposes
  `/dev/kvm`). Reference host: Ubuntu 22.04, **24 cores / 48 threads, 256 GB RAM** — 8
  emulators were stable on that. A smaller box works for fewer lanes.
- The active NEW-gen host is reachable via `~/vultr3_ssh.py` (paramiko; host/port/creds
  embedded in that file — see §3.5). `config.py:VULTR_SSH` points at it.

### 1.2 Base packages
```bash
apt-get update
apt-get install -y git curl wget unzip rsync openjdk-17-jdk \
                   python3.11 python3.11-venv python3-pip qemu-kvm
```

### 1.3 Directories + env
```bash
mkdir -p /data2/androidworld_eval /data2/android-sdk /data2/android-avd /data2/android-emulator-logs
cat > /data2/androidworld_eval/env.sh <<'EOF'
export ANDROID_HOME=/data2/android-sdk
export ANDROID_SDK_ROOT=/data2/android-sdk
export ANDROID_AVD_HOME=/data2/android-avd
export PATH="$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator:$ANDROID_HOME/cmdline-tools/latest/bin:$PATH"
export PYTHONPATH=/data2/androidworld_eval:/data2/androidworld_eval/android_world:${PYTHONPATH:-}
EOF
source /data2/androidworld_eval/env.sh
```

### 1.4 Android SDK (cmdline-tools + API 33 system image)
Download the Linux **commandlinetools** zip from
https://developer.android.com/studio#command-line-tools-only and unzip so that
`sdkmanager` ends up at `/data2/android-sdk/cmdline-tools/latest/bin/sdkmanager`
(i.e. the zip's top `cmdline-tools/` dir is renamed to `latest/`). Then:
```bash
source /data2/androidworld_eval/env.sh
yes | sdkmanager --licenses
sdkmanager "platform-tools" "emulator" \
           "platforms;android-33" "system-images;android-33;google_apis;x86_64"
```
> The eval uses **Android 33** AVDs named `aw_pixel33_0 .. aw_pixel33_7`.

### 1.5 Install AndroidWorld (PINNED commit)
```bash
cd /data2/androidworld_eval
git clone https://github.com/google-research/android_world.git
cd android_world
git checkout d9c569f764b3a5629321858de03ff653d0f24056    # PINNED — do not float

cd /data2/androidworld_eval
python3.11 -m venv venv && source venv/bin/activate
pip install -U pip setuptools wheel
pip install -r android_world/requirements.txt
pip install -e android_world
pip install fastapi uvicorn requests opencv-python huggingface_hub
```
Exact pinned versions: `vultr_androidworld_backup/python_freeze.txt`.
The 25 information-retrieval ("answer") tasks need 4 generated protobuf files under
`android_world/task_evals/information_retrieval/proto/*_pb2*.py`. They are produced by
compiling that dir's `.proto` with `protoc` (or are present in the backup `raw/`); if the
IR tasks error on import, regenerate them or copy from the backup.

### 1.6 Deploy THIS repo's Vultr-side scripts onto the host
```bash
# from local, the canonical copies live in aw_eval/vultr/:
rsync -a /home/perelman/aw_eval/vultr/ <VULTR>:/data2/androidworld_eval/
# (places scripts/run_eval_massive_autoopen.sh, scripts/run_one_eval.sh,
#  scripts/summarize_androidworld_run.py, aw_auto_open.py,
#  guiowl_androidworld_agent.py, setup_env_once.py)
chmod +x /data2/androidworld_eval/scripts/*.sh
```
For the OLD/GPU gen scripts (`prepare_avds.sh`, `start_emulators.sh`, `check_emulators.sh`,
`stop_emulators.sh`, `setup_androidworld_envs.sh`), restore from the backup:
```bash
rsync -a /home/perelman/fast-dvlm-guiowl-runpod-backup/vultr_androidworld_backup/raw/ \
         <VULTR>:/data2/androidworld_eval/
```

### 1.7 Create + boot 8 AVDs, install the task apps
```bash
cd /data2/androidworld_eval && source env.sh && source venv/bin/activate
API_LEVEL=33 bash scripts/prepare_avds.sh 8          # -> aw_pixel33_0..7 (pixel_2, 4GB RAM, 8GB data, keyboard on)
API_LEVEL=33 bash scripts/start_emulators.sh 8       # -no-window -no-audio -gpu swiftshader_indirect -no-snapshot -accel on
adb devices && bash scripts/check_emulators.sh       # wait for sys.boot_completed=1 on all 8
```
Expected ports — console: `5554,5556,...,5568`; gRPC: `8554..8561`.

Install the AndroidWorld task apps into each AVD **once** (this is the per-emulator
`--perform_emulator_setup`; uses `vultr/setup_env_once.py`):
```bash
for i in 0 1 2 3 4 5 6 7; do
  python /data2/androidworld_eval/setup_env_once.py \
    --console_port $((5554 + 2*i)) --grpc_port $((8554 + i)) --perform_emulator_setup
done
```

---

## 2. Task sets

`bd_sweep.py --task-set` maps via `config.py:TASK_SETS`:
- `standard_full` → AndroidWorld task_set `standard_full` = **all 116 tasks**, enumerated
  live from the registry by `run_eval_massive_autoopen.sh`:
  ```python
  from android_world import registry
  r = registry.TaskRegistry().get_registry(family=registry.TaskRegistry.ANDROID_WORLD_FAMILY)
  sorted(r.keys())   # 116 task names
  ```
- `smoke` → `smoke_norm_core` = 4 tasks (OpenApp/Clock/Wifi/Bluetooth), from
  `/data2/androidworld_eval/task_sets/smoke_norm_core.txt` (a comma/newline list).
  Deploy `task_sets/*.txt` to the host if absent (copies in the backup `raw/`).

Fixed `TASK_RANDOM_SEED=30` and `N_TASK_COMBINATIONS=1` → deterministic task instances.

---

## 3. Model-serving endpoint (TPU v6e-4 JAX server)

### 3.1 TPU host + repo
- Host: `config.py:TPU_HOST` = `dayeonhwang9@34.84.241.117` (a teammate's v6e-4 pod).
  SSH key must already be authorized for that user; the driver uses plain
  `ssh -o StrictHostKeyChecking=no`. If provisioning a **fresh** TPU: create a v6e-4,
  `git clone` the Fast-dVLM trainer repo to `~/Weasel_toy_experiment`, build a `.venv`
  (JAX-TPU + the repo deps), and place `launch_aw_server.sh` at `~/` (next step).
- The serving code (`androidworld_tpu_jax_server.py`, `dual_stream_decode_jax.py`,
  `dvlm_decode_jax.py`) lives in the trainer repo
  `github.com/MK040412/Weasel_toy_experiment` (branch `aw-blockdiffusion-eval-repro`,
  commit `f817b9c`). Pull it onto the TPU from there.

### 3.2 Deploy the launcher + start the server
```bash
scp launch_aw_server.sh dayeonhwang9@34.84.241.117:~/launch_aw_server.sh   # one-time / when changed
```
`bd_sweep.py` starts/stops the server itself (`start_server()`), but to launch manually:
```bash
ssh dayeonhwang9@34.84.241.117
tmux new -s aw_srv                       # MUST be tmux — nohup/systemd die after model load here
bash ~/launch_aw_server.sh grounded_ar_jit 1     # decode + bd; -> 127.0.0.1:8124
```
`launch_aw_server.sh <decode> <bd> [model_dir]` runs:
```
python androidworld_tpu_jax_server.py --model-path <ckpt> --host 127.0.0.1 --port 8124 \
       --max-pixels 100352 --gen-len 96 --decode <decode> --bd <bd>
```

### 3.3 API contract
- `POST /predict` body `{screenshot_b64, goal, history[], ui_elements_text}` →
  `{raw, latency_ms, model_latency_ms, tokens, nfe, decode, bd, tau, ...}`.
- `GET /health` → `{ok, model, decode, bd, devices, ...}`. The driver polls until
  `"ok":true` **and** the reported `bd`/`decode` match what it asked for.
- **Decode map** (`config.py:decode_for_bd`): `bd==1` → `grounded_ar_jit` (AR);
  `bd>1` → `dual_dvlm_bd4 --bd N` (generalized dual-stream block-diffusion). Also
  supports single-stream `dvlm_bd4`. Caps `NOISY_CAP=448 / PROMPT_CAP=640` in
  `dual_stream_decode_jax.py` — long AW prompts overflow → that task errors (counts as a
  fail). Raise the caps for a clean full sweep (costs HBM + a recompile).

### 3.4 Checkpoints (`config.py:CHECKPOINTS`)
- `boltzmann-final` → `{"tpu_path": "/home/dayeonhwang9/tpu_runs/boltzmann_20260606_092721/final"}`
  (already on the TPU).
- `bd-curric-6000` → `{"hf": ("KMK040412/fast-dvlm-guiowl-kd-tpu",
  "fast-dvlm-kd-tpu/aw-overfit-bdcurric/checkpoint-step006000")}` — auto-fetched to
  `/home/dayeonhwang9/ckpts/<name>` by `ensure_checkpoint()` (needs `HF_TOKEN`).
- `baseline` (pre-curriculum, for the curriculum ablation) — **TODO**: the HF subdir is
  not yet filled in `config.py` (commented placeholder). Fill it
  (e.g. `fast-dvlm-kd-tpu/aw-overfit-baseline/...`) before running that ablation.

> `HF_TOKEN` is read from env in `config.py` (`os.environ.get("HF_TOKEN","")`). Export it
> before a run that fetches HF checkpoints. **Never hardcode it.**

### 3.5 Tunnel chain + SECRETS
`bd_sweep.ensure_tunnel()` auto-starts `~/tunnel_supervisor.sh`, which keeps both hops up:
```
Vultr:18124 ──reverse (~/aw_reverse_tunnel.py)──> local:18124 ──(ssh -L)──> TPU:8124
```
The Vultr-side eval hits `http://127.0.0.1:18124`, which tunnels to the TPU's `:8124`.

**Secrets — source from env / keep out of git:**
- `~/vultr3_ssh.py` **hardcodes the Vultr root password in plaintext.** It must stay in
  `$HOME`, NOT in this repo, and must NEVER be committed (GitHub secret-scanning will
  block the push, and it is a live credential). To rotate/relocate, move the host+password
  into an env-sourced file (e.g. `~/.fastdvlm_secrets.env`, mode 600) and read it there.
- SSH private key `~/.ssh/vultr_vscode_ed25519` — secret, never commit.
- `HF_TOKEN` — env var only.
- The real secrets bundle is `~/.fastdvlm_secrets.env` (mode 600). `source` it before runs:
  `set -a; . ~/.fastdvlm_secrets.env; set +a`.

---

## 4. Run the bd-sweep

```bash
cd /home/perelman/aw_eval
python bd_sweep.py --checkpoints <keys> --bds 1,2,4,8,16,32 --repair {on|off|both} \
                   --task-set {standard_full|smoke}
```
Per `(checkpoint, bd)` the driver:
1. `ensure_checkpoint` → resolve/fetch the ckpt path on the TPU.
2. `start_server` → kill + (separately) tmux-launch the server at `(decode, bd)`, health-poll ≤ ~4.5 min.
3. `run_eval` → on Vultr, `setsid env <VARS> bash scripts/run_eval_massive_autoopen.sh`,
   poll ≤ ~107 min for `MASSIVE_DONE` / `*.pkl.gz`.
4. `summarize` → regenerate `summary.json` then read metrics via `summarize_cell.py`.
5. Append one JSON line to `runs/results.jsonl`.

**Env vars passed to the Vultr eval** (set by `run_eval()`):
`TASK_SET, LANES=8, SERVER_URL=http://127.0.0.1:18124, COORD_MODE=normalized,
AUTO_OPEN=1, GUIOWL_REPAIR={1|0}, N_TASK_COMBINATIONS=1, TASK_RANDOM_SEED=30, OUT_ROOT=...`.

### Gotchas (encoded in the scripts — do not "fix" away)
- **Server restart**: kill and launch are SEPARATE ssh calls (a combined kill+launch drops
  on this TPU). Server runs under **tmux** (nohup dies after model load here).
- **Decode regression-free**: `dual_stream_decode_jax.py` was generalized from hardcoded
  bd4; `bd` appears only in `_turn_indices` (`//bd`) and `active_len` (`min(bd,...)`), so
  `bd=4` is byte-identical to the old code (zero regression).
- **auto-open**: AW tasks start on the HOME screen and the model never learned `open` (the
  curated data had ~0% open-app) → it loops a no-op swipe-up. `vultr/aw_auto_open.py`
  (`GUIOWL_AUTO_OPEN=1`) launches the task's first app at init AND sets
  `TaskEval.start_on_home_screen=False` (else `episode_runner` re-homes via
  `agent.reset(go_home=True)` and undoes the launch). Success eval is start-screen
  independent, so this is safe.
- **Coord mode**: NEW gen uses `COORD_MODE=normalized` (norm-1000 coords) + repair. The
  OLD agent defaulted to `absolute` — do not reuse the OLD default.
- **Tunnel fragility**: the reverse tunnel dies on its own → `tunnel_supervisor.sh`
  auto-restarts both hops; the driver ensures it is running.

---

## 5. Results collection

- Vultr writes per-task `*.pkl.gz` under `OUT_ROOT`.
- `vultr/scripts/summarize_androidworld_run.py <OUT_ROOT>` → `summary.json` +
  `episodes.jsonl` (handles `episode_data` as list-of-dicts OR dict-of-lists). Per-episode:
  `is_successful`, `strict_json_rate`, `mobile_use_rate`, `repair_rate`.
- `summarize_cell.py <OUT_ROOT>` (deployed to Vultr) reads `summary.json` → one JSON:
  `episodes, success, rate, strict_json, mobile_use, repair_rate, succ_repair_rate, succ_tasks`.
- `bd_sweep.summarize()` parses that into `runs/results.jsonl`.

**Honest metrics:** STRICT (raw model output: `strict_json`, and success with `--repair off`)
vs REPAIRED (deployment: success with `--repair on`) are reported separately.
`succ_repair_rate` shows whether wins needed repair.

> `runs/results.jsonl` lives only on the local box (committed to git). Heavy artifacts
> (videos, `.pkl.gz`) were uploaded to HF `KMK040412/fast-dvlm-guiowl-androidworld-artifacts`.
> `config.py:RESULTS_REPO` (`KMK040412/androidworld-bd-sweep`) is aspirational — there is no
> automatic results-upload step yet.

### Headline result (Boltzmann-final, standard_full=116, auto-open, repair ON)

| decode | success/116 | rate | strict_json | succ_repair_rate |
|---|---|---|---|---|
| bd1 (AR) | 7 | 6.0% | 1.000 | 0.000 |
| bd2 | 7 | 6.0% | 0.976 | 0.000 |
| bd4 | 7 | 6.0% | 0.952 | 0.000 |
| bd8 | 6 | 5.2% | 0.978 | 0.000 |
| bd16 | 6 | 5.2% | 0.945 | 0.000 |
| bd32 | 4 | 3.4% | 0.569 | 0.000 |

- Block-diffusion is **near-lossless to bd4 (=AR 6.0%)**; bd2=bd4=bd1; bd8=bd16=5.2%;
  bd32 degrades (3.4%, strict_json 0.569).
- All wins had `succ_repair_rate=0.000` → **repair-independent** (repair never converted a
  fail→success). The "no loss to bd4" is REAL.
- Successes are the short Settings-toggle family (Wifi/Bluetooth on/off ±Verify,
  TurnOffWifi+OnBluetooth).
- **Open question (ablation):** is the bd-robustness from the Boltzmann curriculum? Run
  `--checkpoints baseline,bd-curric-6000,boltzmann-final --bds 1,4,8,16,32` (fill the
  `baseline` ckpt in `config.py` first, §3.4).

The absolute ~6% reflects the overfit checkpoint + barebones harness (no a11y/history/
planner, pixel coords, 2B). The bd-sweep isolates the *decode* effect on top of that ceiling.

---

## 6. Verify everything is backed up (before any TPU/Vultr teardown)

```bash
export HF_TOKEN=...
bash verify_repro.sh
```
Checks local code mirrors, the `aw_eval/` docs+harness (now including `vultr/`), HF
checkpoints, HF datasets, Vultr curation data, and git state. WARN ≠ blocker (most are
"upload in progress" or "git-commit recommended").

---

## 7. Stop / teardown

```bash
# TPU server:
ssh dayeonhwang9@34.84.241.117 'pkill -f androidworld_tpu_jax_server; tmux kill-server' || true
# Vultr emulators:
<VULTR_SSH> 'bash /data2/androidworld_eval/scripts/stop_emulators.sh' || true
# Tunnels (local):
pkill -f tunnel_supervisor.sh; pkill -f aw_reverse_tunnel.py; pkill -f 'L 127.0.0.1:18124'
```

---

## 8. Known issues / next

- **Throughput**: single serial TPU server is the bottleneck. Upgrade = 4 single-chip
  server instances + lanes split across them (the 8-lane harness already supports multiple
  server ports) → ~4×.
- **Scaffold ablation**: re-run with `--include-ui --include-history` on the server to test
  the a11y/history handicap (model wasn't trained with them → expect limited gain; proper
  fix = train with history via the `history` column / episode-packing).
- **`baseline` checkpoint** for the curriculum ablation is still a TODO in `config.py` (§3.4).

---

## 9. Repo / backup map (where each piece lives in git)

- **This harness + JAX serving code** → `github.com/MK040412/Weasel_toy_experiment`, branch
  `aw-blockdiffusion-eval-repro` (commit `f817b9c` added `aw_eval/` +
  `androidworld_tpu_jax_server.py`; the `aw_eval/vultr/` backup scripts were added later).
  The pushed copy is the **nested** `Weasel_toy_experiment/aw_eval/`; the top-level
  `/home/perelman/aw_eval/` is a working copy (NOT its own git repo — its git root is
  `/home/perelman`, an unrelated repo). Keep the two copies in sync; only the nested one is
  tracked/pushable.
- **Vultr emulator/provisioning side** → `github.com/MK040412/fast-dvlm-guiowl-runpod-backup`,
  dir `vultr_androidworld_backup/` (RUNBOOK, env freeze, AVD/emulator scripts, android_world
  commit pin). Fully pushed.
- **Upstream AndroidWorld** → `github.com/google-research/android_world` @ `d9c569f`.
- **Host secrets** (`~/vultr3_ssh.py` password, `~/.ssh/vultr_*`, `HF_TOKEN`) → NEVER in git;
  env / `~/.fastdvlm_secrets.env` only.
