# AndroidWorld bd-sweep / ablation eval — reproducible pipeline

Fast-dVLM (GUI-Owl-1.5-2B block-diffusion VLA) evaluated on the AndroidWorld
benchmark. This dir is the **canonical, re-runnable** harness for the bd-sweep and
checkpoint ablations. **Why it matters:** the bd-sweep showed block-diffusion is
near-lossless up to bd4 (= AR) and that result must be reproducible + extendable to
ablations (baseline vs bd-curric vs Boltzmann; repair on/off).

## TL;DR — reproduce

```bash
cd /home/perelman/aw_eval
# one-time deploy of the helper scripts to TPU + Vultr (see "Deploy" below)
python bd_sweep.py --checkpoints boltzmann-final --bds 1,2,4,8,16,32 --repair on
# ablation: does the Boltzmann curriculum give the bd-robustness?
python bd_sweep.py --checkpoints boltzmann-final,bd-curric-6000 --bds 1,4,16 --repair both
# results: runs/results.jsonl   progress: tail -f runs/driver.log
```

## Architecture (3 machines)

```
 local orchestrator (this dir)
   bd_sweep.py ── ssh ──> TPU v6e-4  (JAX policy server, 127.0.0.1:8124)
        │                   androidworld_tpu_jax_server.py  --decode --bd
        │                   dual_stream_decode_jax.py  (bd-parameterized)
        │
        └── vultr3_ssh ──> Vultr (48-core)  emulator farm aw_pixel33_0..7
                            run_eval_massive_autoopen.sh  (8 lanes, 1 shared server)
                            aw_auto_open.py  (GUIOWL_AUTO_OPEN -> launch task app)
                            guiowl_androidworld_agent.py + summarize_androidworld_run.py

 tunnel chain (kept alive by tunnel_supervisor.sh):
   Vultr:18124 ──reverse(aw_reverse_tunnel.py)──> local:18124 ──fwd(ssh -L)──> TPU:8124
```

The single TPU server processes requests serially (LOCK + batch=1); 8 emulator lanes
overlap env overhead, not model inference. (Throughput upgrade = multi-instance, see
"Known issues".)

## Components & locations

| Concern | File | Host |
|---|---|---|
| Orchestrator (sweep, idempotent, strict/repaired) | `bd_sweep.py` | local |
| Config (hosts/ports/checkpoints/decode-map) | `config.py` | local |
| Per-run metric extractor | `summarize_cell.py` | local→Vultr |
| Tunnel keep-alive | `tunnel_supervisor.sh` (+ `~/aw_reverse_tunnel.py`, `~/vultr3_ssh.py`) | local |
| Generic server launcher `<decode> <bd>` | `launch_aw_server.sh` | local→TPU `~/` |
| JAX policy server (`--decode --bd`) | `androidworld_tpu_jax_server.py` | TPU `~/Weasel_toy_experiment/` |
| Generalized block-diffusion decode (bd param) | `dual_stream_decode_jax.py` | TPU |
| Auto-open workaround (launch task app) | `aw_auto_open.py` | Vultr `/data2/androidworld_eval/` |
| 8-lane parallel eval | `scripts/run_eval_massive_autoopen.sh` | Vultr |
| AW agent (coord norm1000, repair) | `guiowl_androidworld_agent.py` | Vultr |

## Deploy (one-time, after editing local canonical copies)

```bash
cd /home/perelman/aw_eval
scp launch_aw_server.sh dayeonhwang9@34.84.241.117:~/launch_aw_server.sh
scp summarize_cell.py "$(python3 -c 'import config;print(config.VULTR_EVAL)')"/   # via vultr3_ssh / scp
# androidworld_tpu_jax_server.py, dual_stream_decode_jax.py, aw_auto_open.py,
# run_eval_massive_autoopen.sh are already deployed; re-scp from their repos if changed.
```

## Design principles

1. **Config-driven** — every host/port/checkpoint/decode choice in `config.py`; no magic constants in the driver.
2. **Separation of concerns** — serving (TPU) / tunnel / eval (Vultr) / orchestration (local) are independent layers.
3. **Idempotent + resumable** — `(checkpoint, bd, repair)` cells already in `runs/results.jsonl` are skipped; safe to re-run after a crash.
4. **Reproducible** — fixed `TASK_SEED=30`, fixed task set, deterministic `decode_for_bd` map, logged config.
5. **Honest metrics** — STRICT (raw model output: `strict_json`, and success with `--repair off`) vs REPAIRED (deployment: success with `--repair on`) reported separately. `succ_repair_rate` shows whether wins needed repair.

## Key gotchas (learned, encoded in the scripts)

- **Server restart**: kill and launch must be SEPARATE ssh calls (combined kill+launch drops on this TPU). `start_server()` does this; tmux (not nohup — nohup dies after model load here).
- **Decode map**: bd=1 → `grounded_ar_jit` (AR); bd>1 → `dual_dvlm_bd4 --bd N` (generalized dual-stream). `dual_stream_decode_jax.py` was generalized from hardcoded bd4 — bd appears only in `_turn_indices` (`//bd`) and `active_len` (`min(bd,...)`), so bd=4 is byte-identical to the old code (zero regression).
- **NOISY_CAP=448 / PROMPT_CAP=640**: dual-stream caps tuned for the training regime; long-prompt AW tasks overflow → that task errors (counts as fail). Raise the caps in `dual_stream_decode_jax.py` for a clean full sweep (costs memory + recompile).
- **auto-open**: AW tasks start on the HOME screen and the model never learned `open` (curated data had ~0% open) → it loops on home. `aw_auto_open.py` (GUIOWL_AUTO_OPEN=1) launches the task's app at init AND sets `TaskEval.start_on_home_screen=False` (else `episode_runner.agent.reset(go_home=True)` re-homes and undoes the launch).
- **Tunnel fragility**: the reverse tunnel dies on its own → `tunnel_supervisor.sh` auto-restarts both hops; the driver ensures it is running.

## Results (fill from runs/results.jsonl)

Boltzmann-final, standard_full (116 tasks), auto-open, repair ON — bd-sweep:

| decode | success/116 | rate | strict_json | succ_repair_rate |
|---|---|---|---|---|
| bd1 (AR) | 7 | 6.0% | 1.000 | 0.000 |
| bd2 | 7 | 6.0% | 0.976 | 0.000 |
| bd4 | 7 | 6.0% | 0.952 | 0.000 |
| bd8 | 6 | 5.2% | 0.978 | 0.000 |
| bd16 | 6 | 5.2% | 0.945 | 0.000 |
| bd32 | 4 | 3.4% | 0.569 | 0.000 |

- **block-diffusion is near-lossless to bd4 (=AR 6.0%); bd2=bd4=bd1; bd8=bd16=5.2%; bd32 degrades (3.4%, strict_json 0.569).**
- **All wins had `succ_repair_rate=0.000`** → repair-independent; repair never converted a fail→success. The "no loss to bd4" is REAL.
- Successes are the short Settings-toggle family (Wifi/Bluetooth on/off ±Verify, TurnOffWifi+OnBluetooth).
- Open question (needs ablation): is bd-robustness from the Boltzmann curriculum? Run `--checkpoints baseline,bd-curric-6000,boltzmann-final --bds 1,4,8,16,32`.

## Known issues / next

- **Throughput**: single serial TPU server is the bottleneck. Upgrade = 4 single-chip server instances (v6e-4) + lanes split across them (the 8-lane harness already supports multi-server ports) → ~4x.
- **Scaffold ablation** (#1): re-run with `--include-ui --include-history` on the server to test the a11y/history handicap (model wasn't trained with them → expect limited gain; proper fix = train with history via the `history` column / episode-packing).
- The absolute ~6% reflects the overfit checkpoint + barebones (no a11y/history/planner, pixel coords, 2B). The bd-sweep isolates *decode* effect on top of that ceiling.
