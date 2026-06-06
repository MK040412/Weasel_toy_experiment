#!/usr/bin/env python3
"""Reproducible AndroidWorld bd-sweep / ablation driver.

Orchestrates the 3-machine pipeline (local orchestrator + TPU JAX policy server +
Vultr emulator farm) to sweep over (checkpoint x block-size x repair) and record
STRICT vs REPAIRED metrics separately (per the benchmark spec).

Design principles
  - Config-driven: all hosts/ports/checkpoints/decode-map live in config.py.
  - Separation of concerns: serving (TPU) / tunnel / eval (Vultr) / orchestration (here).
  - Idempotent + resumable: a (checkpoint, bd, repair) cell already in results.jsonl is skipped.
  - Reproducible: fixed TASK_SEED, fixed task set, logged config, deterministic decode map.
  - Honest metrics: strict (raw model output) and repaired (deployment) reported separately.

Usage
  python bd_sweep.py --checkpoints boltzmann-final --bds 1,2,4,8,16,32 --repair on
  python bd_sweep.py --checkpoints boltzmann-final,bd-curric-6000 --bds 1,4,16 --repair both   # ablation
  python bd_sweep.py --task-set smoke --bds 4 --repair both                                     # quick check
Progress: tail -f <out>/driver.log ; results: <out>/results.jsonl
"""
import argparse, json, os, re, subprocess, time

import config as C

OUTDIR = "/home/perelman/aw_eval/runs"
os.makedirs(OUTDIR, exist_ok=True)
DRIVER_LOG = os.path.join(OUTDIR, "driver.log")
RESULTS = os.path.join(OUTDIR, "results.jsonl")


def sh(cmd, t):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t).stdout or ""
    except Exception as e:
        return f"(ERR {e})"


def tpu(c, t=120):
    return sh(f"ssh -o ConnectTimeout=12 -o StrictHostKeyChecking=no {C.TPU_HOST} '{c}'", t)


def vultr(c, t=120):
    return sh(f'{C.VULTR_SSH} "{c}" {t}', t + 25)


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(DRIVER_LOG, "a") as f:
        f.write(line + "\n")


def done_cells():
    """Set of (checkpoint, bd, repair) already recorded -> idempotency."""
    cells = set()
    if os.path.exists(RESULTS):
        for ln in open(RESULTS):
            try:
                r = json.loads(ln)
                cells.add((r["checkpoint"], r["bd"], r["repair"]))
            except Exception:
                pass
    return cells


def ensure_tunnel():
    if "tunnel_supervisor.sh" not in sh("pgrep -af tunnel_supervisor.sh || true", 15):
        log("starting tunnel supervisor")
        sh(f"setsid bash {C.TUNNEL_SUPERVISOR} >/tmp/tunnel_sup/sup.log 2>&1 < /dev/null &", 15)
        time.sleep(15)


def ensure_checkpoint(name):
    """Return the TPU-local path for a checkpoint, fetching from HF if needed."""
    spec = C.CHECKPOINTS[name]
    if "tpu_path" in spec:
        return spec["tpu_path"]
    repo, sub = spec["hf"]
    dest = f"/home/dayeonhwang9/ckpts/{name}"
    if "READY" not in tpu(f"test -f {dest}/config.json && echo READY || echo NO", 20):
        log(f"downloading checkpoint {name} ({repo}/{sub}) to TPU ...")
        tpu(f"HF_TOKEN={C.HF_TOKEN} {C.TPU_REPO}/.venv/bin/python -c "
            f"\"from huggingface_hub import snapshot_download as d; "
            f"d(repo_id='{repo}', allow_patterns=['{sub}/*'], local_dir='/tmp/hf_{name}', token='{C.HF_TOKEN}')\"; "
            f"mkdir -p {dest} && cp -r /tmp/hf_{name}/{sub}/* {dest}/", 3600)
    return dest


def start_server(ckpt_path, decode, bd):
    """Restart the TPU policy server at (decode, bd). Kill and launch are SEPARATE
    ssh calls (combined kill+launch tends to drop on this TPU). Returns True if healthy."""
    tpu("pkill -f androidworld_tpu_jax_server 2>/dev/null; tmux kill-server 2>/dev/null; sleep 2", 40)
    tpu(f"rm -f ~/aw_server.log; tmux new-session -d -s aw_srv "
        f"'bash ~/launch_aw_server.sh {decode} {bd} 2>&1 | tee -a ~/aw_server.log'; sleep 3", 40)
    for _ in range(45):  # up to ~4.5 min (model load + warmup compile)
        h = tpu(f"curl -sS --max-time 4 http://127.0.0.1:{C.TPU_PORT}/health 2>/dev/null", 25).replace(" ", "")
        if '"ok":true' in h and f'"bd":{bd}' in h and f'"decode":"{decode}"' in h:
            return True
        time.sleep(6)
    return False


def run_eval(out_root, task_set, repair, auto_open=1):
    swlog = f"/tmp/sweep_{os.path.basename(out_root)}.log"
    env = (f"TASK_SET={C.TASK_SETS[task_set]} LANES={C.LANES} "
           f"SERVER_URL=http://127.0.0.1:{C.TUNNEL_PORT} COORD_MODE=normalized "
           f"AUTO_OPEN={auto_open} GUIOWL_REPAIR={repair} GUIOWL_RECORD_TRAJ=0 "
           f"N_TASK_COMBINATIONS=1 TASK_RANDOM_SEED={C.TASK_SEED} OUT_ROOT={out_root}")
    vultr(f"cd {C.VULTR_EVAL} && setsid env {env} bash scripts/run_eval_massive_autoopen.sh "
          f"> {swlog} 2>&1 < /dev/null & echo LAUNCHED", 60)
    for i in range(160):  # up to ~107 min
        st = vultr(f"grep -c MASSIVE_DONE {swlog} 2>/dev/null; find {out_root} -name '*.pkl.gz' 2>/dev/null | wc -l", 40)
        n = re.findall(r"\d+", st)
        if n and int(n[0]) >= 1:
            return True
        if i % 5 == 0 and len(n) > 1:
            log(f"  ... {n[1]}/116 tasks done")
        time.sleep(40)
    return False


def summarize(out_root):
    """Regenerate summary.json then read metrics via the deployed summarize_cell.py helper."""
    out = vultr(
        f"cd {C.VULTR_EVAL} && source venv/bin/activate 2>/dev/null; "
        f"scripts/summarize_androidworld_run.py {out_root} >/dev/null 2>&1; "
        f"venv/bin/python3 summarize_cell.py {out_root}", 90)
    m = re.search(r"\{.*\}", out)
    return json.loads(m.group(0)) if m else {"raw": out[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="boltzmann-final", help="comma list of CHECKPOINTS keys")
    ap.add_argument("--bds", default="1,2,4,8,16,32", help="comma list of block sizes")
    ap.add_argument("--repair", default="on", choices=["on", "off", "both"])
    ap.add_argument("--task-set", default="standard_full", choices=list(C.TASK_SETS))
    a = ap.parse_args()
    ckpts = [c for c in a.checkpoints.split(",") if c]
    bds = [int(b) for b in a.bds.split(",") if b]
    repairs = {"on": [1], "off": [0], "both": [1, 0]}[a.repair]

    ensure_tunnel()
    cells = done_cells()
    log(f"=== bd-sweep START ckpts={ckpts} bds={bds} repairs={repairs} task_set={a.task_set} ===")
    for ck in ckpts:
        ckpt_path = ensure_checkpoint(ck)
        for bd in bds:
            decode, bd_arg = C.decode_for_bd(bd)
            ready = None
            for rep in repairs:
                if (ck, bd, rep) in cells:
                    log(f"skip (done): {ck} bd{bd} repair{rep}")
                    continue
                if ready is None:  # start server once per (ckpt, bd); reuse across repair on/off
                    log(f"--- {ck} bd{bd} ({decode}) : starting server ---")
                    ready = start_server(ckpt_path, decode, bd_arg)
                    log(f"{ck} bd{bd} server ready={ready}")
                if not ready:
                    rec = {"checkpoint": ck, "bd": bd, "repair": rep, "error": "server_not_ready"}
                else:
                    ts = sh("date +%Y%m%d_%H%M%S", 10).strip()
                    out_root = f"{C.VULTR_EVAL}/runs/{ts}_{ck}_bd{bd}_rep{rep}_{a.task_set}"
                    log(f"{ck} bd{bd} repair{rep} -> eval {out_root}")
                    fin = run_eval(out_root, a.task_set, rep)
                    rec = {"checkpoint": ck, "bd": bd, "repair": rep, "done": fin, "out": out_root, "decode": decode}
                    if fin:
                        rec.update(summarize(out_root))
                with open(RESULTS, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                log(f"RESULT {ck} bd{bd} rep{rep}: success={rec.get('success')}/{rec.get('episodes')} "
                    f"rate={rec.get('rate')}% strict_json={rec.get('strict_json')}")
    log("=== bd-sweep COMPLETE ===")
    print("\n==== SUMMARY ====")
    for ln in open(RESULTS):
        r = json.loads(ln)
        print(f"  {r.get('checkpoint')} bd{r.get('bd')} repair{r.get('repair')}: "
              f"{r.get('success')}/{r.get('episodes')} ({r.get('rate')}%) strict_json={r.get('strict_json')}")


if __name__ == "__main__":
    main()
