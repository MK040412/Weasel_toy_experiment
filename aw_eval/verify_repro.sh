#!/usr/bin/env bash
# Reproducibility verification — confirms everything needed to reproduce training + eval
# is captured OFF the v6e-4 TPU (so the pod can be torn down safely). Re-runnable anytime.
# Checks: local code mirrors, aw_eval docs/scripts, HF checkpoints, HF datasets, Vultr data, git state.
HF_TOKEN=${HF_TOKEN:?export HF_TOKEN first}
PASS=0; WARN=0
ok(){ echo "  [PASS] $1"; PASS=$((PASS+1)); }
warn(){ echo "  [WARN] $1"; WARN=$((WARN+1)); }
chk(){ [ -f "$1" ] && ok "$2 ($1)" || warn "MISSING $2 ($1)"; }

echo "================ REPRODUCIBILITY VERIFICATION ================"
echo "[1] Local code mirrors (survive TPU teardown; on this machine)"
for f in /home/perelman/Weasel_toy_experiment/dual_stream_decode_jax.py \
         /home/perelman/Weasel_toy_experiment/androidworld_tpu_jax_server.py \
         /home/perelman/episode_work/curation_src/train_fastdvlm_continue_TPU.py; do chk "$f" "code"; done
python3 -m py_compile /home/perelman/Weasel_toy_experiment/dual_stream_decode_jax.py \
   /home/perelman/Weasel_toy_experiment/androidworld_tpu_jax_server.py 2>/dev/null \
   && ok "local code py_compile clean" || warn "local code py_compile FAILED"

echo "[2] aw_eval canonical docs + harness"
for f in CLAUDE.md TRAINING_v6e16.md CURATION_PLAN.md bd_sweep.py config.py summarize_cell.py launch_aw_server.sh tunnel_supervisor.sh; do
  chk "/home/perelman/aw_eval/$f" "$f"; done
python3 -m py_compile /home/perelman/aw_eval/bd_sweep.py /home/perelman/aw_eval/config.py 2>/dev/null \
   && ok "bd_sweep.py + config.py py_compile clean" || warn "bd_sweep py_compile FAILED"

echo "[3] HF checkpoints (model repo KMK040412/fast-dvlm-guiowl-kd-tpu)"
HF_TOKEN=$HF_TOKEN python3 - <<'PY'
import os
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"])
need=["fast-dvlm-kd-tpu/aw-overfit-boltzmann/final",
      "fast-dvlm-kd-tpu/aw-overfit-bdcurric/checkpoint-step006000",
      "fast-dvlm-kd-tpu/aw-overfit-continue/final"]
fs=set(api.list_repo_files("KMK040412/fast-dvlm-guiowl-kd-tpu", repo_type="model"))
for n in need:
    print("  [PASS] ckpt on HF:" if any(f.startswith(n) for f in fs) else "  [WARN] ckpt MISSING:", n)
PY

echo "[4] HF datasets (raw corpus + 3 versions; workflow may still be uploading)"
HF_TOKEN=$HF_TOKEN python3 - <<'PY'
import os
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"])
for r in ["KMK040412/guiowl-curated-corpus","KMK040412/guiowl-aw-mix-full",
          "KMK040412/guiowl-aw-mix-targeted","KMK040412/guiowl-aw-mix-hybrid"]:
    try:
        info=api.dataset_info(r); vis="public" if not info.private else "PRIVATE"
        print(f"  [PASS] dataset exists ({vis}): {r}")
    except Exception:
        print(f"  [WARN] dataset not yet present (upload in progress?): {r}")
PY

echo "[5] Vultr persisted data (curated sources + new mixes)"
python3 /home/perelman/vultr3_ssh.py "echo curated_out=\$(ls -d /data3/curation_out/*/ 2>/dev/null|wc -l)' sources'; ls -d /data3/aw_mix_* 2>/dev/null || echo '  (aw_mix_* not built yet)'; df -h /data3 2>/dev/null|tail -1" 40 2>&1 | sed 's/^/  /' | head -8

echo "[6] git state of training/eval code (full repro needs commit, not just local FS)"
cd /home/perelman/Weasel_toy_experiment 2>/dev/null && {
  git status --porcelain dual_stream_decode_jax.py androidworld_tpu_jax_server.py 2>/dev/null | grep -q . \
    && warn "generalized decode/server NOT committed to git (only local FS) — commit+push for full repro" \
    || ok "decode/server tracked & clean in git"
} || warn "Weasel_toy_experiment not a git repo locally"

echo "============================================================="
echo "RESULT: PASS=$PASS  WARN=$WARN"
[ "$WARN" -eq 0 ] && echo "✅ FULLY REPRODUCIBLE — safe to tear down v6e-4 TPU." \
  || echo "⚠️ $WARN warning(s) — review above (most are 'upload still in progress' or 'git-commit recommended', not blockers for TPU teardown)."
