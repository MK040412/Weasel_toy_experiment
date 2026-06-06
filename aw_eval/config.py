"""Shared configuration for the AndroidWorld bd-sweep eval pipeline.

Single source of truth for hosts, ports, checkpoints, and the decode map.
Edit here, not in the driver. (Design principle: config-driven, no magic constants
scattered across scripts.)
"""

# ---- Infrastructure ---------------------------------------------------------
TPU_HOST = "dayeonhwang9@34.84.241.117"          # JAX policy server (v6e-4)
TPU_REPO = "/home/dayeonhwang9/Weasel_toy_experiment"
TPU_PORT = 8124                                   # server binds 127.0.0.1:TPU_PORT
VULTR_SSH = "python3 /home/perelman/vultr3_ssh.py"  # paramiko helper (creds embedded)
VULTR_EVAL = "/data2/androidworld_eval"           # AndroidWorld harness + agent on Vultr
TUNNEL_PORT = 18124                               # Vultr:TUNNEL -> local -> TPU:TPU_PORT
TUNNEL_SUPERVISOR = "/home/perelman/tunnel_supervisor.sh"

# ---- Checkpoints (ablation registry) ---------------------------------------
# name -> {"tpu_path": local dir on TPU} OR {"hf": (repo_id, path_in_repo)}.
# The driver fetches HF checkpoints to TPU on demand if tpu_path is absent.
CHECKPOINTS = {
    "boltzmann-final": {"tpu_path": "/home/dayeonhwang9/tpu_runs/boltzmann_20260606_092721/final"},
    "bd-curric-6000":  {"hf": ("KMK040412/fast-dvlm-guiowl-kd-tpu",
                               "fast-dvlm-kd-tpu/aw-overfit-bdcurric/checkpoint-step006000")},
    # baseline (pre-curriculum) — fill in the HF path/subdir when running that ablation:
    # "baseline":      {"hf": ("KMK040412/fast-dvlm-guiowl-kd-tpu", "fast-dvlm-kd-tpu/aw-overfit-baseline/...")},
}

# ---- Decode map: bd -> (decode_mode, bd_arg) -------------------------------
# bd=1 uses the autoregressive grounded path; bd>1 uses the generalized dual-stream
# block-diffusion decode (dual_stream_decode_jax.py, bd parameterized).
def decode_for_bd(bd: int):
    return ("grounded_ar_jit", 1) if int(bd) == 1 else ("dual_dvlm_bd4", int(bd))

# ---- Eval defaults ----------------------------------------------------------
SERVER_MAX_PIXELS = 100352
SERVER_GEN_LEN = 96
TASK_SEED = 30          # fixed -> reproducible task instances
LANES = 8               # emulator lanes on Vultr (all share the one TPU server)
TASK_SETS = {           # name -> AndroidWorld task_set
    "standard_full": "standard_full",   # all 116 tasks
    "smoke": "smoke_norm_core",          # 4 tasks (OpenApp/Clock/Wifi/Bluetooth)
}
HF_TOKEN = __import__("os").environ.get("HF_TOKEN", "")  # set HF_TOKEN in env; do not hardcode
RESULTS_REPO = "KMK040412/androidworld-bd-sweep"   # optional: upload results here
