"""Quick TPU distributed sanity check.

Run on ALL workers simultaneously:
  gcloud compute tpus tpu-vm ssh <TPU_NAME> --zone=<ZONE> --worker=all \
    --command="cd ~/Weasel_toy_experiment && source .venv/bin/activate && python scripts/test_tpu_distributed.py"

Single-host (skip distributed init):
  python scripts/test_tpu_distributed.py --no-distributed
"""

import argparse

import jax
import jax.numpy as jnp

parser = argparse.ArgumentParser()
parser.add_argument("--no-distributed", action="store_true")
args = parser.parse_args()

if not args.no_distributed:
    print("[worker] Calling jax.distributed.initialize()...")
    jax.distributed.initialize()

n_global = jax.device_count()  # total across all hosts (e.g. 8 for v4-16)
n_local = jax.local_device_count()  # this host only         (e.g. 4 for v4-16)
process_idx = jax.process_index()
n_processes = jax.process_count()

print(f"[worker {process_idx}/{n_processes}] global_devices={n_global}, local_devices={n_local}")
print(f"[worker {process_idx}/{n_processes}] local device list: {jax.local_devices()}")

# pmap input must use local_device_count (NOT global device_count)
pmap_fn = jax.pmap(
    lambda x: jax.lax.psum(jnp.sum(x), axis_name="batch"),
    axis_name="batch",
)

x = jnp.ones((n_local, 8))  # (local_devices, 8)
result = pmap_fn(x)
jax.block_until_ready(result)

# After psum across all hosts: expected = n_global * 8
expected = float(n_global * 8)
got = float(result[0])
status = "OK" if got == expected else f"MISMATCH (expected {expected}, got {got})"
print(f"[worker {process_idx}/{n_processes}] pmap cross-host all-reduce: {status}")

if process_idx == 0:
    print("\n=== RESULT ===")
    print(f"Global devices : {n_global}  (v4-16 → 8)")
    print(f"Local devices  : {n_local}   (per host → 4)")
    print(f"Processes      : {n_processes}   (v4-16 → 2)")
    print("All checks passed!" if got == expected else "FAILED")
