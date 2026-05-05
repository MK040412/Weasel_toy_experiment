"""Quick TPU distributed sanity check.

Run on ALL workers simultaneously:
  gcloud compute tpus tpu-vm ssh <TPU_NAME> --zone=<ZONE> --worker=all \
    --command="cd ~/Weasel_toy_experiment && source .venv/bin/activate && python scripts/test_tpu_distributed.py"

Single-host (skip distributed init):
  python scripts/test_tpu_distributed.py --no-distributed
"""

import argparse
import os

import jax
import jax.numpy as jnp

parser = argparse.ArgumentParser()
parser.add_argument("--no-distributed", action="store_true")
args = parser.parse_args()

if not args.no_distributed:
    print(f"[worker] Calling jax.distributed.initialize()...")
    jax.distributed.initialize()

n_dev = jax.device_count()
process_idx = jax.process_index()
n_processes = jax.process_count()

print(f"[worker {process_idx}/{n_processes}] devices seen: {n_dev}")
print(f"[worker {process_idx}/{n_processes}] device list: {jax.devices()}")

# pmap test: each device computes its own index sum, then all-reduce
@jax.pmap
def pmap_test(x):
    local_sum = jnp.sum(x)
    global_sum = jax.lax.psum(local_sum, axis_name="batch")
    return global_sum

pmap_test = jax.pmap(lambda x: jax.lax.psum(jnp.sum(x), axis_name="batch"), axis_name="batch")

x = jnp.ones((n_dev, 8))  # (n_dev, 8) — one slice per device
result = pmap_test(x)
jax.block_until_ready(result)

expected = float(n_dev * 8)
got = float(result[0])
status = "OK" if got == expected else f"MISMATCH (expected {expected}, got {got})"
print(f"[worker {process_idx}/{n_processes}] pmap all-reduce test: {status}")

if process_idx == 0:
    print(f"\n=== RESULT ===")
    print(f"Total devices: {n_dev}  (expected 16 for v4-16)")
    print(f"Processes: {n_processes}  (expected 2 for v4-16)")
    print("All checks passed!" if got == expected else "FAILED")
