"""Qwen3-VL 2B batch training benchmark on TPU v4-8.

Compares single-device vs 4-device data-parallel training throughput.
Uses synthetic data for pure compute benchmarking.
"""

import os
os.environ["LIBTPU_INIT_ARGS"] = " ".join([
    "--xla_tpu_use_enhanced_launch_barrier=true",
    "--xla_tpu_enable_data_parallel_all_reduce_opt=true",
    "--xla_tpu_scoped_vmem_limit_kib=98304",
])
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import time
import sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from model import modeling

MODEL_PATH = os.environ.get("QWEN3VL_MODEL_PATH", os.path.join(_SCRIPT_DIR, "..", "models", "qwen3-vl-2b"))
SEQ_LEN = 256
VOCAB_SIZE = 151936


def make_optimizer(lr_schedule, kind="sgd"):
    """Create optimizer. SGD for memory-constrained multi-device, AdamW for single."""
    if kind == "adamw_bf16":
        return optax.chain(
            optax.scale_by_adam(mu_dtype=jnp.bfloat16),
            optax.add_decayed_weights(1e-4),
            optax.scale_by_schedule(lr_schedule),
            optax.scale(-1.0),
        )
    elif kind == "adamw":
        return optax.adamw(lr_schedule)
    else:
        return optax.sgd(lr_schedule)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def make_synthetic_batch(batch_size: int, seq_len: int, vocab_size: int, rng_key):
    """Generate random input_ids and labels for benchmarking."""
    k1, k2 = jax.random.split(rng_key)
    input_ids = jax.random.randint(k1, (batch_size, seq_len), 0, vocab_size)
    labels = jax.random.randint(k2, (batch_size, seq_len), 0, vocab_size)
    return input_ids, labels


# ---------------------------------------------------------------------------
# Single-device training (baseline)
# ---------------------------------------------------------------------------

def train_single_device(model, batch_size: int, num_steps: int = 10, warmup_steps: int = 3):
    """Baseline: train on 1 device with nnx.jit."""
    print(f"\n{'='*60}")
    print(f"SINGLE-DEVICE TRAINING (batch_size={batch_size}, seq_len={SEQ_LEN})")
    print(f"{'='*60}")

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=1e-5, warmup_steps=warmup_steps,
        decay_steps=num_steps, end_value=1e-6,
    )
    optimizer = nnx.Optimizer(model, make_optimizer(lr_schedule, kind="adamw_bf16"))

    @nnx.jit
    def train_step(model, optimizer, input_ids, labels):
        def loss_fn(model):
            return modeling.forward_train(model, input_ids, labels)
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    rng = jax.random.PRNGKey(42)
    step_times = []
    losses = []

    for step in range(num_steps):
        rng, step_rng = jax.random.split(rng)
        input_ids, labels = make_synthetic_batch(batch_size, SEQ_LEN, VOCAB_SIZE, step_rng)

        start = time.time()
        loss = train_step(model, optimizer, input_ids, labels)
        jax.block_until_ready(loss)
        elapsed = time.time() - start

        loss_val = float(loss)
        losses.append(loss_val)

        if step < warmup_steps:
            print(f"  [warmup] step {step}: loss={loss_val:.4f}, time={elapsed:.3f}s (JIT compile)")
        else:
            step_times.append(elapsed)
            tokens_per_sec = batch_size * SEQ_LEN / elapsed
            print(f"  step {step}: loss={loss_val:.4f}, time={elapsed:.3f}s, "
                  f"{tokens_per_sec:.0f} tok/s")

    if step_times:
        avg_time = np.mean(step_times)
        avg_tps = batch_size * SEQ_LEN / avg_time
        print(f"\n  >> Average step time: {avg_time:.3f}s")
        print(f"  >> Average throughput: {avg_tps:.0f} tok/s")
        print(f"  >> Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
        return avg_tps, avg_time
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Multi-device training (4-chip data parallel via Mesh + NamedSharding)
# ---------------------------------------------------------------------------

def train_multi_device(model, batch_size_total: int, num_steps: int = 10, warmup_steps: int = 3):
    """4-device data parallel training using JAX Mesh sharding."""
    n_devices = jax.device_count()
    assert batch_size_total % n_devices == 0, \
        f"batch_size_total={batch_size_total} must be divisible by {n_devices} devices"
    batch_per_device = batch_size_total // n_devices

    print(f"\n{'='*60}")
    print(f"MULTI-DEVICE TRAINING ({n_devices} TPUs, batch_total={batch_size_total}, "
          f"batch/device={batch_per_device}, seq_len={SEQ_LEN})")
    print(f"{'='*60}")

    devices = jax.devices()
    mesh = Mesh(np.array(devices), axis_names=("dp",))
    data_sharding = NamedSharding(mesh, P("dp"))
    replicated = NamedSharding(mesh, P())

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=1e-5, warmup_steps=warmup_steps,
        decay_steps=num_steps, end_value=1e-6,
    )
    optimizer = nnx.Optimizer(model, make_optimizer(lr_schedule, kind="sgd"))

    # Replicate model + optimizer state across all devices
    state = nnx.state(model)
    replicated_state = jax.device_put(state, replicated)
    nnx.update(model, replicated_state)

    opt_state = nnx.state(optimizer)
    replicated_opt = jax.device_put(opt_state, replicated)
    nnx.update(optimizer, replicated_opt)

    @nnx.jit
    def train_step(model, optimizer, input_ids, labels):
        def loss_fn(model):
            return modeling.forward_train(model, input_ids, labels)
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    rng = jax.random.PRNGKey(123)
    step_times = []
    losses = []

    for step in range(num_steps):
        rng, step_rng = jax.random.split(rng)
        input_ids, labels = make_synthetic_batch(batch_size_total, SEQ_LEN, VOCAB_SIZE, step_rng)
        # Shard data across devices
        input_ids = jax.device_put(input_ids, data_sharding)
        labels = jax.device_put(labels, data_sharding)

        start = time.time()
        loss = train_step(model, optimizer, input_ids, labels)
        jax.block_until_ready(loss)
        elapsed = time.time() - start

        loss_val = float(loss)
        losses.append(loss_val)

        if step < warmup_steps:
            print(f"  [warmup] step {step}: loss={loss_val:.4f}, time={elapsed:.3f}s (JIT compile)")
        else:
            step_times.append(elapsed)
            tokens_per_sec = batch_size_total * SEQ_LEN / elapsed
            print(f"  step {step}: loss={loss_val:.4f}, time={elapsed:.3f}s, "
                  f"{tokens_per_sec:.0f} tok/s")

    if step_times:
        avg_time = np.mean(step_times)
        avg_tps = batch_size_total * SEQ_LEN / avg_time
        print(f"\n  >> Average step time: {avg_time:.3f}s")
        print(f"  >> Average throughput: {avg_tps:.0f} tok/s")
        print(f"  >> Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
        return avg_tps, avg_time
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "multi", "both"], default="both")
    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3-VL 2B — TPU v4-8 Batch Training Benchmark")
    print("=" * 60)
    print(f"JAX: {jax.__version__}")
    print(f"Devices: {jax.device_count()} x TPU v4")
    print(f"HBM per device: {jax.devices()[0].memory_stats()['bytes_limit'] / 1e9:.1f} GB")

    config = modeling.ModelConfig.qwen3vl_2b()
    results = {}

    if args.mode in ("single", "both"):
        print("\nLoading pretrained model...")
        model = modeling.Qwen3VLForConditionalGeneration.from_pretrained(MODEL_PATH, config=config)
        print("Model loaded.")

        for bs in [1, 2, 4]:
            tps, avg_t = train_single_device(model, batch_size=bs, num_steps=8, warmup_steps=2)
            results[f"single_bs{bs}"] = {"tok/s": tps, "step_time": avg_t}

        # Free single-device model
        del model
        import gc; gc.collect()

    if args.mode in ("multi", "both"):
        for bs_total in [4, 8, 16]:
            print(f"\nLoading model for multi-device bs={bs_total}...")
            model = modeling.Qwen3VLForConditionalGeneration.from_pretrained(MODEL_PATH, config=config)
            tps, avg_t = train_multi_device(model, batch_size_total=bs_total, num_steps=8, warmup_steps=2)
            results[f"multi_bs{bs_total}"] = {"tok/s": tps, "step_time": avg_t}
            del model
            import gc; gc.collect()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Config':<25} {'Throughput':>12} {'Step Time':>12} {'Speedup':>10}")
    print("-" * 60)

    baseline_tps = results.get("single_bs1", {}).get("tok/s", 1.0) or 1.0
    for key, val in results.items():
        tps = val["tok/s"]
        step_t = val["step_time"]
        speedup = tps / baseline_tps if baseline_tps > 0 else 0
        print(f"  {key:<23} {tps:>10.0f} t/s {step_t:>10.3f}s {speedup:>8.2f}x")

    print("=" * 60)


if __name__ == "__main__":
    main()
