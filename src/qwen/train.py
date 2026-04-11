"""Qwen3-VL 2B batch training benchmark on TPU v4-8.

Compares single-device vs 4-device data-parallel training throughput.
Uses synthetic data for pure compute benchmarking.
"""

import gc
import os
import time

os.environ["LIBTPU_INIT_ARGS"] = " ".join(
    [
        "--xla_tpu_use_enhanced_launch_barrier=true",
        "--xla_tpu_enable_data_parallel_all_reduce_opt=true",
        "--xla_tpu_scoped_vmem_limit_kib=98304",
    ]
)
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from qwen.qwen3vl import modeling

_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get(
    "QWEN3VL_MODEL_PATH",
    os.path.join(_ROOT, "..", "..", "..", "models", "qwen3-vl-2b"),
)
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


def make_synthetic_batch(batch_size: int, seq_len: int, vocab_size: int, rng_key):
    k1, k2 = jax.random.split(rng_key)
    input_ids = jax.random.randint(k1, (batch_size, seq_len), 0, vocab_size)
    labels = jax.random.randint(k2, (batch_size, seq_len), 0, vocab_size)
    return input_ids, labels


# ---------------------------------------------------------------------------
# Single-device training (baseline)
# ---------------------------------------------------------------------------


def train_single_device(model, batch_size: int, num_steps: int = 10, warmup_steps: int = 3):
    print(f"\n{'=' * 60}")
    print(f"SINGLE-DEVICE TRAINING (batch_size={batch_size}, seq_len={SEQ_LEN})")
    print(f"{'=' * 60}")

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=1e-5, warmup_steps=warmup_steps, decay_steps=num_steps, end_value=1e-6
    )
    optimizer = nnx.Optimizer(model, make_optimizer(lr_schedule, kind="adamw_bf16"))

    @nnx.jit
    def train_step(model, optimizer, input_ids, labels):
        def loss_fn(m):
            return modeling.forward_train(m, input_ids, labels)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    rng = jax.random.PRNGKey(42)
    step_times, losses = [], []

    for step in range(num_steps):
        rng, step_rng = jax.random.split(rng)
        input_ids, labels = make_synthetic_batch(batch_size, SEQ_LEN, VOCAB_SIZE, step_rng)

        t0 = time.time()
        loss = train_step(model, optimizer, input_ids, labels)
        jax.block_until_ready(loss)
        elapsed = time.time() - t0

        loss_val = float(loss)
        losses.append(loss_val)

        if step < warmup_steps:
            print(f"  [warmup] step {step}: loss={loss_val:.4f}, time={elapsed:.3f}s")
        else:
            step_times.append(elapsed)
            tps = batch_size * SEQ_LEN / elapsed
            print(f"  step {step}: loss={loss_val:.4f}, time={elapsed:.3f}s, {tps:.0f} tok/s")

    if step_times:
        avg_t = np.mean(step_times)
        avg_tps = batch_size * SEQ_LEN / avg_t
        print(f"\n  >> avg step: {avg_t:.3f}s, {avg_tps:.0f} tok/s, loss: {losses[0]:.4f}->{losses[-1]:.4f}")
        return avg_tps, avg_t
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Multi-device training (4-chip data parallel)
# ---------------------------------------------------------------------------


def train_multi_device(model, batch_size_total: int, num_steps: int = 10, warmup_steps: int = 3):
    n_devices = jax.device_count()
    assert batch_size_total % n_devices == 0
    batch_per_device = batch_size_total // n_devices

    print(f"\n{'=' * 60}")
    print(f"MULTI-DEVICE TRAINING ({n_devices} TPUs, bs_total={batch_size_total}, bs/dev={batch_per_device})")
    print(f"{'=' * 60}")

    devices = jax.devices()
    mesh = Mesh(np.array(devices), axis_names=("dp",))
    data_sharding = NamedSharding(mesh, P("dp"))
    replicated = NamedSharding(mesh, P())

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=1e-5, warmup_steps=warmup_steps, decay_steps=num_steps, end_value=1e-6
    )
    optimizer = nnx.Optimizer(model, make_optimizer(lr_schedule, kind="sgd"))

    state = nnx.state(model)
    nnx.update(model, jax.device_put(state, replicated))
    opt_state = nnx.state(optimizer)
    nnx.update(optimizer, jax.device_put(opt_state, replicated))

    @nnx.jit
    def train_step(model, optimizer, input_ids, labels):
        def loss_fn(m):
            return modeling.forward_train(m, input_ids, labels)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    rng = jax.random.PRNGKey(123)
    step_times, losses = [], []

    for step in range(num_steps):
        rng, step_rng = jax.random.split(rng)
        input_ids, labels = make_synthetic_batch(batch_size_total, SEQ_LEN, VOCAB_SIZE, step_rng)
        input_ids = jax.device_put(input_ids, data_sharding)
        labels = jax.device_put(labels, data_sharding)

        t0 = time.time()
        loss = train_step(model, optimizer, input_ids, labels)
        jax.block_until_ready(loss)
        elapsed = time.time() - t0

        loss_val = float(loss)
        losses.append(loss_val)

        if step < warmup_steps:
            print(f"  [warmup] step {step}: loss={loss_val:.4f}, time={elapsed:.3f}s")
        else:
            step_times.append(elapsed)
            tps = batch_size_total * SEQ_LEN / elapsed
            print(f"  step {step}: loss={loss_val:.4f}, time={elapsed:.3f}s, {tps:.0f} tok/s")

    if step_times:
        avg_t = np.mean(step_times)
        avg_tps = batch_size_total * SEQ_LEN / avg_t
        print(f"\n  >> avg step: {avg_t:.3f}s, {avg_tps:.0f} tok/s, loss: {losses[0]:.4f}->{losses[-1]:.4f}")
        return avg_tps, avg_t
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
    print(f"JAX: {jax.__version__}, Devices: {jax.device_count()} x TPU v4")
    print(f"HBM/device: {jax.devices()[0].memory_stats()['bytes_limit'] / 1e9:.1f} GB")

    config = modeling.ModelConfig.qwen3vl_2b()
    results = {}

    if args.mode in ("single", "both"):
        model = modeling.Qwen3VLForConditionalGeneration.from_pretrained(MODEL_PATH, config=config)
        for bs in [1, 2, 4]:
            tps, avg_t = train_single_device(model, batch_size=bs, num_steps=8, warmup_steps=2)
            results[f"single_bs{bs}"] = {"tok/s": tps, "step_time": avg_t}
        del model
        gc.collect()

    if args.mode in ("multi", "both"):
        for bs_total in [4, 8, 16]:
            model = modeling.Qwen3VLForConditionalGeneration.from_pretrained(MODEL_PATH, config=config)
            tps, avg_t = train_multi_device(model, batch_size_total=bs_total, num_steps=8, warmup_steps=2)
            results[f"multi_bs{bs_total}"] = {"tok/s": tps, "step_time": avg_t}
            del model
            gc.collect()

    print(f"\n{'=' * 60}\nBENCHMARK SUMMARY\n{'=' * 60}")
    print(f"{'Config':<25} {'Throughput':>12} {'Step Time':>12} {'Speedup':>10}")
    print("-" * 60)
    baseline = results.get("single_bs1", {}).get("tok/s", 1.0) or 1.0
    for key, val in results.items():
        tps, st = val["tok/s"], val["step_time"]
        print(f"  {key:<23} {tps:>10.0f} t/s {st:>10.3f}s {tps / baseline:>8.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
