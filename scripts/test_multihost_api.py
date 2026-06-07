#!/usr/bin/env python3
"""CPU validation of the JAX multi-host primitives used by train_fastdvlm_tpu.py --multihost.

Forces 4 CPU 'devices' (1 process) to exercise the exact APIs:
  - to_global_array  -> jax.make_array_from_process_local_data (data-parallel batch assembly)
  - _replicate_global -> jax.make_array_from_callback (replicated params), incl. jax.tree.map over a pytree
This cannot test true cross-host collectives (that needs a real pod) but it catches API misuse
(wrong shapes, non-replicated params, sharding errors) before paying for TPU.

Run:  XLA_FLAGS="--xla_force_host_platform_device_count=4" JAX_PLATFORMS=cpu uv run python scripts/test_multihost_api.py
"""
import numpy as np
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def to_global_array(local, sharding, global_batch):
    """Mirror of train_fastdvlm_tpu.to_global_array."""
    if global_batch is not None and sharding is not None:
        gshape = (int(global_batch),) + tuple(local.shape[1:])
        return jax.make_array_from_process_local_data(sharding, np.asarray(local), gshape)
    return jax.device_put(local, sharding) if sharding is not None else jax.numpy.asarray(local)


def main():
    assert jax.device_count() >= 2, "run with XLA_FLAGS=--xla_force_host_platform_device_count=4"
    mesh = Mesh(np.asarray(jax.devices()), axis_names=("dp",))
    data_sharding = NamedSharding(mesh, P("dp"))
    replicated = NamedSharding(mesh, P())

    gb = 8
    local = np.arange(gb * 3, dtype=np.float32).reshape(gb, 3)  # 1 process => local == global
    g = to_global_array(local, data_sharding, gb)
    assert g.shape == (gb, 3) and np.allclose(np.asarray(g), local)
    assert g.sharding.spec == P("dp")
    print("[ok] data: make_array_from_process_local_data assembles a dp-sharded global batch")

    s = to_global_array(local, data_sharding, None)  # single-host path
    assert np.allclose(np.asarray(s), local)
    print("[ok] single-host device_put path intact")

    def _replicate_global(x):
        hl = np.asarray(x)
        return jax.make_array_from_callback(hl.shape, replicated, lambda idx: hl[idx])

    state = {"a": np.ones((2, 2), np.float32), "b": np.zeros((3,), np.float32)}
    rep = jax.tree.map(_replicate_global, state)
    assert rep["a"].sharding.is_fully_replicated and rep["b"].sharding.is_fully_replicated
    assert rep["a"].shape == (2, 2) and rep["b"].shape == (3,)
    print("[ok] params: make_array_from_callback yields fully-replicated arrays over jax.tree.map")
    print("\nMULTIHOST-API CHECKS PASS")


if __name__ == "__main__":
    main()
