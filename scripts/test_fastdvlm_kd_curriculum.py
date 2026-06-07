#!/usr/bin/env python3
"""CPU unit tests for the kd_fewstep loss term and the degree-2 block-size curriculum.

Runs WITHOUT TPU/jax/flax: the pure curriculum function `degree2_bd_probs` is extracted
from `train_fastdvlm_tpu.py` via AST so we test the real source, and the host-side
`lambda_fs` schedule + the pair-0 KD reduction are mirrored in numpy to lock the contract.

Usage:  python3 scripts/test_fastdvlm_kd_curriculum.py
"""
import ast
import pathlib

import numpy as np

TRAINER = pathlib.Path(__file__).with_name("train_fastdvlm_tpu.py")


def _load_func(name: str):
    """Extract a single top-level function from the trainer and exec it with numpy only."""
    tree = ast.parse(TRAINER.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            mod = ast.Module(body=[node], type_ignores=[])
            ns: dict = {"np": np}
            exec(compile(mod, str(TRAINER), "exec"), ns)  # noqa: S102 - trusted local source
            return ns[name]
    raise AssertionError(f"function {name} not found in {TRAINER}")


degree2_bd_probs = _load_func("degree2_bd_probs")
compute_response_block_idx = _load_func("compute_response_block_idx")
asymmetric_allowed = _load_func("asymmetric_allowed")


def test_degree2_reduces_to_power_law():
    """lambda2 = 0  =>  P(b) ∝ b^{-lambda1} (Boltzmann power law)."""
    bd = [1, 2, 4, 8, 16, 32]
    for l1 in (0.0, 0.5, 1.0, 1.5, 2.0):
        got = degree2_bd_probs(bd, l1, 0.0)
        ref = np.asarray(bd, dtype=np.float64) ** (-l1)
        ref = ref / ref.sum()
        assert np.allclose(got, ref, atol=1e-12), (l1, got, ref)
    print("[ok] degree2 lambda2=0 == power law")


def test_degree2_is_a_distribution():
    bd = [1, 2, 4, 8, 16, 32]
    for l1, l2 in [(1.0, 0.3), (-0.5, 0.5), (0.0, 0.0), (2.0, 1.0)]:
        p = degree2_bd_probs(bd, l1, l2)
        assert np.all(p > 0) and abs(p.sum() - 1.0) < 1e-12, (l1, l2, p)
    print("[ok] degree2 is a valid distribution (positive, sums to 1)")


def test_degree2_lambda1_shifts_mass():
    """Larger lambda1 => more mass on small blocks; annealing lambda1 down => mass to large blocks."""
    bd = [1, 2, 4, 8, 16, 32]
    hi = degree2_bd_probs(bd, 1.8, 0.3)   # favors small b
    lo = degree2_bd_probs(bd, -0.5, 0.3)  # favors large b
    # P(b=1) should drop and P(b=32) should rise as lambda1 decreases.
    assert hi[0] > lo[0], (hi[0], lo[0])
    assert lo[-1] > hi[-1], (hi[-1], lo[-1])
    print("[ok] degree2 lambda1 anneal shifts mass small->large")


def lambda_fs(step, step_bd, *, weight=0.25, bd_ref=4.0, bd_cap=4.0, warmup=500):
    """Exact mirror of the host-side schedule in the dispatch loop."""
    ramp = min((step + 1) / max(int(warmup), 1), 1.0)
    bd_factor = min(float(step_bd) / max(float(bd_ref), 1e-9), float(bd_cap))
    return float(weight) * ramp * bd_factor


def test_lambda_fs_b16_conservative():
    """Post-warmup, b16-conservative cap (bd_ref=4, bd_cap=4, lambda0=0.25):
       b4:0.25, b8:0.5, b16:1.0, b32:1.0 (capped)."""
    s = 10_000  # well past warmup
    assert abs(lambda_fs(s, 4) - 0.25) < 1e-9
    assert abs(lambda_fs(s, 8) - 0.50) < 1e-9
    assert abs(lambda_fs(s, 16) - 1.00) < 1e-9
    assert abs(lambda_fs(s, 32) - 1.00) < 1e-9, "b32 must be capped to b16 level"
    assert abs(lambda_fs(s, 1) - 0.0625) < 1e-9
    print("[ok] lambda_fs b16-conservative: b4=.25 b8=.5 b16=1 b32=1(cap)")


def test_lambda_fs_warmup():
    # near 0 at the very first step, full strength after warmup
    assert lambda_fs(0, 16) < 0.01
    assert abs(lambda_fs(499, 16) - 1.0) < 1e-9   # step+1 == warmup -> full
    assert abs(lambda_fs(10_000, 16) - 1.0) < 1e-9
    # off when weight=0
    assert lambda_fs(10_000, 32, weight=0.0) == 0.0
    print("[ok] lambda_fs warmup ramps 0->full; weight=0 disables")


def test_pair0_reduction():
    """Mirror the jnp pair-0 isolation: token_kl/(gb*pb,cap) -> (gb,pb,cap)[:,0,:] masked-avg.
       Confirms pair index 0 (mask_idx, heavy) is selected and weight=0 contributes nothing."""
    gb, pb, cap = 2, 2, 3
    rng = np.random.default_rng(0)
    token_kl = rng.random((gb * pb, cap))
    weights = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.float64)  # (gb*pb, cap)
    tk = token_kl.reshape(gb, pb, cap)
    w = weights.reshape(gb, pb, cap)
    fs_kl, fs_w = tk[:, 0, :], w[:, 0, :]
    kd_fewstep = (fs_kl * fs_w).sum() / max(fs_w.sum(), 1.0)
    # manual: only rows 0 and 2 (pair-0 of each group)
    manual_num = (token_kl[0] * weights[0]).sum() + (token_kl[2] * weights[2]).sum()
    manual_den = weights[0].sum() + weights[2].sum()
    assert abs(kd_fewstep - manual_num / manual_den) < 1e-12
    # weight=0 => zero contribution to the loss, finite kd_fewstep (no div0)
    assert np.isfinite(kd_fewstep)
    assert 0.0 * kd_fewstep == 0.0
    # all-zero-weight group reduces to 0 (clamped denom), not NaN
    z = (np.zeros(cap) * np.zeros(cap)).sum() / max(0.0, 1.0)
    assert z == 0.0
    print("[ok] pair-0 reduction selects mask_idx view; weight=0 => 0; no div0")


def test_multiturn_block_index():
    """Episode-packing relies on compute_response_block_idx assigning increasing block ids across
    MULTIPLE response segments (turns) in one sequence. Build a 2-turn labels vector and check."""
    C = -100
    # [ctx ctx] [turn-A resp x5] [ctx] [turn-B resp x3]
    labels = np.array([C, C, 10, 11, 12, 13, 14, C, 20, 21, 22], dtype=np.int32)
    block_idx, turn_idx, n_blocks = compute_response_block_idx(labels, block_size=4)
    resp = labels != -100
    assert np.all(block_idx[~resp] == -1), "context tokens must have block -1"
    assert np.all(block_idx[resp] >= 0), "response tokens must have block >= 0"
    a_blocks = block_idx[2:7]   # turn A
    b_blocks = block_idx[8:11]  # turn B
    assert a_blocks.min() == 0 and a_blocks.max() == 1, a_blocks  # ceil(5/4)=2 blocks: 0,0,0,0,1
    assert b_blocks.min() > a_blocks.max(), (a_blocks, b_blocks)  # turn B blocks strictly after turn A
    assert turn_idx[10] > turn_idx[2], "turn_idx must increase from turn A to turn B"
    assert n_blocks >= 3
    print("[ok] multi-turn block_idx increases across response segments (episode packing)")


def test_cross_turn_attention():
    """asymmetric_allowed must give: within-turn bidirectional (noisy-noisy same turn),
    cross-turn causal (noisy turn t -> clean turn t' only if t>t'), and clean AR causal."""
    n_noisy = 4
    turn_noisy = np.array([0, 0, 1, 1], dtype=np.int32)  # 2 turns, 2 noisy tokens each
    turn_clean = np.array([0, 0, 1, 1], dtype=np.int32)
    total = n_noisy + turn_clean.shape[0]  # 8
    q = np.arange(total, dtype=np.int32)[:, None]
    kv = np.arange(total, dtype=np.int32)[None, :]
    A = asymmetric_allowed(q, kv, turn_noisy, turn_clean, n_noisy)
    # noisy(turn0) <-> noisy(turn0): bidirectional within turn
    assert A[1, 0] and A[0, 1], "within-turn noisy must be bidirectional"
    # noisy(turn1 q=2) -/- noisy(turn0 kv=0): different turns, noisy-noisy is blocked
    assert not A[2, 0], "cross-turn noisy-noisy must be blocked"
    # noisy(turn1 q=2) -> clean(turn0 kv=4): allowed (attend earlier clean context)
    assert A[2, 4], "noisy must attend earlier-turn clean context"
    # noisy(turn0 q=0) -/- clean(turn1 kv=6): cannot attend FUTURE clean turn
    assert not A[0, 6], "noisy must not attend future-turn clean"
    # clean AR causal: clean q=5(pos1) -> clean kv=4(pos0) yes; reverse no
    assert A[5, 4] and not A[4, 5], "clean branch must be AR causal"
    print("[ok] cross-turn attention: in-turn bidir, noisy->past-clean, clean AR causal")


if __name__ == "__main__":
    test_degree2_reduces_to_power_law()
    test_degree2_is_a_distribution()
    test_degree2_lambda1_shifts_mass()
    test_lambda_fs_b16_conservative()
    test_lambda_fs_warmup()
    test_pair0_reduction()
    test_multiturn_block_index()
    test_cross_turn_attention()
    print("\nALL PASS")
