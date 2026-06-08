"""CPU unit tests for vision-encoder LoRA ("vit-lora") in the Fast-dVLM trainer.

Run:  JAX_PLATFORMS=cpu uv run python scripts/test_vit_lora.py

Validates:
  A) tiny ViT forward runs + finite (baseline)
  B) LoRA injection wraps the right Linears; lora_b zero-init => identical forward at step 0 (zero delta)
  C) trainable filter (ON) selects language params + visual-lora params, NOT other visual params;
     trainable filter (OFF) == original (no visual params)
  D) backward gives finite grads for lora + language params, no grad for frozen visual non-lora params,
     and at least one lora_b grad is nonzero (so the adapter actually trains)
  E) OFF-path structural identity: un-injected model has plain .kernel ViT linears; full forward finite

Exits nonzero on any failed assertion.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from flax import nnx

# Repo src on path (scripts/ -> ../src)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

from qwen.qwen3vl import modeling  # noqa: E402,I001
import train_fastdvlm_tpu as trainer  # noqa: E402,I001


def tiny_config() -> modeling.ModelConfig:
    vcfg = modeling.Qwen3VLVisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        in_channels=3,
        patch_size=4,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=48,
        num_position_embeddings=16,
        deepstack_visual_indexes=(0,),
    )
    tcfg = modeling.Qwen3VLTextConfig(
        vocab_size=64,
        hidden_size=48,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=24,
        tie_word_embeddings=True,
    )
    return modeling.ModelConfig(vision_config=vcfg, text_config=tcfg)


def build_model(cfg: modeling.ModelConfig, seed: int = 0) -> modeling.Qwen3VLForConditionalGeneration:
    return modeling.Qwen3VLForConditionalGeneration(cfg, rngs=nnx.Rngs(seed))


def make_pixels(cfg: modeling.ModelConfig, key) -> tuple[jax.Array, dict]:
    v = cfg.vision_config
    # grid must be divisible by spatial_merge_size on h,w; t=1.
    grid_t, grid_h, grid_w = 1, 4, 4
    seq_len = grid_t * grid_h * grid_w
    patch_dim = v.in_channels * v.temporal_patch_size * v.patch_size * v.patch_size
    pixels = jax.random.normal(key, (seq_len, patch_dim), dtype=jnp.float32)
    return pixels, {"grid_t": grid_t, "grid_h": grid_h, "grid_w": grid_w}


def vis_forward(model, pixels, grid):
    out, ds = model.model.visual.forward_static_with_deepstack(
        pixels, grid_h=grid["grid_h"], grid_w=grid["grid_w"], grid_t=grid["grid_t"]
    )
    return out, ds


def selected_paths(model, filt) -> set[str]:
    st = nnx.state(model, filt)
    return {jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(st)}


def check(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def main() -> int:
    key = jax.random.PRNGKey(0)
    cfg = tiny_config()

    # ---------------- A) baseline forward ----------------
    print("[A] baseline ViT forward")
    model = build_model(cfg, seed=0)
    pixels, grid = make_pixels(cfg, jax.random.fold_in(key, 1))
    out_before, ds_before = vis_forward(model, pixels, grid)
    check(bool(jnp.all(jnp.isfinite(out_before))), "baseline visual output finite")
    check(all(bool(jnp.all(jnp.isfinite(d))) for d in ds_before), "baseline deepstack finite")

    # ---------------- B) inject LoRA, zero delta at init ----------------
    print("[B] LoRA injection + zero-init delta")
    n_wrapped = modeling.inject_vit_lora(model, rank=4, alpha=8.0, targets=("attn", "mlp"), rngs=nnx.Rngs(123))
    check(n_wrapped == cfg.vision_config.depth * 4, f"wrapped depth*4 linears (got {n_wrapped})")
    # structural: a wrapped layer is LoRALinear with .base
    blk0 = model.model.visual.blocks[0]
    check(isinstance(blk0.attn.qkv, modeling.LoRALinear), "block.attn.qkv is LoRALinear")
    check(isinstance(blk0.mlp.linear_fc2, modeling.LoRALinear), "block.mlp.linear_fc2 is LoRALinear")
    # dtype-follows-base invariant (critical for the bf16 production ckpt + ZeRO uniform dtype):
    base_dt = blk0.attn.qkv.base.kernel.value.dtype
    check(blk0.attn.qkv.lora_a.value.dtype == base_dt and blk0.attn.qkv.lora_b.value.dtype == base_dt,
          f"lora_a/lora_b dtype follows base kernel dtype ({base_dt})")
    # scale = alpha / rank
    check(abs(blk0.attn.qkv.scale - (8.0 / 4)) < 1e-9, "LoRA scale == alpha/rank")
    out_after, ds_after = vis_forward(model, pixels, grid)
    check(bool(jnp.all(jnp.isfinite(out_after))), "lora-on visual output finite")
    check(bool(jnp.allclose(out_before, out_after, atol=1e-5, rtol=1e-5)),
          "lora-on output == baseline (lora_b zero-init => zero delta at step 0)")
    for d0, d1 in zip(ds_before, ds_after):
        check(bool(jnp.allclose(d0, d1, atol=1e-5, rtol=1e-5)), "deepstack identical at init")

    # ---------------- C) trainable filter selection ----------------
    print("[C] trainable filter selection (ON vs OFF)")
    filt_on = trainer.make_trainable_filter(True)
    sel_on = selected_paths(model, filt_on)
    check(len(sel_on) > 0, "ON filter selects >0 params")
    # every selected path is (not visual) OR (visual AND lora)
    bad = [p for p in sel_on if ("visual" in p and "lora" not in p)]
    check(not bad, f"ON filter selects no visual-non-lora params (offenders: {bad[:3]})")
    lora_sel = [p for p in sel_on if ("visual" in p and "lora" in p)]
    check(len(lora_sel) > 0, f"ON filter selects visual-lora params ({len(lora_sel)} of them)")
    lang_sel = [p for p in sel_on if "visual" not in p]
    check(len(lang_sel) > 0, f"ON filter selects language/head params ({len(lang_sel)} of them)")
    # explicitly: a known frozen visual base weight is NOT selected
    check(not any("base" in p and "kernel" in p and "visual" in p for p in sel_on),
          "ON filter does NOT select wrapped visual base kernels")

    # OFF filter == original constant selection (set equality)
    filt_off = trainer.make_trainable_filter(False)
    sel_off = selected_paths(model, filt_off)
    sel_orig = selected_paths(model, nnx.All(nnx.Param, nnx.Not(nnx.PathContains("visual"))))
    check(sel_off == sel_orig, "OFF filter selection == original (Not visual) selection")
    check(not any("visual" in p for p in sel_off), "OFF filter selects no visual params")
    # ON must be a strict superset of OFF (adds only lora visual params)
    check(sel_off.issubset(sel_on), "OFF selection is subset of ON selection")
    check(sel_on - sel_off == set(lora_sel), "ON adds exactly the visual-lora params over OFF")

    # ---------------- D) gradients ----------------
    print("[D] gradients: finite for lora+language, absent for frozen visual")
    # tiny token batch for the language path
    input_ids = jax.random.randint(jax.random.fold_in(key, 7), (1, 6), 0, cfg.text_config.vocab_size)

    def loss_fn(m):
        vout, _ = m.model.visual.forward_static_with_deepstack(
            pixels, grid_h=grid["grid_h"], grid_w=grid["grid_w"], grid_t=grid["grid_t"]
        )
        logits = m(input_ids)
        return jnp.sum(vout.astype(jnp.float32) ** 2) + jnp.sum(logits.astype(jnp.float32) ** 2)

    grads = nnx.grad(loss_fn, argnums=nnx.DiffState(0, filt_on))(model)
    gpaths = {jax.tree_util.keystr(p): v for p, v in jax.tree_util.tree_leaves_with_path(grads)}
    check(len(gpaths) == len(sel_on), f"grad leaves match selected params ({len(gpaths)} == {len(sel_on)})")
    # all selected grads finite
    check(all(bool(jnp.all(jnp.isfinite(v))) for v in gpaths.values()), "all selected grads finite")
    # no frozen visual non-lora param appears in grads
    frozen_vis = [p for p in gpaths if ("visual" in p and "lora" not in p)]
    check(not frozen_vis, f"no frozen visual-non-lora grad present (offenders: {frozen_vis[:3]})")
    # lora_b grads: at least one nonzero (adapter trains); lora_a grads finite (expected ~0 at init)
    lora_b_grads = [v for p, v in gpaths.items() if "lora_b" in p]
    lora_a_grads = [v for p, v in gpaths.items() if "lora_a" in p]
    check(len(lora_b_grads) > 0 and len(lora_a_grads) > 0, "lora_a and lora_b grads present")
    check(all(bool(jnp.all(jnp.isfinite(v))) for v in lora_a_grads), "lora_a grads finite")
    max_b = max(float(jnp.max(jnp.abs(v))) for v in lora_b_grads)
    check(max_b > 0.0, f"at least one lora_b grad nonzero (max|grad|={max_b:.3e})")
    # language grads finite & at least one nonzero
    lang_grads = [v for p, v in gpaths.items() if "visual" not in p]
    max_lang = max(float(jnp.max(jnp.abs(v))) for v in lang_grads)
    check(max_lang > 0.0, f"language grads nonzero (max|grad|={max_lang:.3e})")

    # ---------------- E) OFF-path structural identity ----------------
    print("[E] OFF-path: un-injected model structurally unchanged")
    model_off = build_model(cfg, seed=0)  # same seed, NOT injected
    blk0_off = model_off.model.visual.blocks[0]
    check(isinstance(blk0_off.attn.qkv, nnx.Linear) and not isinstance(blk0_off.attn.qkv, modeling.LoRALinear),
          "un-injected block.attn.qkv is plain nnx.Linear (no LoRA)")
    paths_off = {jax.tree_util.keystr(p) for p, _ in
                 jax.tree_util.tree_leaves_with_path(nnx.state(model_off))}
    check(any(("'attn'" in p and "'qkv'" in p and "'kernel'" in p and "'base'" not in p) for p in paths_off),
          "un-injected ViT has plain attn.qkv.kernel path (no .base nesting)")
    check(not any("lora" in p for p in paths_off), "un-injected model has no lora params")
    sel_off2 = selected_paths(model_off, trainer.make_trainable_filter(False))
    check(not any("visual" in p for p in sel_off2), "OFF filter on un-injected model selects no visual params")
    logits_off = model_off(input_ids)
    check(bool(jnp.all(jnp.isfinite(logits_off))), "un-injected full forward finite")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
