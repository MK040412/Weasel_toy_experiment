"""Offline evaluation: predict actions on held-out split, compute metrics.

Uses cached VLM embeddings for the eval split (auto-computes if missing).
Metrics: pos_error (3D), orn_error (3D), gripper_accuracy, total_mse.

Usage:
    PYTHONPATH=src python scripts/eval_offline.py --env calvin-abcd --split val
    PYTHONPATH=src python scripts/eval_offline.py --env calvin-abcd --split val --n-steps 10
    PYTHONPATH=src python scripts/eval_offline.py --env calvin-abcd --checkpoint result/vla_abcd/checkpoint_train_final.npz
"""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from qwen.qwen3vl import modeling as qwen3vl
from qwen.vla.config import PipelineConfig
from qwen.vla.data.protocol import create_dataset
from qwen.vla.models.vla import VLAPolicy
from qwen.vla.training.vlm_cache import VLMCacher


def _load_checkpoint(ckpt_path: str, policy: VLAPolicy):
    """Load checkpoint .npz into policy.obs_proj + action_expert."""
    data = np.load(ckpt_path)
    obs_leaves = jax.tree.leaves(nnx.state(policy.obs_proj))
    expert_leaves = jax.tree.leaves(nnx.state(policy.action_expert))
    n_obs = len(obs_leaves)

    # Load new values
    obs_graphdef = nnx.graphdef(policy.obs_proj)
    expert_graphdef = nnx.graphdef(policy.action_expert)
    obs_state = nnx.state(policy.obs_proj)
    expert_state = nnx.state(policy.action_expert)

    obs_flat, obs_treedef = jax.tree.flatten(obs_state)
    expert_flat, expert_treedef = jax.tree.flatten(expert_state)

    new_obs_flat = [jnp.array(data[f"p{i}"]) for i in range(n_obs)]
    new_expert_flat = [jnp.array(data[f"p{n_obs + i}"]) for i in range(len(expert_flat))]

    new_obs_state = jax.tree.unflatten(obs_treedef, new_obs_flat)
    new_expert_state = jax.tree.unflatten(expert_treedef, new_expert_flat)

    nnx.update(policy.obs_proj, new_obs_state)
    nnx.update(policy.action_expert, new_expert_state)

    return data.get("q01"), data.get("q99"), data.get("q01_state"), data.get("q99_state")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="calvin-abcd", choices=["calvin-debug", "calvin-abcd"])
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--checkpoint", default="result/vla_abcd/checkpoint_train_final.npz")
    parser.add_argument("--n-steps", type=int, default=10, help="Denoising steps")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit eval samples")
    parser.add_argument("--output-dir", default="result/vla_abcd")
    parser.add_argument("--model-path", default="/home/perelman/models/qwen3-vl-2b")
    args = parser.parse_args()

    t_total = time.time()

    if args.env == "calvin-abcd":
        cfg = PipelineConfig.calvin_abcd()
    else:
        cfg = PipelineConfig.calvin_debug()
    cfg.training.output_dir = args.output_dir

    print("=" * 60)
    print(f"Offline Eval — {cfg.env.name} [{args.split}]")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Denoising steps: {args.n_steps}")

    # Dataset
    ds = create_dataset(cfg.env, split=args.split)
    n = len(ds) if args.max_samples is None else min(args.max_samples, len(ds))
    print(f"Samples: {n} / {len(ds)}")

    # VLM cache for eval split (separate from train cache)
    eval_cache_dir = os.path.join(args.output_dir, f"vlm_cache_{args.split}")
    cacher = VLMCacher(eval_cache_dir.replace("vlm_cache_", "").replace(f"/{args.split}", ""))
    # Use dedicated directory
    cacher._cache_dir = eval_cache_dir
    cacher._cache_path = os.path.join(eval_cache_dir, "embeddings.parquet")
    cacher._meta_path = os.path.join(eval_cache_dir, "meta.json")

    # Create policy
    policy = VLAPolicy(
        vlm=None, vlm_hidden_dim=cfg.vlm.hidden_dim,
        action_expert_config={"action_dim": cfg.env.action_dim, "proprio_dim": cfg.env.proprio_dim},
        rngs=nnx.Rngs(params=42),
    )

    if cacher.exists():
        print(f"\nLoading {args.split} VLM cache...")
        cache = cacher.load()
    else:
        print(f"\n{args.split} VLM cache not found, computing...")
        model_config = qwen3vl.ModelConfig.qwen3vl_2b()
        vlm = qwen3vl.Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, config=model_config)
        policy_with_vlm = VLAPolicy(
            vlm=vlm, vlm_hidden_dim=cfg.vlm.hidden_dim,
            action_expert_config={"action_dim": cfg.env.action_dim, "proprio_dim": cfg.env.proprio_dim},
            rngs=nnx.Rngs(params=42),
        )
        # Limit dataset for cache if needed
        if args.max_samples:
            class _Subset:
                def __init__(self, ds, k):
                    self.ds = ds
                    self.k = k
                    self.chunk_size = ds.chunk_size
                    self.action_dim = ds.action_dim
                    self.proprio_dim = ds.proprio_dim
                    self.q01 = ds.q01
                    self.q99 = ds.q99
                    self.q01_state = ds.q01_state
                    self.q99_state = ds.q99_state

                def __len__(self):
                    return self.k

                def __getitem__(self, i):
                    return self.ds[i]

            cache_ds = _Subset(ds, n)
        else:
            cache_ds = ds
        cache = cacher.compute(cache_ds, vlm, policy_with_vlm.obs_proj, cfg.vlm.model_id, cfg.env.image_size)
        # Copy obs_proj weights from the cached policy before loading checkpoint
        policy.obs_proj = policy_with_vlm.obs_proj
        del vlm, policy_with_vlm
        jax.clear_caches()

    # Load trained checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    q01, q99, q01_state, q99_state = _load_checkpoint(args.checkpoint, policy)
    print(f"  q01={q01 is not None}, q99={q99 is not None}")

    # Evaluate
    print(f"\nEvaluating {n} samples (n_steps={args.n_steps})...")
    rng = jax.random.PRNGKey(0)

    pos_errs, orn_errs, grip_accs, action_mses = [], [], [], []

    t_eval = time.time()
    for i in range(n):
        rng, pred_rng = jax.random.split(rng)
        obs_embed = cache.obs[i : i + 1]
        proprio = cache.proprio[i : i + 1]

        acts_pred = policy.action_expert.denoise(
            obs_embed, proprio, chunk_size=cfg.env.chunk_size,
            n_steps=args.n_steps, rng=pred_rng,
        )
        pred = np.array(acts_pred[0])  # (T, 7) normalized
        gt = np.array(cache.actions[i])  # (T, 7) normalized

        # Denormalize using saved quantiles
        if q01 is not None and q99 is not None:
            pred_denorm = (pred + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
            gt_denorm = (gt + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        else:
            pred_denorm, gt_denorm = pred, gt

        # Metrics (in normalized space for MSE, raw for pos/orn)
        pos_err = np.sqrt(((pred_denorm[:, :3] - gt_denorm[:, :3]) ** 2).sum(axis=1)).mean()
        orn_err = np.sqrt(((pred_denorm[:, 3:6] - gt_denorm[:, 3:6]) ** 2).sum(axis=1)).mean()
        grip_acc = ((pred[:, 6] > 0) == (gt[:, 6] > 0)).mean()
        mse = ((pred - gt) ** 2).mean()

        pos_errs.append(pos_err)
        orn_errs.append(orn_err)
        grip_accs.append(grip_acc)
        action_mses.append(mse)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t_eval
            rate = (i + 1) / elapsed
            print(f"  {i + 1}/{n}: pos_err={np.mean(pos_errs):.4f}, grip_acc={np.mean(grip_accs):.3f} "
                  f"({rate:.0f}/s)")

    results = {
        "env": cfg.env.name,
        "split": args.split,
        "n_samples": n,
        "n_steps": args.n_steps,
        "checkpoint": args.checkpoint,
        "metrics": {
            "pos_error_mean": float(np.mean(pos_errs)),
            "pos_error_std": float(np.std(pos_errs)),
            "orn_error_mean": float(np.mean(orn_errs)),
            "orn_error_std": float(np.std(orn_errs)),
            "gripper_acc_mean": float(np.mean(grip_accs)),
            "action_mse_mean": float(np.mean(action_mses)),
        },
        "eval_time_s": time.time() - t_eval,
    }

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    m = results["metrics"]
    print(f"  pos_error:  {m['pos_error_mean']:.4f} ± {m['pos_error_std']:.4f}")
    print(f"  orn_error:  {m['orn_error_mean']:.4f} ± {m['orn_error_std']:.4f}")
    print(f"  grip_acc:   {m['gripper_acc_mean']:.3f}")
    print(f"  action_mse: {m['action_mse_mean']:.4f}")
    print(f"  eval time:  {results['eval_time_s']:.0f}s")

    out_path = os.path.join(args.output_dir, f"eval_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Total: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
