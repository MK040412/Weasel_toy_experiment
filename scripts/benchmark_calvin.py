"""CALVIN Benchmark with batched TPU inference + parallel CPU sim envs.

Architecture:
  - Main process: JAX policy on TPU (batched inference)
  - N CPU sim envs in same process (one at a time, but batched inference)
  - Batched inference: gather obs from all envs → batch through TPU → scatter actions

Usage:
    bash commands/benchmark.sh --num-sequences 20 --num-envs 8 --save-videos 3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
os.environ.pop("DISPLAY", None)

# Project paths
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, "/home/perelman/calvin/calvin_env")
sys.path.insert(0, "/home/perelman/calvin/calvin_models")

import hydra  # noqa: E402
import imageio  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from calvin_agent.evaluation.multistep_sequences import get_sequences  # noqa: E402
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition  # noqa: E402
from flax import nnx  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

IMAGE_TOKEN_ID = 151655
VISION_START_ID = 151652
VISION_END_ID = 151653


def load_policy(ckpt_path, model_path):
    from qwen.qwen3vl import modeling as qwen3vl
    from qwen.vla.models.vla import VLAPolicy

    print(f"Loading Qwen3-VL from {model_path}...")
    mcfg = qwen3vl.ModelConfig.qwen3vl_2b()
    vlm = qwen3vl.Qwen3VLForConditionalGeneration.from_pretrained(model_path, config=mcfg)

    policy = VLAPolicy(
        vlm=vlm,
        vlm_hidden_dim=2048,
        action_expert_config={"action_dim": 7, "proprio_dim": 15},
        rngs=nnx.Rngs(params=42),
    )

    print(f"Loading checkpoint {ckpt_path}...")
    data = np.load(ckpt_path)
    obs_flat, obs_tree = jax.tree.flatten(nnx.state(policy.obs_proj))
    expert_flat, expert_tree = jax.tree.flatten(nnx.state(policy.action_expert))
    n_obs = len(obs_flat)

    new_obs = [jnp.array(data[f"p{i}"]) for i in range(n_obs)]
    new_expert = [jnp.array(data[f"p{n_obs + i}"]) for i in range(len(expert_flat))]
    nnx.update(policy.obs_proj, jax.tree.unflatten(obs_tree, new_obs))
    nnx.update(policy.action_expert, jax.tree.unflatten(expert_tree, new_expert))

    q = {"q01": data["q01"], "q99": data["q99"], "q01_state": data["q01_state"], "q99_state": data["q99_state"]}
    return policy, q


def prepare_vision_input(image_rgb, text_tokens, image_size=320):
    """Resize + patch extract. Returns numpy arrays."""
    if image_rgb.shape[0] != image_size or image_rgb.shape[1] != image_size:
        pil = PILImage.fromarray(image_rgb.astype(np.uint8))
        pil = pil.resize((image_size, image_size), PILImage.BILINEAR)
        image = np.array(pil, dtype=np.float32) / 255.0
    else:
        image = image_rgb.astype(np.float32) / 255.0

    patch_size, tp, ms = 16, 2, 2
    grid_h = image_size // patch_size
    n_vt = (grid_h // ms) ** 2

    input_ids = np.array(
        [[VISION_START_ID] + [IMAGE_TOKEN_ID] * n_vt + [VISION_END_ID] + text_tokens],
        dtype=np.int32,
    )

    img_d = np.stack([image, image], axis=0)
    patches = []
    for h in range(0, image_size, patch_size):
        for w in range(0, image_size, patch_size):
            patch = img_d[:tp, h : h + patch_size, w : w + patch_size, :]
            patches.append(patch.transpose(3, 0, 1, 2).flatten())

    return {
        "input_ids": input_ids,
        "pixel_values": np.array(patches, dtype=np.float32),
        "token_type_ids": (input_ids == IMAGE_TOKEN_ID).astype(np.int32),
        "grid_thw": np.array([[1, grid_h, grid_h]], dtype=np.int32),
    }


def batched_encode(policy, vlm_inputs_list, config):
    """Batch N samples through VLM on TPU. Returns obs_embed (N, seq, 1536)."""
    from qwen.qwen3vl import modeling as qwen3vl

    n = len(vlm_inputs_list)
    # Pad to same seq length
    max_seq = max(inp["input_ids"].shape[1] for inp in vlm_inputs_list)

    input_ids = np.concatenate(
        [np.pad(inp["input_ids"], ((0, 0), (0, max_seq - inp["input_ids"].shape[1]))) for inp in vlm_inputs_list],
        axis=0,
    )
    token_type_ids = np.concatenate(
        [
            np.pad(inp["token_type_ids"], ((0, 0), (0, max_seq - inp["token_type_ids"].shape[1])))
            for inp in vlm_inputs_list
        ],
        axis=0,
    )

    # Each image encoded separately (vision can't batch due to int() issue)
    vision_embeds = []
    grid_h = 20
    for inp in vlm_inputs_list:
        pv = jnp.array(inp["pixel_values"])
        ve = policy.vlm.model.visual.forward_static(pv, grid_h=grid_h, grid_w=grid_h, grid_t=1)
        vision_embeds.append(ve)
    batch_ve = jnp.stack(vision_embeds)  # (N, n_patches, 2048)

    # Batched language
    batch_ids = jnp.array(input_ids)
    batch_tt = jnp.array(token_type_ids)
    positions = jnp.broadcast_to(jnp.arange(max_seq)[None, :], (n, max_seq))
    sin, cos = qwen3vl._generate_rope(positions, config.text_config.head_dim, config.text_config.rope_theta)
    mask = qwen3vl.make_train_causal_mask(max_seq)

    inputs_embeds = policy.vlm.model.language_model.embed_tokens(batch_ids)
    inputs_embeds = qwen3vl.batched_merge_modalities(batch_ve, inputs_embeds, batch_tt)
    hidden = policy.vlm.model.language_model(inputs_embeds, None, sin, cos, mask)
    obs_embed = policy.obs_proj(hidden)
    return obs_embed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="result/vla_abcd/checkpoint_train_final.npz")
    parser.add_argument("--num-sequences", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=8, help="Parallel CALVIN sim envs")
    parser.add_argument("--ep-len", type=int, default=360)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--save-videos", type=int, default=3)
    parser.add_argument("--output-dir", default="result/vla_abcd/benchmark")
    parser.add_argument("--calvin-dir", default=os.environ.get("CALVIN_DIR", "/home/perelman/calvin"))
    parser.add_argument(
        "--model-path",
        default=os.environ.get("QWEN3VL_MODEL_PATH", "/home/perelman/models/qwen3-vl-2b"),
    )
    args = parser.parse_args()

    t_total = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"JAX devices: {jax.devices()}")

    # ── Load env configs ──
    config_dir = str(Path(args.calvin_dir) / "calvin_env" / "conf")
    with initialize_config_dir(config_dir=config_dir):
        env_cfg = compose(
            config_name="config_data_collection.yaml",
            overrides=["cameras=static_and_gripper", "use_vr=False"],
        )
    env_cfg.env.use_egl = False
    env_cfg.env.show_gui = False
    env_cfg.env.use_vr = False
    env_cfg.env.use_scene_info = True

    # Task oracle + language annotations
    conf_dir = Path(args.calvin_dir) / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    # ── Create N parallel envs ──
    n_envs = args.num_envs
    print(f"Creating {n_envs} CALVIN envs...")
    envs = []
    for i in range(n_envs):
        e = hydra.utils.instantiate(env_cfg.env)
        envs.append(e)
    print(f"  {n_envs} envs ready")

    # ── Load policy ──
    policy, q = load_policy(args.checkpoint, args.model_path)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    vlm_config = policy.vlm.config

    # ── Get sequences ──
    eval_sequences = get_sequences(args.num_sequences)
    print(f"Evaluating {len(eval_sequences)} sequences with {n_envs}-way parallelism...")

    # Results tracking
    success_counts = []
    sequence_results = []
    success_videos = []
    failure_videos = []

    # Process sequences in groups of n_envs
    seq_idx = 0
    while seq_idx < len(eval_sequences):
        batch = eval_sequences[seq_idx : seq_idx + n_envs]
        actual_n = len(batch)

        # Reset envs for this batch
        for i, (init_state, _) in enumerate(batch):
            robot_obs, scene_obs = get_env_state_for_initial_condition(init_state)
            envs[i].reset(robot_obs=robot_obs, scene_obs=scene_obs)

        # Track per-env progress through their 5-subtask chain
        env_completed = [0] * actual_n
        env_failed = [False] * actual_n
        env_frames = [[] for _ in range(actual_n)]  # video frames per env (current subtask)

        for subtask_idx in range(5):
            active = [i for i in range(actual_n) if not env_failed[i]]
            if not active:
                break

            # Setup: current subtask + start info for each active env
            subtasks = {i: batch[i][1][subtask_idx] for i in active}
            langs = {i: val_annotations[subtasks[i]][0] for i in active}
            text_tokens_list = {i: tokenizer.encode(langs[i], add_special_tokens=False) for i in active}
            start_infos = {i: envs[i].get_info() for i in active}

            # Clear frames for this subtask
            for i in active:
                env_frames[i] = []

            # Rollout with action chunking (replan every chunk_size steps)
            env_action_buffer = {i: [] for i in active}
            rng = jax.random.PRNGKey(seq_idx * 1000 + subtask_idx)

            step = 0
            while step < args.ep_len and active:
                # Gather envs that need new action chunk
                need_replan = [i for i in active if len(env_action_buffer[i]) == 0]

                if need_replan:
                    # Batched VLM encode
                    vlm_inputs = []
                    proprios = []
                    for i in need_replan:
                        obs = envs[i].get_obs()
                        vlm_inp = prepare_vision_input(obs["rgb_obs"]["rgb_static"], text_tokens_list[i])
                        vlm_inputs.append(vlm_inp)

                        robot_obs = obs["robot_obs"]
                        proprio_norm = (robot_obs - q["q01_state"]) / (
                            q["q99_state"] - q["q01_state"] + 1e-6
                        ) * 2.0 - 1.0
                        proprios.append(proprio_norm.astype(np.float32))

                    obs_embed = batched_encode(policy, vlm_inputs, vlm_config)
                    proprio_batch = jnp.array(np.stack(proprios))[:, None, :]  # (N, 1, 15)

                    rng, pred_rng = jax.random.split(rng)
                    acts_norm = policy.action_expert.denoise(
                        obs_embed,
                        proprio_batch,
                        chunk_size=args.chunk_size,
                        n_steps=args.n_steps,
                        rng=pred_rng,
                    )
                    acts_norm = np.array(acts_norm)  # (N, 50, 7)
                    acts = (acts_norm + 1.0) / 2.0 * (q["q99"] - q["q01"] + 1e-6) + q["q01"]

                    for k, i in enumerate(need_replan):
                        env_action_buffer[i] = list(acts[k])

                # Step each active env
                for i in list(active):
                    action = env_action_buffer[i].pop(0).copy()
                    action[6] = 1.0 if action[6] > 0 else -1.0
                    obs, _, _, info = envs[i].step(action)
                    env_frames[i].append(obs["rgb_obs"]["rgb_static"])

                    # Check completion
                    task_info = task_oracle.get_task_info_for_set(start_infos[i], info, {subtasks[i]})
                    if len(task_info) > 0:
                        env_completed[i] = subtask_idx + 1
                        # Save success video if quota remains
                        if len(success_videos) < args.save_videos:
                            success_videos.append(
                                (seq_idx + i, subtask_idx, subtasks[i], langs[i], list(env_frames[i]))
                            )
                        active.remove(i)

                step += 1

            # Envs that didn't succeed → failed
            for i in active:
                env_failed[i] = True
                if len(failure_videos) < args.save_videos:
                    failure_videos.append((seq_idx + i, subtask_idx, subtasks[i], langs[i], list(env_frames[i])))

        # Record batch results
        for i in range(actual_n):
            success_counts.append(env_completed[i])
            sequence_results.append({"seq_idx": seq_idx + i, "completed": env_completed[i]})

        # Progress
        n_done = len(success_counts)
        avg = np.mean(success_counts)
        rates = [sum(1 for c in success_counts if c > j) / n_done for j in range(5)]
        elapsed = time.time() - t_total
        print(
            f"  [{n_done}/{len(eval_sequences)} | {elapsed:.0f}s] "
            + " ".join(f"{j + 1}={r * 100:.0f}%" for j, r in enumerate(rates))
            + f" avg={avg:.2f}"
        )

        seq_idx += actual_n

    # ── Summary ──
    rates = [sum(1 for c in success_counts if c > i) / len(success_counts) for i in range(5)]
    avg_len = float(np.mean(success_counts))

    print("\n" + "=" * 60)
    print("CALVIN Benchmark Results")
    print("=" * 60)
    for i, r in enumerate(rates):
        print(f"  {i + 1}/5: {r * 100:.1f}%")
    print(f"  Avg chain length: {avg_len:.2f} / 5")
    print(f"  Total time: {time.time() - t_total:.0f}s")

    results = {
        "checkpoint": args.checkpoint,
        "num_sequences": len(eval_sequences),
        "num_envs": n_envs,
        "ep_len": args.ep_len,
        "chunk_size": args.chunk_size,
        "n_steps": args.n_steps,
        "success_rates": {f"{i + 1}/5": rates[i] for i in range(5)},
        "avg_chain_length": avg_len,
        "sequences": sequence_results,
        "total_time_s": time.time() - t_total,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output_dir}/results.json")

    for tag, vids in [("success", success_videos), ("failure", failure_videos)]:
        for i, (seq_i, st_i, subtask, lang, frames) in enumerate(vids):
            out_path = os.path.join(args.output_dir, f"{tag}_{i:02d}_seq{seq_i:03d}_{subtask}.mp4")
            imageio.mimwrite(out_path, frames, fps=30, quality=8, macro_block_size=1)
            print(f"  {tag}: {out_path} ({len(frames)}f)")

    for e in envs:
        e.close()


if __name__ == "__main__":
    main()
