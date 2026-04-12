"""CALVIN Benchmark with multiprocessing parallel sim envs.

Architecture:
  - Main process: JAX policy on TPU + batched inference
  - N worker processes: each runs a CALVIN sim env (pybullet in separate process)
  - IPC: multiprocessing Queue for obs↔action exchange
  - Main gathers obs from all workers → batch TPU inference → scatter actions

Usage:
    bash commands/benchmark.sh calvin-abcd-flower --num-sequences 100 --num-workers 16
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
os.environ.pop("DISPLAY", None)

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "src"))
_CALVIN_DIR = os.environ.get("CALVIN_DIR", "/home/perelman/calvin")
sys.path.insert(0, f"{_CALVIN_DIR}/calvin_env")
sys.path.insert(0, f"{_CALVIN_DIR}/calvin_models")


def sim_worker(worker_id, cmd_q, res_q, calvin_dir):
    """Worker process: runs a CALVIN env, handles reset/step/get_info/get_obs commands."""
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ.pop("DISPLAY", None)

    import hydra
    from hydra import compose, initialize_config_dir
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition

    config_dir = str(Path(calvin_dir) / "calvin_env" / "conf")
    with initialize_config_dir(config_dir=config_dir):
        env_cfg = compose(
            config_name="config_data_collection.yaml",
            overrides=["cameras=static_and_gripper", "use_vr=False"],
        )
    env_cfg.env.use_egl = False
    env_cfg.env.show_gui = False
    env_cfg.env.use_vr = False
    env_cfg.env.use_scene_info = True

    env = hydra.utils.instantiate(env_cfg.env)
    res_q.put(("ready", worker_id))

    while True:
        cmd, data = cmd_q.get()
        if cmd == "stop":
            env.close()
            return
        elif cmd == "reset":
            init_state = data
            robot_obs, scene_obs = get_env_state_for_initial_condition(init_state)
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            res_q.put(("reset_done", worker_id))
        elif cmd == "get_obs":
            obs = env.get_obs()
            res_q.put(("obs", worker_id, {
                "rgb_static": obs["rgb_obs"]["rgb_static"],
                "rgb_gripper": obs["rgb_obs"]["rgb_gripper"],
                "robot_obs": obs["robot_obs"],
            }))
        elif cmd == "get_info":
            res_q.put(("info", worker_id, env.get_info()))
        elif cmd == "step":
            obs, _, _, info = env.step(data)
            res_q.put(("step_done", worker_id, {
                "rgb_static": obs["rgb_obs"]["rgb_static"],
                "rgb_gripper": obs["rgb_obs"]["rgb_gripper"],
                "robot_obs": obs["robot_obs"],
                "info": info,
            }))


def load_policy(ckpt_path, model_path, proprio_dim=8, chunk_size=10):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from qwen.qwen3vl import modeling as qwen3vl
    from qwen.vla.models.vla import VLAPolicy

    print(f"Loading Qwen3-VL from {model_path}...")
    mcfg = qwen3vl.ModelConfig.qwen3vl_2b()
    vlm = qwen3vl.Qwen3VLForConditionalGeneration.from_pretrained(model_path, config=mcfg)

    policy = VLAPolicy(
        vlm=vlm, vlm_hidden_dim=2048,
        action_expert_config={"action_dim": 7, "proprio_dim": proprio_dim},
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

    return policy, {
        "q01": data["q01"], "q99": data["q99"],
        "q01_state": data["q01_state"], "q99_state": data["q99_state"],
    }


def compose_image(rgb_static, rgb_gripper, image_size=320):
    """Vstack top (upper half) + wrist (lower half) → (image_size, image_size, 3) float32."""
    import numpy as np
    from PIL import Image as PILImage

    half = image_size // 2
    top = PILImage.fromarray(rgb_static.astype(np.uint8)).resize((image_size, half), PILImage.BILINEAR)
    wrist = PILImage.fromarray(rgb_gripper.astype(np.uint8)).resize((image_size, half), PILImage.BILINEAR)
    return np.vstack([
        np.array(top, dtype=np.float32) / 255.0,
        np.array(wrist, dtype=np.float32) / 255.0,
    ])


def prepare_vision(image, text_tokens, image_size=320):
    import jax.numpy as jnp
    import numpy as np

    IMAGE_TOKEN_ID = 151655
    VISION_START_ID = 151652
    VISION_END_ID = 151653

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
    }


def batched_encode(policy, vlm_inputs_list, config):
    import jax.numpy as jnp
    import numpy as np
    from qwen.qwen3vl import modeling as qwen3vl

    n = len(vlm_inputs_list)
    max_seq = max(inp["input_ids"].shape[1] for inp in vlm_inputs_list)

    input_ids = np.concatenate([
        np.pad(inp["input_ids"], ((0, 0), (0, max_seq - inp["input_ids"].shape[1])))
        for inp in vlm_inputs_list
    ], axis=0)
    tt_ids = np.concatenate([
        np.pad(inp["token_type_ids"], ((0, 0), (0, max_seq - inp["token_type_ids"].shape[1])))
        for inp in vlm_inputs_list
    ], axis=0)

    vision_embeds = []
    grid_h = 20
    for inp in vlm_inputs_list:
        pv = jnp.array(inp["pixel_values"])
        ve = policy.vlm.model.visual.forward_static(pv, grid_h=grid_h, grid_w=grid_h, grid_t=1)
        vision_embeds.append(ve)
    batch_ve = jnp.stack(vision_embeds)

    batch_ids = jnp.array(input_ids)
    batch_tt = jnp.array(tt_ids)
    positions = jnp.broadcast_to(jnp.arange(max_seq)[None, :], (n, max_seq))
    sin, cos = qwen3vl._generate_rope(positions, config.text_config.head_dim, config.text_config.rope_theta)
    mask = qwen3vl.make_train_causal_mask(max_seq)
    inputs_embeds = policy.vlm.model.language_model.embed_tokens(batch_ids)
    inputs_embeds = qwen3vl.batched_merge_modalities(batch_ve, inputs_embeds, batch_tt)
    hidden = policy.vlm.model.language_model(inputs_embeds, None, sin, cos, mask)
    return policy.obs_proj(hidden)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-sequences", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--ep-len", type=int, default=360)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--proprio-dim", type=int, default=8)
    parser.add_argument("--execute-horizon", type=int, default=0, help="Execute first N of chunk before replan; 0=full chunk")
    parser.add_argument("--n-steps", type=int, default=4, help="Denoising steps")
    parser.add_argument("--save-videos", type=int, default=3)
    parser.add_argument("--output-dir", default="result/vla_abcd_flower/benchmark")
    parser.add_argument("--calvin-dir", default=os.environ.get("CALVIN_DIR", "/home/perelman/calvin"))
    parser.add_argument("--model-path", default=os.environ.get("QWEN3VL_MODEL_PATH", "/home/perelman/models/qwen3-vl-2b"))
    args = parser.parse_args()

    import imageio
    import jax
    import jax.numpy as jnp
    import numpy as np
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    import hydra
    from calvin_agent.evaluation.multistep_sequences import get_sequences

    t_total = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"JAX devices: {jax.devices()}")

    # ── Spawn N sim workers ──
    n_w = args.num_workers
    ctx = mp.get_context("spawn")
    cmd_qs = [ctx.Queue() for _ in range(n_w)]
    res_q = ctx.Queue()
    workers = []
    print(f"Spawning {n_w} sim workers...")
    for i in range(n_w):
        p = ctx.Process(target=sim_worker, args=(i, cmd_qs[i], res_q, args.calvin_dir))
        p.start()
        workers.append(p)

    # Wait for all workers ready
    ready_count = 0
    while ready_count < n_w:
        msg = res_q.get()
        if msg[0] == "ready":
            ready_count += 1
    print(f"  {n_w} workers ready")

    # ── Task oracle + annotations ──
    conf_dir = Path(args.calvin_dir) / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    # ── Load policy ──
    policy, q = load_policy(args.checkpoint, args.model_path,
                             proprio_dim=args.proprio_dim, chunk_size=args.chunk_size)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    vlm_config = policy.vlm.config
    execute_horizon = args.execute_horizon or args.chunk_size

    # ── Sequences ──
    eval_sequences = get_sequences(args.num_sequences)
    print(f"Evaluating {len(eval_sequences)} sequences with {n_w} workers...")

    success_counts = []
    seq_results = []
    success_videos = []
    failure_videos = []

    seq_idx = 0
    while seq_idx < len(eval_sequences):
        batch = eval_sequences[seq_idx : seq_idx + n_w]
        actual_n = len(batch)

        # Reset workers with initial states
        for i, (init_state, _) in enumerate(batch):
            cmd_qs[i].put(("reset", init_state))
        # Wait for all resets
        resets_done = 0
        while resets_done < actual_n:
            msg = res_q.get()
            if msg[0] == "reset_done":
                resets_done += 1

        env_completed = [0] * actual_n
        env_failed = [False] * actual_n
        env_frames = [[] for _ in range(actual_n)]

        for subtask_idx in range(5):
            active = [i for i in range(actual_n) if not env_failed[i]]
            if not active:
                break

            subtasks = {i: batch[i][1][subtask_idx] for i in active}
            langs = {i: val_annotations[subtasks[i]][0] for i in active}
            text_tokens = {i: tokenizer.encode(langs[i], add_special_tokens=False) for i in active}

            # Get start_info from all active workers
            for i in active:
                cmd_qs[i].put(("get_info", None))
            start_infos = {}
            remaining = set(active)
            while remaining:
                msg = res_q.get()
                if msg[0] == "info":
                    wid = msg[1]
                    if wid in remaining:
                        start_infos[wid] = msg[2]
                        remaining.remove(wid)

            for i in active:
                env_frames[i] = []

            action_buffers = {i: [] for i in active}
            rng = jax.random.PRNGKey(seq_idx * 1000 + subtask_idx)

            step = 0
            while step < args.ep_len and active:
                # Workers needing replan
                need_replan = [i for i in active if len(action_buffers[i]) == 0]

                if need_replan:
                    # Get current obs from those workers
                    for i in need_replan:
                        cmd_qs[i].put(("get_obs", None))
                    obs_dict = {}
                    remaining = set(need_replan)
                    while remaining:
                        msg = res_q.get()
                        if msg[0] == "obs":
                            wid = msg[1]
                            if wid in remaining:
                                obs_dict[wid] = msg[2]
                                remaining.remove(wid)

                    # Prepare batched VLM inputs
                    vlm_inputs = []
                    proprios = []
                    for i in need_replan:
                        obs = obs_dict[i]
                        # Composite image (top + wrist vstack)
                        composite = compose_image(obs["rgb_static"], obs["rgb_gripper"])
                        vlm_inp = prepare_vision(composite, text_tokens[i])
                        vlm_inputs.append(vlm_inp)

                        # Proprio: extract FLOWER dims [0:7] + [14:15]
                        robot_obs = obs["robot_obs"]
                        if args.proprio_dim == 8:
                            state_subset = np.concatenate([robot_obs[0:7], robot_obs[14:15]])
                        else:
                            state_subset = robot_obs
                        pn = (state_subset - q["q01_state"]) / (q["q99_state"] - q["q01_state"] + 1e-6) * 2.0 - 1.0
                        proprios.append(pn.astype(np.float32))

                    obs_embed = batched_encode(policy, vlm_inputs, vlm_config)
                    proprio_batch = jnp.array(np.stack(proprios))[:, None, :]

                    rng, pred_rng = jax.random.split(rng)
                    acts_norm = policy.action_expert.denoise(
                        obs_embed, proprio_batch,
                        chunk_size=args.chunk_size, n_steps=args.n_steps, rng=pred_rng,
                    )
                    acts_norm = np.array(acts_norm)
                    acts = (acts_norm + 1.0) / 2.0 * (q["q99"] - q["q01"] + 1e-6) + q["q01"]

                    for k, i in enumerate(need_replan):
                        action_buffers[i] = list(acts[k, :execute_horizon])  # execute only N steps

                # Send step commands to all active
                for i in list(active):
                    action = action_buffers[i].pop(0).copy()
                    action[6] = 1.0 if action[6] > 0 else -1.0
                    cmd_qs[i].put(("step", action))

                # Collect results
                remaining = set(active)
                new_infos = {}
                while remaining:
                    msg = res_q.get()
                    if msg[0] == "step_done":
                        wid = msg[1]
                        if wid in remaining:
                            new_infos[wid] = msg[2]
                            remaining.remove(wid)

                # Check completion + update frames
                for i in list(active):
                    data = new_infos[i]
                    env_frames[i].append(data["rgb_static"])

                    task_info = task_oracle.get_task_info_for_set(
                        start_infos[i], data["info"], {subtasks[i]}
                    )
                    if len(task_info) > 0:
                        env_completed[i] = subtask_idx + 1
                        if len(success_videos) < args.save_videos:
                            success_videos.append((
                                seq_idx + i, subtask_idx, subtasks[i], langs[i],
                                list(env_frames[i])
                            ))
                        active.remove(i)

                step += 1

            # Unfinished → failed
            for i in active:
                env_failed[i] = True
                if len(failure_videos) < args.save_videos:
                    failure_videos.append((
                        seq_idx + i, subtask_idx, subtasks[i], langs[i],
                        list(env_frames[i])
                    ))

        for i in range(actual_n):
            success_counts.append(env_completed[i])
            seq_results.append({"seq_idx": seq_idx + i, "completed": env_completed[i]})

        n_done = len(success_counts)
        avg = np.mean(success_counts)
        rates = [sum(1 for c in success_counts if c > j) / n_done for j in range(5)]
        elapsed = time.time() - t_total
        print(f"  [{n_done}/{len(eval_sequences)} | {elapsed:.0f}s] "
              + " ".join(f"{j+1}={r * 100:.0f}%" for j, r in enumerate(rates))
              + f" avg={avg:.2f}")

        seq_idx += actual_n

    # ── Summary ──
    rates = [sum(1 for c in success_counts if c > i) / len(success_counts) for i in range(5)]
    avg_len = float(np.mean(success_counts))

    print("\n" + "=" * 60)
    print("CALVIN Benchmark Results")
    print("=" * 60)
    for i, r in enumerate(rates):
        print(f"  {i+1}/5: {r * 100:.1f}%")
    print(f"  Avg chain length: {avg_len:.2f} / 5")
    print(f"  Total time: {time.time() - t_total:.0f}s")

    results = {
        "checkpoint": args.checkpoint,
        "num_sequences": len(eval_sequences),
        "num_workers": n_w,
        "ep_len": args.ep_len,
        "chunk_size": args.chunk_size,
        "execute_horizon": execute_horizon,
        "n_steps": args.n_steps,
        "success_rates": {f"{i+1}/5": rates[i] for i in range(5)},
        "avg_chain_length": avg_len,
        "sequences": seq_results,
        "total_time_s": time.time() - t_total,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {args.output_dir}/results.json")

    for tag, vids in [("success", success_videos), ("failure", failure_videos)]:
        for i, (seq_i, st_i, subtask, lang, frames) in enumerate(vids):
            out_path = os.path.join(args.output_dir, f"{tag}_{i:02d}_seq{seq_i:03d}_{subtask}.mp4")
            imageio.mimwrite(out_path, frames, fps=30, quality=8, macro_block_size=1)
            print(f"  {tag}: {out_path} ({len(frames)}f)")

    # Cleanup workers
    for i in range(n_w):
        cmd_qs[i].put(("stop", None))
    for w in workers:
        w.join(timeout=5)


if __name__ == "__main__":
    main()
