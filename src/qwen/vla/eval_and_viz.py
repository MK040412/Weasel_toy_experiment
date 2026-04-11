"""Train VLA + visualize predicted vs GT trajectories.

Uses PipelineConfig for all settings. Outputs debug_with_RTC.mp4.
"""

import os

import imageio
import jax
import numpy as np
from flax import nnx
from PIL import Image as PILImage
from PIL import ImageDraw

from qwen.vla.config import PipelineConfig
from qwen.vla.data.protocol import create_dataset
from qwen.vla.models.vla import VLAPolicy
from qwen.vla.training.trainer import VLATrainer
from qwen.vla.training.vlm_cache import VLMCacher

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
MODEL_PATH = os.environ.get("QWEN3VL_MODEL_PATH", "/home/perelman/models/qwen3-vl-2b")


def draw_frame(image, language, step, total, pred_pos, gt_pos, grip_pred, grip_gt, pred_traj, gt_traj):
    h, w = image.shape[:2]
    img = PILImage.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    draw.rectangle([(0, 0), (w, 28)], fill=(0, 0, 0))
    draw.text((4, 4), f"Task: {language[:60]}", fill=(255, 255, 255))

    gp, gg = ("CLOSE" if grip_pred > 0 else "OPEN"), ("CLOSE" if grip_gt > 0 else "OPEN")
    color = (0, 255, 0) if gp == gg else (255, 80, 80)
    draw.rectangle([(0, h - 42), (w, h)], fill=(0, 0, 0))
    draw.text((4, h - 40), f"Step {step + 1}/{total}", fill=(200, 200, 200))
    draw.text((4, h - 22), f"Grip pred:{gp} gt:{gg}", fill=color)

    pos_err = np.sqrt(((pred_pos - gt_pos) ** 2).sum())
    draw.text((w - 140, h - 40), f"pos_err: {pos_err:.3f}", fill=(255, 255, 100))

    def to_px(pos, scale=150, cx=w // 2, cy=h // 2):
        return int(cx + pos[0] * scale), int(cy - pos[1] * scale)

    for j in range(1, len(gt_traj)):
        draw.line([to_px(gt_traj[j - 1]), to_px(gt_traj[j])], fill=(0, 200, 0), width=2)
    for j in range(1, len(pred_traj)):
        draw.line([to_px(pred_traj[j - 1]), to_px(pred_traj[j])], fill=(255, 60, 60), width=2)

    gt_px, pred_px = to_px(gt_pos), to_px(pred_pos)
    draw.ellipse([gt_px[0] - 4, gt_px[1] - 4, gt_px[0] + 4, gt_px[1] + 4], fill=(0, 255, 0))
    draw.ellipse([pred_px[0] - 4, pred_px[1] - 4, pred_px[0] + 4, pred_px[1] + 4], fill=(255, 0, 0))

    draw.rectangle([(w - 120, 32), (w, 62)], fill=(0, 0, 0))
    draw.text((w - 116, 34), "GT", fill=(0, 255, 0))
    draw.text((w - 80, 34), "Pred", fill=(255, 60, 60))
    return np.array(img)


def main():
    cfg = PipelineConfig.calvin_debug()
    cfg.training.lr = 1e-4
    cfg.flow_matching.simulated_delay = 15
    cfg.vlm.model_path = MODEL_PATH

    print("=" * 60)
    print(f"VLA Train + Eval + Viz — {cfg.env.name}")
    print("=" * 60)

    cacher = VLMCacher(cfg.training.output_dir)
    dataset = create_dataset(cfg.env, split="train")
    print(f"  {len(dataset)} chunks")

    if cacher.exists():
        print("VLM cache found.")
        cache = cacher.load()
        policy = VLAPolicy(
            vlm=None, vlm_hidden_dim=cfg.vlm.hidden_dim,
            action_expert_config={"action_dim": cfg.env.action_dim, "proprio_dim": cfg.env.proprio_dim},
            rngs=nnx.Rngs(params=42),
        )
    else:
        print("Loading Qwen3-VL 2B...")
        from qwen.qwen3vl import modeling as qwen3vl

        model_config = qwen3vl.ModelConfig.qwen3vl_2b()
        vlm = qwen3vl.Qwen3VLForConditionalGeneration.from_pretrained(cfg.vlm.model_path, config=model_config)
        policy = VLAPolicy(
            vlm=vlm, vlm_hidden_dim=cfg.vlm.hidden_dim,
            action_expert_config={"action_dim": cfg.env.action_dim, "proprio_dim": cfg.env.proprio_dim},
            rngs=nnx.Rngs(params=42),
        )
        cache = cacher.compute(dataset, vlm, policy.obs_proj, cfg.vlm.model_id, cfg.env.image_size)
        policy.vlm = None
        jax.clear_caches()

    trainer = VLATrainer(policy, cache, cfg, dataset=dataset)
    trainer.train()

    # Eval + viz
    print("\n--- Evaluation + Visualization ---")
    rng = jax.random.PRNGKey(0)
    all_frames = []
    chunk_size = cfg.env.chunk_size
    n_eval = min(5, len(dataset))

    for i in range(n_eval):
        sample = dataset[i]
        obs_embed = cache.obs[i : i + 1]
        proprio = cache.proprio[i : i + 1]
        rng, pred_rng = jax.random.split(rng)
        acts_pred = policy.action_expert.denoise(obs_embed, proprio, chunk_size=chunk_size, n_steps=10, rng=pred_rng)

        pred = np.array(acts_pred[0])
        gt = sample["actions"]
        pos_err = np.sqrt(((pred[:, :3] - gt[:, :3]) ** 2).sum(axis=1))
        grip_acc = ((pred[:, 6] > 0) == (gt[:, 6] > 0)).mean()

        print(f'  Sample {i}: ep={sample["episode"]}, pos_err={pos_err.mean():.4f}, grip_acc={grip_acc:.2f}')

        top_img = sample["images"][0]
        for t in range(chunk_size):
            frame = draw_frame(
                top_img, sample["language"], t, chunk_size,
                pred[t, :3], gt[t, :3], float(pred[t, 6]), float(gt[t, 6]),
                pred[: t + 1, :2], gt[: t + 1, :2],
            )
            all_frames.append(frame)
        for _ in range(5):
            all_frames.append(np.zeros_like(all_frames[-1]))

    os.makedirs(cfg.training.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.training.output_dir, "debug_with_RTC.mp4")
    imageio.mimwrite(out_path, all_frames, fps=15, quality=8, macro_block_size=1)
    print(f"\n  Video saved: {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
