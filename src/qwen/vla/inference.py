"""VLA inference + visualization for JAX/TPU.

Features:
- Predict action chunks from cached VLM embeddings
- Visualize: rollout video + language annotation + predicted vs GT trajectory
- Memorization check: evaluate on training data
"""

import argparse
import json
import os

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from PIL import Image, ImageDraw

from qwen.qwen3vl import modeling as qwen3vl
from qwen.vla.data.lerobot_calvin import CalvinDataset
from qwen.vla.models.vla import VLAPolicy

_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.environ.get(
    "QWEN3VL_MODEL_PATH",
    os.path.join(_ROOT, "..", "..", "..", "models", "qwen3-vl-2b"),
)


def load_policy(model_path: str, seed: int = 42) -> VLAPolicy:
    """Load VLA policy with Qwen3-VL backbone."""
    config = qwen3vl.ModelConfig.qwen3vl_2b()
    vlm = qwen3vl.Qwen3VLForConditionalGeneration.from_pretrained(model_path, config=config)
    rngs = nnx.Rngs(params=seed)
    return VLAPolicy(vlm=vlm, vlm_hidden_dim=2048, rngs=rngs)


def render_frame_with_annotation(
    image: np.ndarray,
    language: str,
    step: int,
    total_steps: int,
    gripper_pred: float,
    gripper_gt: float,
) -> np.ndarray:
    """Render a single frame with language annotation and gripper status overlay."""
    img = Image.fromarray((image * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # Language annotation (top)
    draw.rectangle([(0, 0), (img.width, 30)], fill=(0, 0, 0, 180))
    draw.text((5, 5), f"Task: {language}", fill=(255, 255, 255))

    # Step counter + gripper (bottom)
    grip_pred_str = "CLOSE" if gripper_pred > 0.5 else "OPEN"
    grip_gt_str = "CLOSE" if gripper_gt > 0.5 else "OPEN"
    grip_color = (0, 255, 0) if grip_pred_str == grip_gt_str else (255, 0, 0)
    draw.rectangle([(0, img.height - 35), (img.width, img.height)], fill=(0, 0, 0, 180))
    draw.text((5, img.height - 30), f"Step {step}/{total_steps}", fill=(255, 255, 255))
    draw.text((5, img.height - 15), f"Grip: {grip_pred_str} (GT: {grip_gt_str})", fill=grip_color)

    return np.array(img)


def visualize_trajectory(
    sample: dict,
    actions_pred: np.ndarray,
    gripper_pred: np.ndarray,
    output_path: str,
):
    """Create visualization video: images + language + predicted vs GT trajectory."""
    images = sample["images"]  # (n_cameras, H, W, 3)
    actions_gt = sample["raw_actions"]  # (T, 7)
    language = sample["language"]
    chunk_size = actions_gt.shape[0]

    frames = []
    for t in range(chunk_size):
        img = images[0]  # top camera
        grip_p = float(gripper_pred[t, 0]) if t < len(gripper_pred) else 0.0
        grip_g = float(actions_gt[t, 6] > 0)

        frame = render_frame_with_annotation(img, language, t, chunk_size, grip_p, grip_g)
        frames.append(frame)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimwrite(output_path, frames, fps=10, quality=8, macro_block_size=1)
    print(f"  Video saved: {output_path}")


def evaluate_samples(
    policy: VLAPolicy,
    dataset: CalvinDataset,
    n_samples: int = 10,
    output_dir: str = "result/vla",
    visualize: bool = False,
):
    """Evaluate on dataset samples, compute trajectory error."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    results = []
    rng = jax.random.PRNGKey(0)

    n_samples = min(n_samples, len(dataset))

    for i in range(n_samples):
        sample = dataset[i]

        # Prepare VLM inputs
        pil_imgs = [Image.fromarray((img * 255).astype(np.uint8)) for img in sample["images"]]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_imgs[0]},
                    {"type": "text", "text": sample["language"]},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="np"
        )
        input_ids = jnp.array(inputs["input_ids"])
        pixel_values = jnp.array(inputs["pixel_values"]) if "pixel_values" in inputs else None
        image_grid_thw = jnp.array(inputs["image_grid_thw"]) if "image_grid_thw" in inputs else None
        token_type_ids = (
            (input_ids == policy.vlm.config.image_token_id).astype(jnp.int32) if pixel_values is not None else None
        )

        # Encode + predict
        obs_embed = policy.encode_observations(input_ids, pixel_values, image_grid_thw, token_type_ids)
        rng, pred_rng = jax.random.split(rng)
        actions_cont, gripper_probs = policy.predict_actions(obs_embed, chunk_size=dataset.chunk_size, rng=pred_rng)

        # Convert to numpy
        actions_pred_np = np.array(actions_cont[0])  # (T, 6)
        gripper_pred_np = np.array(gripper_probs[0])  # (T, 1)

        # Denormalize predicted continuous actions
        full_pred = np.zeros((dataset.chunk_size, 7), dtype=np.float32)
        full_pred[:, :6] = dataset.denormalize_actions(
            np.concatenate([actions_pred_np, np.zeros((actions_pred_np.shape[0], 1))], axis=1)
        )[:, :6]
        full_pred[:, 6] = (gripper_pred_np[:, 0] > 0.5).astype(np.float32)

        # Metrics
        gt = sample["raw_actions"]
        pos_err = np.sqrt(((full_pred[:, :3] - gt[:, :3]) ** 2).sum(axis=1)).mean()
        orn_err = np.sqrt(((full_pred[:, 3:6] - gt[:, 3:6]) ** 2).sum(axis=1)).mean()
        grip_acc = (full_pred[:, 6] == (gt[:, 6] > 0).astype(np.float32)).mean()

        results.append(
            {
                "episode": int(sample["episode"]),
                "language": sample["language"],
                "pos_error_m": float(pos_err),
                "orn_error_rad": float(orn_err),
                "gripper_accuracy": float(grip_acc),
            }
        )
        print(f"  [{i + 1}/{n_samples}] ep={sample['episode']}, pos_err={pos_err:.4f}, grip_acc={grip_acc:.2f}")

        if visualize:
            vid_path = os.path.join(output_dir, f"rollout_ep{sample['episode']}.mp4")
            visualize_trajectory(sample, actions_pred_np, gripper_pred_np, vid_path)

    # Save summary
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "eval_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {summary_path}")
    print(f"Avg pos_error: {np.mean([r['pos_error_m'] for r in results]):.4f}")
    print(f"Avg gripper_acc: {np.mean([r['gripper_accuracy'] for r in results]):.2f}")


def main():
    parser = argparse.ArgumentParser(description="VLA Inference (JAX/TPU)")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--visualize", action="store_true", help="Generate rollout videos")
    parser.add_argument("--memorization-check", action="store_true", help="Evaluate on train split")
    parser.add_argument("--output-dir", default="result/vla")
    args = parser.parse_args()

    print("=" * 60)
    print("VLA Inference — JAX/TPU v4-8")
    print("=" * 60)

    policy = load_policy(args.model_path)

    split = "train" if args.memorization_check else args.split
    dataset = CalvinDataset(repo_id="fywang/calvin-debug-lerobot", split=split)
    print(f"Dataset: {len(dataset)} chunks ({split} split)")

    evaluate_samples(policy, dataset, args.n_samples, args.output_dir, args.visualize)


if __name__ == "__main__":
    main()
