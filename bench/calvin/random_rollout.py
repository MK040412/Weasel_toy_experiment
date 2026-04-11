"""CALVIN random action rollout -> MP4 verification script."""

import argparse
import os

os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
os.environ.pop("DISPLAY", None)

from pathlib import Path

import hydra
import imageio
import numpy as np
from hydra import compose, initialize_config_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-dir", default="/home/perelman/calvin")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    config_dir = str(Path(args.calvin_dir) / "calvin_env" / "conf")
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent

    with initialize_config_dir(config_dir=config_dir):
        cfg = compose(
            config_name="config_data_collection.yaml",
            overrides=["cameras=static_and_gripper", "use_vr=False"],
        )

    cfg.env.use_egl = False
    cfg.env.show_gui = False
    cfg.env.use_vr = False
    cfg.env.use_scene_info = True

    print("Instantiating CALVIN PlayTableSimEnv (DIRECT mode)...")
    env = hydra.utils.instantiate(cfg.env)

    try:
        obs = env.reset()
        print(f"Reset OK. Obs keys: {list(obs.keys())}")
        for k, v in obs["rgb_obs"].items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

        frames = []
        for i in range(args.steps):
            action = np.concatenate(
                [
                    np.random.uniform(-0.15, 0.15, 3),
                    np.random.uniform(-0.05, 0.05, 3),
                    [np.random.choice([-1.0, 1.0])],
                ]
            )
            obs, reward, done, info = env.step(action)
            frames.append(obs["rgb_obs"]["rgb_static"])

            if (i + 1) % 50 == 0:
                print(f"  Step {i + 1}/{args.steps}")

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "random_rollout.mp4"
        imageio.mimwrite(str(out_path), frames, fps=30, quality=8, macro_block_size=1)
        print(f"Done! {len(frames)} frames -> {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    finally:
        env.close()


if __name__ == "__main__":
    main()
