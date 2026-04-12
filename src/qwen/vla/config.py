"""Structured config system for the VLA pipeline.

Usage:
    cfg = PipelineConfig.calvin_debug()    # debug preset
    cfg = PipelineConfig.calvin_abcd()     # full CALVIN
    cfg = PipelineConfig()                 # default (calvin debug)

    # Override fields:
    cfg.training.lr = 1e-4
    cfg.training.epochs = 200
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Environment-specific dimensions and data config."""

    name: str = "calvin"
    action_dim: int = 7
    proprio_dim: int = 15
    cameras: list[str] = field(default_factory=lambda: ["top"])
    image_size: int = 320
    chunk_size: int = 50
    stride: int = 0  # 0 = chunk_size // 2 (default), >0 = explicit
    repo_id: str = "fywang/calvin-debug-lerobot"
    local_path: str = ""  # if set, skip HF download

    @classmethod
    def calvin_debug(cls) -> EnvConfig:
        return cls(
            name="calvin-debug",
            action_dim=7,
            proprio_dim=15,
            cameras=["top"],
            image_size=320,
            chunk_size=50,
            repo_id="fywang/calvin-debug-lerobot",
        )

    @classmethod
    def calvin_abcd(cls, local_path: str = "/dev/shm/calvin_abcd") -> EnvConfig:
        return cls(
            name="calvin-abcd",
            action_dim=7,
            proprio_dim=15,
            cameras=["top"],
            image_size=320,
            chunk_size=50,
            repo_id="fywang/calvin-task-ABCD-D-lerobot",
            local_path=local_path,
        )

    @classmethod
    def calvin_abcd_flower(cls, local_path: str = "/dev/shm/calvin_abcd") -> EnvConfig:
        """FLOWER-VLA recipe: chunk=10, proprio 8-dim, top+wrist cameras.

        Uses stride=25 to match baseline's sample count (~15k chunks).
        Each chunk predicts 10 actions; overlap allows multiple views per episode.
        """
        return cls(
            name="calvin-abcd-flower",
            action_dim=7,
            proprio_dim=8,
            cameras=["top", "wrist"],
            image_size=320,
            chunk_size=10,
            stride=25,  # keep sample count manageable (~15k chunks)
            repo_id="fywang/calvin-task-ABCD-D-lerobot",
            local_path=local_path,
        )


@dataclass
class ModelConfig:
    """Action expert architecture."""

    d_model: int = 1536
    n_layers: int = 12
    d_ff: int = 4096
    n_heads: int = 12
    n_kv_heads: int = 4
    head_dim: int = 128


@dataclass
class VLMConfig:
    """VLM backbone."""

    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    hidden_dim: int = 2048
    model_path: str = ""  # local path, empty = auto-detect


@dataclass
class FlowMatchingConfig:
    """Flow matching scheduler."""

    beta_a: float = 1.5
    beta_b: float = 1.0
    t_min: float = 0.001
    t_max: float = 1.0
    denoise_steps: int = 10
    simulated_delay: int = 0  # 0=disabled, >0=RTC


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 100
    batch_size: int = 32
    lr: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    ema_decay: float | None = None  # None=disabled
    log_interval: int = 10
    seed: int = 42
    output_dir: str = "result/vla"


@dataclass
class PipelineConfig:
    """Top-level config composing all sub-configs."""

    env: EnvConfig = field(default_factory=EnvConfig.calvin_debug)
    model: ModelConfig = field(default_factory=ModelConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    flow_matching: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def calvin_debug(cls) -> PipelineConfig:
        return cls(
            env=EnvConfig.calvin_debug(),
            training=TrainingConfig(output_dir="result/vla"),
        )

    @classmethod
    def calvin_abcd(cls) -> PipelineConfig:
        return cls(
            env=EnvConfig.calvin_abcd(),
            training=TrainingConfig(
                epochs=200,
                batch_size=128,
                lr=5e-5,
                output_dir="result/vla_abcd",
            ),
            flow_matching=FlowMatchingConfig(simulated_delay=15),
        )

    @classmethod
    def calvin_abcd_flower(cls) -> PipelineConfig:
        """FLOWER recipe preset: chunk=10, 2 cameras, longer training."""
        return cls(
            env=EnvConfig.calvin_abcd_flower(),
            training=TrainingConfig(
                epochs=200,  # ~95k steps @ bs=32, steps_per_epoch=475
                batch_size=32,
                lr=1e-4,
                output_dir="result/vla_abcd_flower",
            ),
            flow_matching=FlowMatchingConfig(simulated_delay=0, denoise_steps=4),
        )
