"""Dataclass configuration system for the VLA pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ActionExpertConfig:
    d_model: int = 1536
    n_layers: int = 12
    d_ff: int = 4096
    n_heads: int = 12
    n_kv_heads: int = 4
    head_dim: int = 128  # n_heads * head_dim == d_model
    action_dim: int = 7


@dataclass
class VLMConfig:
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    hidden_dim: int = 2048
    freeze: bool = True


@dataclass
class DataConfig:
    repo_id: str = "fywang/calvin-debug-lerobot"
    cameras: list[str] = field(default_factory=lambda: ["top", "wrist"])
    resize_to: int = 336
    action_dim: int = 7
    chunk_size: int = 50
    num_workers: int = 0


@dataclass
class FlowMatchingConfig:
    denoise_steps_inference: int = 10
    beta_a: float = 1.5
    beta_b: float = 1.0
    t_min: float = 0.001
    t_max: float = 1.0
    simulated_delay: int = 0  # 0=disabled, >0=training-time RTC (max prefix length)


@dataclass
class TrainingConfig:
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lr: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    stage1_epochs: int = 30
    stage2_epochs: int = 20
    warmup_steps: int = 100
    mixed_precision: bool = True
    log_interval: int = 10
    save_interval: int = 500
    seed: int = 42


@dataclass
class PipelineConfig:
    action_expert: ActionExpertConfig = field(default_factory=ActionExpertConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    flow_matching: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
