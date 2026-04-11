"""VLATrainer — 2-stage training loop for VLA flow matching (openpi0.5 style).

Stage 1: Cache VLM embeddings, train obs_proj + action_expert (fast, no VLM forward)
Stage 2: Fine-tuning with lower LR (VLM still frozen, cached)
"""

from __future__ import annotations

import gc
import math
import random
import time

import torch

from qwen.vla.config import PipelineConfig
from qwen.vla.data.lerobot_calvin import LeRobotCalvinDataset
from qwen.vla.models.vla import VLAPolicy
from qwen.vla.training.flow_matching import FlowMatchingScheduler


class VLATrainer:
    """2-stage training loop for VLA policy with flow matching."""

    def __init__(self, config: PipelineConfig, device: torch.device | None = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = VLAPolicy(config.vlm, config.action_expert)
        self.policy.obs_proj.to(self.device, dtype=torch.bfloat16)
        self.policy.action_expert.to(self.device, dtype=torch.bfloat16)

        self.flow_matching = FlowMatchingScheduler(config.flow_matching)

        trainable_params = list(self.policy.obs_proj.parameters()) + list(self.policy.action_expert.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

        self.global_step = 0

    def _get_lr(self, base_lr: float, stage_step: int, total_steps: int) -> float:
        """Cosine learning rate with linear warmup (per-stage step counter)."""
        warmup = self.config.training.warmup_steps
        if stage_step < warmup:
            return base_lr * stage_step / max(warmup, 1)
        progress = (stage_step - warmup) / max(1, total_steps - warmup)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    def _update_lr(self, base_lr: float, stage_step: int, total_steps: int) -> None:
        lr = self._get_lr(base_lr, stage_step, total_steps)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def train(self) -> None:
        """Main 2-stage training loop."""
        cfg = self.config

        print("Loading dataset...")
        dataset = LeRobotCalvinDataset(cfg.data, split="train")
        print(f"  Dataset: {len(dataset)} samples, action_q01={dataset.action_q01.tolist()}")
        print(f"  action_q99={dataset.action_q99.tolist()}")

        self.action_q01 = dataset.action_q01
        self.action_q99 = dataset.action_q99

        # ===== Stage 1: Warmup with cached VLM embeddings =====
        print("\n" + "=" * 60)
        print("Stage 1: Caching VLM embeddings...")
        print("=" * 60)
        cached_embeddings = self._cache_vlm_embeddings(dataset)

        print("  Offloading VLM to CPU for Stage 1...")
        self.policy.vlm.to("cpu")
        torch.cuda.empty_cache()

        print(f"\nStage 1: Training action expert with cached embeddings ({cfg.training.stage1_epochs} epochs)...")
        self._train_cached(
            dataset,
            cached_embeddings,
            num_epochs=cfg.training.stage1_epochs,
            lr=cfg.training.lr,
            stage_name="S1",
        )

        # ===== Stage 2: Fine-tuning with lower LR =====
        print("\n" + "=" * 60)
        print(f"Stage 2: Fine-tuning with lower LR ({cfg.training.stage2_epochs} epochs)...")
        print("=" * 60)
        self._train_cached(
            dataset,
            cached_embeddings,
            num_epochs=cfg.training.stage2_epochs,
            lr=cfg.training.lr * 0.1,
            stage_name="S2",
        )

        del cached_embeddings
        gc.collect()
        torch.cuda.empty_cache()

        self.save_checkpoint("checkpoint_final.pt")
        print("\nTraining complete!")

    def _cache_vlm_embeddings(self, dataset: LeRobotCalvinDataset) -> dict[int, torch.Tensor]:
        """One-time VLM forward for all samples -> cache hidden states on CPU."""
        cache: dict[int, torch.Tensor] = {}
        self.policy.vlm.eval()

        t_start = time.perf_counter()
        for idx in range(len(dataset)):
            sample = dataset[idx]
            hidden = self.policy.vlm_forward_hidden(sample["images"], sample["language"])
            cache[idx] = hidden.cpu()
            del hidden
            torch.cuda.empty_cache()

            if (idx + 1) % 10 == 0 or idx == len(dataset) - 1:
                elapsed = time.perf_counter() - t_start
                print(f"  Cached {idx + 1}/{len(dataset)} samples ({elapsed:.1f}s)")

        return cache

    def _train_cached(
        self,
        dataset: LeRobotCalvinDataset,
        cached_embeddings: dict[int, torch.Tensor],
        num_epochs: int,
        lr: float,
        stage_name: str = "S1",
    ) -> None:
        """Train obs_proj + action_expert using cached VLM hidden states."""
        cfg = self.config
        grad_accum = cfg.training.gradient_accumulation_steps
        total_steps = num_epochs * len(dataset) // grad_accum

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        self.policy.action_expert.train()
        self.policy.obs_proj.train()

        indices = list(range(len(dataset)))
        stage_step = 0

        print(
            f"  [{stage_name}] {num_epochs} epochs, {len(dataset)} samples/epoch, "
            f"grad_accum={grad_accum}, lr={lr:.2e}, total_steps~{total_steps}"
        )

        for epoch in range(num_epochs):
            random.shuffle(indices)
            epoch_loss = 0.0
            n_steps_epoch = 0
            t_start = time.perf_counter()

            self.optimizer.zero_grad()
            accum_loss = 0.0

            for i, idx in enumerate(indices):
                self._update_lr(lr, stage_step, total_steps)

                hidden = cached_embeddings[idx].to(self.device)
                sample = dataset[idx]
                obs_embed = self.policy.obs_proj(hidden)

                gt_actions = sample["actions"].unsqueeze(0).to(self.device, dtype=torch.bfloat16)
                x_t, t, target_velocity, loss_mask = self.flow_matching.sample_training_pair(gt_actions)
                predicted_velocity = self.policy.action_expert.forward_joint(obs_embed, x_t, t)

                loss = self.flow_matching.compute_loss(predicted_velocity, target_velocity, loss_mask)
                loss = loss / grad_accum
                loss.backward()

                accum_loss += loss.item()

                if (i + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.policy.obs_proj.parameters()) + list(self.policy.action_expert.parameters()),
                        cfg.training.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1
                    stage_step += 1
                    epoch_loss += accum_loss
                    n_steps_epoch += 1

                    if self.global_step % cfg.training.log_interval == 0:
                        current_lr = self._get_lr(lr, stage_step, total_steps)
                        print(f"  [{stage_name}] step={self.global_step} loss={accum_loss:.4f} lr={current_lr:.2e}")

                    accum_loss = 0.0

                    if self.global_step % cfg.training.save_interval == 0:
                        self.save_checkpoint(f"checkpoint_{stage_name.lower()}_{self.global_step}.pt")

            elapsed = time.perf_counter() - t_start
            avg_loss = epoch_loss / max(n_steps_epoch, 1)
            print(f"  [{stage_name}] Epoch {epoch + 1}/{num_epochs} — avg_loss={avg_loss:.4f} time={elapsed:.1f}s")

    def save_checkpoint(self, path: str) -> None:
        """Save trainable parameters + action normalization stats."""
        checkpoint = {
            "global_step": self.global_step,
            "obs_proj": self.policy.obs_proj.state_dict(),
            "action_expert": self.policy.action_expert.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "action_q01": getattr(self, "action_q01", None),
            "action_q99": getattr(self, "action_q99", None),
        }
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load trainable parameters."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.obs_proj.load_state_dict(checkpoint["obs_proj"])
        self.policy.action_expert.load_state_dict(checkpoint["action_expert"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint["global_step"]
        if checkpoint.get("action_q01") is not None:
            self.action_q01 = checkpoint["action_q01"]
            self.action_q99 = checkpoint["action_q99"]
        print(f"  Checkpoint loaded: {path} (step={self.global_step})")
