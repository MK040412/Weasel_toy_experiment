# Reasoning SFT recipe — Fast-dVLM 2B think-then-act (run AFTER the action SFT finishes)

> Fresh Claude/engineer: this is the exact, verified plan for the NEXT phase — teaching the block-diffusion
> dVLM to REASON (`<think>…</think>` then act) on AndroidWorld. Everything here was decided + verified during
> the action-SFT run (token check, dataset scan, memory limits). Follow top to bottom.

## Decision summary (why this shape)
- **Dataset = `KMK040412/gui-libra-reasoning-phone`** (HF). Vultr-verified full scan: ~**23,948 episodes /
  41,876 steps**, packed parquet, `screenshot` JPEG bytes, `episode_id`, `step_id`(int), `instruction`,
  `target_json` action, **`reasoning` CoT column (0% empty, ~186 tok/step)**. Sources: aitw, amex, gui_odyssey
  (mobile, portrait). Coords already 0–1000. It is **already distilled from a strong reasoner (GUI-Libra ASFT)**.
- **Loss = PURE Fast-dVLM, distillation OFF.** `--kd-noisy-weight 0 --kd-fewstep-weight 0` (keep ce_noisy +
  ce_clean). WHY: the kd terms are SELF-distillation (teacher = the model's own clean/AR branch); the base 2B
  has no reasoning, so the AR branch can't teach reasoning and its KL would fight the CoT cross-entropy.
  Reasoning is learned ONLY from CoT data via CE. (GUI-Libra Stage-1 ASFT is itself plain CE → fully compatible.)
- **No external 8B-Think teacher.** Offline distillation from 8B-Think = SFT on teacher CoT = what
  gui-libra-reasoning-phone ALREADY is; online co-hosting a 9B teacher is hardware-infeasible here
  (won't fit the busy 16 chips; teacher has `tie_word_embeddings=False` → needs a new lm_head logit path) and
  2–4× slower. Marginal gain, huge cost → rejected.

## `<think>` tokens — NATIVE in Qwen3-VL (verified on the boltzmann-final tokenizer)
| token | id | status |
|---|---|---|
| `<think>` | 151667 | single special token ✓ |
| `</think>` | 151668 | single special token ✓ |
| `<tool_call>` | 151657 | single special token ✓ |
| `</tool_call>` | 151658 | single special token ✓ |
| `<answer>` | [27,9217,29] | **splits into 3 — DO NOT USE** |

→ Use the think-then-act format `<think>…</think>` + `<tool_call>…</tool_call>` (all native). Never `<answer>`.

## REQUIRED loader patch (inject the CoT into the assistant turn so it is supervised)
The episode loader currently puts ONLY the action in the assistant turn, so the `reasoning` column is ignored
(CoT gets zero loss). In `scripts/train_fastdvlm_tpu.py`, `_build_episode_messages` (~L522), replace:
```python
        messages.append({"role": "assistant", "content": native_action(step)})
```
with:
```python
        reasoning = str(row_value(step, "reasoning", "") or "").strip()
        action = native_action(step)
        content_str = f"<think>\n{reasoning}\n</think>\n\n{action}" if reasoning else action
        messages.append({"role": "assistant", "content": content_str})
```
`assistant_labels` masks the whole assistant span, so the `<think>…</think>` CoT is then supervised by
ce_noisy + ce_clean automatically. (Backward-compatible: datasets without a `reasoning` column fall back to
action-only.) Verify after a step: decode the `labels != -100` span and confirm it contains the `<think>` CoT.

## Stable hyperparameters (near-zero failure — do NOT exceed these)
| knob | value | why |
|---|---|---|
| epochs | **2** (~3,000 steps; ~1,497/epoch) | 1 under-trains CoT, 3+ overfits the CoT templates |
| `--batch-size` | **16** (1/chip) | 32 (2/chip) OOMs the vision-constrained TPU_0 |
| `--pair-batch` | **1** | pair 2 doubles the f32[2,16,total,total] attn buffer → OOM (independent of kd) |
| `--optim` | `adamw_bf16` `--shard-opt-state` `--skip-nonfinite` | proven; never fp32 |
| `--lr` | **3e-6** | continued-SFT conservative; proven stable |
| `--noisy-pad-to` | **1536** (or 2048 if OOM headroom) | CoT lengthens the noisy/text branch; 1024 truncates 59–85% of episodes |
| `--ctx-cap` / `--pad-to` | **4096** | bigger = total_dual_len² memory; keep 4096 for the FIRST reasoning run |
| `--bd-curriculum` | degree2, eval-centered bd16 (same λ2=1.04, λ1 anneal) OR Boltzmann | match the eval block size |
| `--kd-noisy-weight` / `--kd-fewstep-weight` | **0** / **0** | distillation OFF (see above) |

**Removing distillation does NOT free memory** (the cost is the dual-stream attention buffer, not kd) →
you CANNOT raise batch/pairs. The speed levers are epochs / ctx / noisy-pad, not batch.

**Context length is the known future bottleneck:** longer CoT needs longer context, but total_dual_len²
memory caps it at batch 16/pair 1. Do the FIRST run at ctx 4096 + noisy 1536; train LONGER context as a
SEPARATE later stage (stage by length).

## Prereqs (same as the action run — see `tpu_v6e16_fastdvlm_zero1_recipe.md`)
- CPU torch+torchvision in the venv; `~/.fastdvlm_secrets.env` (HF_TOKEN) on every worker.
- Data on every worker: `~/data/gui-libra-reasoning-phone` (download via Vultr-relay; **nested layout** →
  use `--data-pattern "*/*.parquet"`; **`--data-mode episode` is mandatory** — row mode KeyErrors on `screenshot`).
- Start checkpoint = the BEST action-SFT checkpoint (from `KMK040412/fastdvlm-aw-guiowlvit`), or boltzmann-final.

## Launch (adapt `commands/launch_fastdvlm_v6e16.sh`)
```bash
... same env block ...
uv run --no-sync python scripts/train_fastdvlm_tpu.py --multihost --data-parallel \
  --model-dir ~/models/<best-action-sft-ckpt> \
  --data ~/data/gui-libra-reasoning-phone --data-pattern "*/*.parquet" \
  --out /dev/shm/v6e16_reasoning --data-mode episode --max-turns 12 \
  --max-samples 0 --samples-per-window 64 \
  --batch-size 16 --max-steps 3000 --epochs 2 \
  --bd-curriculum degree2 --bd-values "1,2,4,8,16,32" --bd-lambda1 0.0 --bd-lambda2 1.04 --bd-lambda1-end -5.77 --bd-anneal-steps 1500 \
  --ctx-cap 4096 --pad-to 4096 --noisy-pad-to 1536 --vision-pad-to 1280 \
  --vision-precompute-batch-size 16 --pair-batch 1 --loss-token-cap 256 \
  --dtype bf16 --optim adamw_bf16 --lr 3e-6 --weight-decay 0.01 --shard-opt-state --skip-nonfinite \
  --ce-noisy-weight 1.0 --ce-clean-weight 0.75 --kd-noisy-weight 0.0 --kd-fewstep-weight 0.0 \
  --hf-upload-repo KMK040412/fastdvlm-aw-reasoning --hf-upload-every-steps 750 --hf-upload-final \
  --prefetch-windows 1 --log-every 1 --monitor-every 5
```
Launch PER-WORKER in parallel (NOT `--worker=all`), with the idempotency guard from the action recipe.

## Checkpointing = DECOUPLED (already in place — see `CHECKPOINT_DECOUPLED.md`)
Saving is the dumb per-host shard dump to `/dev/shm` + the external `scripts/stitch_and_ship_checkpoint.py`.
Same procedure. A save/ship bug is fixed externally WITHOUT restarting training.

## ETA
~3,000 steps × ~2 s/step (longer ctx) + ~14–28 min compile ≈ **~2 hours** (much faster than the 3-epoch
action run because the dataset is smaller and it's 2 epochs).
