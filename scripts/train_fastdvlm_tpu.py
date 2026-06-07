"""Fast-dVLM dual-stream continuation on TPU for Qwen3-VL/GUI-Owl checkpoints.

This is a practical JAX port of the PyTorch Fast-dVLM objective used in
fast-dvlm-guiowl/src/fast_dvlm/forward.py:

  layout = [noisy_text_tokens | clean_full_multimodal_tokens]
  loss   = 0.5 * CE(noisy masked response tokens) + CE(clean AR response tokens)

The current Weasel Qwen3-VL JAX model supports Qwen3-VL weights and TPU training,
but its text path is still the simplified 1D-RoPE path and does not inject
DeepStack features into text layers. This script logs those exactness flags so
results cannot be confused with the H100 PyTorch path.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import queue
import random
import shutil
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault(
    "LIBTPU_INIT_ARGS",
    " ".join(
        [
            "--xla_tpu_use_enhanced_launch_barrier=true",
            "--xla_tpu_enable_data_parallel_all_reduce_opt=true",
            "--xla_tpu_scoped_vmem_limit_kib=98304",
        ]
    ),
)
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

# Import torch/torchvision BEFORE jax. transformers' Qwen3-VL AutoProcessor pulls in torch+torchvision
# for the video sub-processor; importing them AFTER jax/libtpu has claimed the TPU aborts the process
# (no Python traceback). Pre-importing initializes torch's runtime first and avoids the conflict.
try:
    import torch  # noqa: F401
    import torchvision  # noqa: F401
except Exception:
    pass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from PIL import Image

from qwen.qwen3vl import modeling, params

IMG_TOKEN_ID = 151655
IM_END_ID = 151645
MASK_TOKEN_ID = 151665
MIN_NOISE = 1e-3
_VISION_PMAP_CACHE: dict[tuple[int, int, int, int], Any] = {}
# Multihost host-local vision precompute: jitted single-(local-)device forwards, keyed by image shape.
# jit (NOT pmap) keeps the forward process-local (no global collective) so the 4 independent hosts
# never deadlock; params are passed as an ARG (operand) so they are NOT re-baked per shape variant.
_LOCAL_VIS_FWD_CACHE: dict[tuple[int, int, int, int], Any] = {}
_LOCAL_VIS_SPLIT_CACHE: dict[int, Any] = {}

# The vision encoder is FROZEN (vision is precomputed; train_step never forwards through it). Restrict
# both the gradient and the optimizer to the non-visual params so no adam moments (~2.4GB/chip) or grad
# buffers (~1.2GB) are allocated for the dead-weight vision encoder — that headroom is what lets the
# train_step program fit on the (vision-precompute) chip that is ~5GB tighter than the others.
_TRAINABLE_FILTER = nnx.All(nnx.Param, nnx.Not(nnx.PathContains("visual")))

MOBILE_USE_TOOL = {
    "type": "function",
    "function": {
        "name": "mobile_use",
        "description": "Use a touchscreen to interact with a mobile device. Coordinates are 0-1000.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["click", "long_press", "swipe", "type", "system_button", "open", "wait", "terminate"],
                },
                "coordinate": {"type": "array"},
                "coordinate2": {"type": "array"},
                "text": {"type": "string"},
                "button": {"type": "string", "enum": ["Back", "Home", "Menu", "Enter"]},
                "status": {"type": "string", "enum": ["success", "failure"]},
            },
            "required": ["action"],
        },
    },
}

SYSTEM_PROMPT = (
    "You are a GUI agent operating an Android phone. Given the goal, the action history "
    "and the current screenshot, output the next action by calling the mobile_use function."
)


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=True, default=json_default) + "\n")


def write_json(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=True, indent=2, default=json_default)


def parse_bd_schedule(spec: str | None, default_bd: int) -> tuple[list[int], np.ndarray]:
    """Parse e.g. '4:0.1,8:0.2,16:0.3,32:0.4'."""
    if not spec:
        return [int(default_bd)], np.asarray([1.0], dtype=np.float64)
    bds: list[int] = []
    probs: list[float] = []
    for item in spec.split(","):
        if not item.strip():
            continue
        if ":" in item:
            bd_s, p_s = item.split(":", 1)
            bds.append(int(bd_s.strip()))
            probs.append(float(p_s.strip()))
        else:
            bds.append(int(item.strip()))
            probs.append(1.0)
    if not bds:
        raise ValueError(f"Empty bd schedule: {spec!r}")
    arr = np.asarray(probs, dtype=np.float64)
    if np.any(arr < 0) or arr.sum() <= 0:
        raise ValueError(f"Invalid bd schedule probabilities: {spec!r}")
    arr = arr / arr.sum()
    return bds, arr


def degree2_bd_probs(bd_values: list[int], lambda1: float, lambda2: float) -> np.ndarray:
    """Degree-2 Gaussian-in-log-b block-size curriculum (W0 paper, Prop. ~L1920-1941):

        P(b) ∝ exp(−λ1·ln b − λ2·(ln b)²)

    λ2 = 0 reduces exactly to the Boltzmann / Gibbs power law P(b) ∝ b^{−λ1}. Larger λ1
    shifts mass toward small blocks (AR-like, easy); λ2 > 0 concentrates around a
    log-b mode. Annealing λ1 downward over training shifts mass to large blocks.
    """
    lb = np.log(np.asarray(bd_values, dtype=np.float64))
    logits = -float(lambda1) * lb - float(lambda2) * (lb * lb)
    logits = logits - logits.max()  # stabilize before exp
    p = np.exp(logits)
    return p / p.sum()


# Fixed vision input size (W, H), both multiples of 28 (patch*merge) and W*H <= max_pixels (100352).
# Every screenshot is resized to this single size so the ViT precompute compiles ONE executable
# instead of one per image aspect-ratio (~10 distinct -> ~18-30GB of reserved TPU program memory that
# OOMs train_step). 196x448 is exactly the grid the processor already produces for the dominant
# 1080x2400 phone screenshots (~48% of images), so those are unchanged; minority aspects are squished
# (grounding is unaffected — target coordinates are normalized to 0-1000, i.e. resize-invariant).
FIXED_VISION_WH: tuple[int, int] = (196, 448)


def decode_image(raw: bytes, width: int = 540) -> Image.Image | None:
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB").resize(FIXED_VISION_WH)
    except Exception:
        pass
    arr = np.frombuffer(raw, np.uint8)
    if arr.size % (width * 3) != 0:
        return None
    height = arr.size // 3 // width
    return Image.fromarray(arr.reshape(height, width, 3), "RGB")


def row_has(row: Any, key: str) -> bool:
    try:
        return key in row.index
    except Exception:
        try:
            return key in row
        except Exception:
            return False


def row_value(row: Any, key: str, default: Any = None) -> Any:
    if not row_has(row, key):
        return default
    value = row[key]
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
    except Exception:
        pass
    return value


def target_json_action(row: Any) -> str | None:
    payload = row_value(row, "target_json")
    if payload is None:
        return None
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    payload = str(payload).strip()
    if not payload:
        return None
    if payload.startswith("<tool_call>"):
        return payload
    try:
        parsed = json.loads(payload)
        payload = json.dumps(parsed, ensure_ascii=True, separators=(",", ":"))
    except Exception:
        pass
    return "<tool_call>\n" + payload + "\n</tool_call>"


def _xy(yx: Any) -> list[int]:
    arr = np.asarray(yx, dtype=np.float32)
    return [int(round(float(arr[1]) * 1000)), int(round(float(arr[0]) * 1000))]


def native_action(row: Any) -> str:
    classified_action = target_json_action(row)
    if classified_action is not None:
        return classified_action
    action_type = int(row["results_action_type"])
    if action_type == 4:
        touch = np.asarray(row["results_yx_touch"], dtype=np.float32)
        lift = np.asarray(row["results_yx_lift"], dtype=np.float32)
        if np.allclose(touch, lift, atol=1e-3):
            args = {"action": "click", "coordinate": _xy(touch)}
        else:
            args = {"action": "swipe", "coordinate": _xy(touch), "coordinate2": _xy(lift)}
    elif action_type == 3:
        typed = row["results_type_action"]
        if isinstance(typed, np.ndarray):
            typed = typed.tolist()
        args = {"action": "type", "text": "".join(typed)[:100]}
    elif action_type == 5:
        args = {"action": "system_button", "button": "Back"}
    elif action_type == 6:
        args = {"action": "system_button", "button": "Home"}
    elif action_type == 7:
        args = {"action": "system_button", "button": "Enter"}
    elif action_type == 10:
        args = {"action": "terminate", "status": "success"}
    elif action_type == 11:
        args = {"action": "terminate", "status": "failure"}
    else:
        args = {"action": "wait", "time": 1}
    return "<tool_call>\n" + json.dumps({"name": "mobile_use", "arguments": args}, ensure_ascii=True) + "\n</tool_call>"


def resolve_parquet(args: argparse.Namespace) -> Path | None:
    if args.data:
        path = Path(args.data).expanduser()
        if path.is_file():
            return path
        matches = sorted(path.glob("*.parquet"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"No parquet file found at {path}")
    if args.hf_file:
        from huggingface_hub import hf_hub_download

        local_dir = Path(args.download_dir).expanduser()
        local_dir.mkdir(parents=True, exist_ok=True)
        return Path(
            hf_hub_download(
                repo_id=args.hf_repo,
                filename=args.hf_file,
                repo_type="dataset",
                local_dir=str(local_dir),
                token=os.environ.get("HF_TOKEN"),
            )
        )
    return None


def resolve_parquet_files(args: argparse.Namespace) -> list[Path]:
    if args.data:
        path = Path(args.data).expanduser()
        if path.is_file():
            return [path]
        pattern = args.data_pattern or "*.parquet"
        matches = sorted(path.glob(pattern))
        if matches:
            return matches
        raise FileNotFoundError(f"No parquet files matching {pattern!r} at {path}")
    if args.hf_file:
        return [resolve_parquet(args)]
    return []


def assistant_labels(input_ids: np.ndarray, tokenizer: Any) -> np.ndarray:
    assist = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    seq = input_ids.tolist()
    labels = np.full_like(input_ids, -100, dtype=np.int32)
    i = 0
    while i <= len(seq) - len(assist):
        if seq[i : i + len(assist)] == assist:
            start = end = i + len(assist)
            while end < len(seq) and seq[end] != im_end:
                end += 1
            end = min(end + 1, len(seq))
            labels[start:end] = input_ids[start:end]
            i = end
        else:
            i += 1
    return labels


def build_row_sample(row: Any, processor: Any, ctx_cap: int | None, max_pixels: int) -> dict[str, Any] | None:
    image = decode_image(bytes(row["image"]))
    if image is None:
        return None
    goal = str(row_value(row, "goal_info", row_value(row, "instruction", "")))
    if not goal:
        return None
    action = native_action(row)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": f"Goal: {goal}"}]},
        {"role": "assistant", "content": action},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=[MOBILE_USE_TOOL])
    try:
        from qwen_vl_utils import process_vision_info

        images, videos = process_vision_info(messages)
    except Exception:
        images, videos = [image], None

    batch = processor(
        text=[text],
        images=images,
        videos=videos,
        return_tensors="np",
    )
    input_ids = np.asarray(batch["input_ids"][0], dtype=np.int32)
    if ctx_cap and len(input_ids) > ctx_cap:
        return None
    labels = assistant_labels(input_ids, processor.tokenizer)
    if np.count_nonzero(labels != -100) == 0:
        return None
    mm_token_type_ids = batch.get("mm_token_type_ids")
    if mm_token_type_ids is None:
        mm_types = (input_ids == IMG_TOKEN_ID).astype(np.int32)
    else:
        mm_types = np.asarray(mm_token_type_ids[0], dtype=np.int32)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "mm_token_type_ids": mm_types,
        "token_type_ids": (mm_types > 0).astype(np.bool_),
        "pixel_values": np.asarray(batch["pixel_values"]),
        "image_grid_thw": np.asarray(batch["image_grid_thw"], dtype=np.int32),
        "goal": goal,
        "target": action,
        "source": "aitw_row",
    }


def load_row_samples(parquet_path: Path, processor: Any, max_samples: int, ctx_cap: int | None, max_pixels: int) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    samples: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        sample = build_row_sample(row, processor, ctx_cap, max_pixels)
        if sample is not None:
            samples.append(sample)
        if max_samples and len(samples) >= max_samples:
            break
    return samples


def iter_row_sample_windows(
    parquet_paths: list[Path],
    processor: Any,
    max_samples: int,
    samples_per_window: int,
    ctx_cap: int | None,
    max_pixels: int,
) -> Iterator[tuple[list[dict[str, Any]], dict[str, Any]]]:
    import pyarrow.parquet as pq

    window: list[dict[str, Any]] = []
    emitted = 0
    seen_rows = 0
    skipped = 0
    window_idx = 0
    target = max(samples_per_window, 0)
    stop = False
    for parquet_idx, parquet_path in enumerate(parquet_paths):
        pf = pq.ParquetFile(parquet_path)
        for row_group_idx in range(pf.num_row_groups):
            df = pf.read_row_group(row_group_idx).to_pandas()
            for _, row in df.iterrows():
                seen_rows += 1
                if max_samples and emitted >= max_samples:
                    stop = True
                    break
                sample = build_row_sample(row, processor, ctx_cap, max_pixels)
                if sample is None:
                    skipped += 1
                    continue
                sample["parquet_path"] = str(parquet_path)
                sample["parquet_idx"] = parquet_idx
                sample["row_group_idx"] = row_group_idx
                sample["global_sample_idx"] = emitted
                window.append(sample)
                emitted += 1
                if target and len(window) >= target:
                    meta = {
                        "window_idx": window_idx,
                        "parquet_path": str(parquet_path),
                        "parquet_idx": parquet_idx,
                        "row_group_idx": row_group_idx,
                        "n_samples": len(window),
                        "global_emitted": emitted,
                        "seen_rows": seen_rows,
                        "skipped_rows": skipped,
                    }
                    yield window, meta
                    window_idx += 1
                    window = []
            if stop:
                break
        if stop:
            break
    if window:
        meta = {
            "window_idx": window_idx,
            "parquet_path": str(parquet_paths[-1]) if parquet_paths else None,
            "parquet_idx": None,
            "row_group_idx": None,
            "n_samples": len(window),
            "global_emitted": emitted,
            "seen_rows": seen_rows,
            "skipped_rows": skipped,
        }
        yield window, meta


def _episode_image_bytes(step: Any) -> bytes | None:
    """Packed-hybrid stores the screenshot under 'screenshot'; older row mixes use 'image'."""
    raw = row_value(step, "image", row_value(step, "screenshot"))
    if isinstance(raw, dict):
        raw = raw.get("bytes")
    if raw is None:
        return None
    try:
        return bytes(raw)
    except Exception:
        return None


def _build_episode_messages(steps: list[Any]) -> tuple[list[dict[str, Any]] | None, str]:
    """Multi-turn messages for an episode: [system] + per-turn user[image(+goal@turn0)], assistant[action]."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    goal = ""
    for t, step in enumerate(steps):
        raw = _episode_image_bytes(step)
        if raw is None:
            return None, ""
        image = decode_image(raw)
        if image is None:
            return None, ""
        content: list[dict[str, Any]] = [{"type": "image", "image": image}]
        if t == 0:
            goal = str(row_value(step, "goal_info", row_value(step, "instruction", "")) or "")
            if goal:
                content.append({"type": "text", "text": f"Goal: {goal}"})
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": native_action(step)})
    return messages, goal


def build_episode_sample(
    steps: list[Any], processor: Any, ctx_cap: int | None, max_pixels: int, max_turns: int = 0
) -> tuple[dict[str, Any] | None, int, int]:
    """Pack an ordered episode (list of step rows) into ONE multi-turn sample.

    Returns (sample | None, n_turns_used, n_turns_full). To honor ctx_cap we keep the MOST RECENT
    turns: drop the oldest turn and re-tokenize until it fits (the goal text rides on the first
    retained turn). A single turn that still overflows ctx_cap is skipped. assistant_labels marks
    EVERY assistant turn, so compute_response_block_idx assigns turn_idx/block_idx across all of
    them and asymmetric_allowed gives within-turn block-diagonal + cross-turn causal attention.
    """
    full_n = len(steps)
    if max_turns and full_n > max_turns:
        steps = steps[-max_turns:]
    while steps:
        messages, goal = _build_episode_messages(steps)
        if messages is None:
            return None, 0, full_n
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=[MOBILE_USE_TOOL]
        )
        try:
            from qwen_vl_utils import process_vision_info

            images, videos = process_vision_info(messages)
        except Exception:
            images = [
                c["image"]
                for m in messages
                if isinstance(m["content"], list)
                for c in m["content"]
                if isinstance(c, dict) and c.get("type") == "image"
            ]
            videos = None
        batch = processor(text=[text], images=images, videos=videos, return_tensors="np")
        input_ids = np.asarray(batch["input_ids"][0], dtype=np.int32)
        if ctx_cap and len(input_ids) > ctx_cap:
            if len(steps) == 1:
                return None, 0, full_n  # even a single turn overflows ctx_cap -> skip
            steps = steps[1:]  # drop oldest turn, keep most-recent-K, retry
            continue
        labels = assistant_labels(input_ids, processor.tokenizer)
        if np.count_nonzero(labels != -100) == 0:
            return None, 0, full_n
        mm_token_type_ids = batch.get("mm_token_type_ids")
        if mm_token_type_ids is None:
            mm_types = (input_ids == IMG_TOKEN_ID).astype(np.int32)
        else:
            mm_types = np.asarray(mm_token_type_ids[0], dtype=np.int32)
        sample = {
            "input_ids": input_ids,
            "labels": labels,
            "mm_token_type_ids": mm_types,
            "token_type_ids": (mm_types > 0).astype(np.bool_),
            "pixel_values": np.asarray(batch["pixel_values"]),
            "image_grid_thw": np.asarray(batch["image_grid_thw"], dtype=np.int32),
            "goal": goal,
            "target": "<episode>",
            "source": "episode",
        }
        return sample, len(steps), full_n
    return None, 0, full_n


def iter_episode_windows(
    parquet_paths: list[Path],
    processor: Any,
    max_samples: int,
    samples_per_window: int,
    ctx_cap: int | None,
    max_pixels: int,
    max_turns: int,
) -> Iterator[tuple[list[dict[str, Any]], dict[str, Any]]]:
    """Stream EPISODE-PACKED samples: group packed-parquet rows by episode_id (contiguous within a
    shard), order by step_id, and emit one multi-turn sample per episode. Whole-shard read is safe
    because the packed builder writes episode-complete rows contiguously at ~4000 rows/shard."""
    from collections import OrderedDict

    import pyarrow.parquet as pq

    window: list[dict[str, Any]] = []
    emitted = 0
    seen_eps = 0
    skipped = 0
    truncated = 0
    window_idx = 0
    target = max(samples_per_window, 0)
    stop = False
    for parquet_idx, parquet_path in enumerate(parquet_paths):
        df = pq.ParquetFile(parquet_path).read().to_pandas()
        if "episode_id" not in df.columns:
            raise ValueError(
                f"--data-mode episode requires an 'episode_id' column; {parquet_path} has {list(df.columns)}"
            )
        groups: "OrderedDict[str, list[Any]]" = OrderedDict()
        for _, row in df.iterrows():
            eid = str(row_value(row, "episode_id", "?"))
            groups.setdefault(eid, []).append(row)
        for eid, steps in groups.items():
            if max_samples and emitted >= max_samples:
                stop = True
                break
            seen_eps += 1
            if steps and row_has(steps[0], "step_id"):
                steps = sorted(steps, key=lambda r: int(row_value(r, "step_id", 0)))
            sample, n_used, n_full = build_episode_sample(steps, processor, ctx_cap, max_pixels, max_turns)
            if sample is None:
                skipped += 1
                continue
            if n_used != n_full:
                truncated += 1
            sample["episode_id"] = eid
            sample["parquet_path"] = str(parquet_path)
            sample["parquet_idx"] = parquet_idx
            sample["global_sample_idx"] = emitted
            sample["n_turns"] = n_used
            sample["n_turns_full"] = n_full
            window.append(sample)
            emitted += 1
            if target and len(window) >= target:
                yield window, {
                    "window_idx": window_idx,
                    "parquet_path": str(parquet_path),
                    "parquet_idx": parquet_idx,
                    "n_samples": len(window),
                    "global_emitted": emitted,
                    "seen_episodes": seen_eps,
                    "skipped_episodes": skipped,
                    "truncated_episodes": truncated,
                }
                window_idx += 1
                window = []
        del df, groups
        if stop:
            break
    if window:
        yield window, {
            "window_idx": window_idx,
            "parquet_path": str(parquet_paths[-1]) if parquet_paths else None,
            "parquet_idx": None,
            "n_samples": len(window),
            "global_emitted": emitted,
            "seen_episodes": seen_eps,
            "skipped_episodes": skipped,
            "truncated_episodes": truncated,
        }


def make_synthetic_sample(seq_len: int, vocab_size: int, response_len: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    input_ids = rng.integers(100, min(vocab_size - 1, 10000), size=(seq_len,), dtype=np.int32)
    labels = np.full((seq_len,), -100, dtype=np.int32)
    start = max(1, seq_len - response_len)
    labels[start:] = input_ids[start:]
    labels[-1] = IM_END_ID
    input_ids[-1] = IM_END_ID
    return {
        "input_ids": input_ids,
        "labels": labels,
        "mm_token_type_ids": np.zeros((seq_len,), dtype=np.int32),
        "token_type_ids": np.zeros((seq_len,), dtype=np.bool_),
        "pixel_values": None,
        "image_grid_thw": None,
        "vision_embeds": np.zeros((0, 2048), dtype=np.float32),
        "deepstack_embeds": [np.zeros((0, 2048), dtype=np.float32) for _ in range(3)],
        "goal": "synthetic",
        "target": "synthetic",
        "source": "synthetic",
    }


def compute_response_block_idx(labels: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    response_mask = labels != -100
    response_block_idx = np.full((labels.shape[0],), -1, dtype=np.int32)
    turn_idx = np.zeros((labels.shape[0],), dtype=np.int32)
    current_block = 0
    in_response = False
    pos_in_seg = 0
    for i in range(labels.shape[0]):
        if response_mask[i]:
            if not in_response:
                in_response = True
                pos_in_seg = 0
            response_block_idx[i] = current_block + (pos_in_seg // block_size)
            pos_in_seg += 1
        elif in_response:
            current_block += (pos_in_seg + block_size - 1) // block_size
            in_response = False
    for i in range(1, labels.shape[0]):
        turn_idx[i] = turn_idx[i - 1] + (1 if response_block_idx[i] != response_block_idx[i - 1] else 0)
    if in_response:
        current_block += (pos_in_seg + block_size - 1) // block_size
    return response_block_idx, turn_idx, current_block


def asymmetric_allowed(q_idx: np.ndarray, kv_idx: np.ndarray, turn_idx_noisy: np.ndarray, turn_idx_clean: np.ndarray, n_noisy: int) -> np.ndarray:
    x0_q = q_idx >= n_noisy
    x0_kv = kv_idx >= n_noisy
    pos_q = np.where(x0_q, q_idx - n_noisy, q_idx)
    pos_kv = np.where(x0_kv, kv_idx - n_noisy, kv_idx)
    tq = np.where(
        x0_q,
        turn_idx_clean[np.clip(pos_q, 0, turn_idx_clean.shape[0] - 1)],
        turn_idx_noisy[np.clip(pos_q, 0, turn_idx_noisy.shape[0] - 1)],
    )
    tk = np.where(
        x0_kv,
        turn_idx_clean[np.clip(pos_kv, 0, turn_idx_clean.shape[0] - 1)],
        turn_idx_noisy[np.clip(pos_kv, 0, turn_idx_noisy.shape[0] - 1)],
    )
    block_diagonal = (~x0_q) & (~x0_kv) & (tq == tk)
    offset_block_causal = (tq > tk) & x0_kv & (~x0_q)
    x0_causal = x0_q & x0_kv & (pos_q >= pos_kv)
    return block_diagonal | offset_block_causal | x0_causal


def get_vision_position_ids_np(start_position: int, grid_thw: np.ndarray, spatial_merge_size: int) -> np.ndarray:
    grid_t, grid_h, grid_w = [int(x) for x in grid_thw.tolist()]
    llm_grid_t = grid_t
    llm_grid_h = grid_h // spatial_merge_size
    llm_grid_w = grid_w // spatial_merge_size
    position_temporal = np.arange(llm_grid_t, dtype=np.int32).repeat(llm_grid_h * llm_grid_w) + start_position
    position_height = np.arange(llm_grid_h, dtype=np.int32) + start_position
    position_height = np.repeat(position_height, llm_grid_w)
    position_height = np.tile(position_height, llm_grid_t)
    position_width = np.arange(llm_grid_w, dtype=np.int32) + start_position
    position_width = np.tile(position_width, llm_grid_h * llm_grid_t)
    return np.stack([position_temporal, position_height, position_width], axis=0)


def compute_mrope_position_ids_np(
    mm_token_type_ids: np.ndarray,
    image_grid_thw: np.ndarray | None,
    *,
    spatial_merge_size: int,
    pad_to: int,
) -> np.ndarray:
    position_ids = np.zeros((3, pad_to), dtype=np.int32)
    current_pos = 0
    out_chunks: list[np.ndarray] = []
    image_i = 0
    i = 0
    n = int(mm_token_type_ids.shape[0])
    while i < n:
        modality = int(mm_token_type_ids[i])
        j = i + 1
        while j < n and int(mm_token_type_ids[j]) == modality:
            j += 1
        if modality == 0:
            text_len = j - i
            out_chunks.append(np.broadcast_to(np.arange(text_len, dtype=np.int32)[None, :] + current_pos, (3, text_len)))
            current_pos += text_len
        elif modality == 1:
            if image_grid_thw is None or image_i >= len(image_grid_thw):
                raise ValueError("Image token type present but image_grid_thw is missing.")
            grid = image_grid_thw[image_i]
            vp = get_vision_position_ids_np(current_pos, grid, spatial_merge_size)
            out_chunks.append(vp)
            current_pos += max(int(grid[1]), int(grid[2])) // spatial_merge_size
            image_i += 1
        else:
            raise ValueError(f"Unsupported mm_token_type_id={modality}; video is not handled in this TPU trainer.")
        i = j
    if out_chunks:
        pos = np.concatenate(out_chunks, axis=1)
        if pos.shape[1] != n:
            raise ValueError(f"mRoPE length mismatch: built {pos.shape[1]}, expected {n}")
        position_ids[:, :n] = pos
    return position_ids


def prepare_dual_arrays(
    sample: dict[str, Any],
    bd: int,
    rng: np.random.Generator,
    min_noise: float,
    *,
    pad_to: int | None = None,
    noisy_pad_to: int | None = None,
    pad_token_id: int = 0,
    loss_token_cap: int = 128,
    pair_batch: int = 2,
) -> dict[str, np.ndarray]:
    ids = np.asarray(sample["input_ids"], dtype=np.int32)
    labels = np.asarray(sample["labels"], dtype=np.int32)
    if pad_to is not None and ids.shape[0] > pad_to:
        raise ValueError(f"Sample length {ids.shape[0]} exceeds --pad-to {pad_to}")
    vision_mask = np.asarray(sample["token_type_ids"], dtype=np.bool_) | (ids == IMG_TOKEN_ID)
    text_positions = np.nonzero(~vision_mask)[0].astype(np.int32)
    lt_actual = int(text_positions.shape[0])
    clean_len = int(pad_to or ids.shape[0])
    lt = int(noisy_pad_to or lt_actual)
    if lt_actual > lt:
        raise ValueError(f"Noisy text length {lt_actual} exceeds noisy_pad_to {lt}")
    total = int(lt + clean_len)

    clean_ids = np.full((clean_len,), pad_token_id, dtype=np.int32)
    clean_labels = np.full((clean_len,), -100, dtype=np.int32)
    clean_vision_mask = np.zeros((clean_len,), dtype=np.bool_)
    clean_ids[: ids.shape[0]] = ids
    clean_labels[: labels.shape[0]] = labels
    clean_vision_mask[: vision_mask.shape[0]] = vision_mask
    clean_mm_types = np.zeros((clean_len,), dtype=np.int32)
    mm_types = np.asarray(sample.get("mm_token_type_ids", sample["token_type_ids"]), dtype=np.int32)
    clean_mm_types[: mm_types.shape[0]] = mm_types

    response_block_idx, turn_idx, n_blocks = compute_response_block_idx(labels, bd)
    turn_idx_clean = np.zeros((clean_len,), dtype=np.int32)
    turn_idx_clean[: turn_idx.shape[0]] = turn_idx
    pblk = (1.0 - min_noise) * rng.random(max(n_blocks, 1), dtype=np.float32) + min_noise
    block_lookup = np.maximum(response_block_idx, 0)
    mask_idx = (rng.random(ids.shape[0]) < pblk[block_lookup]) & (response_block_idx >= 0)
    response = labels != -100
    im_end = (ids == IM_END_ID) & response
    mask_idx = mask_idx | im_end
    comp_idx = (response & ~mask_idx) | im_end

    def make_pair(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        noisy_ids = ids.copy()
        noisy_ids[mask] = MASK_TOKEN_ID
        noisy_labels = labels.copy()
        noisy_labels[~mask] = -100
        noisy_ids_out = np.full((lt,), pad_token_id, dtype=np.int32)
        noisy_labels_out = np.full((lt,), -100, dtype=np.int32)
        noisy_ids_out[:lt_actual] = noisy_ids[text_positions].astype(np.int32)
        noisy_labels_out[:lt_actual] = noisy_labels[text_positions].astype(np.int32)
        return noisy_ids_out, noisy_labels_out

    # pair_batch noised views per sample: pair-0 = mask_idx (heavily-masked, large-block; the kd_fewstep
    # student), pair-1 = comp_idx (complementary mask). pair_batch=1 keeps only pair-0 — a standard
    # single-mask diffusion step that HALVES the train_step noisy forward (memory) at a small coverage cost.
    pair_masks = [mask_idx] if int(pair_batch) <= 1 else [mask_idx, comp_idx]
    n_pairs = len(pair_masks)
    noisy_ids_list: list[np.ndarray] = []
    noisy_labels_list: list[np.ndarray] = []
    for _pm in pair_masks:
        _nid, _nlab = make_pair(_pm)
        noisy_ids_list.append(_nid)
        noisy_labels_list.append(_nlab)

    turn_noisy = np.zeros((lt,), dtype=np.int32)
    turn_noisy[:lt_actual] = turn_idx[text_positions]
    q_idx = np.arange(total, dtype=np.int32)[:, None]
    kv_idx = np.arange(total, dtype=np.int32)[None, :]
    attn_mask = asymmetric_allowed(q_idx, kv_idx, turn_noisy, turn_idx_clean, lt)[None, None, :, :]
    noisy_valid = np.zeros((lt,), dtype=np.bool_)
    noisy_valid[:lt_actual] = True
    clean_valid = np.zeros((clean_len,), dtype=np.bool_)
    clean_valid[: ids.shape[0]] = True
    valid_dual = np.concatenate([noisy_valid, clean_valid], axis=0)
    attn_mask = attn_mask & valid_dual[None, None, :, None] & valid_dual[None, None, None, :]

    clean_pos3 = compute_mrope_position_ids_np(
        mm_types,
        sample.get("image_grid_thw", None),
        spatial_merge_size=2,
        pad_to=clean_len,
    )
    noisy_pos3 = np.zeros((3, lt), dtype=np.int32)
    noisy_pos3[:, :lt_actual] = clean_pos3[:, text_positions]
    position_ids_3d = np.concatenate([noisy_pos3, clean_pos3], axis=1)
    position_ids_3d = np.broadcast_to(position_ids_3d[:, None, :], (3, n_pairs, total)).copy()
    teacher_idx = np.zeros((lt,), dtype=np.int32)
    teacher_idx[:lt_actual] = np.maximum(text_positions - 1, 0)

    def shifted_loss_arrays(label_vec: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        label_pos = np.nonzero(label_vec[1:] != -100)[0].astype(np.int32) + 1
        n_loss = int(label_pos.shape[0])
        if loss_token_cap > 0:
            label_pos = label_pos[:loss_token_cap]
            cap = loss_token_cap
        else:
            cap = max(n_loss, 1)
        pos = np.zeros((cap,), dtype=np.int32)
        tgt = np.zeros((cap,), dtype=np.int32)
        mask = np.zeros((cap,), dtype=np.bool_)
        if label_pos.shape[0] > 0:
            used = min(int(label_pos.shape[0]), cap)
            pos[:used] = np.clip(label_pos[:used] - 1, 0, seq_len - 1)
            tgt[:used] = label_vec[label_pos[:used]].astype(np.int32)
            mask[:used] = True
        return pos, tgt, mask, n_loss

    noisy_loss_pos = []
    noisy_loss_labels = []
    noisy_loss_mask = []
    noisy_teacher_pos = []
    noisy_loss_counts = []
    for nl in noisy_labels_list:
        pos, tgt, msk, n_loss = shifted_loss_arrays(nl, lt)
        label_pos = np.clip(pos + 1, 0, lt - 1)
        noisy_loss_pos.append(pos)
        noisy_loss_labels.append(tgt)
        noisy_loss_mask.append(msk)
        noisy_teacher_pos.append(teacher_idx[label_pos].astype(np.int32))
        noisy_loss_counts.append(n_loss)
    clean_loss_pos, clean_loss_labels, clean_loss_mask, clean_loss_count = shifted_loss_arrays(clean_labels, clean_len)

    return {
        "input_ids": clean_ids,
        "labels": clean_labels,
        "vision_mask": clean_vision_mask,
        "noisy_ids": np.stack(noisy_ids_list, axis=0),
        "noisy_labels": np.stack(noisy_labels_list, axis=0),
        "attn_mask": attn_mask.astype(np.bool_),
        "position_ids_3d": position_ids_3d.astype(np.int32),
        "teacher_idx": teacher_idx.astype(np.int32),
        "noisy_loss_pos": np.stack(noisy_loss_pos, axis=0),
        "noisy_loss_labels": np.stack(noisy_loss_labels, axis=0),
        "noisy_loss_mask": np.stack(noisy_loss_mask, axis=0),
        "noisy_teacher_pos": np.stack(noisy_teacher_pos, axis=0),
        "clean_loss_pos": clean_loss_pos,
        "clean_loss_labels": clean_loss_labels,
        "clean_loss_mask": clean_loss_mask,
        "lt": np.asarray(lt, dtype=np.int32),
        "lt_actual": np.asarray(lt_actual, dtype=np.int32),
        "total": np.asarray(total, dtype=np.int32),
        "n_blocks": np.asarray(n_blocks, dtype=np.int32),
        "masked_tokens": np.asarray(np.count_nonzero(mask_idx) + np.count_nonzero(comp_idx), dtype=np.int32),
        "noisy_loss_tokens": np.asarray(sum(noisy_loss_counts), dtype=np.int32),
        "clean_loss_tokens": np.asarray(clean_loss_count, dtype=np.int32),
        "loss_tokens_truncated": np.asarray(
            int(any(n > loss_token_cap for n in noisy_loss_counts + [clean_loss_count])) if loss_token_cap > 0 else 0,
            dtype=np.int32,
        ),
    }


def masked_ce(logits: jax.Array, labels: jax.Array) -> jax.Array:
    shift_logits = logits[:, :-1, :].astype(jnp.float32)
    shift_labels = labels[:, 1:]
    safe_labels = jnp.maximum(shift_labels, 0)
    token_loss = optax.softmax_cross_entropy_with_integer_labels(shift_logits, safe_labels)
    mask = (shift_labels != -100).astype(jnp.float32)
    return (token_loss * mask).sum() / jnp.maximum(mask.sum(), 1.0)


def masked_kl(student_logits: jax.Array, teacher_logits: jax.Array, labels: jax.Array, temp: jax.Array) -> jax.Array:
    s = student_logits[:, :-1, :].astype(jnp.float32) / temp
    t = jax.lax.stop_gradient(teacher_logits[:, 1:, :].astype(jnp.float32) / temp)
    shift_labels = labels[:, 1:]
    logp_s = jax.nn.log_softmax(s, axis=-1)
    logp_t = jax.nn.log_softmax(t, axis=-1)
    p_t = jnp.exp(logp_t)
    token_kl = jnp.sum(p_t * (logp_t - logp_s), axis=-1) * (temp * temp)
    mask = (shift_labels != -100).astype(jnp.float32)
    return (token_kl * mask).sum() / jnp.maximum(mask.sum(), 1.0)


def sparse_ce_from_hidden(
    hidden: jax.Array,
    positions: jax.Array,
    targets: jax.Array,
    mask: jax.Array,
    emb_table: jax.Array,
) -> jax.Array:
    batch = hidden.shape[0]
    hidden_sel = hidden[jnp.arange(batch)[:, None], positions, :]
    logits = hidden_sel @ emb_table.T
    token_loss = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), targets)
    weights = mask.astype(jnp.float32)
    return (token_loss * weights).sum() / jnp.maximum(weights.sum(), 1.0)


def sparse_ce_kd_from_hidden(
    student_hidden: jax.Array,
    teacher_hidden: jax.Array,
    student_positions: jax.Array,
    teacher_positions: jax.Array,
    targets: jax.Array,
    mask: jax.Array,
    emb_table: jax.Array,
    temp: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    batch = student_hidden.shape[0]
    student_sel = student_hidden[jnp.arange(batch)[:, None], student_positions, :]
    teacher_sel = jax.lax.stop_gradient(teacher_hidden[jnp.arange(batch)[:, None], teacher_positions, :])
    student_logits = student_sel @ emb_table.T
    teacher_logits = teacher_sel @ emb_table.T

    token_ce = optax.softmax_cross_entropy_with_integer_labels(student_logits.astype(jnp.float32), targets)
    s = student_logits.astype(jnp.float32) / temp
    t = jax.lax.stop_gradient(teacher_logits.astype(jnp.float32) / temp)
    logp_s = jax.nn.log_softmax(s, axis=-1)
    logp_t = jax.nn.log_softmax(t, axis=-1)
    p_t = jnp.exp(logp_t)
    token_kl = jnp.sum(p_t * (logp_t - logp_s), axis=-1) * (temp * temp)
    weights = mask.astype(jnp.float32)
    denom = jnp.maximum(weights.sum(), 1.0)
    # Also return the per-token KL and weights (unreduced) so the caller can
    # isolate a single noisy pair (e.g. pair-0 = heavily-masked, large-block
    # few-step student) for the step-axis kd_fewstep term.
    return (token_ce * weights).sum() / denom, (token_kl * weights).sum() / denom, token_kl, weights


def dual_stream_loss_jax(
    model: modeling.Qwen3VLForConditionalGeneration,
    input_ids: jax.Array,
    labels: jax.Array,
    vision_mask: jax.Array,
    vision_embeds: jax.Array,
    deepstack_0: jax.Array,
    deepstack_1: jax.Array,
    deepstack_2: jax.Array,
    noisy_ids: jax.Array,
    noisy_labels: jax.Array,
    attn_mask: jax.Array,
    position_ids_3d: jax.Array,
    teacher_idx: jax.Array,
    noisy_loss_pos: jax.Array,
    noisy_loss_labels: jax.Array,
    noisy_loss_mask: jax.Array,
    noisy_teacher_pos: jax.Array,
    clean_loss_pos: jax.Array,
    clean_loss_labels: jax.Array,
    clean_loss_mask: jax.Array,
    dtype: jnp.dtype,
    ce_noisy_weight: jax.Array,
    ce_clean_weight: jax.Array,
    kd_noisy_weight: jax.Array,
    kd_temp: jax.Array,
    kd_fewstep_weight: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]
        labels = labels[None, :]
        vision_mask = vision_mask[None, :]
        vision_embeds = vision_embeds[None, :, :]
        deepstack_0 = deepstack_0[None, :, :]
        deepstack_1 = deepstack_1[None, :, :]
        deepstack_2 = deepstack_2[None, :, :]
        noisy_ids = noisy_ids[None, :, :]
        noisy_labels = noisy_labels[None, :, :]
        attn_mask = attn_mask[None, :, :, :, :]
        position_ids_3d = position_ids_3d[None, :, :, :]
        teacher_idx = teacher_idx[None, :]
        noisy_loss_pos = noisy_loss_pos[None, :, :]
        noisy_loss_labels = noisy_loss_labels[None, :, :]
        noisy_loss_mask = noisy_loss_mask[None, :, :]
        noisy_teacher_pos = noisy_teacher_pos[None, :, :]
        clean_loss_pos = clean_loss_pos[None, :]
        clean_loss_labels = clean_loss_labels[None, :]
        clean_loss_mask = clean_loss_mask[None, :]

    global_batch = input_ids.shape[0]
    pair_batch = noisy_ids.shape[1]
    lt = noisy_ids.shape[2]
    clean_len = input_ids.shape[1]
    total_len = lt + clean_len

    embed = model.model.language_model.embed_tokens
    clean_emb = embed(input_ids)
    if vision_embeds.shape[1] > 0:
        clean_emb = modeling.batched_merge_modalities(vision_embeds, clean_emb, vision_mask)
    noisy_flat = noisy_ids.reshape(global_batch * pair_batch, lt)
    noisy_emb = embed(noisy_flat).reshape(global_batch, pair_batch, lt, -1)
    clean_pair = jnp.broadcast_to(clean_emb[:, None, :, :], (global_batch, pair_batch, clean_len, clean_emb.shape[-1]))
    demb = jnp.concatenate([noisy_emb, clean_pair], axis=2)
    demb = demb.reshape(global_batch * pair_batch, total_len, demb.shape[-1]).astype(dtype)

    pos3 = jnp.transpose(position_ids_3d, (1, 0, 2, 3)).reshape(3, global_batch * pair_batch, total_len)
    sin, cos = modeling._generate_interleaved_mrope(
        pos3,
        model.config.text_config.head_dim,
        model.config.text_config.rope_theta,
        model.config.text_config.mrope_section,
    )
    visual_pos_masks = jnp.concatenate(
        [jnp.zeros((global_batch, lt), dtype=jnp.bool_), vision_mask], axis=1
    )
    visual_pos_masks = jnp.broadcast_to(
        visual_pos_masks[:, None, :], (global_batch, pair_batch, total_len)
    ).reshape(global_batch * pair_batch, total_len)
    deepstack_visual_embeds = [
        jnp.broadcast_to(ds[:, None, :, :], (global_batch, pair_batch, ds.shape[1], ds.shape[2])).reshape(
            global_batch * pair_batch, ds.shape[1], ds.shape[2]
        )
        for ds in (deepstack_0, deepstack_1, deepstack_2)
    ]
    attn = jnp.broadcast_to(attn_mask, (global_batch, pair_batch, 1, total_len, total_len)).reshape(
        global_batch * pair_batch, 1, total_len, total_len
    )
    # NB: per-layer gradient checkpointing already happens INSIDE language_model when cache is None
    # (modeling.Qwen3VLTextModel: jax.checkpoint(layer) per decoder layer during training). A second
    # top-level remat here is redundant — it doesn't lower the peak further (the backward already
    # recomputes per layer) and it ~triples XLA compile time. So call the model directly.
    hidden = model.model.language_model(
        demb,
        None,
        sin,
        cos,
        attn,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
    )
    emb_table = embed.embedding[...]
    hidden = hidden.reshape(global_batch, pair_batch, total_len, hidden.shape[-1])
    noisy_hidden = hidden[:, :, :lt, :].reshape(global_batch * pair_batch, lt, hidden.shape[-1])
    clean_hidden = hidden[:, 0, lt:, :]
    ce_clean = sparse_ce_from_hidden(
        clean_hidden,
        clean_loss_pos,
        clean_loss_labels,
        clean_loss_mask,
        emb_table,
    )
    ce_noisy, kd_noisy, token_kl, kd_weights = sparse_ce_kd_from_hidden(
        noisy_hidden,
        jnp.broadcast_to(clean_hidden[:, None, :, :], (global_batch, pair_batch, clean_len, hidden.shape[-1])).reshape(
            global_batch * pair_batch, clean_len, hidden.shape[-1]
        ),
        noisy_loss_pos.reshape(global_batch * pair_batch, noisy_loss_pos.shape[-1]),
        noisy_teacher_pos.reshape(global_batch * pair_batch, noisy_teacher_pos.shape[-1]),
        noisy_loss_labels.reshape(global_batch * pair_batch, noisy_loss_labels.shape[-1]),
        noisy_loss_mask.reshape(global_batch * pair_batch, noisy_loss_mask.shape[-1]),
        emb_table,
        kd_temp,
    )
    # kd_fewstep (step-axis self-distillation): forward-KL(clean/AR teacher || noisy student)
    # evaluated ONLY on pair-0 (the mask_idx view = heavily-masked / large-block / few-effective-step
    # state, closest to the bd32 failure regime). kd_noisy above already averages the SAME KL over
    # BOTH pairs (context-axis). The host-side caller scales kd_fewstep_weight = lambda_fs(bd) so this
    # term concentrates on the large-block tail where strict-JSON validity collapses. See PAPER_BIB.md.
    cap = token_kl.shape[-1]
    token_kl_pairs = token_kl.reshape(global_batch, pair_batch, cap)
    kd_weights_pairs = kd_weights.reshape(global_batch, pair_batch, cap)
    fs_kl = token_kl_pairs[:, 0, :]
    fs_w = kd_weights_pairs[:, 0, :]
    kd_fewstep = (fs_kl * fs_w).sum() / jnp.maximum(fs_w.sum(), 1.0)
    loss = (
        ce_noisy_weight * ce_noisy
        + ce_clean_weight * ce_clean
        + kd_noisy_weight * kd_noisy
        + kd_fewstep_weight * kd_fewstep
    )
    return loss, {"ce_noisy": ce_noisy, "ce_clean": ce_clean, "kd_noisy": kd_noisy, "kd_fewstep": kd_fewstep}


def make_optimizer(lr: float, kind: str, weight_decay: float) -> optax.GradientTransformation:
    if kind == "adamw_bf16":
        return optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(mu_dtype=jnp.bfloat16),
            optax.add_decayed_weights(weight_decay),
            optax.scale(-lr),
        )
    if kind == "adamw":
        return optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=weight_decay))
    return optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(lr))


@nnx.jit
def train_step(
    model: modeling.Qwen3VLForConditionalGeneration,
    optimizer: nnx.Optimizer,
    input_ids: jax.Array,
    labels: jax.Array,
    vision_mask: jax.Array,
    vision_embeds: jax.Array,
    deepstack_0: jax.Array,
    deepstack_1: jax.Array,
    deepstack_2: jax.Array,
    noisy_ids: jax.Array,
    noisy_labels: jax.Array,
    attn_mask: jax.Array,
    position_ids_3d: jax.Array,
    teacher_idx: jax.Array,
    noisy_loss_pos: jax.Array,
    noisy_loss_labels: jax.Array,
    noisy_loss_mask: jax.Array,
    noisy_teacher_pos: jax.Array,
    clean_loss_pos: jax.Array,
    clean_loss_labels: jax.Array,
    clean_loss_mask: jax.Array,
    ce_noisy_weight: jax.Array,
    ce_clean_weight: jax.Array,
    kd_noisy_weight: jax.Array,
    kd_temp: jax.Array,
    kd_fewstep_weight: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    def loss_fn(m):
        return dual_stream_loss_jax(
            m,
            input_ids,
            labels,
            vision_mask,
            vision_embeds,
            deepstack_0,
            deepstack_1,
            deepstack_2,
            noisy_ids,
            noisy_labels,
            attn_mask,
            position_ids_3d,
            teacher_idx,
            noisy_loss_pos,
            noisy_loss_labels,
            noisy_loss_mask,
            noisy_teacher_pos,
            clean_loss_pos,
            clean_loss_labels,
            clean_loss_mask,
            jnp.bfloat16,
            ce_noisy_weight,
            ce_clean_weight,
            kd_noisy_weight,
            kd_temp,
            kd_fewstep_weight,
        )

    (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, _TRAINABLE_FILTER), has_aux=True)(model)
    optimizer.update(grads)
    return loss, aux["ce_noisy"], aux["ce_clean"], aux["kd_noisy"], aux["kd_fewstep"]


def compute_vision_embeds(model: modeling.Qwen3VLForConditionalGeneration, sample: dict[str, Any], dtype: jnp.dtype, visual_override: Any = None) -> tuple[jax.Array, list[jax.Array]]:
    # visual_override: a HOST-LOCAL copy of the vision encoder (multihost). The frozen vision encoder
    # is NOT used by the jitted train_step (vision is precomputed), so running it with local params
    # keeps the per-host precompute off the global mesh and avoids a cross-host abort.
    visual = visual_override if visual_override is not None else model.model.visual
    if sample.get("pixel_values") is None:
        empty = jnp.zeros((0, model.config.text_config.hidden_size), dtype=dtype)
        return empty, [empty, empty, empty]
    pixel_values = jnp.asarray(sample["pixel_values"], dtype=dtype)
    grid = np.asarray(sample["image_grid_thw"], dtype=np.int32)
    if grid.shape[0] == 1:
        grid_jax = jnp.asarray(grid)
        try:
            vision_embeds, deepstack = visual.forward_static_with_deepstack(
                pixel_values,
                grid_t=int(grid[0, 0]),
                grid_h=int(grid[0, 1]),
                grid_w=int(grid[0, 2]),
            )
        except Exception:
            vision_embeds, deepstack = visual(pixel_values, grid_jax)
        return vision_embeds.astype(dtype), [d.astype(dtype) for d in deepstack]
    # Multi-image (episode packing): forward each image separately and concatenate the
    # vision + deepstack tokens IN IMAGE ORDER. The downstream merge (merge_modalities /
    # add_visual_embeds) scatters tokens via cumsum(mask)-1, i.e. consumes them in this exact
    # order against the image-token positions of the packed sequence; trailing pad rows are
    # never indexed. pixel_values is the row-concatenation of all images' patches, split here
    # by per-image patch counts (grid_t * grid_h * grid_w).
    patches = (grid[:, 0].astype(np.int64) * grid[:, 1].astype(np.int64) * grid[:, 2].astype(np.int64))
    offs = np.concatenate([[0], np.cumsum(patches)]).astype(np.int64)
    v_parts: list[jax.Array] = []
    d_parts: list[list[jax.Array]] = [[], [], []]
    for i in range(grid.shape[0]):
        pv_i = pixel_values[int(offs[i]) : int(offs[i + 1])]
        try:
            ve_i, ds_i = visual.forward_static_with_deepstack(
                pv_i,
                grid_t=int(grid[i, 0]),
                grid_h=int(grid[i, 1]),
                grid_w=int(grid[i, 2]),
            )
        except Exception:
            ve_i, ds_i = visual(pv_i, jnp.asarray(grid[i : i + 1]))
        v_parts.append(ve_i.astype(dtype))
        for k in range(3):
            d_parts[k].append(ds_i[k].astype(dtype))
    vision_embeds = jnp.concatenate(v_parts, axis=0)
    deepstack = [jnp.concatenate(d_parts[k], axis=0) for k in range(3)]
    return vision_embeds, deepstack


def make_pmap_vision_forward(model: modeling.Qwen3VLForConditionalGeneration, grid_t: int, grid_h: int, grid_w: int):
    cache_key = (id(model), int(grid_t), int(grid_h), int(grid_w))
    cached = _VISION_PMAP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    def _one(pixels: jax.Array):
        vision, deepstack = model.model.visual.forward_static_with_deepstack(
            pixels,
            grid_t=grid_t,
            grid_h=grid_h,
            grid_w=grid_w,
        )
        return vision.astype(jnp.bfloat16), tuple(d.astype(jnp.bfloat16) for d in deepstack)

    mapped = jax.pmap(_one)
    _VISION_PMAP_CACHE[cache_key] = mapped
    return mapped


def _local_vis_split(vis_local: Any) -> tuple[Any, Any]:
    """Cache nnx.split(vis_local) so graphdef identity is STABLE across windows (jit-cache hits)."""
    key = id(vis_local)
    cached = _LOCAL_VIS_SPLIT_CACHE.get(key)
    if cached is None:
        cached = nnx.split(vis_local)
        _LOCAL_VIS_SPLIT_CACHE[key] = cached
    return cached


def make_local_vision_forward(vis_graphdef: Any, grid_t: int, grid_h: int, grid_w: int):
    """JITted single-(local-)device vision forward for the MULTIHOST precompute.

    Why jit and not pmap: the 4 hosts run precompute INDEPENDENTLY over disjoint data shards, so any
    global collective (pmap) would deadlock waiting for lockstep participation. A jit whose inputs and
    params live on one local device is a process-local computation (no cross-host op). Params (`state`)
    are an ARGUMENT, not a closure constant, so they are shared operands rather than constants re-baked
    into every shape variant (which would blow up HBM). One compile per unique (grid_t,grid_h,grid_w).
    """
    key = (id(vis_graphdef), int(grid_t), int(grid_h), int(grid_w))
    fn = _LOCAL_VIS_FWD_CACHE.get(key)
    if fn is not None:
        return fn

    @jax.jit
    def _f(state: Any, pixels: jax.Array):
        visual = nnx.merge(vis_graphdef, state)
        v, ds = visual.forward_static_with_deepstack(pixels, grid_t=grid_t, grid_h=grid_h, grid_w=grid_w)
        return v.astype(jnp.bfloat16), tuple(d.astype(jnp.bfloat16) for d in ds)

    _LOCAL_VIS_FWD_CACHE[key] = _f
    return _f


def compute_vision_embeds_for_window(
    model: modeling.Qwen3VLForConditionalGeneration,
    samples: list[dict[str, Any]],
    dtype: jnp.dtype,
    batch_size: int,
    vis_local: Any = None,
) -> dict[str, int | str]:
    pending = [i for i, sample in enumerate(samples) if sample.get("vision_embeds") is None]
    stats: dict[str, int | str] = {
        "vision_precompute_backend": "pmap",
        "vision_pmap_batches": 0,
        "vision_pmap_samples": 0,
        "vision_fallback_samples": 0,
    }
    if not pending:
        return stats
    if vis_local is not None:
        # MULTIHOST: run vision on THIS process's first TPU chip via a jitted host-local forward. Because every
        # image is resized to FIXED_VISION_WH, only ONE ViT executable compiles (~2-3GB) — it coexists with the
        # replicated model and the remat'd train_step. Embeds are pulled to host numpy. Eager fallback on error.
        stats["vision_precompute_backend"] = "tpu_local_jit"
        vis_dev = jax.local_devices()[0]
        vis_gd, vis_st = _local_vis_split(vis_local)
        for idx in pending:
            sample = samples[idx]
            if sample.get("pixel_values") is None:
                ve, ds = compute_vision_embeds(model, sample, dtype, visual_override=vis_local)
                sample["vision_embeds"] = np.asarray(jax.device_get(ve), dtype=np.float32)
                sample["deepstack_embeds"] = [np.asarray(jax.device_get(d), dtype=np.float32) for d in ds]
                stats["vision_fallback_samples"] = int(stats["vision_fallback_samples"]) + 1
                continue
            pv = jax.device_put(np.asarray(sample["pixel_values"], dtype=dtype), vis_dev)
            grid = np.asarray(sample["image_grid_thw"], dtype=np.int32)
            patches = grid[:, 0].astype(np.int64) * grid[:, 1].astype(np.int64) * grid[:, 2].astype(np.int64)
            offs = np.concatenate([[0], np.cumsum(patches)]).astype(np.int64)
            v_parts: list[np.ndarray] = []
            d_parts: list[list[np.ndarray]] = [[], [], []]
            for i in range(grid.shape[0]):
                pv_i = pv[int(offs[i]) : int(offs[i + 1])]
                try:
                    fwd = make_local_vision_forward(vis_gd, int(grid[i, 0]), int(grid[i, 1]), int(grid[i, 2]))
                    ve_i, ds_i = fwd(vis_st, pv_i)
                except Exception:
                    ve_i, ds_i = vis_local.forward_static_with_deepstack(
                        pv_i, grid_t=int(grid[i, 0]), grid_h=int(grid[i, 1]), grid_w=int(grid[i, 2])
                    )
                v_parts.append(np.asarray(jax.device_get(ve_i), dtype=np.float32))
                for k in range(3):
                    d_parts[k].append(np.asarray(jax.device_get(ds_i[k]), dtype=np.float32))
            sample["vision_embeds"] = np.concatenate(v_parts, axis=0)
            sample["deepstack_embeds"] = [np.concatenate(d_parts[k], axis=0) for k in range(3)]
            stats["vision_pmap_samples"] = int(stats["vision_pmap_samples"]) + 1
        return stats
    batch_size = max(int(batch_size), 1)
    pmap_width = min(max(1, jax.local_device_count()), batch_size)
    if pmap_width < 2:
        stats["vision_precompute_backend"] = "single"
    groups: dict[tuple[Any, ...], list[int]] = {}
    for idx in pending:
        sample = samples[idx]
        if sample.get("pixel_values") is None:
            sample["vision_embeds"], sample["deepstack_embeds"] = compute_vision_embeds(model, sample, dtype)
            stats["vision_fallback_samples"] = int(stats["vision_fallback_samples"]) + 1
            continue
        grid = np.asarray(sample["image_grid_thw"], dtype=np.int32)
        if grid.shape[0] != 1:
            sample["vision_embeds"], sample["deepstack_embeds"] = compute_vision_embeds(model, sample, dtype)
            stats["vision_fallback_samples"] = int(stats["vision_fallback_samples"]) + 1
            continue
        key = (tuple(sample["pixel_values"].shape), int(grid[0, 0]), int(grid[0, 1]), int(grid[0, 2]))
        groups.setdefault(key, []).append(idx)

    for key, indices in groups.items():
        _, grid_t, grid_h, grid_w = key
        pmap_forward = None
        if pmap_width >= 2:
            pmap_forward = make_pmap_vision_forward(model, grid_t, grid_h, grid_w)
        for start in range(0, len(indices), pmap_width):
            chunk = indices[start : start + pmap_width]
            if pmap_forward is None:
                sample = samples[chunk[0]]
                sample["vision_embeds"], sample["deepstack_embeds"] = compute_vision_embeds(model, sample, dtype)
                stats["vision_fallback_samples"] = int(stats["vision_fallback_samples"]) + 1
                continue
            padded_chunk = list(chunk)
            while len(padded_chunk) < pmap_width:
                padded_chunk.append(padded_chunk[-1])
            try:
                pixel_values = jnp.asarray(
                    np.stack([samples[idx]["pixel_values"] for idx in padded_chunk], axis=0),
                    dtype=dtype,
                )
                vision_batch, deepstack_batch = pmap_forward(pixel_values)
                jax.block_until_ready(vision_batch)
                vision_np = np.asarray(jax.device_get(vision_batch))
                deepstack_np = [np.asarray(jax.device_get(ds)) for ds in deepstack_batch]
                for local_idx, sample_idx in enumerate(chunk):
                    samples[sample_idx]["vision_embeds"] = vision_np[local_idx]
                    samples[sample_idx]["deepstack_embeds"] = [ds[local_idx] for ds in deepstack_np]
                stats["vision_pmap_batches"] = int(stats["vision_pmap_batches"]) + 1
                stats["vision_pmap_samples"] = int(stats["vision_pmap_samples"]) + len(chunk)
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "event": "vision_pmap_fallback",
                            "reason": type(exc).__name__,
                            "batch_size": len(chunk),
                            "grid_t": grid_t,
                            "grid_h": grid_h,
                            "grid_w": grid_w,
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
                for sample_idx in chunk:
                    samples[sample_idx]["vision_embeds"], samples[sample_idx]["deepstack_embeds"] = compute_vision_embeds(
                        model,
                        samples[sample_idx],
                        dtype,
                    )
                    stats["vision_fallback_samples"] = int(stats["vision_fallback_samples"]) + 1
    return stats


def pad_vision_embeds(vision_embeds: jax.Array, target_len: int, hidden_size: int, dtype: jnp.dtype) -> jax.Array:
    arr = np.asarray(jax.device_get(vision_embeds))
    if arr.shape[0] > target_len:
        raise ValueError(f"Vision embedding length {arr.shape[0]} exceeds target {target_len}")
    out = np.zeros((target_len, hidden_size), dtype=arr.dtype if arr.size else np.float32)
    if arr.shape[0] > 0:
        out[: arr.shape[0]] = arr
    return jnp.asarray(out, dtype=dtype)


def stack_prepared_batch(prepared_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = {
        "input_ids",
        "labels",
        "vision_mask",
        "noisy_ids",
        "noisy_labels",
        "attn_mask",
        "position_ids_3d",
        "teacher_idx",
        "noisy_loss_pos",
        "noisy_loss_labels",
        "noisy_loss_mask",
        "noisy_teacher_pos",
        "clean_loss_pos",
        "clean_loss_labels",
        "clean_loss_mask",
    }
    return {key: np.stack([p[key] for p in prepared_list], axis=0) for key in keys}


def to_global_array(local: np.ndarray, sharding: NamedSharding | None, global_batch: int | None) -> jax.Array:
    """Place a process-LOCAL batched array onto devices.

    Single-host (global_batch is None): plain device_put over the local mesh (or host array).
    Multi-host (global_batch given): this process holds global_batch // process_count leading rows
    (its data shard); assemble the GLOBAL data-parallel array from the per-process shards via
    make_array_from_process_local_data so the jitted train_step runs over all hosts' chips and XLA
    inserts the cross-host gradient all-reduce. process_index ordering matches Mesh(jax.devices())."""
    if global_batch is not None and sharding is not None:
        gshape = (int(global_batch),) + tuple(local.shape[1:])
        return jax.make_array_from_process_local_data(sharding, np.asarray(local), gshape)
    return jax.device_put(local, sharding) if sharding is not None else jnp.asarray(local)


def put_batch_arrays(
    batch: dict[str, np.ndarray], sharding: NamedSharding | None, global_batch: int | None = None
) -> dict[str, jax.Array]:
    return {key: to_global_array(value, sharding, global_batch) for key, value in batch.items()}


def make_sample_batch_indices(order: list[int], start: int, batch_size: int) -> list[int]:
    idx = order[start : start + batch_size]
    if not idx:
        return idx
    if len(idx) < batch_size:
        idx = idx + idx[: batch_size - len(idx)]
    return idx


def local_mem(tag: str, proc_index: int = 0) -> None:
    """Per-PROCESS HBM probe (jax.local_devices()[0]) — memory_record() only sees global device 0,
    which hides per-host asymmetry. Used to localize where the train_step headroom disappears."""
    try:
        for d in jax.local_devices():
            s = d.memory_stats() or {}
            print(json.dumps({
                "event": "mem_probe", "tag": tag, "proc": proc_index, "device": str(d),
                "in_use": round(float(s.get("bytes_in_use", 0)) / 1e9, 2),
                "free_block": round(float(s.get("largest_free_block_bytes", 0)) / 1e9, 2),
                "reservable": round(float(s.get("bytes_reservable_limit", 0)) / 1e9, 2),
            }), flush=True)
    except Exception as exc:
        print(json.dumps({"event": "mem_probe", "tag": tag, "proc": proc_index, "err": repr(exc)[:120]}), flush=True)


def memory_record() -> dict[str, Any]:
    out: dict[str, Any] = {"device_count": jax.device_count(), "platform": jax.default_backend()}
    try:
        stats = jax.devices()[0].memory_stats() or {}
        for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit"):
            if key in stats and stats[key] is not None:
                out[key] = int(stats[key])
                out[key.replace("bytes", "gb")] = float(stats[key]) / 1e9
    except Exception as exc:
        out["memory_stats_error"] = repr(exc)
    return out


def shell_snapshot() -> dict[str, Any]:
    rec: dict[str, Any] = {}
    for name, cmd in {
        "free": "free -h | sed -n '1,2p'",
        "top_python": "ps -eo pid,pcpu,pmem,cmd --sort=-pcpu | head -8",
    }.items():
        try:
            rec[name] = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT, timeout=5)
        except Exception as exc:
            rec[name] = repr(exc)
    return rec


def flatten_arrays(tree: Any, prefix: str = "") -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            child = f"{prefix}/{key}" if prefix else str(key)
            arrays.update(flatten_arrays(value, child))
    elif isinstance(tree, (list, tuple)):
        for idx, value in enumerate(tree):
            child = f"{prefix}/{idx}" if prefix else str(idx)
            arrays.update(flatten_arrays(value, child))
    elif isinstance(tree, jax.Array):
        arrays[prefix] = np.asarray(jax.device_get(tree))
    elif hasattr(tree, "value") and isinstance(tree.value, jax.Array):
        arrays[prefix] = np.asarray(jax.device_get(tree.value))
    return arrays


def save_nnx_state_npz(model: modeling.Qwen3VLForConditionalGeneration, out_dir: Path, step: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = nnx.to_pure_dict(nnx.state(model))
    arrays = flatten_arrays(state)
    path = out_dir / f"jax_model_state_step{step}.npz"
    np.savez(path, **arrays)
    return path


def _tree_get(tree: Any, path: str) -> Any:
    value = tree
    for part in path.split("."):
        key: Any = int(part) if part.isdigit() else part
        value = value[key]
    return value


def _as_np(tree: Any, path: str, transform: str = "default") -> np.ndarray:
    arr = np.asarray(jax.device_get(_tree_get(tree, path)))
    if transform == "linear":
        arr = arr.transpose(1, 0)
    elif transform == "conv3d":
        arr = arr.transpose(4, 3, 0, 1, 2)
    elif transform != "default":
        raise ValueError(f"Unknown transform: {transform}")
    return np.ascontiguousarray(arr)


def export_hf_safetensors(
    model: modeling.Qwen3VLForConditionalGeneration,
    source_model_dir: Path,
    out_dir: Path,
    step: int,
) -> Path:
    """Export the current NNX model state back to HF-style safetensors.

    This is the inverse of qwen.qwen3vl.params.create_model_from_safe_tensors for
    the Qwen3-VL 2B architecture used here.
    """
    from safetensors.numpy import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "video_preprocessor_config.json",
        "vit_lora_merged_config.json",
    ]:
        src = source_model_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    state = nnx.to_pure_dict(nnx.state(model))
    tensors: dict[str, np.ndarray] = {}
    add = tensors.__setitem__

    # Vision tower.
    add("model.visual.patch_embed.proj.weight", _as_np(state, "model.visual.patch_embed.proj.kernel", "conv3d"))
    add("model.visual.patch_embed.proj.bias", _as_np(state, "model.visual.patch_embed.proj.bias"))
    add("model.visual.pos_embed.weight", _as_np(state, "model.visual.pos_embed.embedding"))
    for i in range(model.config.vision_config.depth):
        p = f"model.visual.blocks.{i}"
        tp = f"model.visual.blocks.{i}"
        add(f"{tp}.norm1.weight", _as_np(state, f"{p}.norm1.scale"))
        add(f"{tp}.norm1.bias", _as_np(state, f"{p}.norm1.bias"))
        add(f"{tp}.norm2.weight", _as_np(state, f"{p}.norm2.scale"))
        add(f"{tp}.norm2.bias", _as_np(state, f"{p}.norm2.bias"))
        add(f"{tp}.attn.qkv.weight", _as_np(state, f"{p}.attn.qkv.kernel", "linear"))
        add(f"{tp}.attn.qkv.bias", _as_np(state, f"{p}.attn.qkv.bias"))
        add(f"{tp}.attn.proj.weight", _as_np(state, f"{p}.attn.proj.kernel", "linear"))
        add(f"{tp}.attn.proj.bias", _as_np(state, f"{p}.attn.proj.bias"))
        add(f"{tp}.mlp.linear_fc1.weight", _as_np(state, f"{p}.mlp.linear_fc1.kernel", "linear"))
        add(f"{tp}.mlp.linear_fc1.bias", _as_np(state, f"{p}.mlp.linear_fc1.bias"))
        add(f"{tp}.mlp.linear_fc2.weight", _as_np(state, f"{p}.mlp.linear_fc2.kernel", "linear"))
        add(f"{tp}.mlp.linear_fc2.bias", _as_np(state, f"{p}.mlp.linear_fc2.bias"))

    p = "model.visual.merger"
    add("model.visual.merger.norm.weight", _as_np(state, f"{p}.norm.scale"))
    add("model.visual.merger.norm.bias", _as_np(state, f"{p}.norm.bias"))
    add("model.visual.merger.linear_fc1.weight", _as_np(state, f"{p}.linear_fc1.kernel", "linear"))
    add("model.visual.merger.linear_fc1.bias", _as_np(state, f"{p}.linear_fc1.bias"))
    add("model.visual.merger.linear_fc2.weight", _as_np(state, f"{p}.linear_fc2.kernel", "linear"))
    add("model.visual.merger.linear_fc2.bias", _as_np(state, f"{p}.linear_fc2.bias"))
    for i in range(len(model.config.vision_config.deepstack_visual_indexes)):
        p = f"model.visual.deepstack_merger_list.{i}"
        tp = f"model.visual.deepstack_merger_list.{i}"
        add(f"{tp}.norm.weight", _as_np(state, f"{p}.norm.scale"))
        add(f"{tp}.norm.bias", _as_np(state, f"{p}.norm.bias"))
        add(f"{tp}.linear_fc1.weight", _as_np(state, f"{p}.linear_fc1.kernel", "linear"))
        add(f"{tp}.linear_fc1.bias", _as_np(state, f"{p}.linear_fc1.bias"))
        add(f"{tp}.linear_fc2.weight", _as_np(state, f"{p}.linear_fc2.kernel", "linear"))
        add(f"{tp}.linear_fc2.bias", _as_np(state, f"{p}.linear_fc2.bias"))

    # Language model.
    emb = _as_np(state, "model.language_model.embed_tokens.embedding")
    add("model.language_model.embed_tokens.weight", emb)
    add("lm_head.weight", emb)
    for i in range(model.config.text_config.num_hidden_layers):
        p = f"model.language_model.layers.{i}"
        tp = f"model.language_model.layers.{i}"
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            add(f"{tp}.self_attn.{name}.weight", _as_np(state, f"{p}.self_attn.{name}.kernel", "linear"))
        add(f"{tp}.self_attn.q_norm.weight", _as_np(state, f"{p}.self_attn.q_norm.weight"))
        add(f"{tp}.self_attn.k_norm.weight", _as_np(state, f"{p}.self_attn.k_norm.weight"))
        for name in ["gate_proj", "up_proj", "down_proj"]:
            add(f"{tp}.mlp.{name}.weight", _as_np(state, f"{p}.mlp.{name}.kernel", "linear"))
        add(f"{tp}.input_layernorm.weight", _as_np(state, f"{p}.input_layernorm.weight"))
        add(f"{tp}.post_attention_layernorm.weight", _as_np(state, f"{p}.post_attention_layernorm.weight"))
    add("model.language_model.norm.weight", _as_np(state, "model.language_model.norm.weight"))

    ckpt_path = out_dir / "model.safetensors"
    save_file(tensors, ckpt_path, metadata={"format": "pt", "jax_export_step": str(step)})
    write_json(
        out_dir / "jax_export_summary.json",
        {
            "step": step,
            "tensor_count": len(tensors),
            "source_model_dir": str(source_model_dir),
            "mrope_exact": True,
            "deepstack_exact": True,
            "note": "Exported from Weasel JAX state. Training path uses Qwen3-VL interleaved mRoPE and DeepStack injection.",
        },
    )
    return ckpt_path


def save_checkpoint_bundle(
    model: modeling.Qwen3VLForConditionalGeneration,
    source_model_dir: Path,
    run_out_dir: Path,
    step: int,
    *,
    final: bool,
) -> Path:
    ckpt_dir = run_out_dir / ("final" if final else f"checkpoint-step{step:06d}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = export_hf_safetensors(model, source_model_dir, ckpt_dir, step)
    for name in ["train_log.jsonl", "tpu_usage.jsonl", "run_config.json", "data_summary.json", "summary.json"]:
        src = run_out_dir / name
        if src.exists():
            shutil.copy2(src, ckpt_dir / name)
    write_json(
        ckpt_dir / "checkpoint_manifest.json",
        {
            "step": step,
            "final": final,
            "checkpoint": str(ckpt_path),
            "source_model_dir": str(source_model_dir),
            "contains_logs": True,
            "format": "hf_safetensors",
        },
    )
    return ckpt_dir


def maybe_upload_checkpoint_bundle(
    model: modeling.Qwen3VLForConditionalGeneration,
    source_model_dir: Path,
    run_out_dir: Path,
    step: int,
    args: argparse.Namespace,
    *,
    final: bool = False,
) -> dict[str, Any]:
    if not args.hf_upload_repo:
        return {"uploaded": False, "reason": "hf_upload_repo_not_set"}
    event: dict[str, Any] = {
        "event": "hf_checkpoint_upload_start",
        "step": step,
        "final": final,
        "repo_id": args.hf_upload_repo,
        "repo_type": args.hf_upload_repo_type,
        "path_in_repo": None,
    }
    append_jsonl(run_out_dir / "train_log.jsonl", event)
    try:
        from huggingface_hub import HfApi

        token = os.environ.get(args.hf_token_env) or os.environ.get("HF_TOKEN")
        ckpt_dir = save_checkpoint_bundle(model, source_model_dir, run_out_dir, step, final=final)
        path_in_repo = f"{args.hf_upload_prefix.rstrip('/')}/{ckpt_dir.name}" if args.hf_upload_prefix else ckpt_dir.name
        api = HfApi(token=token)
        api.create_repo(
            repo_id=args.hf_upload_repo,
            repo_type=args.hf_upload_repo_type,
            private=bool(args.hf_upload_private),
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=args.hf_upload_repo,
            repo_type=args.hf_upload_repo_type,
            folder_path=str(ckpt_dir),
            path_in_repo=path_in_repo,
            token=token,
        )
        rec = {
            "event": "hf_checkpoint_upload_done",
            "step": step,
            "final": final,
            "repo_id": args.hf_upload_repo,
            "repo_type": args.hf_upload_repo_type,
            "path_in_repo": path_in_repo,
            "local_dir": str(ckpt_dir),
        }
        append_jsonl(run_out_dir / "train_log.jsonl", rec)
        if args.delete_local_uploaded_checkpoints and not final:
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            append_jsonl(
                run_out_dir / "train_log.jsonl",
                {"event": "local_uploaded_checkpoint_deleted", "step": step, "local_dir": str(ckpt_dir)},
            )
        return rec
    except Exception as exc:
        rec = {
            "event": "hf_checkpoint_upload_failed",
            "step": step,
            "final": final,
            "repo_id": args.hf_upload_repo,
            "error": repr(exc),
        }
        append_jsonl(run_out_dir / "train_log.jsonl", rec)
        if args.hf_upload_strict:
            raise
        return rec


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out", default="~/tpu_fastdvlm_runs/continue")
    p.add_argument("--data", default=None)
    p.add_argument("--data-pattern", default="*.parquet", help="Glob used when --data is a directory.")
    p.add_argument("--data-mode", choices=["row", "episode"], default="row",
                   help="row = one step per sample (history-as-text, back-compat). episode = pack a whole episode "
                        "(multi-turn: N images + N assistant turns in one sequence, cross-turn attention).")
    p.add_argument("--max-turns", type=int, default=0,
                   help="episode mode: keep at most the most-recent N turns/episode (0=no cap; ctx-cap still trims).")
    p.add_argument("--hf-repo", default="cjfcsjt/AITW_General")
    p.add_argument("--hf-file", default=None)
    p.add_argument("--download-dir", default="~/data/aitw_general")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--max-samples", type=int, default=64)
    p.add_argument(
        "--samples-per-window",
        type=int,
        default=0,
        help="RAM-resident streaming window size. 0 loads the selected data as one window.",
    )
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--start-step", type=int, default=0,
                   help="Resume offset: initialize the step counter here so kd_fewstep warmup + --max-steps "
                        "CONTINUE (not restart) after a spot preemption. Must be identical on all hosts.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1, help="Global batch size. Use a multiple of TPU device count for data-parallel sharding.")
    p.add_argument("--data-parallel", action="store_true", help="Shard batch axis over all local TPU devices and replicate model/optimizer state.")
    p.add_argument("--multihost", action="store_true",
                   help="JAX multi-host data-parallel across a TPU pod (e.g. v6e-16 = 4 hosts x 4 chips). "
                        "Calls jax.distributed.initialize(); each process reads a disjoint file shard and feeds "
                        "global arrays. Requires --data-parallel and --max-steps>0 (fixed global step budget = "
                        "deadlock-free). IO (checkpoint/HF upload) runs on process 0 only.")
    p.add_argument("--bd", type=int, default=32)
    p.add_argument("--bd-schedule", default=None, help="Comma schedule, e.g. '4:0.1,8:0.2,16:0.3,32:0.4'.")
    # degree-2 Gaussian-in-log-b block-size curriculum (W0 paper). static = use --bd-schedule fixed probs.
    p.add_argument("--bd-curriculum", choices=["static", "degree2"], default="static",
                   help="static=--bd-schedule fixed probs (back-compat). degree2=P(b) ∝ exp(-λ1 ln b - λ2 (ln b)^2).")
    p.add_argument("--bd-values", default="1,2,4,8,16,32", help="Support set for --bd-curriculum degree2 (comma ints).")
    p.add_argument("--bd-lambda1", type=float, default=1.0, help="degree2 λ1 (larger => favor small blocks).")
    p.add_argument("--bd-lambda2", type=float, default=0.3,
                   help="degree2 λ2 (log-quadratic; 0 => Boltzmann power law).")
    p.add_argument("--bd-lambda1-end", type=float, default=None,
                   help="With --bd-anneal-steps, cosine-anneal λ1 from --bd-lambda1 to this (mass -> large blocks).")
    p.add_argument("--bd-anneal-steps", type=int, default=0, help="Steps to cosine-anneal λ1 over. 0 disables.")
    p.add_argument("--ctx-cap", type=int, default=2048)
    p.add_argument("--pad-to", type=int, default=0)
    p.add_argument("--noisy-pad-to", type=int, default=0)
    p.add_argument("--vision-pad-to", type=int, default=0)
    p.add_argument(
        "--vision-precompute-batch-size",
        type=int,
        default=16,
        help="Batch size for no-grad/window-level ViT+DeepStack precompute. Does not train vision params.",
    )
    p.add_argument("--loss-token-cap", type=int, default=128, help="Max supervised shifted tokens per branch/sample for sparse LM-head loss. 0 disables truncation.")
    p.add_argument("--pair-batch", type=int, default=2, choices=(1, 2), help="Noised views per sample in the noisy stream. 2=mask+complement (full coverage); 1=mask only (standard single-mask diffusion, ~half the train_step noisy-forward memory).")
    p.add_argument("--pad-token-id", type=int, default=0)
    p.add_argument("--max-pixels", type=int, default=100352)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--response-len", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optim", choices=["sgd", "adamw", "adamw_bf16"], default="sgd")
    p.add_argument("--ce-noisy-weight", type=float, default=1.0)
    p.add_argument("--ce-clean-weight", type=float, default=0.75)
    p.add_argument("--kd-noisy-weight", type=float, default=0.25)
    p.add_argument("--kd-temp", type=float, default=2.0)
    # kd_fewstep: step-axis self-distillation. clean/AR branch (stop-grad) teaches pair-0 (heavily-masked,
    # large-block, few-effective-step) noisy student via forward-KL, on top of the kept kd_noisy.
    # Default 0.0 = OFF = byte-identical to the 3-term loss. Set 0.25 to enable.
    p.add_argument("--kd-fewstep-weight", type=float, default=0.0,
                   help="lambda0 for step-axis AR-teacher->few-step KD on pair-0. 0=off (byte-identical). 0.25=on.")
    p.add_argument("--kd-fewstep-bd-ref", type=float, default=4.0,
                   help="Reference block size; lambda_fs scales with step_bd/bd_ref (bd4 = lossless anchor).")
    p.add_argument("--kd-fewstep-bd-cap", type=float, default=4.0,
                   help="Cap on step_bd/bd_ref. 4.0 = b16-conservative (lambda_fs saturates at lambda0*4 for b>=16).")
    p.add_argument("--kd-fewstep-warmup-steps", type=int, default=500,
                   help="Linear warmup of lambda_fs from 0 over N steps (lets the clean/AR teacher settle first).")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--min-noise", type=float, default=MIN_NOISE)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-final", action="store_true")
    p.add_argument("--save-hf-final", action="store_true")
    p.add_argument("--hf-upload-repo", default=os.environ.get("HF_UPLOAD_REPO"))
    p.add_argument("--hf-upload-repo-type", choices=["model", "dataset"], default="model")
    p.add_argument("--hf-upload-prefix", default="fast-dvlm-kd-tpu")
    p.add_argument("--hf-token-env", default="HF_TOKEN")
    p.add_argument("--hf-upload-private", action="store_true")
    p.add_argument("--hf-upload-strict", action="store_true")
    p.add_argument("--hf-upload-every-steps", type=int, default=0)
    p.add_argument("--hf-upload-final", action="store_true")
    p.add_argument("--delete-local-uploaded-checkpoints", action="store_true")
    p.add_argument("--monitor-every", type=int, default=5)
    p.add_argument(
        "--prefetch-prep",
        action="store_true",
        help="Prepare the next CPU/noising batch in a background thread while the TPU runs the current step.",
    )
    p.add_argument(
        "--prefetch-windows",
        type=int,
        default=0,
        help=(
            "Ouroboros-style raw sample window prefetch depth. This overlaps parquet/image/token "
            "loading for future windows with current-window TPU training. It intentionally does "
            "not run ViT/DeepStack precompute in the background because that uses the same TPU/model "
            "state as training."
        ),
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    # Multi-host: initialize the JAX distributed runtime BEFORE any other jax call so jax.devices()
    # returns the GLOBAL device set (e.g. v6e-16 = 4 hosts x 4 chips = 16). No-arg auto-config on GCP TPU.
    if args.multihost:
        jax.distributed.initialize()
    proc_count = jax.process_count()
    proc_index = jax.process_index()
    is_primary = proc_index == 0
    multihost = bool(args.multihost) and proc_count > 1
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    random.seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    bd_values, bd_probs = parse_bd_schedule(args.bd_schedule, args.bd)
    if args.bd_curriculum == "degree2":
        bd_values = [int(x.strip()) for x in str(args.bd_values).split(",") if x.strip()]
        if not bd_values:
            raise ValueError(f"--bd-values is empty: {args.bd_values!r}")

        def bd_probs_fn(cur_step: int) -> np.ndarray:
            l1 = float(args.bd_lambda1)
            if args.bd_lambda1_end is not None and args.bd_anneal_steps > 0:
                frac = min(max(cur_step / float(args.bd_anneal_steps), 0.0), 1.0)
                cos = 0.5 * (1.0 - np.cos(np.pi * frac))  # 0 -> 1
                l1 = float(args.bd_lambda1) + (float(args.bd_lambda1_end) - float(args.bd_lambda1)) * cos
            return degree2_bd_probs(bd_values, l1, float(args.bd_lambda2))

        bd_probs = bd_probs_fn(0)
    else:
        def bd_probs_fn(cur_step: int) -> np.ndarray:
            return bd_probs

    n_devices = jax.device_count()
    global_batch_size = max(int(args.batch_size), 1)
    if args.data_parallel and global_batch_size % n_devices != 0:
        rounded = ((global_batch_size + n_devices - 1) // n_devices) * n_devices
        print(json.dumps({"event": "batch_size_rounded", "requested": global_batch_size, "rounded": rounded, "device_count": n_devices}))
        global_batch_size = rounded

    # Multi-host data-parallel: each process owns global_batch//proc_count rows (its shard); the global
    # batch is reassembled device-side (see to_global_array). per_process_batch is what THIS process
    # samples/stacks each step. mh_global drives the global-array assembly (None => single-host path).
    per_process_batch = (global_batch_size // proc_count) if multihost else global_batch_size
    mh_global = global_batch_size if multihost else None
    if multihost:
        if not args.data_parallel:
            raise ValueError("--multihost requires --data-parallel")
        if args.max_steps <= 0:
            raise ValueError(
                "--multihost requires --max-steps > 0: a FIXED global step budget keeps all hosts in "
                "lockstep on the per-step collective. Use a high --epochs so a process that exhausts its "
                "file shard re-iterates instead of exiting early (which would deadlock the pod)."
            )
        if global_batch_size % proc_count != 0:
            raise ValueError(f"global batch {global_batch_size} not divisible by process_count {proc_count}")
        if per_process_batch % jax.local_device_count() != 0:
            raise ValueError(
                f"per-process batch {per_process_batch} not divisible by local_device_count {jax.local_device_count()}"
            )
        # SHAPE LOCK: the per-step train_step is the only cross-host collective and runs over arrays
        # assembled by make_array_from_process_local_data((global,)+local.shape[1:]). The TRAILING dims
        # (clean_len, noisy lt, vision rows, loss-token cap) must be byte-identical on every host, else
        # make_array assembles an inconsistent global shape and the SPMD collective hangs. With the pad
        # flags at 0 they are derived per-window from each host's DISJOINT shard and diverge -> require them.
        if not (args.pad_to and (args.noisy_pad_to or args.pad_to) and args.vision_pad_to and args.loss_token_cap):
            raise ValueError(
                "--multihost requires fixed shapes on every host: set --pad-to, --noisy-pad-to (or --pad-to), "
                "--vision-pad-to, and --loss-token-cap > 0. Per-window-max padding diverges across data "
                "shards and hangs the cross-host collective."
            )
        print(json.dumps({
            "event": "multihost_init", "proc_index": proc_index, "proc_count": proc_count,
            "local_devices": jax.local_device_count(), "global_devices": n_devices,
            "global_batch": global_batch_size, "per_process_batch": per_process_batch,
        }))

    run_config = {
        "model_dir": str(Path(args.model_dir).expanduser()),
        "bd": args.bd,
        "bd_curriculum": args.bd_curriculum,
        "bd_schedule": [{"bd": bd, "prob": float(prob)} for bd, prob in zip(bd_values, bd_probs)],
        "bd_degree2": (
            {"lambda1": args.bd_lambda1, "lambda2": args.bd_lambda2,
             "lambda1_end": args.bd_lambda1_end, "anneal_steps": args.bd_anneal_steps}
            if args.bd_curriculum == "degree2" else None
        ),
        "kd_fewstep": {
            "weight": args.kd_fewstep_weight, "bd_ref": args.kd_fewstep_bd_ref,
            "bd_cap": args.kd_fewstep_bd_cap, "warmup_steps": args.kd_fewstep_warmup_steps,
        },
        "batch_size": global_batch_size,
        "data_parallel": bool(args.data_parallel),
        "lr": args.lr,
        "optim": args.optim,
        "dtype": args.dtype,
        "max_samples": args.max_samples,
        "samples_per_window": args.samples_per_window,
        "max_steps": args.max_steps,
        "epochs": args.epochs,
        "loss_token_cap": args.loss_token_cap,
        "jax_version": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "mrope_exact": True,
        "mrope_note": "Qwen3-VL interleaved 3D mRoPE is computed from mm_token_type_ids/image_grid_thw.",
        "deepstack_exact": True,
        "deepstack_note": "Vision pooler embeddings are merged and DeepStack features are injected into early text layers.",
        "ce_noisy_weight": args.ce_noisy_weight,
        "ce_clean_weight": args.ce_clean_weight,
        "kd_noisy_weight": args.kd_noisy_weight,
        "kd_temp": args.kd_temp,
        "teacher": "same-model clean branch with stop_gradient",
        "vision_grad": False,
        "vision_grad_note": "Vision embeddings are precomputed per sample to avoid tracing dynamic image grids.",
        "vision_precompute_batch_size": args.vision_precompute_batch_size,
        "hf_upload_repo": args.hf_upload_repo,
        "hf_upload_every_steps": args.hf_upload_every_steps,
        "hf_upload_prefix": args.hf_upload_prefix,
        "prefetch_prep": bool(args.prefetch_prep),
    }
    write_json(out_dir / "run_config.json", run_config)

    print(json.dumps({"event": "load_model_start", **run_config}, ensure_ascii=True))
    config = modeling.ModelConfig.qwen3vl_2b()
    model = params.create_model_from_safe_tensors(str(Path(args.model_dir).expanduser()), config)
    # Optimizer is created AFTER the model is replicated (below). Building it here, before replication,
    # would force the multihost replication step to duplicate the (large) adamw moment state in HBM
    # transiently (old local copy + new global copy coexisting), spiking the load peak to ~29.8GB/chip
    # and stranding HBM so train_step can't fit. Built on the already-replicated model, the moments are
    # allocated directly as global arrays -> no duplicate, ~10GB lower peak.
    optimizer: nnx.Optimizer | None = None
    data_sharding = None
    vis_local = None  # host-local vision encoder for multihost precompute (set below); None = use model.model.visual
    if args.data_parallel:
        mesh = Mesh(np.asarray(jax.devices()), axis_names=("dp",))
        data_sharding = NamedSharding(mesh, P("dp"))
        replicated = NamedSharding(mesh, P())
        if multihost:
            # Fail fast if the 4 hosts did NOT load byte-identical weights (stale/partial per-host copy):
            # make_array_from_callback would build a 'replicated' array from divergent data -> every chip
            # holds a different replica -> all collectives silently corrupt. Cheap cross-host checksum.
            from jax.experimental import multihost_utils as _mhu
            _leaves = jax.tree.leaves(nnx.state(model))
            _chk = np.asarray([sum(float(np.asarray(v).sum()) for v in _leaves[:16])], dtype=np.float64)
            _all = np.asarray(_mhu.process_allgather(_chk))
            if float(_all.max() - _all.min()) > 1e-3 * (abs(float(_all.mean())) + 1.0):
                raise ValueError(
                    f"multihost weight checksum mismatch across hosts {_all.ravel().tolist()}: hosts loaded "
                    "different --model-dir contents. Ensure every worker has the SAME checkpoint."
                )

            # Replicate params/optimizer across ALL hosts' chips. jax.device_put can't target
            # non-addressable (remote-host) devices, so build each leaf as a globally-replicated array
            # via make_array_from_callback. Every process loaded the SAME checkpoint, so the replicas
            # are identical -> consistent global replicated state.
            def _replicate_global(x: jax.Array) -> jax.Array:
                host_local = np.asarray(x)
                return jax.make_array_from_callback(host_local.shape, replicated, lambda idx: host_local[idx])

            nnx.update(model, jax.tree.map(_replicate_global, nnx.state(model)))
        else:
            nnx.update(model, jax.device_put(nnx.state(model), replicated))
        if multihost:
            # HOST-LOCAL copy of the frozen vision encoder on this process's first TPU chip. Per-host-local
            # params keep the vision forward process-local (no cross-host abort). Because every screenshot is
            # resized to a SINGLE fixed size (FIXED_VISION_WH), the jit precompute compiles exactly ONE ViT
            # executable (~2-3GB) instead of one per aspect-ratio (~18-30GB) — so it now fits on-chip alongside
            # the replicated model and the remat'd train_step. Vision stays on the TPU (fast), not the CPU
            # (which was ~290s/window and starved the host).
            _vis_dev = jax.local_devices()[0]
            _vis_gd, _vis_state = nnx.split(model.model.visual)
            _local_vis_state = jax.tree.map(
                lambda x: jax.device_put(np.asarray(x), _vis_dev), _vis_state
            )
            vis_local = nnx.merge(_vis_gd, _local_vis_state)
            # Drop the globally-replicated vision encoder, replacing it with the host-local copy. Safe now
            # that the vision encoder is excluded from BOTH the optimizer (wrt=_TRAINABLE_FILTER) and the
            # grad (DiffState) — so no adam moments/grad buffers are made for it; the old global arrays are
            # unreferenced and freed (~1.2GB/chip). train_step never reads it; checkpoint export still sees
            # the (unchanged, frozen) visual params.
            model.model.visual = vis_local
            gc.collect()
            print(json.dumps({"event": "vis_local_built", "device": str(_vis_dev)}), flush=True)
        print(
            json.dumps(
                {
                    "event": "data_parallel_enabled",
                    "device_count": n_devices,
                    "batch_size": global_batch_size,
                    "per_device_batch": global_batch_size // n_devices,
                },
                ensure_ascii=True,
            )
        )
    # Create the optimizer on the (now replicated, if data-parallel) model so its moment state is
    # allocated directly with the final sharding -> no transient duplicate during replication.
    optimizer = nnx.Optimizer(model, make_optimizer(args.lr, args.optim, args.weight_decay), wrt=_TRAINABLE_FILTER)
    print(json.dumps({"event": "load_model_done", **memory_record()}, ensure_ascii=True))

    # --start-step: spot-resume offset (identical on all hosts -> lockstep preserved). The kd_fewstep
    # warmup ramp and the --max-steps budget CONTINUE from here instead of restarting on every preemption.
    # NOTE: Adam moment state is NOT in the HF safetensors export, so resuming reloads weights but restarts
    # the optimizer moments from zero (acceptable for a short 1-epoch SFT; flagged in the playbook).
    step = int(args.start_step)
    t_start = time.time()
    last_monitor = 0.0
    last_upload_step = step
    source_model_dir = Path(args.model_dir).expanduser()
    synthetic_samples: list[dict[str, Any]] | None = None
    processor = None
    parquet_files: list[Path] = []

    if args.synthetic:
        synthetic_samples = [
            make_synthetic_sample(args.seq_len, config.text_config.vocab_size, args.response_len, args.seed + i)
            for i in range(max(args.max_samples, 1))
        ]
    else:
        from transformers import AutoProcessor

        parquet_files = resolve_parquet_files(args)
        if not parquet_files:
            raise ValueError("Pass --synthetic, --data, or --hf-file.")
        if multihost:
            # Disjoint file shard per process so the 4 hosts train on different episodes (data-parallel).
            sharded = parquet_files[proc_index::proc_count]
            if not sharded:
                raise ValueError(
                    f"process {proc_index}: 0 parquet files after sharding {len(parquet_files)} across "
                    f"{proc_count} processes — use more shards or fewer hosts."
                )
            print(json.dumps({"event": "file_shard", "proc_index": proc_index,
                              "files_total": len(parquet_files), "files_this_proc": len(sharded)}))
            parquet_files = sharded
        processor = AutoProcessor.from_pretrained(
            str(source_model_dir),
            trust_remote_code=True,
            max_pixels=args.max_pixels,
        )

    write_json(
        out_dir / "data_summary.json",
        {
            "synthetic": bool(args.synthetic),
            "data_mode": args.data_mode,
            "max_turns": args.max_turns,
            "parquet_files": [str(p) for p in parquet_files],
            "n_parquet_files": len(parquet_files),
            "data_pattern": args.data_pattern,
            "max_samples": args.max_samples,
            "samples_per_window": args.samples_per_window,
            "ctx_cap": args.ctx_cap,
            "pad_to": args.pad_to,
            "noisy_pad_to": args.noisy_pad_to or args.pad_to,
            "vision_pad_to": args.vision_pad_to,
        },
    )

    def sample_windows_for_epoch(epoch: int) -> Iterator[tuple[list[dict[str, Any]], dict[str, Any]]]:
        if synthetic_samples is not None:
            yield list(synthetic_samples), {
                "window_idx": 0,
                "n_samples": len(synthetic_samples),
                "parquet_path": None,
                "global_emitted": len(synthetic_samples),
            }
            return
        assert processor is not None
        if args.data_mode == "episode":
            yield from iter_episode_windows(
                parquet_files,
                processor,
                args.max_samples,
                args.samples_per_window,
                args.ctx_cap,
                args.max_pixels,
                args.max_turns,
            )
        else:
            yield from iter_row_sample_windows(
                parquet_files,
                processor,
                args.max_samples,
                args.samples_per_window,
                args.ctx_cap,
                args.max_pixels,
            )

    def prepare_sample_window(samples: list[dict[str, Any]], epoch: int, window_meta: dict[str, Any]) -> list[dict[str, Any]]:
        window_t0 = time.time()
        before = len(samples)
        local_mem("prep_start", proc_index)
        vision_t0 = time.time()
        vision_stats = compute_vision_embeds_for_window(model, samples, dtype, args.vision_precompute_batch_size, vis_local=vis_local)
        vision_precompute_sec = time.time() - vision_t0
        local_mem("after_vision", proc_index)
        if args.pad_to:
            samples = [s for s in samples if len(s["input_ids"]) <= args.pad_to]
            if not samples:
                raise RuntimeError(f"No samples fit --pad-to {args.pad_to} in window {window_meta}")
        if args.vision_pad_to or args.pad_to:
            vision_target = args.vision_pad_to or max(int(s["vision_embeds"].shape[0]) for s in samples)
            for sample in samples:
                sample["vision_embeds"] = pad_vision_embeds(
                    sample["vision_embeds"],
                    vision_target,
                    config.text_config.hidden_size,
                    dtype,
                )
                sample["deepstack_embeds"] = [
                    pad_vision_embeds(ds, vision_target, config.text_config.hidden_size, dtype)
                    for ds in sample["deepstack_embeds"]
                ]
        else:
            vision_target = max(int(s["vision_embeds"].shape[0]) for s in samples)
        for sample in samples:
            sample["vision_embeds"] = np.asarray(jax.device_get(sample["vision_embeds"]), dtype=np.float32)
            sample["deepstack_embeds"] = [
                np.asarray(jax.device_get(ds), dtype=np.float32)
                for ds in sample["deepstack_embeds"]
            ]
        rec = {
            "event": "window_ready",
            "epoch": epoch,
            **window_meta,
            "n_samples_before_pad_filter": before,
            "n_samples_after_pad_filter": len(samples),
            "first_lengths": [int(len(s["input_ids"])) for s in samples[:8]],
            "input_len_min": int(min(len(s["input_ids"]) for s in samples)),
            "input_len_max": int(max(len(s["input_ids"]) for s in samples)),
            "vision_pad_to": int(vision_target),
            "vision_precompute_sec": vision_precompute_sec,
            "vision_precompute_batch_size": args.vision_precompute_batch_size,
            **vision_stats,
            "prepare_window_sec": time.time() - window_t0,
            "window_prefetch_raw": bool(args.prefetch_windows),
            "ram_snapshot": shell_snapshot().get("free"),
            **memory_record(),
        }
        append_jsonl(out_dir / "data_windows.jsonl", rec)
        append_jsonl(log_path, rec)
        return samples

    def train_window(samples: list[dict[str, Any]], epoch: int, window_meta: dict[str, Any]) -> None:
        nonlocal step, last_monitor, last_upload_step

        def prepare_batch_host(sample_indices: list[int], step_bd: int, seed: int) -> dict[str, Any]:
            local_rng = np.random.default_rng(seed)
            prep_t0 = time.time()
            prepared_list = [
                prepare_dual_arrays(
                    samples[sample_idx],
                    step_bd,
                    local_rng,
                    args.min_noise,
                    pad_to=args.pad_to or None,
                    noisy_pad_to=args.noisy_pad_to or args.pad_to or None,
                    pad_token_id=args.pad_token_id,
                    loss_token_cap=args.loss_token_cap,
                    pair_batch=args.pair_batch,
                )
                for sample_idx in sample_indices
            ]
            stacked = stack_prepared_batch(prepared_list)
            return {
                "sample_indices": sample_indices,
                "step_bd": step_bd,
                "prepared_list": prepared_list,
                "stacked": stacked,
                "prep_sec": time.time() - prep_t0,
            }

        def make_prep_request(order: list[int], batch_start: int) -> tuple[list[int], int, int] | None:
            sample_indices = make_sample_batch_indices(order, batch_start, per_process_batch)
            if not sample_indices:
                return None
            # Masking bd (per-host, decided at prep time on this host's own data shard — does NOT affect
            # the fixed padded array shapes, so cross-host consistency is unnecessary here). The kd_fewstep
            # loss WEIGHT (lambda_fs) is instead derived from the consumption step at dispatch time so it
            # stays identical across hosts.
            step_bd = int(np_rng.choice(bd_values, p=bd_probs_fn(step)))
            seed = int(np_rng.integers(0, np.iinfo(np.uint32).max))
            return sample_indices, step_bd, seed

        def submit_prep(
            executor: ThreadPoolExecutor | None,
            request: tuple[list[int], int, int] | None,
        ) -> Future[dict[str, Any]] | dict[str, Any] | None:
            if request is None:
                return None
            sample_indices, step_bd, seed = request
            if executor is None:
                return prepare_batch_host(sample_indices, step_bd, seed)
            return executor.submit(prepare_batch_host, sample_indices, step_bd, seed)

        prep_executor = ThreadPoolExecutor(max_workers=1) if args.prefetch_prep else None
        try:
            order = list(range(len(samples)))
            random.shuffle(order)
            batch_starts = list(range(0, len(order), per_process_batch))
            batch_pos = 0
            pending_prep = submit_prep(
                prep_executor,
                make_prep_request(order, batch_starts[batch_pos]) if batch_pos < len(batch_starts) else None,
            )
            while pending_prep is not None:
                if args.max_steps and step >= args.max_steps:
                    break
                wall_t0 = time.time()
                prep_wait_t0 = time.time()
                if isinstance(pending_prep, Future):
                    prepared_payload = pending_prep.result()
                else:
                    prepared_payload = pending_prep
                prep_wait_sec = time.time() - prep_wait_t0
                batch_pos += 1
                pending_prep = submit_prep(
                    prep_executor,
                    make_prep_request(order, batch_starts[batch_pos]) if batch_pos < len(batch_starts) else None,
                )

                sample_indices = prepared_payload["sample_indices"]
                step_bd = prepared_payload["step_bd"]
                prepared_list = prepared_payload["prepared_list"]
                prep_sec = prepared_payload["prep_sec"]
                put_t0 = time.time()
                arrays = put_batch_arrays(prepared_payload["stacked"], data_sharding, mh_global)
                vision_embeds = to_global_array(
                    np.stack([samples[sample_idx]["vision_embeds"] for sample_idx in sample_indices], axis=0),
                    data_sharding,
                    mh_global,
                ).astype(dtype)
                deepstack = [
                    to_global_array(
                        np.stack([samples[sample_idx]["deepstack_embeds"][i] for sample_idx in sample_indices], axis=0),
                        data_sharding,
                        mh_global,
                    ).astype(dtype)
                    for i in range(3)
                ]
                device_put_sec = time.time() - put_t0

                # kd_fewstep: host-side lambda_fs(bd) = lambda0 * warmup_ramp * min(bd/bd_ref, bd_cap).
                # bd is only known host-side (step_bd), so we fold the bd-scaling + warmup into a single
                # scalar weight passed to the (jitted) train_step -> no bd threaded into JIT, no recompile.
                # bd_cap=4.0 with bd_ref=4 = b16-conservative: lambda_fs saturates at lambda0*4 for b>=16
                # (b4:0.25, b8:0.5, b16:1.0, b32:1.0 when lambda0=0.25). weight=0 => byte-identical 3-term loss.
                fs_warmup = max(int(args.kd_fewstep_warmup_steps), 1)
                fs_ramp = min((step + 1) / fs_warmup, 1.0)
                # weight_bd is drawn from the CONSUMPTION step (this exact global train step, identical and
                # in lockstep across hosts), NOT the per-host masking step_bd (decided at prep time on each
                # host's own shard). This guarantees lambda_fs is the SAME scalar on every host so the
                # gradient all-reduce averages one consistently-weighted objective.
                if multihost:
                    _wbd_rng = np.random.default_rng((int(args.seed), int(step)))
                    weight_bd = int(_wbd_rng.choice(bd_values, p=bd_probs_fn(step)))
                else:
                    weight_bd = step_bd
                fs_bd_factor = min(
                    float(weight_bd) / max(float(args.kd_fewstep_bd_ref), 1e-9), float(args.kd_fewstep_bd_cap)
                )
                lambda_fs = float(args.kd_fewstep_weight) * fs_ramp * fs_bd_factor

                t0 = time.time()
                loss, ce_noisy, ce_clean, kd_noisy, kd_fewstep = train_step(
                    model,
                    optimizer,
                    arrays["input_ids"],
                    arrays["labels"],
                    arrays["vision_mask"],
                    vision_embeds,
                    deepstack[0],
                    deepstack[1],
                    deepstack[2],
                    arrays["noisy_ids"],
                    arrays["noisy_labels"],
                    arrays["attn_mask"],
                    arrays["position_ids_3d"],
                    arrays["teacher_idx"],
                    arrays["noisy_loss_pos"],
                    arrays["noisy_loss_labels"],
                    arrays["noisy_loss_mask"],
                    arrays["noisy_teacher_pos"],
                    arrays["clean_loss_pos"],
                    arrays["clean_loss_labels"],
                    arrays["clean_loss_mask"],
                    jnp.asarray(args.ce_noisy_weight, dtype=jnp.float32),
                    jnp.asarray(args.ce_clean_weight, dtype=jnp.float32),
                    jnp.asarray(args.kd_noisy_weight, dtype=jnp.float32),
                    jnp.asarray(args.kd_temp, dtype=jnp.float32),
                    jnp.asarray(lambda_fs, dtype=jnp.float32),
                )
                jax.block_until_ready(loss)
                if step < 4:
                    local_mem(f"after_step{step}", proc_index)  # watch per-step HBM growth on first steps
                compute_sec = time.time() - t0
                wall_step_sec = time.time() - wall_t0
                step += 1

                total_tokens = sum(int(prepared["total"]) * 2 for prepared in prepared_list)
                rec = {
                    "event": "train_step",
                    "step": step,
                    "epoch": epoch,
                    "window_idx": window_meta.get("window_idx"),
                    "window_sample_count": len(samples),
                    "sample_idx": sample_indices,
                    "global_sample_idx": [int(samples[i].get("global_sample_idx", i)) for i in sample_indices],
                    "bd": step_bd,
                    "loss": float(loss),
                    "ce_noisy": float(ce_noisy),
                    "ce_clean": float(ce_clean),
                    "kd_noisy": float(kd_noisy),
                    "kd_fewstep": float(kd_fewstep),
                    "ce_noisy_weight": args.ce_noisy_weight,
                    "ce_clean_weight": args.ce_clean_weight,
                    "kd_noisy_weight": args.kd_noisy_weight,
                    "kd_fewstep_weight": float(args.kd_fewstep_weight),
                    "kd_fewstep_lambda": lambda_fs,
                    "kd_temp": args.kd_temp,
                    "teacher": "same-model clean branch with stop_gradient",
                    "elapsed_sec": compute_sec,
                    "compute_sec": compute_sec,
                    "prep_sec": prep_sec,
                    "prep_wait_sec": prep_wait_sec,
                    "prefetch_prep": bool(args.prefetch_prep),
                    "device_put_sec": device_put_sec,
                    "wall_step_sec": wall_step_sec,
                    "tokens_per_sec": total_tokens / max(compute_sec, 1e-9),
                    "wall_tokens_per_sec": total_tokens / max(wall_step_sec, 1e-9),
                    "host_overhead_sec": max(wall_step_sec - compute_sec, 0.0),
                    "host_overhead_frac": max(wall_step_sec - compute_sec, 0.0) / max(wall_step_sec, 1e-9),
                    "batch_size": len(sample_indices),
                    "per_device_batch": len(sample_indices) // n_devices if args.data_parallel else None,
                    "input_len_min": int(min(len(samples[i]["input_ids"]) for i in sample_indices)),
                    "input_len_max": int(max(len(samples[i]["input_ids"]) for i in sample_indices)),
                    "total_dual_len": int(prepared_list[0]["total"]),
                    "n_blocks_mean": float(np.mean([int(prepared["n_blocks"]) for prepared in prepared_list])),
                    "masked_tokens_pair_sum": int(sum(int(prepared["masked_tokens"]) for prepared in prepared_list)),
                    "noisy_loss_tokens_sum": int(sum(int(prepared["noisy_loss_tokens"]) for prepared in prepared_list)),
                    "clean_loss_tokens_sum": int(sum(int(prepared["clean_loss_tokens"]) for prepared in prepared_list)),
                    "loss_tokens_truncated_count": int(sum(int(prepared["loss_tokens_truncated"]) for prepared in prepared_list)),
                    "mrope_exact": True,
                    "deepstack_exact": True,
                    **memory_record(),
                }
                append_jsonl(log_path, rec)
                if step % args.log_every == 0:
                    print(json.dumps(rec, ensure_ascii=True))
                now = time.time()
                if args.monitor_every > 0 and now - last_monitor >= args.monitor_every:
                    append_jsonl(out_dir / "tpu_usage.jsonl", {"event": "usage", "step": step, **memory_record(), **shell_snapshot()})
                    last_monitor = now
                if (
                    is_primary
                    and args.hf_upload_every_steps
                    and args.hf_upload_repo
                    and step % args.hf_upload_every_steps == 0
                    and step != last_upload_step
                ):
                    # process-0 only: the model state is replicated, so device_get on host 0 yields the
                    # full weights with no collective. Other hosts must NOT also upload (HF repo conflict).
                    maybe_upload_checkpoint_bundle(model, source_model_dir, out_dir, step, args, final=False)
                    last_upload_step = step
        finally:
            if prep_executor is not None:
                prep_executor.shutdown(wait=True)

    # Epoch driver. Multi-host: the loop is bounded ONLY by --max-steps (every host must emit EXACTLY
    # max_steps cross-host train_step collectives), and the data source cycles indefinitely so a host
    # that exhausts its disjoint file shard re-iterates instead of dropping out of the collective (which
    # would hang the pod). Single-host: bounded by --epochs (unchanged). Any host-local fatal error in
    # multihost calls os._exit(1) to crash the WHOLE pod fast rather than leave peers blocked in all-reduce.
    epoch = 0
    try:
        while not (args.max_steps and step >= args.max_steps) and (multihost or epoch < args.epochs):
            windows_this_epoch = 0
            window_iter = iter(sample_windows_for_epoch(epoch))
            if args.prefetch_windows > 0 and synthetic_samples is None:
                # Ouroboros-style producer/consumer at window granularity: future raw windows are
                # prepared on host threads while the TPU trains the current one. ViT/DeepStack precompute
                # stays serial on the main thread (it mutates the shared NNX model/optimizer state).
                raw_queue: queue.Queue[tuple[list[dict[str, Any]], dict[str, Any]] | None] = queue.Queue(
                    maxsize=max(1, args.prefetch_windows)
                )
                stop_raw_prefetch = threading.Event()

                def raw_window_producer() -> None:
                    try:
                        for payload in window_iter:
                            while not stop_raw_prefetch.is_set():
                                try:
                                    raw_queue.put(payload, timeout=0.5)
                                    break
                                except queue.Full:
                                    continue
                            if stop_raw_prefetch.is_set():
                                break
                    finally:
                        while not stop_raw_prefetch.is_set():
                            try:
                                raw_queue.put(None, timeout=0.5)
                                break
                            except queue.Full:
                                continue

                producer_thread = threading.Thread(target=raw_window_producer, daemon=True)
                producer_thread.start()
                try:
                    while True:
                        if args.max_steps and step >= args.max_steps:
                            break
                        try:
                            raw_payload = raw_queue.get(timeout=30)
                        except queue.Empty:
                            if not producer_thread.is_alive():
                                break
                            continue
                        if raw_payload is None:
                            break
                        raw_samples, window_meta = raw_payload
                        if not raw_samples:
                            continue
                        samples = prepare_sample_window(raw_samples, epoch, window_meta)
                        local_mem("before_train", proc_index)
                        train_window(samples, epoch, window_meta)
                        windows_this_epoch += 1
                        del samples
                        del raw_samples
                        gc.collect()
                finally:
                    stop_raw_prefetch.set()
            else:
                for raw_samples, window_meta in window_iter:
                    if args.max_steps and step >= args.max_steps:
                        break
                    if not raw_samples:
                        continue
                    samples = prepare_sample_window(raw_samples, epoch, window_meta)
                    train_window(samples, epoch, window_meta)
                    windows_this_epoch += 1
                    del samples
                    del raw_samples
                    gc.collect()
            if multihost and windows_this_epoch == 0:
                # No trainable windows this whole pass => this host can never reach max_steps while
                # peers keep calling the collective. Crash the pod fast instead of hanging it.
                raise RuntimeError(
                    f"proc {proc_index}: data shard produced 0 trainable windows in epoch {epoch}; "
                    "cannot keep the cross-host collective in lockstep."
                )
            epoch += 1
    except BaseException as exc:  # multihost: any host-local failure must abort ALL hosts, not hang them
        if multihost:
            print(json.dumps({"event": "multihost_fatal", "proc_index": proc_index, "step": step,
                              "error": repr(exc)}, ensure_ascii=True), flush=True)
            os._exit(1)
        raise

    summary = {
        "event": "done",
        "steps": step,
        "elapsed_sec": time.time() - t_start,
        "log_path": str(log_path),
        **memory_record(),
    }
    # Final checkpoint/upload on process 0 only (multihost). Replicated state -> host-0 device_get is
    # complete and collective-free; other hosts skip to avoid duplicate HF commits.
    if args.save_final and is_primary:
        ckpt_path = save_nnx_state_npz(model, out_dir, step)
        summary["jax_npz_checkpoint"] = str(ckpt_path)
        summary["checkpoint_format_note"] = "JAX NNX state checkpoint."
    if args.save_hf_final and is_primary:
        hf_ckpt_path = export_hf_safetensors(model, source_model_dir, out_dir, step)
        summary["hf_safetensors_checkpoint"] = str(hf_ckpt_path)
    if is_primary and args.hf_upload_repo and (args.hf_upload_final or args.hf_upload_every_steps):
        summary["hf_final_upload"] = maybe_upload_checkpoint_bundle(model, source_model_dir, out_dir, step, args, final=True)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
