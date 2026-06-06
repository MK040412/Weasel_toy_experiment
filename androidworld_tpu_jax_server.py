#!/usr/bin/env python3
"""TPU-only JAX server for AndroidWorld GUI-Owl/Fast-dVLM evaluation.

This intentionally has no PyTorch import path. It serves the JAX checkpoint from
``fast-dvlm-kd-tpu/aw-overfit-boltzmann/final`` or a local exported directory and
keeps generation on TPU through fixed-shape grounded AR decode.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("PJRT_DEVICE", "TPU")

import jax
import numpy as np
from fastapi import FastAPI, Request
from PIL import Image
import uvicorn

REPO = Path(__file__).resolve().parent
for path in (REPO / "src", Path.home() / "Weasel_toy_experiment" / "src"):
    if path.exists():
        sys.path.insert(0, str(path))

from qwen.qwen3vl import modeling  # noqa: E402
from grounded_ar_jax import GEN_LEN, PROMPT_CAP, grounded_ar_decode  # noqa: E402


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

app = FastAPI()
STATE: dict[str, object] = {}
LOCK = threading.Lock()


def _limit_ui_text(ui_text: str, max_lines: int | None) -> str:
    if max_lines is None:
        return ui_text
    if max_lines <= 0:
        return ""
    return "\n".join(ui_text.splitlines()[:max_lines])


def build_inputs(
    processor,
    image: Image.Image,
    goal: str,
    history: list[str],
    ui_text: str,
    *,
    include_history: bool | None = None,
    include_ui: bool | None = None,
    history_limit: int = 8,
    ui_line_limit: int | None = None,
):
    if include_history is None:
        include_history = bool(STATE.get("include_history"))
    if include_ui is None:
        include_ui = bool(STATE.get("include_ui"))
    text_parts = [f"Goal: {goal}"]
    if include_history and history and history_limit > 0:
        text_parts.insert(0, "\n".join(history[-history_limit:]))
    ui_text = _limit_ui_text(ui_text, ui_line_limit)
    if include_ui and ui_text:
        text_parts.append(f"Visible UI elements:\n{ui_text}")
    user_text = "\n".join(part for part in text_parts if part)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]},
    ]
    template = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=[MOBILE_USE_TOOL],
    )
    try:
        from qwen_vl_utils import process_vision_info

        images, videos = process_vision_info(messages)
    except Exception:
        images, videos = [image], None
    return processor(text=[template], images=images, videos=videos, return_tensors="np")


def build_inputs_capped(processor, image: Image.Image, goal: str, history: list[str], ui_text: str):
    """Build a prompt that fits the fixed JAX decode shape.

    The training-style prompt is just goal + image. History/UI are optional
    serving aids; if they make the prompt too long, trim optional context instead
    of failing the AndroidWorld episode with HTTP 500.
    """
    want_history = bool(STATE.get("include_history"))
    want_ui = bool(STATE.get("include_ui"))
    target_prompt_cap = int(STATE.get("prompt_cap", PROMPT_CAP))
    history_limits = [8, 4, 2, 0] if want_history else [0]
    ui_limits = [None, 48, 32, 16, 8, 4, 0] if want_ui else [0]

    for history_limit in history_limits:
        for ui_line_limit in ui_limits:
            enc = build_inputs(
                processor,
                image,
                goal,
                history,
                ui_text,
                include_history=want_history and history_limit > 0,
                include_ui=want_ui and ui_line_limit != 0,
                history_limit=history_limit,
                ui_line_limit=ui_line_limit,
            )
            prompt_len = int(np.asarray(enc["input_ids"][0]).shape[0])
            if prompt_len <= target_prompt_cap:
                meta = {
                    "prompt_len": prompt_len,
                    "history_count": history_limit if want_history and history_limit > 0 else 0,
                    "ui_line_limit": ui_line_limit if want_ui and ui_line_limit is not None else None,
                    "ui_enabled": bool(want_ui and ui_line_limit != 0),
                    "history_enabled": bool(want_history and history_limit > 0),
                }
                return enc, meta

    enc = build_inputs(
        processor,
        image,
        goal,
        history,
        ui_text,
        include_history=False,
        include_ui=False,
    )
    prompt_len = int(np.asarray(enc["input_ids"][0]).shape[0])
    if prompt_len > target_prompt_cap:
        raise ValueError(f"prompt_len {prompt_len} exceeds prompt_cap {target_prompt_cap} without optional context")
    return enc, {
        "prompt_len": prompt_len,
        "history_count": 0,
        "ui_line_limit": 0,
        "ui_enabled": False,
        "history_enabled": False,
    }


@app.post("/predict")
async def predict(req: Request):
    body = await req.json()
    image = Image.open(io.BytesIO(base64.b64decode(body["screenshot_b64"]))).convert("RGB")
    enc, context_meta = build_inputs_capped(
        STATE["processor"],
        image,
        str(body.get("goal", "")),
        list(body.get("history", []) or []),
        str(body.get("ui_elements_text", "") or ""),
    )
    t0 = time.time()
    with LOCK:
        if STATE["decode"] == "dvlm_bd4":
            from dvlm_decode_jax import dvlm_decode

            d0 = time.time()
            raw, ntok, nfe = dvlm_decode(
                STATE["model"],
                STATE["config"],
                enc,
                STATE["processor"],
                gen_len=int(STATE["gen_len"]),
                tau=float(STATE["tau"]),
                block_size=int(STATE["bd"]),
            )
            model_ms = (time.time() - d0) * 1000.0
        elif STATE["decode"] == "dual_dvlm_bd4":
            from dual_stream_decode_jax import dual_dvlm_decode

            d0 = time.time()
            raw, ntok, nfe = dual_dvlm_decode(
                STATE["model"],
                STATE["config"],
                enc,
                STATE["processor"],
                gen_len=int(STATE["gen_len"]),
                tau=float(STATE["tau"]),
                block_size=int(STATE["bd"]),
            )
            model_ms = (time.time() - d0) * 1000.0
        else:
            raw, ntok, nfe, model_ms = grounded_ar_decode(
                STATE["model"],
                STATE["config"],
                enc,
                STATE["processor"],
                gen_len=int(STATE["gen_len"]),
            )
    return {
        "raw": raw,
        "latency_ms": round((time.time() - t0) * 1000.0, 1),
        "model_latency_ms": round(model_ms, 1),
        "tokens": int(ntok),
        "nfe": int(nfe),
        "decode": STATE["decode"],
        "bd": int(STATE["bd"]) if STATE["decode"] in ("dvlm_bd4", "dual_dvlm_bd4") else 1,
        "tau": float(STATE["tau"]) if STATE["decode"] in ("dvlm_bd4", "dual_dvlm_bd4") else None,
        "prompt_cap": int(STATE.get("prompt_cap", PROMPT_CAP)),
        "gen_len": int(STATE["gen_len"]),
        "include_history": bool(STATE["include_history"]),
        "include_ui": bool(STATE["include_ui"]),
        "context": context_meta,
    }


@app.get("/health")
async def health():
    return {
        "ok": True,
        "model": STATE.get("model_path"),
        "processor": STATE.get("processor_path"),
        "decode": STATE.get("decode"),
        "bd": int(STATE.get("bd", 4)),
        "devices": jax.device_count(),
        "platform": jax.default_backend(),
        "prompt_cap": int(STATE.get("prompt_cap", PROMPT_CAP)),
        "gen_len": int(STATE.get("gen_len", GEN_LEN)),
        "max_pixels": int(STATE.get("max_pixels", 0)),
        "include_history": bool(STATE.get("include_history")),
        "include_ui": bool(STATE.get("include_ui")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--processor-path",
        default=None,
        help="Processor/tokenizer source. Defaults to --model-path so the exported checkpoint chat template is used.",
    )
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--gen-len", type=int, default=GEN_LEN)
    parser.add_argument(
        "--decode",
        choices=["grounded_ar_jit", "dvlm_bd4", "dual_dvlm_bd4"],
        default="grounded_ar_jit",
    )
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--bd", type=int, default=4,
                        help="block size for dvlm_bd4/dual_dvlm_bd4 decode (GEN_LEN must be divisible by it)")
    parser.add_argument("--include-history", action="store_true")
    parser.add_argument("--include-ui", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8124)
    args = parser.parse_args()

    if args.gen_len != GEN_LEN:
        raise ValueError(f"fixed grounded AR server expects --gen-len {GEN_LEN}")

    from transformers import AutoProcessor

    print(f"[server] loading JAX model: {args.model_path}", flush=True)
    print(f"[server] devices={jax.devices()}", flush=True)
    config = modeling.ModelConfig.qwen3vl_2b()
    model = modeling.Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, config=config)
    processor_path = args.processor_path or args.model_path
    print(f"[server] loading processor: {processor_path}", flush=True)
    processor = AutoProcessor.from_pretrained(processor_path, max_pixels=args.max_pixels)
    STATE.update(
        model=model,
        config=config,
        processor=processor,
        processor_path=processor_path,
        model_path=args.model_path,
        gen_len=args.gen_len,
        decode=args.decode,
        tau=args.tau,
        bd=args.bd,
        max_pixels=args.max_pixels,
        include_history=args.include_history,
        include_ui=args.include_ui,
    )
    if args.decode == "dual_dvlm_bd4":
        from dual_stream_decode_jax import PROMPT_CAP as DUAL_PROMPT_CAP

        STATE["prompt_cap"] = DUAL_PROMPT_CAP
    else:
        STATE["prompt_cap"] = PROMPT_CAP

    print(f"[server] warmup {args.decode}...", flush=True)
    dummy = Image.new("RGB", (512, 1024), (128, 128, 128))
    enc = build_inputs(processor, dummy, "Open Settings.", [], "")
    t0 = time.time()
    if args.decode == "dvlm_bd4":
        from dvlm_decode_jax import dvlm_decode

        d0 = time.time()
        raw, ntok, nfe = dvlm_decode(model, config, enc, processor, gen_len=args.gen_len, tau=args.tau, block_size=args.bd)
        model_ms = (time.time() - d0) * 1000.0
    elif args.decode == "dual_dvlm_bd4":
        from dual_stream_decode_jax import dual_dvlm_decode

        d0 = time.time()
        raw, ntok, nfe = dual_dvlm_decode(
            model,
            config,
            enc,
            processor,
            gen_len=args.gen_len,
            tau=args.tau,
            block_size=args.bd,
        )
        model_ms = (time.time() - d0) * 1000.0
    else:
        raw, ntok, nfe, model_ms = grounded_ar_decode(model, config, enc, processor, gen_len=args.gen_len)
    print(
        f"[server] warmup done total={time.time() - t0:.1f}s model_ms={model_ms:.1f} "
        f"tok={ntok} nfe={nfe} raw={raw[:120]!r}",
        flush=True,
    )
    print(f"[server] READY on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
