"""AndroidWorld agent wrapper for GUI-Owl/Fast-dVLM policy server."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
from PIL import Image
import requests

from android_world.agents import base_agent
from android_world.env import interface
from android_world.env import json_action


TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
CONCLUSION_RE = re.compile(r"<conclusion>\s*(.*?)\s*</conclusion>", re.DOTALL | re.IGNORECASE)
ACTIONS = (
    "click",
    "long_press",
    "swipe",
    "type",
    "answer",
    "system_button",
    "open",
    "key",
    "wait",
    "terminate",
)
BUTTONS = ("Home", "Back", "Enter", "Menu")


def _first_json_object(text: str) -> str | None:
    match = TOOL_RE.search(text)
    if match:
        return match.group(1)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_action(raw: str) -> str | None:
    low = raw.lower()
    for action in ACTIONS:
        if re.search(rf'["\']?action["\']?\s*:{{0,2}}\s*["\']?{re.escape(action)}', low):
            return action
    for action in ACTIONS:
        if action in low:
            return action
    return None


def _extract_coord_pairs(raw: str) -> list[list[int]]:
    pairs: list[list[int]] = []
    for match in re.finditer(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", raw):
        x = int(round(float(match.group(1))))
        y = int(round(float(match.group(2))))
        pairs.append([x, y])
    return pairs


def _extract_button(raw: str) -> str | None:
    low = raw.lower()
    for button in BUTTONS:
        if button.lower() in low:
            return button
    return None


def _extract_text(raw: str) -> str:
    match = re.search(r'"(?:text|app_name)"\s*:\s*"([^"]{0,200})"', raw)
    return match.group(1) if match else ""


def _extract_conclusion(raw: str) -> str | None:
    match = CONCLUSION_RE.search(raw or "")
    if not match:
        return None
    text = re.sub(r"\s+", " ", match.group(1)).strip()
    return text[:240] or None


def repair_mobile_use(raw: str) -> dict[str, Any] | None:
    action = _extract_action(raw)
    if not action:
        return None
    args: dict[str, Any] = {"action": action}
    pairs = _extract_coord_pairs(raw)
    if action in {"click", "long_press"}:
        if not pairs:
            return None
        args["coordinate"] = pairs[0]
    elif action == "swipe":
        if len(pairs) < 2:
            return None
        args["coordinate"] = pairs[0]
        args["coordinate2"] = pairs[1]
    elif action == "system_button":
        button = _extract_button(raw)
        if not button:
            return None
        args["button"] = button
    elif action in {"type", "answer", "key"}:
        args["text"] = _extract_text(raw)
    elif action == "open":
        text = _extract_text(raw)
        if text:
            args["text"] = text
    elif action == "wait":
        match = re.search(r'"time"\s*:\s*(\d+(?:\.\d+)?)', raw)
        if match:
            args["time"] = float(match.group(1))
    elif action == "terminate":
        args["status"] = "failure" if "failure" in raw.lower() else "success"
    return args


def parse_mobile_use(raw: str, repair: bool = True) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    meta: dict[str, Any] = {
        "raw": raw,
        "has_tool_call": bool(TOOL_RE.search(raw)),
        "valid_json": False,
        "valid_mobile_use": False,
        "repaired": False,
        "parse_error": None,
    }
    obj_text = _first_json_object(raw)
    if not obj_text:
        meta["parse_error"] = "no_json_object"
    else:
        try:
            obj = json.loads(obj_text)
            meta["valid_json"] = True
        except Exception as exc:
            meta["parse_error"] = f"json_error:{type(exc).__name__}:{exc}"
        else:
            if obj.get("name") == "mobile_use" and isinstance(obj.get("arguments"), dict):
                args = obj["arguments"]
                meta["valid_mobile_use"] = True
                meta["tool_call"] = obj
                return args, meta

            if isinstance(obj.get("action"), str):
                meta["valid_mobile_use"] = True
                meta["tool_call"] = {"name": "mobile_use", "arguments": obj}
                return obj, meta

            meta["parse_error"] = "not_mobile_use"
            meta["tool_call"] = obj

    if repair:
        fixed = repair_mobile_use(raw)
        if fixed is not None:
            meta["repaired"] = True
            meta["valid_mobile_use"] = True
            meta["tool_call"] = {"name": "mobile_use", "arguments": fixed}
            return fixed, meta
    return None, meta


def _coord_to_pixel(coord: Any, screen_size: tuple[int, int]) -> tuple[int, int] | None:
    if not isinstance(coord, (list, tuple)) or len(coord) < 2:
        return None
    width, height = screen_size
    mode = os.environ.get("GUIOWL_COORD_MODE", "absolute").lower()
    x = float(coord[0])
    y = float(coord[1])
    if mode in {"normalized", "qwen-vl", "qwen_vl", "0-1000", "relative"}:
        x = x / 1000.0 * width
        y = y / 1000.0 * height
    # GUI-Owl/MobileAgent coordinates are absolute pixels; clamp only at actuation.
    return int(round(max(0, min(width - 1, x)))), int(round(max(0, min(height - 1, y))))


def _swipe_direction(args: dict[str, Any]) -> str:
    start = args.get("coordinate")
    end = args.get("coordinate2")
    if isinstance(start, (list, tuple)) and isinstance(end, (list, tuple)) and len(start) >= 2 and len(end) >= 2:
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        return "down" if dy > 0 else "up"
    return str(args.get("direction") or "up")


def to_android_action(
    args: dict[str, Any] | None,
    screen_size: tuple[int, int],
) -> tuple[json_action.JSONAction, dict[str, Any]]:
    meta: dict[str, Any] = {"android_action_valid": False, "action_error": None}
    if not args:
        meta["action_error"] = "no_arguments"
        return json_action.JSONAction(action_type="wait"), meta

    action = str(args.get("action", "")).lower()
    try:
        if action == "click":
            xy = _coord_to_pixel(args.get("coordinate"), screen_size)
            if xy is None:
                raise ValueError("click_missing_coordinate")
            out = json_action.JSONAction(action_type="click", x=xy[0], y=xy[1])
        elif action == "long_press":
            xy = _coord_to_pixel(args.get("coordinate"), screen_size)
            if xy is None:
                raise ValueError("long_press_missing_coordinate")
            out = json_action.JSONAction(action_type="long_press", x=xy[0], y=xy[1])
        elif action == "swipe":
            out = json_action.JSONAction(
                action_type="swipe",
                direction=_swipe_direction(args),
            )
        elif action == "type":
            out = json_action.JSONAction(
                action_type="input_text",
                text=str(args.get("text", "")),
                clear_text=True,
            )
        elif action == "answer":
            out = json_action.JSONAction(
                action_type="answer",
                text=str(args.get("text", "")),
            )
        elif action == "system_button":
            button = str(args.get("button", "")).lower()
            if button == "home":
                out = json_action.JSONAction(action_type="navigate_home")
            elif button == "back":
                out = json_action.JSONAction(action_type="navigate_back")
            elif button == "enter":
                out = json_action.JSONAction(action_type="keyboard_enter")
            else:
                out = json_action.JSONAction(action_type="wait")
        elif action == "open":
            out = json_action.JSONAction(
                action_type="open_app",
                app_name=str(args.get("text") or args.get("app_name") or ""),
            )
        elif action == "key":
            key = str(args.get("text") or args.get("button") or "").lower()
            if key in {"enter", "keycode_enter"}:
                out = json_action.JSONAction(action_type="keyboard_enter")
            elif key in {"back", "keycode_back"}:
                out = json_action.JSONAction(action_type="navigate_back")
            elif key in {"home", "keycode_home"}:
                out = json_action.JSONAction(action_type="navigate_home")
            else:
                raise ValueError(f"unsupported_key:{key}")
        elif action == "wait":
            out = json_action.JSONAction(action_type="wait")
        elif action == "terminate":
            status = str(args.get("status", "success")).lower()
            out = json_action.JSONAction(
                action_type="status",
                goal_status="complete" if status == "success" else "infeasible",
            )
        else:
            raise ValueError(f"unsupported_action:{action}")
        meta["android_action_valid"] = True
        meta["android_action"] = out.as_dict()
        return out, meta
    except Exception as exc:
        meta["action_error"] = f"{type(exc).__name__}:{exc}"
        return json_action.JSONAction(action_type="wait"), meta


def _screenshot_b64(pixels: np.ndarray) -> str:
    image = Image.fromarray(pixels.astype(np.uint8), "RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _format_history_summary(
    step_index: int,
    args: dict[str, Any] | None,
    action: json_action.JSONAction,
    raw: str,
) -> str | None:
    conclusion = _extract_conclusion(raw)
    if conclusion:
        return f"Step {step_index}: {conclusion}"
    if not args:
        return f"Step {step_index}: the previous model output could not be converted, so the agent waited."

    kind = str(args.get("action", "")).lower()
    if kind == "click":
        desc = "clicked a visible screen location"
    elif kind == "long_press":
        desc = "long-pressed a visible screen location"
    elif kind == "swipe":
        desc = f"swiped {getattr(action, 'direction', '') or 'on the screen'}"
    elif kind == "type":
        desc = f"typed text {str(args.get('text', ''))[:80]!r}"
    elif kind == "answer":
        desc = f"answered {str(args.get('text', ''))[:120]!r}"
    elif kind == "open":
        desc = f"opened app {str(args.get('text') or args.get('app_name') or '')[:80]!r}"
    elif kind == "system_button":
        desc = f"pressed system button {str(args.get('button', ''))}"
    elif kind == "terminate":
        desc = f"terminated with status {str(args.get('status', 'success'))}"
    else:
        desc = f"performed action {kind or action.action_type}"
    return f"Step {step_index}: {desc}. Observe the current screenshot before choosing the next action; do not blindly repeat failed actions."


def _format_ui_elements(ui_elements: list[Any], screen_size: tuple[int, int], limit: int = 80) -> str:
    rows: list[str] = []
    width, height = screen_size
    for idx, el in enumerate(ui_elements[:limit]):
        label = (getattr(el, "text", None) or getattr(el, "content_description", None)
                 or getattr(el, "hint_text", None) or getattr(el, "resource_name", None)
                 or getattr(el, "class_name", None) or "")
        label = str(label).replace("\n", " ").strip()[:80]
        bbox = getattr(el, "bbox", None)
        bp = getattr(el, "bbox_pixels", None)
        if bbox is not None:
            box = [int(round(bbox.x_min)), int(round(bbox.y_min)), int(round(bbox.x_max)), int(round(bbox.y_max))]
        elif bp is not None and width and height:
            box = [
                int(round(float(bp.x_min) / width * 1000)),
                int(round(float(bp.y_min) / height * 1000)),
                int(round(float(bp.x_max) / width * 1000)),
                int(round(float(bp.y_max) / height * 1000)),
            ]
        else:
            box = None
        flags = []
        for name, short in (("is_clickable", "click"), ("is_editable", "edit"), ("is_scrollable", "scroll"), ("is_checked", "checked")):
            if getattr(el, name, None):
                flags.append(short)
        if label or box or flags:
            rows.append(f"{idx}: label={label!r} box_0_1000={box} flags={','.join(flags)}")
    return "\n".join(rows)


class GuiOwlAgent(base_agent.EnvironmentInteractingAgent):
    def __init__(
        self,
        env: interface.AsyncEnv,
        server_url: str = "http://127.0.0.1:8123",
        timeout: float = 120.0,
    ):
        super().__init__(env, name="guiowl_dvlm")
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.history: list[str] = []
        self.last_args: dict[str, Any] | None = None
        self.record_dir = os.environ.get("GUIOWL_RECORD_DIR")
        self._episode_dir: Path | None = None

    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home)
        self.history.clear()
        self.last_args = None
        self._episode_dir = None

    def _get_episode_dir(self, goal: str) -> Path | None:
        if not self.record_dir:
            return None
        if self._episode_dir is not None:
            return self._episode_dir
        digest = hashlib.sha1(goal.encode("utf-8")).hexdigest()[:10]
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", goal).strip("_")[:80] or "episode"
        self._episode_dir = Path(self.record_dir) / f"{slug}_{digest}"
        (self._episode_dir / "frames").mkdir(parents=True, exist_ok=True)
        return self._episode_dir

    def _record_step(self, goal: str, step_data: dict[str, Any], pixels: np.ndarray) -> None:
        episode_dir = self._get_episode_dir(goal)
        if episode_dir is None:
            return
        step_idx = int(step_data.get("local_step_index", len(self.history)))
        frame_path = episode_dir / "frames" / f"step_{step_idx:03d}.png"
        Image.fromarray(pixels.astype(np.uint8), "RGB").save(frame_path)
        serializable = dict(step_data)
        serializable["frame"] = str(frame_path)
        with (episode_dir / "actions.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(serializable, ensure_ascii=False, default=str) + "\n")

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        state = self.get_post_transition_state()
        screen_size = self.env.logical_screen_size
        if self.last_args and str(self.last_args.get("action", "")).lower() == "answer":
            args = {"action": "terminate", "status": "success"}
            action = json_action.JSONAction(action_type="status", goal_status="complete")
            step_data = {
                "goal": goal,
                "screen_size": screen_size,
                "server_prediction": {
                    "raw": '<tool_call>\n{"name":"mobile_use","arguments":{"action":"terminate","status":"success"}}\n</tool_call>',
                    "decode": "local_answer_finalize",
                    "tokens": 0,
                    "nfe": 0,
                    "latency_ms": 0,
                },
                "parsed_arguments": args,
                "structural": {
                    "valid_json": True,
                    "valid_mobile_use": True,
                    "repaired": False,
                    "local_finalize_after_answer": True,
                },
                "android_action": action.as_dict(),
                "action_meta": {"android_action_valid": True, "action_error": None, "android_action": action.as_dict()},
                "latency_total_ms": 0,
                "coord_mode": os.environ.get("GUIOWL_COORD_MODE", "absolute"),
                "history_summary": "Previous step answered the user question; terminating successfully.",
                "local_step_index": len(self.history),
            }
            self.history.append(f"Step {len(self.history) + 1}: previous answer submitted; terminate success.")
            self.last_args = args
            self._record_step(goal, step_data, state.pixels)
            return base_agent.AgentInteractionResult(True, step_data)

        request = {
            "goal": goal,
            "screen_width": screen_size[0],
            "screen_height": screen_size[1],
            "history": self.history[-8:],
            "ui_elements_text": _format_ui_elements(state.ui_elements, screen_size),
            "screenshot_b64": _screenshot_b64(state.pixels),
        }
        t0 = time.time()
        response = requests.post(
            f"{self.server_url}/predict",
            json=request,
            timeout=self.timeout,
        )
        response.raise_for_status()
        pred = response.json()
        raw = pred.get("raw", "")
        repair = os.environ.get("GUIOWL_REPAIR", "1") != "0"
        args, parse_meta = parse_mobile_use(raw, repair=repair)
        action, action_meta = to_android_action(args, screen_size)
        done = action.action_type == "status"

        step_data = {
            "goal": goal,
            "screen_size": screen_size,
            "server_prediction": pred,
            "parsed_arguments": args,
            "structural": parse_meta,
            "android_action": action.as_dict(),
            "action_meta": action_meta,
            "latency_total_ms": int((time.time() - t0) * 1000),
            "coord_mode": os.environ.get("GUIOWL_COORD_MODE", "absolute"),
            "local_step_index": len(self.history),
        }

        history_summary = _format_history_summary(len(self.history) + 1, args, action, raw)
        if history_summary:
            self.history.append(history_summary)
            step_data["history_summary"] = history_summary
        self.last_args = args
        self._record_step(goal, step_data, state.pixels)
        if action.action_type != "status":
            self.env.execute_action(action)
        return base_agent.AgentInteractionResult(done, step_data)


if __name__ == "__main__":
    examples = [
        '<tool_call>\n{"name":"mobile_use","arguments":{"action":"click","coordinate":[695,793]}}\n</tool_call>',
        '<tool_call>\n{"name":"mobile_use","arguments":{"action":"system_button","button":"Home"}}\n</tool_call>',
        "broken",
    ]
    for raw in examples:
        args, meta = parse_mobile_use(raw)
        action, ameta = to_android_action(args, (540, 1080))
        print(raw)
        print(args)
        print(meta)
        print(action, ameta)
