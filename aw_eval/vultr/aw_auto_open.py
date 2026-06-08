"""Immediate workaround: auto-launch each task's target app at init.

The fine-tuned GUI-Owl/Fast-dVLM model never learned the `open` action (the
training mix has zero open-app actions), so from the AndroidWorld home screen it
loops a no-op swipe-up and never reaches the app. In-app grounding is fine
(near-pixel-perfect), so launching the task's primary app before the agent runs
unblocks navigation.

Self-applying on import, gated by env GUIOWL_AUTO_OPEN=1. Patches the base-class
TaskEval.initialize_task so every task launches its first non-clipper app_name
(then a short settle sleep) after the framework's own setup.
"""
from __future__ import annotations

import logging
import os
import time

if os.environ.get("GUIOWL_AUTO_OPEN", "0") == "1":
    from android_world.task_evals import task_eval as _te
    from android_world.env import adb_utils as _adb

    _settle = float(os.environ.get("GUIOWL_AUTO_OPEN_SETTLE", "2.0"))

    # CRITICAL: tasks default start_on_home_screen=True, which makes the episode
    # runner call agent.reset(go_home=True) AFTER initialize_task — navigating
    # back home and undoing our app launch. Force False so the launched app stays
    # foregrounded for the agent's first screenshot. (Success eval is independent
    # of the start screen, so this is safe.)
    _te.TaskEval.start_on_home_screen = False

    _orig_init = _te.TaskEval.initialize_task

    def _initialize_task_with_launch(self, env):  # type: ignore[no-untyped-def]
        _orig_init(self, env)
        apps = [a for a in (self.app_names or ()) if a and a != "clipper"]
        if not apps:
            logging.warning("[auto-open] task=%s has no launchable app_names", self.name)
            return
        app = apps[0]
        try:
            _adb.launch_app(app, env.controller)
            time.sleep(_settle)
            logging.warning("[auto-open] launched %r for task=%s (app_names=%s)",
                            app, self.name, list(self.app_names))
        except Exception as exc:  # noqa: BLE001
            logging.warning("[auto-open] FAILED launch %r for task=%s: %r", app, self.name, exc)

    _te.TaskEval.initialize_task = _initialize_task_with_launch
    logging.warning("[auto-open] ENABLED: TaskEval.initialize_task patched (settle=%.1fs)", _settle)
else:
    logging.info("[auto-open] disabled (set GUIOWL_AUTO_OPEN=1 to enable)")
