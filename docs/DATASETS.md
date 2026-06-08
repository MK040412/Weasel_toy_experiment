# DATASETS — Fast-dVLM (action SFT + reasoning SFT)

> Complete inventory for a fresh account/engineer. Two datasets feed the two training phases:
> **(1) ACTION** continued-SFT (`packed-*.parquet`, phone-only GUI-agent mix) and
> **(2) REASONING** continued-SFT (`gui-libra-reasoning-phone`, think-then-act CoT).
> All facts below were Vultr-verified during the action-SFT run. For *how to train* read
> `commands/REASONING_SFT_RECIPE.md`; for *how to save/recover checkpoints* read
> `commands/CHECKPOINT_DECOUPLED.md`. This file is the data-side companion to both — do not duplicate them.

Both datasets share the same on-disk contract:
- **One row = one step**; columns include `episode_id`, `step_id`, `instruction`, `screenshot` (raw **JPEG
  bytes**, never base64), `target_json` (the native action), and 0–1000 coordinates (GUI-Owl norm1000).
- **`--data-mode episode` is MANDATORY** for both. Episode mode packs all steps of an `episode_id` into one
  multi-turn sequence. Row mode `KeyError`s on `screenshot` and breaks multi-turn history.
- Vision tower is FROZEN/precomputed → screenshots are decoded once and cached; the loader needs raw JPEG.

---

## 1. ACTION dataset — phone-only packed GUI-agent mix

| property | value |
|---|---|
| **Canonical HF repo** | `KMK040412/guiowl-aw-mix-phoneonly-packed` (dataset) |
| Worker/local path | `~/data/phoneonly` (the action TPU run reads from here) |
| Files | **151** `packed-*.parquet` (`packed-0000.parquet` … `packed-0150.parquet`) + `curation_manifest.json` |
| Size | ~64 GB |
| Episodes / rows (after curation) | **53,991 episodes / 540,780 rows** (≈54k phone-only episodes) |
| Sources (5) | `aitw`, `amex`, `androidcontrol_par`, `gui_odyssey`, `openmobile` |
| Screenshots | 100% raw **JPEG** (`ffd8ff…`), **0 base64** — verified across all sources |
| Coordinates | GUI-Owl **norm1000** (x,y ∈ [0,1000]), out-of-bounds = 0 |
| Geometry | phone / **portrait** only (`screen_width < screen_height`, AR 0.40–0.58 inclusive) |
| Train flags | `--data-pattern "packed-*.parquet"` `--data-mode episode` |

### Provenance (how it was built)
`phoneonly` is **the phone-only filtered + re-packed action mix**. It is NOT a hand-collected set: it is
derived from `KMK040412/guiowl-aw-mix-hybrid-packed` (`src_repo` in the manifest) by a curation pass
(`curation_manifest.json`, `script_version 2.0.0`, full pass over all 151 shards). The pass applied a
12-filter pipeline — `source_whitelist`, `null_metadata`, `instruction_presence`, `phone_geometry_metadata`,
`image_integrity`, `metadata_pixel_consistency`, `phone_geometry_decoded`, `target_json_validity`,
`action_type_validity`, `coordinate_bounds`, `episode_structural_integrity`, `cross_shard_episode` —
with **whole-episode drop on any failing step** and **cross-shard episodes dropped everywhere**. Net effect:
600,040 → 540,780 rows (−9.88%), 57,802 → 53,991 episodes (−6.59%). Action vocabulary (9):
`answer, click, long_press, open, swipe, system_button, terminate, type, wait`. The local worker copy lives
at `~/data/phoneonly`; the canonical, reproducible source of record is the HF repo above. (`curation_manifest.json`
inside the repo carries the full per-source/per-filter breakdown — fetch it for an exact audit.)

### Download (action)
```bash
source ~/.fastdvlm_secrets.env           # exports HF_TOKEN (never print its value)
HF_HUB_DISABLE_XET=1 python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="KMK040412/guiowl-aw-mix-phoneonly-packed",
    repo_type="dataset",
    local_dir="/root/data/phoneonly",     # = ~/data/phoneonly on the worker
    allow_patterns=["packed-*.parquet", "curation_manifest.json"],
    max_workers=16,
)
PY
```
> `HF_HUB_DISABLE_XET=1` avoids the Xet download stalls seen on the TPU/Vultr workers. The flat layout means
> the train pattern is the bare `--data-pattern "packed-*.parquet"` (no subdir glob).

---

## 2. REASONING dataset — GUI-Libra CoT (think-then-act), phone-portrait

| property | value |
|---|---|
| **Canonical HF repo** | `KMK040412/gui-libra-reasoning-phone` (dataset) |
| Worker/local path | `~/data/gui-libra-reasoning-phone` |
| Size | ~8.93 GB |
| Files | **11** parquet, **NESTED** per-source: `aitw/` (2), `amex/` (5), `gui_odyssey/` (4) — named `guilibra-shard-NNNN.parquet` |
| Episodes / steps (after filter) | **23,948 episodes / 41,876 steps** |
| Per-source steps | aitw 7,626 · amex 19,694 · gui_odyssey 14,556 (Vultr full-scan verified) |
| `reasoning` CoT column | **100% non-empty** (0/41,876 empty); ~847-char median, ~900–1150 chars typical |
| Screenshots | raw **JPEG** (`ffd8ff…`), verified NOT base64 (3300/3300 sampled, 100% decode) |
| Coordinates | norm1000, x,y ∈ [0,1000], OOB = 0 |
| Geometry | phone / **portrait** only |
| `source` column | fully-qualified: `GUI-Libra/GUI-Libra-81K-SFT::{aitw,amex,gui_odyssey}` |
| Train flags | `--data-pattern "*/*.parquet"` `--data-mode episode` |

### Provenance (how it was built)
Phone-portrait filter of the GUI-Libra reasoning/CoT SFT shards (aitz-canon schema), maximizing AndroidWorld
relevance. Drop rule: an **episode** is landscape (dropped) if **any** step has `screen_height/screen_width <
1.0`; episodes kept contiguous and intact. Schema is **byte-identical** to the source (same columns/dtypes,
`target_json` unchanged, coords `guiowl_norm1000_xy`). Filter stats: 24,993 → 23,948 episodes (−1,045),
48,472 → 41,876 steps (−6,596). `aitw`/`amex` had 0% landscape (copied verbatim); `gui_odyssey` was
filtered + re-packed episode-contiguous (3,342 → 2,297 ep). `android_control` and `coat` are **report-only**
(no parquet shards in this set). The CoT is **already distilled from a strong reasoner (GUI-Libra ASFT)**, so
the reasoning phase learns CoT by plain cross-entropy on the `reasoning` column — distillation is OFF
(`--kd-noisy-weight 0 --kd-fewstep-weight 0`); see `commands/REASONING_SFT_RECIPE.md` for why.

### Download (reasoning)
```bash
source ~/.fastdvlm_secrets.env           # exports HF_TOKEN (never print its value)
HF_HUB_DISABLE_XET=1 python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="KMK040412/gui-libra-reasoning-phone",
    repo_type="dataset",
    local_dir="/root/data/gui-libra-reasoning-phone",   # = ~/data/gui-libra-reasoning-phone
    allow_patterns=["*/*.parquet", "README.md", "balance_report.json"],
    max_workers=16,
)
PY
```

### Gotchas (reasoning)
- **NESTED layout** → the train pattern is `--data-pattern "*/*.parquet"` (the `*/` matches `aitw/`, `amex/`,
  `gui_odyssey/`). The flat `packed-*.parquet` pattern from the action run will match **zero** files here.
- **`--data-mode episode` is mandatory** — row mode `KeyError`s on `screenshot`.
- The CoT lengthens the noisy/text branch: use `--noisy-pad-to 1536` (recipe pre-authorizes 2048). At 1536,
  12.5% of episodes exceed the cap and are handled by drop-oldest-turn → 0% skipped, 0% lose CoT.
- Loader patch (already on `reasoning-sft`) injects `<think>\n{reasoning}\n</think>\n\n{action}` into the
  supervised assistant turn so the CoT gets CE loss. Native Qwen3-VL tokens: `<think>`=151667 /
  `</think>`=151668 / `<tool_call>`=151657 / `</tool_call>`=151658 (all single). NEVER use `<answer>` (splits).

---

## 3. Verified properties (quick reference)

| property | ACTION (`guiowl-aw-mix-phoneonly-packed`) | REASONING (`gui-libra-reasoning-phone`) |
|---|---|---|
| files | 151 `packed-*.parquet` (flat) | 11 `*/*.parquet` (nested aitw/amex/gui_odyssey) |
| size | ~64 GB | ~8.93 GB |
| episodes / steps | 53,991 / 540,780 | 23,948 / 41,876 |
| screenshots | 100% JPEG, 0 base64 | 100% JPEG, 0 base64 (3300/3300 decode) |
| coords | norm1000 (0–1000) | norm1000 (0–1000) |
| `reasoning` CoT | — (action only) | 100% non-empty |
| sources | aitw, amex, androidcontrol_par, gui_odyssey, openmobile | aitw, amex, gui_odyssey |
| `--data-pattern` | `"packed-*.parquet"` | `"*/*.parquet"` |
| `--data-mode` | `episode` | `episode` |
| download | `snapshot_download` + `HF_HUB_DISABLE_XET=1` | `snapshot_download` + `HF_HUB_DISABLE_XET=1` |

---

## 4. Checkpoint repos & decoupled-save note

| phase | HF model repo | path prefix / checkpoints |
|---|---|---|
| ACTION | `KMK040412/fastdvlm-aw-guiowlvit` | `fast-dvlm-kd-tpu/checkpoint-step{N}`; shipped: 843, 1686, 2529, 3372, 4215, 5058, 5901 (~4.89 GB each) |
| REASONING | `KMK040412/fastdvlm-aw-reasoning` | created on first upload (does not exist yet); `--hf-upload-every-steps 750`, `--hf-upload-final` |

- The reasoning run's **start checkpoint** is the **best action-SFT checkpoint** from
  `KMK040412/fastdvlm-aw-guiowlvit` (or `~/models/boltzmann-final`), set as `--model-dir`.
- **Missing-files gotcha:** shipped action checkpoints may LACK the image-processor files
  (`preprocessor_config.json`, `special_tokens_map.json`, `video_preprocessor_config.json`). Before using a
  checkpoint as `--model-dir` for any image-processing run, **complete the processor** by copying those files
  from the base `Qwen/Qwen3-VL-2B-Instruct` (or `boltzmann-final`).
- **EPOCH selection:** 2 epochs is a CEILING, not a proven optimum (small + already-distilled set → overfit
  risk past ~1–1.5 ep). Checkpoint every ~0.5 ep (`--hf-upload-every-steps 750`) and **EVAL-SELECT** the best
  checkpoint on AndroidWorld; 1 epoch may already be best. Do not assume 2.

### Decoupled checkpointing (multihost + ZeRO-1) — see `commands/CHECKPOINT_DECOUPLED.md`
Saving is **decoupled**: each host dumps only its local addressable shards (dumb, no collective, no gather,
try/except so a save bug can never kill training) to `/dev/shm`; an **external**
`scripts/stitch_and_ship_checkpoint.py` reassembles them into HF safetensors and ships. This is required
because ZeRO-1 leaves the vocab embedding dp-sharded across all 16 chips, so a naive in-process save
deadlocks/fails. A save/ship bug is fixed externally on the already-dumped shards **without restarting
training**. Full procedure (including the per-host rsync gather and resume-after-preemption steps) is in
`commands/CHECKPOINT_DECOUPLED.md` — read it before touching any checkpoint code.

---

## See also
- `commands/REASONING_SFT_RECIPE.md` — the exact verified launch for the reasoning phase (loss shape, HPs, curriculum, loader patch).
- `commands/CHECKPOINT_DECOUPLED.md` — canonical save/recover procedure for the multihost ZeRO-1 run.
- `docs/COORDINATE_CONVENTION.md` — the norm1000 coordinate contract used by both datasets.
