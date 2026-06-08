"""Inspect the ACTUAL data of a (NaN-)skipped training batch — directly, not by metadata.

Given a skipped step's log line (window_idx, sample_idx, bd), reconstruct the exact episodes that
fed it and look at the real content: decoded instruction + action turns, pixel finiteness, token
range, image-token vs grid-token consistency, loss-token counts, and the exact bd dual-stream arrays
over many mask-seeds (any non-finite array / zero-loss / all-masked attention row). CPU-only — no TPU,
no model weights (only the processor/tokenizer). Reuses the trainer's own data functions for fidelity.

This is the tool behind the v6e-16 NaN finding (`commands/tpu_v6e16_fastdvlm_zero1_recipe.md`):
the data is clean; the rare bd=2 NaN is a bf16 forward-compute edge, handled by `--skip-nonfinite`.

Example (on a worker, host-0 batch of the step-196 skip):
  uv run --no-sync python scripts/inspect_nan_batch.py \
    --data ~/data/aw_mix_hybrid_packed --model-dir ~/models/boltzmann-final \
    --proc-index 0 --proc-count 4 --bd 2 --sample-idx 10,21,48,55
"""
import argparse
import glob
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import train_fastdvlm_tpu as T  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data-pattern", default="packed-*.parquet")
    ap.add_argument("--proc-index", type=int, default=0)
    ap.add_argument("--proc-count", type=int, default=4)
    ap.add_argument("--bd", type=int, default=2)
    ap.add_argument("--sample-idx", default="10,21,48,55", help="comma-separated sample_idx within the window")
    ap.add_argument("--max-samples", type=int, default=64)
    ap.add_argument("--samples-per-window", type=int, default=64)
    ap.add_argument("--ctx-cap", type=int, default=4096)
    ap.add_argument("--max-pixels", type=int, default=100352)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--pad-to", type=int, default=4096)
    ap.add_argument("--noisy-pad-to", type=int, default=1024)
    ap.add_argument("--loss-token-cap", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=300)
    args = ap.parse_args()

    flag = [int(x) for x in args.sample_idx.split(",") if x.strip() != ""]
    files = sorted(glob.glob(os.path.join(args.data, args.data_pattern)))
    host = files[args.proc_index::args.proc_count]
    print(f"host-{args.proc_index}/{args.proc_count} shards (first 3): "
          f"{[os.path.basename(p) for p in host[:3]]} (total {len(host)})")

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    vocab = len(processor.tokenizer)
    print(f"tokenizer vocab len = {vocab}")

    window = None
    for w, meta in T.iter_episode_windows(
        [Path(p) for p in host], processor, args.max_samples, args.samples_per_window,
        args.ctx_cap, args.max_pixels, args.max_turns,
    ):
        window = w
        print("window meta:", json.dumps({k: meta.get(k) for k in
              ("window_idx", "n_samples", "seen_episodes", "skipped_episodes", "truncated_episodes")}))
        break
    assert window is not None and len(window) > max(flag), f"window too small: {None if window is None else len(window)}"

    def img_tok(ids):
        return int(np.count_nonzero(ids == T.IMG_TOKEN_ID))

    def grid_tok(grid):
        g = np.asarray(grid)
        return int((g[:, 0] * g[:, 1] * g[:, 2]).sum() // 4)  # spatial_merge_size=2 -> /4

    # --- whole-window anomaly scan ---
    bad_finite, bad_grid, bad_oob = [], [], []
    for i, s in enumerate(window):
        ids = np.asarray(s["input_ids"]); pix = np.asarray(s["pixel_values"], dtype=np.float32)
        if not np.isfinite(pix).all():
            bad_finite.append(i)
        if img_tok(ids) != grid_tok(s["image_grid_thw"]):
            bad_grid.append(i)
        if int(np.count_nonzero(ids >= vocab)) > 0:
            bad_oob.append(i)
    print(f"\nWINDOW anomaly scan ({len(window)} samples): non_finite_pixels={bad_finite} "
          f"img!=grid_tok={bad_grid} oob_token={bad_oob}")

    # --- deep dive on flagged samples ---
    for i in flag:
        s = window[i]
        ids = np.asarray(s["input_ids"]); labels = np.asarray(s["labels"])
        pix = np.asarray(s["pixel_values"], dtype=np.float32)
        print(f"\n----- sample_idx {i}  episode_id={s.get('episode_id')}  "
              f"turns={s.get('n_turns')}/{s.get('n_turns_full')} -----")
        print(f"  len={ids.shape[0]} tok[min,max]=[{int(ids.min())},{int(ids.max())}] "
              f"oob={int(np.count_nonzero(ids >= vocab))} losstok={int(np.count_nonzero(labels != -100))}")
        print(f"  pixels finite={bool(np.isfinite(pix).all())} min={float(pix.min()):.3f} "
              f"max={float(pix.max()):.3f}  img_tok={img_tok(ids)} grid_tok={grid_tok(s['image_grid_thw'])} "
              f"match={img_tok(ids) == grid_tok(s['image_grid_thw'])}")
        print(f"  GOAL: {str(s.get('goal'))[:200]}")
        loss_pos = np.nonzero(labels != -100)[0]
        if loss_pos.size:
            print(f"  ASSISTANT(first 100 loss toks): "
                  f"{processor.tokenizer.decode([int(t) for t in ids[loss_pos][:100]])[:300]}")
        nonfinite = zero_noisy = zero_clean = allmask = 0
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            arr = T.prepare_dual_arrays(s, args.bd, rng, T.MIN_NOISE, pad_to=args.pad_to,
                                        noisy_pad_to=args.noisy_pad_to, pad_token_id=0,
                                        loss_token_cap=args.loss_token_cap, pair_batch=1)
            if any(np.issubdtype(np.asarray(v).dtype, np.floating) and not np.isfinite(np.asarray(v)).all()
                   for v in arr.values()):
                nonfinite += 1
            zero_noisy += int(int(np.asarray(arr["noisy_loss_mask"]).sum()) == 0)
            zero_clean += int(int(np.asarray(arr["clean_loss_mask"]).sum()) == 0)
            am = np.asarray(arr["attn_mask"])[0, 0]
            lt_a, lt_v, tot = int(arr["lt_actual"]), int(arr["lt"]), int(arr["total"])
            cr = int(np.asarray(s["input_ids"]).shape[0])
            valid = np.zeros(tot, bool); valid[:lt_a] = True; valid[lt_v:lt_v + cr] = True
            allmask += int(np.any((am.sum(axis=1) == 0) & valid))
        print(f"  bd={args.bd} over {args.seeds} mask-seeds: nonfinite_arrays={nonfinite} "
              f"zero_noisy_loss={zero_noisy} zero_clean_loss={zero_clean} valid_qrow_all_masked={allmask}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
