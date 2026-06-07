#!/usr/bin/env python3
"""Extrapolate v6e-16 1-epoch wall-time + cost from the smoke-measured step time.

Use AFTER the smoke run: read the median steady `compute_sec`/step from worker-0's
train_log.jsonl (ignore the first ~3 steps = XLA compile), then:

    python commands/v6e16_cost.py --step-sec 6.0 --batch 32 --price 2.70

--price is USD per CHIP-hour (v6e-16 = 16 chips). Spot is typically much cheaper than on-demand;
pass your actual spot rate. Prints steps/epoch, wall-time, and $ so you decide BEFORE the full run.
No TPU is touched by this script.
"""
import argparse, math

EPISODES = 57_669  # KMK040412/guiowl-aw-mix-hybrid-packed (balance_report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-sec", type=float, required=True, help="median steady compute_sec/step from smoke")
    ap.add_argument("--batch", type=int, default=32, help="global batch size")
    ap.add_argument("--price", type=float, default=2.70, help="USD per CHIP-hour (your actual spot rate)")
    ap.add_argument("--chips", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--setup-min", type=float, default=35.0, help="paid setup before steady steps (download+compile)")
    a = ap.parse_args()

    steps = math.ceil(a.episodes / a.batch)
    train_h = steps * a.step_sec / 3600.0
    setup_h = a.setup_min / 60.0
    wall_h = train_h + setup_h
    cost = wall_h * a.chips * a.price
    print(f"episodes={a.episodes}  batch={a.batch}  steps/epoch={steps}")
    print(f"step_sec={a.step_sec}  -> train={train_h:.2f}h  + setup={setup_h:.2f}h  = wall {wall_h:.2f}h")
    print(f"price=${a.price}/chip-hr x {a.chips} chips  ->  ~${cost:,.0f} for 1 epoch")
    print(f"(checkpoint every 300 steps => a spot preemption loses <= {300*a.step_sec/60:.1f} min of compute)")


if __name__ == "__main__":
    main()
