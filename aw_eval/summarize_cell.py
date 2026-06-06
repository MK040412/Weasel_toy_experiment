#!/usr/bin/env python3
"""Summarize one AndroidWorld run dir into strict/repaired metrics (JSON to stdout).

Deployed to Vultr; called by bd_sweep.py. Reads summary.json (regenerating it from
the per-task .pkl.gz via summarize_androidworld_run.py first if absent).
"""
import glob
import json
import sys


def _avg(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 3) if xs else None


def main():
    out_root = sys.argv[1]
    fs = glob.glob(f"{out_root}/summary.json")
    if not fs:
        print(json.dumps({"error": "no summary.json"}))
        return
    d = json.load(open(fs[0]))
    eps = d.get("episodes", [])

    def ok(e):
        s = e.get("success")
        return (s is True) or (isinstance(s, (int, float)) and s > 0)

    succ = [e for e in eps if ok(e)]
    rec = {
        "episodes": len(eps),
        "success": len(succ),
        "rate": round(100 * len(succ) / len(eps), 1) if eps else 0.0,
        "strict_json": _avg([e.get("strict_json_rate") for e in eps]),      # raw-output validity
        "mobile_use": _avg([e.get("mobile_use_rate") for e in eps]),
        "repair_rate": _avg([e.get("repair_rate") for e in eps]),            # overall repair usage
        "succ_repair_rate": _avg([e.get("repair_rate") for e in succ]),      # did wins need repair?
        "succ_tasks": sorted(set(str(e.get("task")) for e in succ)),
    }
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
