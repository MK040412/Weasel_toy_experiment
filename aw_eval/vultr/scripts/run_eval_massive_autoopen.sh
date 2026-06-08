#!/usr/bin/env bash
# Large-scale AndroidWorld eval: shard tasks across N emulator lanes, ALL lanes
# hitting a SINGLE shared model server (the TPU JAX server via tunnel), with the
# auto-open workaround enabled so the model starts in-app.
set -euo pipefail
source /data2/androidworld_eval/env.sh
source /data2/androidworld_eval/venv/bin/activate

TASK_SET=${TASK_SET:-standard_full}
SUITE_FAMILY=${SUITE_FAMILY:-android_world}
N_TASK_COMBINATIONS=${N_TASK_COMBINATIONS:-1}
TASK_RANDOM_SEED=${TASK_RANDOM_SEED:-30}
LANES=${LANES:-8}
SERVER_URL=${SERVER_URL:-http://127.0.0.1:18124}   # single shared server
COORD_MODE=${COORD_MODE:-normalized}
AUTO_OPEN=${AUTO_OPEN:-1}
OUT_ROOT=${OUT_ROOT:-/data2/androidworld_eval/runs/$(date +%Y%m%d_%H%M%S)_${TASK_SET}_massive}
mkdir -p "$OUT_ROOT/shards"
echo "OUT_ROOT=$OUT_ROOT"

get_tasks() {
  if [ "$TASK_SET" = "standard_full" ]; then
    python - <<'PY'
from android_world import registry
r = registry.TaskRegistry().get_registry(family=registry.TaskRegistry.ANDROID_WORLD_FAMILY)
print(",".join(sorted(r.keys())))
PY
  elif [ -f "/data2/androidworld_eval/task_sets/${TASK_SET}.txt" ]; then
    tr -d '\n ' < "/data2/androidworld_eval/task_sets/${TASK_SET}.txt"
  else
    echo "$TASK_SET"
  fi
}

TASKS_CSV=$(get_tasks)
python - "$TASKS_CSV" "$LANES" "$OUT_ROOT/shards" <<'PY'
import json, pathlib, sys
tasks=[t for t in sys.argv[1].split(",") if t]
lanes=int(sys.argv[2]); out=pathlib.Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
shards=[]
for i in range(lanes):
    shard=tasks[i::lanes]
    (out/f"shard_{i}.txt").write_text(",".join(shard), encoding="utf-8")
    shards.append({"shard": i, "n_tasks": len(shard)})
(out/"shards.json").write_text(json.dumps({"total_tasks": len(tasks), "lanes": lanes, "shards": shards}, indent=2))
print(json.dumps({"total_tasks": len(tasks), "lanes": lanes, "shard_sizes": [s["n_tasks"] for s in shards]}))
PY

cat > "$OUT_ROOT/plan.txt" <<PLAN
TASK_SET=$TASK_SET
SUITE_FAMILY=$SUITE_FAMILY
N_TASK_COMBINATIONS=$N_TASK_COMBINATIONS
TASK_RANDOM_SEED=$TASK_RANDOM_SEED
LANES=$LANES
SERVER_URL=$SERVER_URL (single shared)
COORD_MODE=$COORD_MODE
AUTO_OPEN=$AUTO_OPEN
MODEL=boltzmann_final_AR_grounded
PLAN

consoles=(5554 5556 5558 5560 5562 5564 5566 5568)
grpcs=(8554 8555 8556 8557 8558 8559 8560 8561)
pids=()
for i in $(seq 0 $((LANES - 1))); do
  tasks=$(cat "$OUT_ROOT/shards/shard_${i}.txt")
  [ -z "$tasks" ] && { echo "[lane $i] empty shard, skip"; continue; }
  name="massive_${TASK_SET}_shard${i}"
  echo "[lane $i] console=${consoles[$i]} grpc=${grpcs[$i]} url=$SERVER_URL ntasks=$(echo $tasks | tr ',' '\n' | grep -c .)"
  NAME="$name" SERVER_URL="$SERVER_URL" CONSOLE_PORT="${consoles[$i]}" GRPC_PORT="${grpcs[$i]}" \
  GUIOWL_COORD_MODE="$COORD_MODE" GUIOWL_REPAIR=1 GUIOWL_RECORD_TRAJ="${GUIOWL_RECORD_TRAJ:-0}" GUIOWL_AUTO_OPEN="$AUTO_OPEN" \
  TASKS="$tasks" SUITE_FAMILY="$SUITE_FAMILY" N_TASK_COMBINATIONS="$N_TASK_COMBINATIONS" TASK_RANDOM_SEED="$TASK_RANDOM_SEED" \
  OUT_ROOT="$OUT_ROOT" /data2/androidworld_eval/scripts/run_one_eval.sh \
  >"$OUT_ROOT/${name}.runner.log" 2>&1 &
  pids+=("$!")
done

status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done

/data2/androidworld_eval/scripts/summarize_androidworld_run.py "$OUT_ROOT" > "$OUT_ROOT/summary.stdout" 2>&1 || true
echo "MASSIVE_DONE $OUT_ROOT status=$status"
