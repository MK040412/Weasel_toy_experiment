#!/usr/bin/env bash
set -euo pipefail
source /data2/androidworld_eval/env.sh
source /data2/androidworld_eval/venv/bin/activate
NAME=${NAME:-run}
TASKS=${TASKS:-ContactsAddContact,ClockStopWatchRunning}
SUITE_FAMILY=${SUITE_FAMILY:-android_world}
N_TASK_COMBINATIONS=${N_TASK_COMBINATIONS:-1}
TASK_RANDOM_SEED=${TASK_RANDOM_SEED:-30}
FIXED_TASK_SEED=${FIXED_TASK_SEED:-true}
SERVER_URL=${SERVER_URL:-http://127.0.0.1:8123}
GUIOWL_COORD_MODE=${GUIOWL_COORD_MODE:-absolute}
CONSOLE_PORT=${CONSOLE_PORT:-5554}
GRPC_PORT=${GRPC_PORT:-8554}
OUT_ROOT=${OUT_ROOT:-/data2/androidworld_eval/runs}
RUN_DIR="$OUT_ROOT/$NAME"
mkdir -p "$RUN_DIR"
export GUIOWL_COORD_MODE
if [ "${GUIOWL_RECORD_TRAJ:-0}" = "1" ]; then
  export GUIOWL_RECORD_DIR="$RUN_DIR/trajectories"
  mkdir -p "$GUIOWL_RECORD_DIR"
else
  unset GUIOWL_RECORD_DIR
fi
python /data2/androidworld_eval/run_guiowl_androidworld.py \
  --adb_path "$ANDROID_HOME/platform-tools/adb" \
  --console_port "$CONSOLE_PORT" \
  --grpc_port "$GRPC_PORT" \
  --suite_family "$SUITE_FAMILY" \
  --task_random_seed "$TASK_RANDOM_SEED" \
  --tasks "$TASKS" \
  --n_task_combinations "$N_TASK_COMBINATIONS" \
  --fixed_task_seed="$FIXED_TASK_SEED" \
  --output_path "$RUN_DIR" \
  --guiowl_server_url "$SERVER_URL" \
  2>&1 | tee "$RUN_DIR/eval.log"
