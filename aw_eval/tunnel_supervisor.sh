#!/usr/bin/env bash
# Keep the AndroidWorld eval tunnel chain alive for long runs:
#   Vultr:18124  --(reverse, aw_reverse_tunnel.py)-->  thismachine:18124
#   thismachine:18124  --(ssh -L)-->  TPU:8124
# Re-establishes whichever hop drops. Run in background; kill to stop.
TPU=dayeonhwang9@34.84.241.117
LOGD=/tmp/tunnel_sup
mkdir -p "$LOGD"
echo "[sup] started $(date)"
while true; do
  # 1) local forward (this:18124 -> TPU:8124): health-check end to end
  if ! curl -sf --max-time 5 http://127.0.0.1:18124/health >/dev/null 2>&1; then
    # if the local forward itself is missing, (re)start it
    if ! pgrep -f "L 127.0.0.1:18124:127.0.0.1:8124" >/dev/null 2>&1; then
      echo "[sup] $(date) local-forward down -> restart"
      ssh -fN -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
        -o StrictHostKeyChecking=no -L 127.0.0.1:18124:127.0.0.1:8124 "$TPU" >>"$LOGD/fwd.log" 2>&1 || true
    fi
  fi
  # 2) reverse tunnel (Vultr:18124 -> this:18124): restart if process gone
  if ! pgrep -f "aw_reverse_tunnel.py --remote-port 18124" >/dev/null 2>&1; then
    echo "[sup] $(date) reverse-tunnel down -> restart"
    setsid python3 /home/perelman/aw_reverse_tunnel.py --remote-port 18124 --local-port 18124 \
      >>"$LOGD/rev.log" 2>&1 < /dev/null &
  fi
  sleep 10
done
