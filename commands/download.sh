#!/bin/bash
# Download dataset to /dev/shm (RAM)
#
# Usage:
#   bash commands/download.sh calvin-abcd

set -euo pipefail
cd "$(dirname "$0")/.."

ENV="${1:-calvin-abcd}"
HF_TOKEN="${HF_TOKEN:-}"

case "$ENV" in
    calvin-abcd)
        REPO="fywang/calvin-task-ABCD-D-lerobot"
        DST="/dev/shm/calvin_abcd"
        ;;
    *)
        echo "Unknown env: $ENV"
        exit 1
        ;;
esac

echo "=== Download: $REPO → $DST ==="
echo "Available /dev/shm: $(df -h /dev/shm | tail -1 | awk '{print $4}')"

python << PYEOF
import os, time, urllib.request, urllib.error, http.client
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi

REPO = "$REPO"
DST = "$DST"
TOKEN = "$HF_TOKEN"
WORKERS = 64

api = HfApi(token=TOKEN or None)
files = api.list_repo_files(REPO, repo_type="dataset")
targets = [f for f in files if f.endswith(".parquet") or f.startswith("meta/")]
print(f"Files: {len(targets)}")

# Create dirs
dirs = set(os.path.dirname(os.path.join(DST, f)) for f in targets)
for d in dirs:
    os.makedirs(d, exist_ok=True)

# Check existing
missing = []
for f in targets:
    path = os.path.join(DST, f)
    if not os.path.exists(path) or os.path.getsize(path) < 100:
        missing.append(f)
print(f"Missing: {len(missing)} (skipping {len(targets)-len(missing)} existing)")

def download_one(rel):
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{rel}"
    dst = os.path.join(DST, rel)
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
                if len(data) < 100:
                    raise ValueError("too small")
                with open(dst, "wb") as fout:
                    fout.write(data)
            return True
        except (urllib.error.URLError, ValueError, TimeoutError, OSError,
                http.client.IncompleteRead, ConnectionError):
            time.sleep(1.0 * (attempt + 1))
    return False

t0 = time.time()
done, failed = 0, 0
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {pool.submit(download_one, f): f for f in missing}
    for future in as_completed(futures):
        if future.result():
            done += 1
        else:
            failed += 1
        total = done + failed
        if total % 1000 == 0 or total == len(missing):
            elapsed = time.time() - t0
            rate = total / elapsed if elapsed > 0 else 0
            print(f"  {total}/{len(missing)} (ok={done}, fail={failed}, {rate:.0f}/s)")

import shutil
usage = shutil.disk_usage(DST)
print(f"\nDone: ok={done}, failed={failed}, time={time.time()-t0:.0f}s")
print(f"{DST}: {(usage.total - usage.free) / 1024**3:.1f} GB")
PYEOF
