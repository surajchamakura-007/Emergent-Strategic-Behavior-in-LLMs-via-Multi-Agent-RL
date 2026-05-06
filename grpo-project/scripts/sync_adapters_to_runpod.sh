#!/bin/bash
# rsync the final adapter from cluster $SCRATCH to a RunPod pod.
#
# Authority: STAGE1_EXECUTION_PLAN_v3.1 §5.6.
#
# Usage:
#   RUNPOD_HOST=user@1.2.3.4 RUNPOD_PORT=22 \
#       bash scripts/sync_adapters_to_runpod.sh drgrpo_T5_seed1_stage1
#
# Required env:
#   RUNPOD_HOST   - SSH target, e.g., "user@1.2.3.4" (as exposed by RunPod).
#   RUNPOD_PORT   - SSH port (defaults to 22 if unset).
#   SCRATCH       - cluster scratch root (Bridges-2 sets this automatically).
#
# We only rsync the `final/` directory (~80 MB). Snapshots stay on cluster.

set -euo pipefail

RUN_NAME="${1:-}"
if [[ -z "${RUN_NAME}" ]]; then
    echo "Usage: $0 <run_name>" >&2
    exit 2
fi
if [[ -z "${RUNPOD_HOST:-}" ]]; then
    echo "ERROR: RUNPOD_HOST not set" >&2
    exit 2
fi
PORT="${RUNPOD_PORT:-22}"

SRC="${SCRATCH}/grpo-project/checkpoints/stage1/${RUN_NAME}/final/"
DST="${RUNPOD_HOST}:/workspace/grpo-project/checkpoints/stage1/${RUN_NAME}/final/"

if [[ ! -d "${SRC}" ]]; then
    echo "ERROR: source missing: ${SRC}" >&2
    exit 1
fi

# rsync flags:
#   -a archive (preserve perms/timestamps)
#   -v verbose
#   -z compress in transit (LoRA weight files compress to ~50%)
#   --partial: resumable mid-transfer
#   --mkpath: create remote dirs (rsync ≥3.2.3)
#   -e "ssh -p $PORT"
echo "[$(date -Iseconds)] rsync ${SRC} → ${DST}"
rsync -avz --partial --mkpath \
    -e "ssh -p ${PORT} -o StrictHostKeyChecking=accept-new" \
    "${SRC}" "${DST}"
echo "[$(date -Iseconds)] rsync complete"
