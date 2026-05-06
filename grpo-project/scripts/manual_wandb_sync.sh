#!/bin/bash
# Manual W&B sync loop — runs on the Bridges-2 LOGIN node (which has internet).
#
# Authority: STAGE1_EXECUTION_PLAN_v3.1 §3.4 + §6.1.
#
# Compute nodes are firewalled; W&B is set to offline mode so each training
# job writes locally to $SCRATCH/grpo-project/logs/<run>/wandb/. This script,
# run on the login node, periodically uploads the offline runs.
#
# Sync cadence:
#   - Hour 0–1 of any active run: every 30 min (catches format-violation
#     >30% in first 20 steps and early advantage-collapse alerts).
#   - Hour 1+: every 6 hours.
#
# Usage (typical):
#   ssh bridges-login
#   tmux new -s wandb-sync
#   bash $SCRATCH/grpo-project/scripts/manual_wandb_sync.sh
#
# Detach with Ctrl-b d. Re-attach with `tmux attach -t wandb-sync`.

set -euo pipefail

PROJECT_ROOT="${SCRATCH}/grpo-project"
SLEEP_FAST="${SLEEP_FAST:-1800}"    # 30 min
SLEEP_SLOW="${SLEEP_SLOW:-21600}"   # 6 hr
FAST_WINDOW_HOURS="${FAST_WINDOW_HOURS:-1}"

if ! command -v wandb >/dev/null; then
    echo "ERROR: wandb CLI not on PATH on login node. Install in a venv first." >&2
    exit 1
fi

start_ts=$(date +%s)
echo "[$(date -Iseconds)] Starting wandb-sync loop. Project: ${PROJECT_ROOT}"

while true; do
    # Sync any offline run dir found under any run's logs/.
    for d in "${PROJECT_ROOT}"/logs/*/wandb; do
        [[ -d "$d" ]] || continue
        echo "[$(date -Iseconds)] wandb sync --include-offline ${d}"
        wandb sync --include-offline "$d" || \
            echo "[$(date -Iseconds)] wandb sync exited nonzero; retrying next loop"
    done

    elapsed_s=$(( $(date +%s) - start_ts ))
    elapsed_h=$(( elapsed_s / 3600 ))
    if (( elapsed_h < FAST_WINDOW_HOURS )); then
        echo "[$(date -Iseconds)] sleeping ${SLEEP_FAST}s (fast cadence)"
        sleep "${SLEEP_FAST}"
    else
        echo "[$(date -Iseconds)] sleeping ${SLEEP_SLOW}s (slow cadence)"
        sleep "${SLEEP_SLOW}"
    fi
done
