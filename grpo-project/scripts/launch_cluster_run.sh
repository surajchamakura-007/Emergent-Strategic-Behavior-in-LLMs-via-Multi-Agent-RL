#!/bin/bash
# SLURM submission for one Stage 1 training run on Bridges-2 V100-32GB.
#
# Authority: STAGE1_EXECUTION_PLAN_v3.1 §3.4 + §5.4.
#
# Usage:
#   sbatch scripts/launch_cluster_run.sh configs/config_drgrpo_T5_seed1.yaml
#
# This script:
#   - requests the GPU partition with V100-32GB (PRD v6.1 D10)
#   - 36-hour walltime (cap=48h on Bridges-2 GPU partition)
#   - writes logs under $SCRATCH/grpo-project/logs/{run_name}/
#   - runs in W&B offline mode (PRD v6.1: compute nodes have no outbound HTTPS)
#   - on success: rsyncs the final adapter to RunPod and triggers wandb sync.

#SBATCH --job-name=grpo_stage1
#SBATCH --partition=GPU
#SBATCH --gres=gpu:v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# 1) Args
# ---------------------------------------------------------------------------
CONFIG_PATH="${1:-}"
if [[ -z "$CONFIG_PATH" ]]; then
    echo "Usage: sbatch $0 <path-to-config.yaml>" >&2
    exit 2
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: config not found: $CONFIG_PATH" >&2
    exit 2
fi

# Derive run name from the config filename (drop dir + extension).
RUN_NAME=$(basename "$CONFIG_PATH" .yaml | sed 's/^config_//')

# ---------------------------------------------------------------------------
# 2) Environment
# ---------------------------------------------------------------------------
PROJECT_ROOT="${SCRATCH}/grpo-project"
LOG_DIR="${PROJECT_ROOT}/logs/${RUN_NAME}"
WANDB_DIR="${LOG_DIR}/wandb"
mkdir -p "${LOG_DIR}" "${WANDB_DIR}"

cd "${PROJECT_ROOT}"
module purge
# These module names are placeholders — confirm via `module avail` on Bridges-2.
module load anaconda3 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true

source activate stage1

# Sanity: log resolved versions to the job log.
python -c "import torch, trl, vllm, peft, transformers; \
    print(f'torch={torch.__version__}, trl={trl.__version__}, vllm={vllm.__version__}, \
peft={peft.__version__}, transformers={transformers.__version__}')"
nvidia-smi

# ---------------------------------------------------------------------------
# 3) W&B offline (PRD v6.1 §3.4)
# ---------------------------------------------------------------------------
export WANDB_MODE=offline
export WANDB_DIR="${WANDB_DIR}"
export WANDB_PROJECT="grpo-social-dilemmas"

# ---------------------------------------------------------------------------
# 4) Train
# ---------------------------------------------------------------------------
echo "[$(date -Iseconds)] Starting training: ${RUN_NAME}"
python -u training/train.py --config "${CONFIG_PATH}" \
    2>&1 | tee -a "${LOG_DIR}/train.log"

TRAIN_RC=${PIPESTATUS[0]}
echo "[$(date -Iseconds)] Training exited rc=${TRAIN_RC}"

# ---------------------------------------------------------------------------
# 5) Post-job hooks (success only)
# ---------------------------------------------------------------------------
if [[ ${TRAIN_RC} -eq 0 ]]; then
    # Sync W&B logs from the LOGIN NODE (compute nodes are firewalled).
    echo "[$(date -Iseconds)] Triggering wandb sync via login-node helper..."
    touch "${LOG_DIR}/.READY_FOR_WANDB_SYNC"

    # Rsync final adapter to RunPod (only if the network volume is reachable).
    if [[ -n "${RUNPOD_HOST:-}" ]]; then
        bash scripts/sync_adapters_to_runpod.sh "${RUN_NAME}"
    else
        echo "[$(date -Iseconds)] RUNPOD_HOST unset; skipping rsync. Run scripts/sync_adapters_to_runpod.sh manually."
    fi
fi

exit ${TRAIN_RC}
