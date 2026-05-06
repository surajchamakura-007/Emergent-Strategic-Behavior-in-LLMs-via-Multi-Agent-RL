# Stage 1 — GRPO Social-Dilemma Training Pipeline

**Spec authority:** PRD v6.1 + STAGE1_EXECUTION_PLAN_v3.1 + STAGE1_IMPLEMENTATION_MAP.

This is the production code for the 4-run Dr. GRPO matrix on IPD:
T ∈ {5, 9} × seed ∈ {1, 2}, 500 steps each, on Bridges-2 V100-32GB.

## Layout

```
configs/                  Run config + version pinning
  config.py               Frozen Config dataclass (all hyperparams)
  stack_versions.py       Version range checks for trl/vllm/transformers/peft
  config_drgrpo_T*_seed*.yaml   The 4 run configs (only T + seed differ)

envs/                     Game environments (same step() interface)
  prisoners_dilemma.py    IPD with stochastic horizon (p=0.95, cap=60)
  stag_hunt.py            Tier C transfer env (coordination)
  public_goods.py         Tier C transfer env (free-riding)
  history.py              Round dataclass + Map §4 truncation policy

training/                 Training pipeline
  reward.py               FORMAT_RE + multiplicative gated reward
  prompt_builder.py       Prompt assembly (system + history + format spec)
  snapshot_buffer.py      Ring buffer with atomic-persist invariants (§7.1)
  frozen_snapshot_trainer.py   GRPOTrainer subclass with multi-LoRA dispatch
  train.py                Orchestrator (canonical hook order, Map §3.1)
  callbacks/              4 callbacks (diagnostic, format-warmup, R2, snapshot)

evaluation/               Tier A/B/C eval
  opponents.py            TfT, AC, AD, R50, GenerousTfT, GrimTrigger, GPT-4o-mini
  eval.py                 Tier A: fixed-opponent matchups on IPD
  trace_eval.py           Per-round reasoning-trace dumper (REMUL/RFEval)
  elo.py                  Tier B: round-robin Elo across 4 adapters
  transfer.py             Tier C: Stag Hunt + Public Goods

analysis/
  threshold_calibration.py   Map §6: 5th-percentile threshold from 250-step pilot
  fit_sigmoids.py            D7: per-run R(C) sigmoid fits with C_final/C_mid
  parity_audit.py            R11: cluster ↔ RunPod 10-step diff verdict

utils/
  seed.py                 deterministic seeding across all RNGs
  lora_io.py              atomic adapter save (tmp → fsync → rename → fsync_dir)
  experiment_logger.py    W&B init (offline mode default)

preflight/                10 gating smoke tests (Map §7)
  01_vllm_colocate_smoke.py    Riskiest single bet — gates everything
  02_multi_lora_smoke.py       PRD v6.1 §7.2 verification
  03_trl_flags_smoke.py        scale_rewards=False (S1-12)
  04_buffer_resume_smoke.py    §7.1 resume integrity (S1-15)
  05_temp_callback_smoke.py    R2 runtime mutability probe (Issue #4)
  06_format_gate_smoke.py      Bimodal reward distribution
  07_episode_dist_smoke.py     Geometric horizon mean/std/trunc
  08_max_seq_len_smoke.py      VRAM at max_seq=1024
  09_parity_audit.py           Per-platform 10-step record dump
  10_untrained_baseline.py     RFEval reference point (blocking dep)
  README.md                    DAG + per-script exit criteria

scripts/
  launch_cluster_run.sh         SLURM submit (V100-32GB, 36 hr walltime)
  sync_adapters_to_runpod.sh    rsync final/ adapter cluster → RunPod
  manual_wandb_sync.sh          Login-node tmux loop for offline W&B sync

tests/                    Unit tests for the determinism-critical pieces
  test_reward_format_gate.py
  test_history_truncation.py
  test_episode_horizon.py
  test_snapshot_buffer.py
  test_temp_callback_gate.py
```

## Quickstart

### 1. Build env on Bridges-2

```bash
ssh bridges-login
cd $SCRATCH
git clone <repo-url> grpo-project
cd grpo-project
conda env create -f environment.yml -n stage1   # see stack_versions.py for pins
conda activate stage1
```

### 2. Run preflights (CPU-only first, then GPU)

```bash
# Pure-CPU; no model load
python preflight/06_format_gate_smoke.py
python preflight/07_episode_dist_smoke.py

# Threshold from the 250-step pilot — required before training start
python analysis/threshold_calibration.py \
    --run-id <pilot_run_id> --project grpo-social-dilemmas \
    --entity suraj_chamakura-university-of-california-berkeley \
    --out configs/calibrated_threshold.json

# GPU smokes — run on a V100 dev node
python preflight/01_vllm_colocate_smoke.py --config configs/config_drgrpo_T5_seed1.yaml
python preflight/02_multi_lora_smoke.py    --config configs/config_drgrpo_T5_seed1.yaml
python preflight/03_trl_flags_smoke.py     --config configs/config_drgrpo_T5_seed1.yaml
python preflight/04_buffer_resume_smoke.py --config configs/config_drgrpo_T5_seed1.yaml
python preflight/05_temp_callback_smoke.py --config configs/config_drgrpo_T5_seed1.yaml
python preflight/08_max_seq_len_smoke.py   --config configs/config_drgrpo_T5_seed1.yaml
python preflight/09_parity_audit.py        --config configs/config_drgrpo_T5_seed1.yaml \
    --platform-label bridges_v100

# Same on RunPod 4090
python preflight/09_parity_audit.py --config configs/config_drgrpo_T5_seed1.yaml \
    --platform-label runpod_4090
python preflight/10_untrained_baseline.py --config configs/config_drgrpo_T5_seed1.yaml
```

After every preflight passes, run the parity audit (offline, on either box):

```bash
python analysis/parity_audit.py \
    --cluster logs/preflight/09_parity_bridges_v100.json \
    --runpod  logs/preflight/09_parity_runpod_4090.json
```

A `verdict: "pass"` is the gate before submitting cluster jobs.

### 3. Submit the 4 cluster jobs

```bash
cd $SCRATCH/grpo-project
sbatch scripts/launch_cluster_run.sh configs/config_drgrpo_T5_seed1.yaml
sbatch scripts/launch_cluster_run.sh configs/config_drgrpo_T5_seed2.yaml
# Hold T=9 jobs until T=5 seed=1 has passed Q+12 health check (plan §5.2):
sbatch --hold scripts/launch_cluster_run.sh configs/config_drgrpo_T9_seed1.yaml
sbatch --hold scripts/launch_cluster_run.sh configs/config_drgrpo_T9_seed2.yaml
```

In a separate tmux session on the login node:

```bash
bash scripts/manual_wandb_sync.sh
```

### 4. Tier A/B/C eval on RunPod

```bash
# After rsync, four final adapters land at /workspace/.../<run>/final/
for run in drgrpo_T5_seed1 drgrpo_T5_seed2 drgrpo_T9_seed1 drgrpo_T9_seed2; do
    python evaluation/eval.py \
        --config configs/config_${run}.yaml \
        --adapter /workspace/grpo-project/checkpoints/stage1/${run}_stage1/final \
        --episodes 20
    python evaluation/trace_eval.py \
        --config configs/config_${run}.yaml \
        --adapter /workspace/grpo-project/checkpoints/stage1/${run}_stage1/final \
        --also-untrained
    python evaluation/transfer.py \
        --config configs/config_${run}.yaml \
        --adapter /workspace/grpo-project/checkpoints/stage1/${run}_stage1/final
done

# Elo across all 4 + GPT-4o-mini
python evaluation/elo.py \
    --config configs/config_drgrpo_T5_seed1.yaml \
    --adapters '{"T5_s1": "...", "T5_s2": "...", "T9_s1": "...", "T9_s2": "..."}' \
    --include-gpt4o
```

### 5. Sigmoid fits (D7)

```bash
# manifest.json: list of {run_name, T, seed, wandb_run_id}
python analysis/fit_sigmoids.py --R0 <untrained_avg> --runs-json manifest.json
```

## Invariants the code enforces

These are not optional — they're checked at runtime and will halt the job:

1. **Stack versions** (`configs/stack_versions.py`): trl ≥1.0, vllm ≥0.10.2, etc.
   Halts at startup if anything is outside the locked range.
2. **Calibrated threshold present** (`configs/config.py:load_calibrated_threshold`):
   raises `CalibratedThresholdMissingError` if `configs/calibrated_threshold.json`
   doesn't exist. Forces preflight 0 to run before training.
3. **Atomic adapter writes** (`utils/lora_io.py`): every snapshot save uses
   tmp → fsync(file) → rename → fsync(parent_dir). No torn writes.
4. **Buffer-state consistency** (`training/snapshot_buffer.py`): every
   `commit()` call follows a `dry_run_add()` + `persist_atomic()`; vLLM
   add fires AFTER persist; on vLLM failure we roll back the JSON.
5. **Buffer load-or-halt** (PRD v6.1 §7.1): on resume, missing or corrupted
   `buffer_state.json` halts the job (`BufferStateMissingError` /
   `BufferIntegrityError`); we never silently start with an empty buffer.
6. **Format gate** (`training/reward.py`): single regex source of truth;
   reward = `format_gate * payoff_lookup`; no additive bonus path.
7. **R2 runtime mutability** (Map §5, Issue #4): preflight 05 writes
   `configs/r2_runtime_mutable.json`; if False, `TempBumpCallback` is
   constructed with `bumped_temp=None` (log-only) instead of silently no-op.

## Where to look when something goes wrong

| Symptom                                            | Look in                                                   |
|----------------------------------------------------|-----------------------------------------------------------|
| `CalibratedThresholdMissingError` at startup       | Run `analysis/threshold_calibration.py` first             |
| `StackVersionError`                                | `configs/stack_versions.py` — bump or pin                  |
| `format_violation_rate` > 30% in first 20 steps    | `FormatWarmupCallback` raised; SFT format warmup needed   |
| `opponent_diversity` ≠ 0.0 in first 40 steps       | empty-buffer rule violated; inspect `SnapshotBuffer.sample_opponent` |
| `opponent_diversity` < 0.3 after step 320          | vLLM `lora_request` dispatch broken; engage §7.2 fallback |
| `R2MitigationFailedError`                          | preflight 05 said mutable, runtime says no — halt loudly  |
| Buffer `BufferIntegrityError` on resume            | snapshot adapter dir was deleted; check `$SCRATCH` quota   |
| TIS importance ratios skewed                       | `vllm_kwargs.logprobs_mode` not `processed_logprobs`      |
| Step time > 200 s on V100                          | reduce G to 4 (S1-14 mitigation)                          |
| Cluster ↔ RunPod parity > 5%                       | `analysis/parity_audit.py` reports breaches; halt fan-out |

## Companion docs in project knowledge

- `STAGE1_IMPLEMENTATION_MAP.md` — file-level contracts, hook order, build order
- `PRD_v6_1.md` — locked decisions D1–D10, fallback policies
- `STAGE1_EXECUTION_PLAN_v3_1.md` — schedule, walltime, monitoring thresholds
- `MODEL_AND_LIMITATIONS.md` — caveats acknowledged in writeup
- `RFEVAL_EXTENSION.md` — Stage 2 faithfulness extension
