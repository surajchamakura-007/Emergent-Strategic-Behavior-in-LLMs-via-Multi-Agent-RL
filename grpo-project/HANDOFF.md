# HANDOFF — Stage 1 GRPO training on Bridges-2

This is everything you need to run the 4-job training fan-out. Treat the
`<bracketed_placeholders>` as fields you fill in for your account/cluster.

## What this is

4 training runs of Dr. GRPO on Iterated Prisoner's Dilemma:

| Job  | Config                                       | T  | Seed |
|------|----------------------------------------------|----|------|
| C-1  | `configs/config_drgrpo_T5_seed1.yaml`        | 5  | 1    |
| C-2  | `configs/config_drgrpo_T5_seed2.yaml`        | 5  | 2    |
| C-3  | `configs/config_drgrpo_T9_seed1.yaml`        | 9  | 1    |
| C-4  | `configs/config_drgrpo_T9_seed2.yaml`        | 9  | 2    |

Each run is **500 steps**, **~30 GPU-hours on V100-32GB** (request 36-hour
walltime for buffer). Output = a ~80 MB LoRA adapter under
`$SCRATCH/grpo-project/checkpoints/stage1/<run_name>/final/`.

## Cluster requirements

- 1 × V100-32GB (32 GB VRAM) **or** 1 × A100. Code runs FP16 throughout.
- 8 CPU cores, 64 GB RAM.
- ~25 GB scratch disk: ~15 GB HF model cache + ~4 GB checkpoints + logs.
- CUDA 12.1+ available via `module load`.
- No outbound internet from compute nodes is fine — W&B runs in offline
  mode and is synced from the login node.

## One-time setup

```bash
ssh <your-bridges-username>@bridges2.psc.edu
cd $SCRATCH

# 1. Get the code.
tar xzf <path-to>/grpo-project-stage1.tar.gz
cd grpo-project

# 2. Build the conda env. Site module names — confirm with `module avail`.
module load anaconda3
module load cuda/12.1
conda create -n stage1 python=3.11 -y
conda activate stage1

# Install torch first so bitsandbytes builds against the right CUDA.
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch>=2.4.0
pip install -r requirements.txt

# Sanity check the version pins.
python -c "from configs.stack_versions import assert_versions; assert_versions()"

# 3. Hugging Face — Qwen2.5 is gated, accept the license once.
#    Visit https://huggingface.co/Qwen/Qwen2.5-7B-Instruct, click "Agree".
#    Then on the login node:
export HF_HOME=$SCRATCH/hf_cache       # add to ~/.bashrc so SLURM jobs see it
mkdir -p "$HF_HOME"
huggingface-cli login                  # paste your HF token

# 4. W&B login.
#    Option A (your own W&B account): wandb login, paste your API key from
#      https://wandb.ai/authorize. Update wandb_entity in the YAMLs to your username.
#    Option B (Suraj's account): use the API key Suraj sent you.
wandb login
```

## Pre-flight (do this on a dev/interactive GPU before queuing the real jobs)

```bash
# Calibrated threshold — Suraj should ship calibrated_threshold.json with the tar.
# If it's missing, ask him for it (or for his W&B run-id of the 250-step pilot
# so you can run analysis/threshold_calibration.py).
ls configs/calibrated_threshold.json   # must exist before training

# Cheap CPU smokes — no GPU needed.
python preflight/06_format_gate_smoke.py
python preflight/07_episode_dist_smoke.py

# GPU smokes — grab an interactive 2-hour V100 session.
srun --partition=GPU --gres=gpu:v100-32:1 --time=2:00:00 --pty bash

cd $SCRATCH/grpo-project
conda activate stage1
python preflight/01_vllm_colocate_smoke.py --config configs/config_drgrpo_T5_seed1.yaml
python preflight/02_multi_lora_smoke.py    --config configs/config_drgrpo_T5_seed1.yaml
python preflight/03_trl_flags_smoke.py     --config configs/config_drgrpo_T5_seed1.yaml
python preflight/04_buffer_resume_smoke.py --config configs/config_drgrpo_T5_seed1.yaml
python preflight/05_temp_callback_smoke.py --config configs/config_drgrpo_T5_seed1.yaml
python preflight/08_max_seq_len_smoke.py   --config configs/config_drgrpo_T5_seed1.yaml
python preflight/09_parity_audit.py        --config configs/config_drgrpo_T5_seed1.yaml \
    --platform-label bridges_v100

# Quick aggregate — every preflight should report `passed: true`:
python -c "import glob, json; \
  print({p: json.load(open(p)).get('passed') for p in sorted(glob.glob('logs/preflight/*.json'))})"
exit                                   # release the interactive session
```

If any preflight fails, see the symptom→action table at the bottom of
`README.md` or the per-script diagnostics in `preflight/README.md`.

## Submit the real jobs

The submit script expects the SLURM partition `GPU` and `--gres=gpu:v100-32`.
**Confirm those names on your cluster** with `sinfo` before submitting; if
they differ, edit the `#SBATCH` lines at the top of
`scripts/launch_cluster_run.sh`.

```bash
cd $SCRATCH/grpo-project

# Submit T=5 jobs immediately.
sbatch scripts/launch_cluster_run.sh configs/config_drgrpo_T5_seed1.yaml
sbatch scripts/launch_cluster_run.sh configs/config_drgrpo_T5_seed2.yaml

# Submit T=9 with --hold; release after T=5 seed=1 looks healthy at hour ~12.
T9_S1=$(sbatch --hold --parsable scripts/launch_cluster_run.sh configs/config_drgrpo_T9_seed1.yaml)
T9_S2=$(sbatch --hold --parsable scripts/launch_cluster_run.sh configs/config_drgrpo_T9_seed2.yaml)
echo "Held T=9 job ids: $T9_S1 $T9_S2 — release with: scontrol release <id>"
```

In a separate tmux session **on the login node** (compute nodes are firewalled):

```bash
tmux new -s wandb-sync
bash $SCRATCH/grpo-project/scripts/manual_wandb_sync.sh
# detach: Ctrl-b d
```

## Health check at hour ~12 (before releasing T=9)

W&B dashboard for run `drgrpo_T5_seed1_stage1` should show:
- `format_violation_rate` < 0.10 sustained
- `advantage_mean_abs` non-zero (i.e., the policy is still learning)
- `opponent_diversity` = 0.0 in steps 1–39, ramping to ~0.5 from step 40
- step time < 200 s/step

If all green, release the held T=9 jobs:

```bash
scontrol release $T9_S1 $T9_S2
```

If anything's red, ping Suraj before releasing.

## What to send back

When all 4 finish (~36 hours after submission), the deliverables for Suraj are:

```
$SCRATCH/grpo-project/checkpoints/stage1/
  drgrpo_T5_seed1_stage1/final/    (~80 MB LoRA adapter)
  drgrpo_T5_seed2_stage1/final/
  drgrpo_T9_seed1_stage1/final/
  drgrpo_T9_seed2_stage1/final/

$SCRATCH/grpo-project/logs/<each-run>/wandb/   (offline W&B runs)
```

The `manual_wandb_sync.sh` loop should already have uploaded all logs to
W&B. Confirm by checking the W&B project dashboard. Then either:
- Suraj rsyncs the adapters down himself (give him SSH access to your
  scratch dir), or
- You run `scripts/sync_adapters_to_runpod.sh <run_name>` for each run with
  `RUNPOD_HOST` set to the IP Suraj sends you.

## Troubleshooting one-liners

| Symptom                                              | What to try                                                                                |
|------------------------------------------------------|--------------------------------------------------------------------------------------------|
| `StackVersionError` at startup                       | `pip install -r requirements.txt --upgrade`                                                |
| `CalibratedThresholdMissingError`                    | Get `configs/calibrated_threshold.json` from Suraj, copy it in                             |
| OOM during training                                  | Reduce `group_size` from 8 to 4 in the YAML (S1-14 mitigation)                             |
| Job runs > 36h walltime                              | Same — drop `group_size`, halve throughput cost                                            |
| HF download stalls / 401                             | Re-run `huggingface-cli login`; confirm you accepted the Qwen2.5 license on the model page |
| `wandb sync` says nothing to upload                  | Check `WANDB_DIR` matches what the job actually wrote; default is `$SCRATCH/.../wandb/`    |
| Preflight 02 fails (multi-LoRA)                      | Suraj has a fallback path; ping him before proceeding                                      |
| SLURM partition or gres flag rejected                | Edit `#SBATCH --partition=` and `--gres=` in `scripts/launch_cluster_run.sh` to match `sinfo` |

## Contacts

- Code/research questions: **Suraj** — `<suraj-contact>`
- Cluster account/queue issues: **PSC Bridges-2 helpdesk** — help@psc.edu
