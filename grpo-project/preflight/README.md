# Preflight DAG — Stage 1

Authority: Implementation Map §7, PRD v6.1 §10.1, STAGE1_EXECUTION_PLAN_v3.1 §3.

These ten scripts gate every dependency the 4-run training fan-out relies on.
The DAG below shows the order; arrows mean "must pass before".

```
              ┌──────────────────────┐
              │ 06 format_gate_smoke │   (pure unit; no GPU)
              │ 07 episode_dist_smoke│
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ 01 vllm_colocate_smoke│  (riskiest single bet; gates everything)
              └──────────┬───────────┘
                         │
            ┌────────────┼─────────────┐
            ▼            ▼             ▼
    ┌────────────┐  ┌──────────┐  ┌──────────────┐
    │ 02 multi_  │  │ 03 trl_  │  │ 08 max_seq_  │
    │   lora     │  │   flags  │  │   len (VRAM) │
    └─────┬──────┘  └────┬─────┘  └──────────────┘
          │              │
          ▼              ▼
    ┌────────────────────────────┐
    │ 04 buffer_resume_smoke     │
    │ 05 temp_callback (Issue#4) │
    └──────────┬─────────────────┘
               │
               ▼
    ┌────────────────────────┐
    │ 09 parity_audit        │  (run on EACH platform: cluster + RunPod)
    │ 10 untrained_baseline  │  (RunPod; blocking dep for RFEval)
    └────────────────────────┘
```

## Per-script exit criteria

| #  | Script                          | Exit criteria                                                                                                                                | Failure → action                                                                                                |
|----|---------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| 01 | `01_vllm_colocate_smoke.py`     | 10 steps complete, median step time < 200 s, advantage_mean_abs logged, FP16 confirmed                                                       | Halt fan-out. Inspect TRL/vLLM error; check `vllm_kwargs={"dtype": "float16", "logprobs_mode": "processed_logprobs"}` |
| 02 | `02_multi_lora_smoke.py`        | Outputs differ across two LoRA adapters at same seed; trainable grad>0, snapshot grad=0; post-backward weight sync ok                          | Engage PRD v6.1 §7.2 fallback: `peft.set_adapter()`. Re-run smoke against fallback. Pre-flight gates path.       |
| 03 | `03_trl_flags_smoke.py`         | `scale_rewards=False` accepted by GRPOConfig AND advantage std varies across steps                                                            | Implement custom advantage callback (S1-12 fallback). Add to `train.py` before fan-out.                          |
| 04 | `04_buffer_resume_smoke.py`     | After step 80: `buffer_state.json` has 2 entries; resume reloads; sampling uniform; integrity error raised on tampered state                  | Inspect atomic-write flow in `utils/lora_io.py` and `training/snapshot_buffer.py`. Pre-flight catches it.        |
| 05 | `05_temp_callback_smoke.py`     | Probe completes; emits `configs/r2_runtime_mutable.json` with verdict                                                                        | If False: R2 callback constructed with `bumped_temp=None` (log-only path). Document in run log.                  |
| 06 | `06_format_gate_smoke.py`       | All rewards ∈ {0, S, P, R, T}; format-violation rate within ±10% of synthetic 40%                                                            | Inspect `training/reward.py:FORMAT_RE`. If leakage: tighten regex.                                               |
| 07 | `07_episode_dist_smoke.py`      | empirical mean within ±5% of analytic geometric-truncated mean; std 12–17; trunc 2–8%                                                        | Inspect `envs/prisoners_dilemma.py:step()` Bernoulli logic.                                                      |
| 08 | `08_max_seq_len_smoke.py`       | 5-step pass with max_seq=1024; peak VRAM under per-device budget (22 GB on 4090, 30 GB on V100-32GB)                                          | Drop max_seq to 768 OR drop G to 4. Update PRD §6 caveat.                                                        |
| 09 | `09_parity_audit.py`            | 10-step JSON written on each platform (run on cluster AND RunPod separately). The actual diff is in `analysis/parity_audit.py`.              | If `analysis/parity_audit.py` reports >5% drift on any of {reward, advantage_mean_abs, group_reward_std}: HALT.  |
| 10 | `10_untrained_baseline.py`      | All 4 fixed opponents complete 20 episodes each; per-opponent JSON populated                                                                  | Required for downstream RFEval. Without it, no faithfulness comparison is possible.                              |

## How to run

```bash
# Pure-CPU preflights (no model load):
python preflight/06_format_gate_smoke.py --config configs/config_drgrpo_T5_seed1.yaml
python preflight/07_episode_dist_smoke.py

# Model-load preflights (need a GPU):
python preflight/01_vllm_colocate_smoke.py --config configs/config_drgrpo_T5_seed1.yaml
python preflight/02_multi_lora_smoke.py    --config configs/config_drgrpo_T5_seed1.yaml
python preflight/03_trl_flags_smoke.py     --config configs/config_drgrpo_T5_seed1.yaml
python preflight/04_buffer_resume_smoke.py --config configs/config_drgrpo_T5_seed1.yaml
python preflight/05_temp_callback_smoke.py --config configs/config_drgrpo_T5_seed1.yaml
python preflight/08_max_seq_len_smoke.py   --config configs/config_drgrpo_T5_seed1.yaml
python preflight/09_parity_audit.py        --config configs/config_drgrpo_T5_seed1.yaml
python preflight/10_untrained_baseline.py  --config configs/config_drgrpo_T5_seed1.yaml
python analysis/parity_audit.py \
    --cluster logs/preflight/09_parity_Tesla_V100-SXM2-32GB.json \
    --runpod  logs/preflight/09_parity_NVIDIA_GeForce_RTX_4090.json
```

All preflights write JSON under `logs/preflight/`. Aggregate them after the
last one runs:

```bash
python -c "import glob, json; out={}; \
    [out.setdefault(p, json.load(open(p))) for p in glob.glob('logs/preflight/*.json')]; \
    print(json.dumps({k: v.get('passed') for k,v in out.items()}, indent=2))"
```

A green light on every script + the parity-audit verdict is the gate before
submitting the 4 cluster jobs.
