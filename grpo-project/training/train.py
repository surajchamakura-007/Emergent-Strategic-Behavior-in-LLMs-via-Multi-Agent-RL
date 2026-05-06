"""Stage 1 training orchestrator.

Authority: Implementation Map §2.7 ("does not contain business logic") +
§3.1 (training-start hook ordering, the second correctness invariant).

What lives here. Just the wiring: load config, assert versions, seed, build
model + vLLM, attach callbacks, run trainer.train(). Anything cleverer goes
in the imported modules.

What does NOT live here. The reward function (training/reward.py), the
opponent sampling rule (training/snapshot_buffer.py), the IPD env
(envs/prisoners_dilemma.py), the truncation policy (envs/history.py),
the R2 logic (training/callbacks/temp_bump_callback.py), etc.

Hook ordering at startup (Map §3.1).
    1. assert_versions()
    2. seed_all(cfg.seed)
    3. load_calibrated_threshold()    — fails fast if calibration didn't run
    4. build_model() + LoRA attach
    5. build_vllm_colocate(enable_lora=True, max_loras=10)
    6. save trainable adapter atomically
    7. vllm.add_lora(trainable @ int_id=1)
    8. construct buffer; if resuming → load_or_halt + re-register snapshots
    9. construct trainer
    10. attach callbacks (in order: Diagnostics, R1, R2, Snapshot)
    11. trainer.train()

Reasons for the order are documented in Map §3.1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from configs.config import Config, CalibratedThresholdMissingError, load_calibrated_threshold
from configs.stack_versions import assert_versions
from training.callbacks.diagnostic_logging import DiagnosticLoggingCallback, RunState
from training.callbacks.format_warmup_callback import FormatWarmupCallback
from training.callbacks.snapshot_callback import SnapshotCallback
from training.callbacks.temp_bump_callback import TempBumpCallback
from training.frozen_snapshot_trainer import (
    FrozenSnapshotGRPOTrainer,
    TRAINABLE_LORA_INT_ID,
    TRAINABLE_LORA_NAME,
)
from training.reward import batch_compute_rewards
from training.snapshot_buffer import (
    BufferStateMissingError,
    SnapshotBuffer,
)
from utils.experiment_logger import init_wandb_offline, log_static_run_config, manual_sync_helper
from utils.lora_io import save_adapter_atomically
from utils.seed import seed_all


# ---------------------------------------------------------------------------
# Builders — kept intentionally thin; pre-flight 01 verifies the vLLM args.
# ---------------------------------------------------------------------------

def build_model_and_tokenizer(cfg: Config):
    """4-bit base + LoRA-attached PeftModel."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=bnb_dtype,
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def build_grpo_config(cfg: Config, output_dir: Path):
    """TRL GRPOConfig with vLLM colocate + Dr.GRPO advantage normalization off."""
    from trl import GRPOConfig

    args = GRPOConfig(
        output_dir=str(output_dir),
        run_name=cfg.run_name,
        seed=cfg.seed,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        warmup_steps=cfg.warmup_steps,
        save_steps=cfg.save_steps,
        save_strategy="steps",
        logging_steps=1,
        max_prompt_length=cfg.max_seq_length - cfg.max_completion_length,
        max_completion_length=cfg.max_completion_length,
        num_generations=cfg.group_size,
        temperature=cfg.sampling_temp_default,
        top_p=cfg.sampling_top_p,
        # vLLM colocate (PRD v6.1 §5.1.2)
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        # Dr. GRPO: disable std normalization. The flag name is verified in
        # pre-flight 03; this is the canonical name in trl 1.0.x.
        scale_rewards=False,
        # Reporting
        report_to=["wandb"],
        bf16=False,
        fp16=True,  # V100 has no BF16 (D10)
    )
    return args


# ---------------------------------------------------------------------------
# Resume detection
# ---------------------------------------------------------------------------

def detect_resume(cfg: Config) -> bool:
    """A run is resumable iff its checkpoint dir exists and contains
    `buffer_state.json`. The buffer is the earliest-failure-mode signal."""
    ckpt = cfg.checkpoint_dir()
    return ckpt.exists() and (ckpt / "buffer_state.json").exists()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg_path: str) -> int:
    # 1. Stack version assertions (PRD v6.1).
    versions = assert_versions(verbose=True)

    # 2. Load + seed + threshold (read JSON now so we fail before model build).
    cfg = Config.from_yaml(cfg_path)
    seed_all(cfg.seed)
    threshold = load_calibrated_threshold(cfg)

    # Output paths
    cfg.checkpoint_dir().mkdir(parents=True, exist_ok=True)
    cfg.snapshot_dir().mkdir(parents=True, exist_ok=True)
    cfg.log_dir().mkdir(parents=True, exist_ok=True)

    # 3. W&B (offline by default).
    wandb_run = init_wandb_offline(cfg)
    log_static_run_config(wandb_run, {
        "calibrated_advantage_threshold": threshold,
        "stack_versions": versions,
    })
    manual_sync_helper(cfg)  # prints the wandb sync command

    # 4-5. Model + vLLM colocate engine (built by TRL/GRPOTrainer).
    model, tokenizer = build_model_and_tokenizer(cfg)
    grpo_args = build_grpo_config(cfg, cfg.checkpoint_dir())

    # 6-7. Save trainable adapter to disk atomically; vLLM will be created
    # by GRPOTrainer with `enable_lora=True`. We pass the path so the
    # orchestrator can register the trainable adapter at int_id=1
    # immediately after trainer construction.
    trainable_path = cfg.checkpoint_dir() / "trainable_adapter"
    save_adapter_atomically(model, trainable_path)

    # 8. Snapshot buffer + resume detection.
    buffer = SnapshotBuffer(
        capacity=cfg.buffer_capacity,
        persist_path=cfg.buffer_state_path(),
    )
    rng = np.random.default_rng(cfg.seed)
    run_state = RunState(window_size=cfg.r2_window_steps)

    resuming = detect_resume(cfg)
    if resuming:
        try:
            buffer.load_or_halt()
        except BufferStateMissingError as e:
            print(f"[train] Halt: {e}", file=sys.stderr)
            return 2
        print(f"[train] Resumed buffer with {len(buffer)} entries", file=sys.stderr)

    # 9. Trainer — TRL constructs the colocate vLLM engine internally given
    # use_vllm=True.
    trainer = FrozenSnapshotGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_args,
        # `reward_funcs` and `train_dataset` are user-supplied. The reward
        # closure pulls payoffs from cfg and binds compute_reward over the
        # current batch. The dataset is a stream of IPD-prompt records; see
        # docs/training_dataset_format.md for the schema.
        reward_funcs=_make_reward_callable(cfg),
        train_dataset=_make_ipd_dataset(cfg, tokenizer),
        snapshot_buffer=buffer,
        run_state=run_state,
        rng=rng,
        opponent_p_buffer=cfg.opponent_p_buffer,
    )
    trainer.register_trainable_adapter_path(str(trainable_path))

    # Register the trainable adapter with vLLM at int_id=1.
    _register_trainable_with_vllm(trainer, trainable_path)

    # If resuming, re-register snapshots with vLLM in their original int_ids.
    if resuming:
        _reregister_snapshots_with_vllm(trainer, buffer)

    # 10. Callbacks (order matters: diagnostics first so other callbacks see
    #     the rolling windows after on_log fires).
    trainer.add_callback(DiagnosticLoggingCallback(run_state, wandb_run=wandb_run))
    trainer.add_callback(FormatWarmupCallback(run_state, wandb_run=wandb_run))
    trainer.add_callback(
        TempBumpCallback(
            run_state=run_state,
            threshold=threshold,
            coop_ceiling=cfg.coop_ceiling_gate,
            bumped_temp=cfg.sampling_temp_bumped if cfg.r2_runtime_mutable else None,
            window_steps=cfg.r2_window_steps,
            trainer_ref=trainer,
            wandb_run=wandb_run,
            runtime_mutability_verified=cfg.r2_runtime_mutable,
        )
    )
    trainer.add_callback(
        SnapshotCallback(
            N=cfg.snapshot_N,
            buffer=buffer,
            snapshots_root=cfg.snapshot_dir(),
            trainer_ref=trainer,
            wandb_run=wandb_run,
        )
    )

    # 11. Train.
    trainer.train(resume_from_checkpoint=resuming)

    # Final adapter (publishes a final/ directory next to the snapshots)
    save_adapter_atomically(model, cfg.checkpoint_dir() / "final")
    wandb_run.finish()
    return 0


# ---------------------------------------------------------------------------
# Helpers — kept private; thin wrappers around env / dataset / vLLM ops.
# ---------------------------------------------------------------------------

def _make_reward_callable(cfg: Config):
    """Wrap `batch_compute_rewards` in TRL's expected reward_funcs signature.

    TRL 1.0 expects callables of signature:
        reward(prompts, completions, **kwargs) -> list[float]

    The IPD dataset must put the opponent's action this round in
    `kwargs['opp_action']` per-prompt — see _make_ipd_dataset.
    """
    payoffs = cfg.payoffs

    def reward_fn(prompts, completions, **kwargs):
        opp_actions = kwargs.get("opp_action")
        if opp_actions is None:
            raise KeyError(
                "Dataset records must include 'opp_action'; got "
                f"keys={list(kwargs)}"
            )
        rewards, _ = batch_compute_rewards(completions, list(opp_actions), payoffs)
        return rewards

    return [reward_fn]


def _make_ipd_dataset(cfg: Config, tokenizer):
    """Build (or load) the per-prompt IPD dataset.

    For Stage 1 the dataset is generated on-the-fly from rollouts of the
    PrisonersDilemmaEnv against either the trainable adapter (warmup) or
    a frozen snapshot. The actual rollout loop is implemented inside
    FrozenSnapshotGRPOTrainer._generate_and_score_completions; the
    HuggingFace dataset is just a thin index of prompt seeds.

    Returns:
        A list-style dataset of `cfg.max_steps * cfg.group_size` records.
        Each record carries (prompt, opp_action, episode_id, round_idx).

    NOTE: The full dataset construction is intentionally minimal here — the
    prompt content is built per-rollout inside the trainer (since the
    prompt depends on the unfolding episode's history, which is not known
    at dataset-construction time). This stub provides the iteration count.
    """
    from datasets import Dataset

    # Each step samples `group_size` rollouts; trainer needs `max_steps * group_size`
    # records to drive its per-step iteration counter. The actual prompts are
    # rebuilt each rollout from the live env state.
    n_records = cfg.max_steps * cfg.group_size
    return Dataset.from_dict({
        "prompt": [""] * n_records,        # filled in by the trainer
        "opp_action": ["C"] * n_records,   # placeholder, overwritten by trainer
        "episode_id": [str(i) for i in range(n_records)],
        "round_idx": [0] * n_records,
    })


def _register_trainable_with_vllm(trainer: FrozenSnapshotGRPOTrainer, path) -> None:
    """Add the trainable adapter to vLLM's LoRA registry at int_id=1."""
    from vllm.lora.request import LoRARequest

    request = LoRARequest(TRAINABLE_LORA_NAME, TRAINABLE_LORA_INT_ID, str(path))
    engine = trainer.vllm_engine
    if hasattr(engine, "add_lora"):
        engine.add_lora(request)
    elif hasattr(engine, "add_lora_request"):
        engine.add_lora_request(request)
    else:
        raise RuntimeError("vLLM engine has no add_lora API")


def _reregister_snapshots_with_vllm(
    trainer: FrozenSnapshotGRPOTrainer,
    buffer: SnapshotBuffer,
) -> None:
    from vllm.lora.request import LoRARequest

    engine = trainer.vllm_engine
    for entry in buffer.current_state.entries:
        request = LoRARequest(entry.snapshot_id, entry.lora_int_id, entry.adapter_path)
        if hasattr(engine, "add_lora"):
            engine.add_lora(request)
        elif hasattr(engine, "add_lora_request"):
            engine.add_lora_request(request)


# ---------------------------------------------------------------------------
# Smoke harness — used by preflight scripts (Map §7).
# ---------------------------------------------------------------------------

def build_smoke_trainer(cfg: Config, num_steps: int = 10):
    """Run the full setup pipeline but return (trainer, run_state) instead
    of calling `.train()`.

    Used by `preflight/0[1-5,8,9].py` to do short, isolated runs without
    duplicating construction logic. Honors all of `main()`'s invariants:
        - version assertion
        - seed_all
        - W&B init (offline)
        - vLLM colocate engine
        - trainable adapter registered at int_id=1
        - all 4 callbacks installed
    Differences from `main()`:
        - max_steps overridden to `num_steps`
        - save_steps disabled (so smoke doesn't litter the checkpoint dir)
        - calibrated_threshold not required: if the JSON is missing, falls
          back to a permissive default (0.10) and notes it in the trainer.
        - Resume detection skipped (smoke is always fresh).
    """
    import dataclasses

    versions = assert_versions(verbose=False)
    cfg = dataclasses.replace(cfg, max_steps=num_steps, save_steps=10**9)
    seed_all(cfg.seed)

    try:
        threshold = load_calibrated_threshold(cfg)
    except CalibratedThresholdMissingError:
        # Smoke runs predate calibration — use a permissive fallback.
        threshold = 0.10

    cfg.checkpoint_dir().mkdir(parents=True, exist_ok=True)
    cfg.snapshot_dir().mkdir(parents=True, exist_ok=True)
    cfg.log_dir().mkdir(parents=True, exist_ok=True)

    wandb_run = init_wandb_offline(cfg)
    log_static_run_config(wandb_run, {
        "smoke": True, "num_steps": num_steps,
        "calibrated_advantage_threshold": threshold,
        "stack_versions": versions,
    })

    model, tokenizer = build_model_and_tokenizer(cfg)
    grpo_args = build_grpo_config(cfg, cfg.checkpoint_dir())

    trainable_path = cfg.checkpoint_dir() / "trainable_init"
    save_adapter_atomically(model, trainable_path)

    buffer = SnapshotBuffer(
        capacity=cfg.buffer_capacity,
        persist_path=cfg.buffer_state_path(),
    )
    rng = np.random.default_rng(cfg.seed)
    run_state = RunState(window_size=cfg.r2_window_steps)

    trainer = FrozenSnapshotGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_args,
        reward_funcs=_make_reward_callable(cfg),
        train_dataset=_make_ipd_dataset(cfg, tokenizer),
        snapshot_buffer=buffer,
        run_state=run_state,
        rng=rng,
        opponent_p_buffer=cfg.opponent_p_buffer,
    )
    trainer.register_trainable_adapter_path(str(trainable_path))
    _register_trainable_with_vllm(trainer, trainable_path)

    trainer.add_callback(DiagnosticLoggingCallback(run_state, wandb_run=wandb_run))
    trainer.add_callback(FormatWarmupCallback(run_state, wandb_run=wandb_run))
    trainer.add_callback(
        TempBumpCallback(
            run_state=run_state,
            threshold=threshold,
            coop_ceiling=cfg.coop_ceiling_gate,
            bumped_temp=cfg.sampling_temp_bumped if cfg.r2_runtime_mutable else None,
            window_steps=cfg.r2_window_steps,
            trainer_ref=trainer,
            wandb_run=wandb_run,
            runtime_mutability_verified=cfg.r2_runtime_mutable,
        )
    )
    trainer.add_callback(
        SnapshotCallback(
            N=cfg.snapshot_N,
            buffer=buffer,
            snapshots_root=cfg.snapshot_dir(),
            trainer_ref=trainer,
            wandb_run=wandb_run,
        )
    )
    return trainer, run_state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="Path to YAML config (e.g. configs/config_drgrpo_T5_seed1.yaml)")
    args = ap.parse_args()
    sys.exit(main(args.config))
