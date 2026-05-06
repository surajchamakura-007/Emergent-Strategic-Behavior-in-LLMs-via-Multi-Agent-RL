"""Stage 1 hyperparameter configuration.

Authority: Implementation Map §2.1 + PRD v6.1 §2.3 (locked configuration table).

This module defines the single source of truth for every hyperparameter that
varies across runs (T, seed) or that is locked at the project level (group size,
episode cap, etc.). The trainer never hardcodes a hyperparameter — it always
reads from this Config object.

Usage:
    cfg = Config.from_yaml("configs/config_drgrpo_T5_seed1.yaml")
    cfg.to_wandb_dict()  # logged once at run start

Frozen dataclass: a Config instance is immutable after construction. This
prevents the silent drift class of bug where a callback mutates a config field
and breaks the run config snapshot.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Payoff matrix (PD, T-parameterized)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PayoffMatrix:
    """Iterated PD payoff matrix.

    Conventions (PROJECT_CONCEPTS.md §3):
        T (temptation)  — defect while opp cooperates
        R (reward)      — mutual cooperation
        P (punishment)  — mutual defection
        S (sucker)      — cooperate while opp defects

    Standard ordering: T > R > P > S, and 2R > T + S to make cooperation
    sustainable in the iterated game.
    """
    T: float
    R: float = 3.0
    P: float = 1.0
    S: float = 0.0

    def __post_init__(self) -> None:
        if not (self.T > self.R > self.P > self.S):
            raise ValueError(
                f"Invalid PD payoffs: require T>R>P>S, got "
                f"T={self.T} R={self.R} P={self.P} S={self.S}"
            )
        # 2R > T+S not strictly required for IPD, but warn on violation later.

    def lookup(self, my_action: str, opp_action: str) -> float:
        """Payoff to ME given (my_action, opp_action) ∈ {C,D}^2."""
        if my_action == "C" and opp_action == "C":
            return self.R
        if my_action == "C" and opp_action == "D":
            return self.S
        if my_action == "D" and opp_action == "C":
            return self.T
        if my_action == "D" and opp_action == "D":
            return self.P
        raise ValueError(f"Bad actions: my={my_action!r}, opp={opp_action!r}")


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """Stage 1 run configuration.

    Locked values from PRD v6.1 §2.3 (do not edit per-run):
        algorithm, group_size, geometric_p, episode_cap, max_seq_length,
        max_steps, snapshot_N, buffer_capacity, lora_rank, lora_alpha.

    Per-run values (vary across the 4-run matrix):
        T (5.0 or 9.0), seed (1 or 2).
    """

    # --- Identity ---
    run_name: str
    seed: int

    # --- Model + LoRA ---
    model_path: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"  # V100 has no BF16; D6 dropped (PRD v6.1)

    # --- Game (per-run varies T) ---
    T: float = 5.0
    R: float = 3.0
    P: float = 1.0
    S: float = 0.0

    # --- Environment (D5, D8) ---
    geometric_p: float = 0.95          # P(continue) per round; PRD §6.1 D5
    episode_cap: int = 60              # hard cap; PRD §6.1 D8 v6 update

    # --- Training (D9, group size locked) ---
    algorithm: str = "drgrpo"          # D1
    max_steps: int = 500               # D9
    group_size: int = 8                # PRD v6.1 §2.3
    learning_rate: float = 1e-5
    grad_accum_steps: int = 1
    per_device_batch_size: int = 1
    warmup_steps: int = 10
    save_steps: int = 25               # checkpoint every 25 steps (R12)

    # --- Sequence length (D5+D8 v6 consequence) ---
    max_seq_length: int = 1024
    max_completion_length: int = 400   # generation budget per Map §4.1

    # --- History truncation policy (Map §4) ---
    history_token_budget: int = 400
    keep_last_k: int = 25

    # --- Sampling (rollout) ---
    sampling_temp_default: float = 0.9
    sampling_temp_bumped: float = 1.2  # R2 mitigation target
    sampling_top_p: float = 1.0
    sampling_top_k: int = -1           # disabled
    repetition_penalty: float = 1.0

    # --- Frozen-snapshot self-play (D2) ---
    snapshot_N: int = 40               # save cadence
    buffer_capacity: int = 8           # ring buffer size
    opponent_p_buffer: float = 0.5     # P(opp ∈ B | |B|>0); PRD §7.1

    # --- R2 callback gates ---
    coop_ceiling_gate: float = 0.85    # Map §5.3
    r2_window_steps: int = 20
    # advantage_collapse_threshold is NOT in this dataclass — it is read at
    # runtime from configs/calibrated_threshold.json (Map §2.1, §6).

    # --- vLLM colocate ---
    vllm_gpu_memory_utilization: float = 0.40
    vllm_max_loras: int = 10           # 1 trainable + 8 snapshots + 1 spare
    vllm_dtype: str = "float16"
    vllm_logprobs_mode: str = "processed_logprobs"  # PRD v6.1 §5.1.2

    # --- W&B ---
    wandb_project: str = "grpo-social-dilemmas"
    wandb_entity: str = "suraj_chamakura-university-of-california-berkeley"
    wandb_mode: str = "offline"        # PRD v6.1 default
    wandb_tags: tuple[str, ...] = ("stage1", "drgrpo", "prd_v6_1")

    # --- Paths (resolved at runtime against $SCRATCH or /workspace) ---
    project_root: str = "/workspace/grpo-project"
    checkpoint_root: str = "checkpoints/stage1"
    snapshot_root: str = "snapshots"
    log_root: str = "logs"
    threshold_json: str = "configs/calibrated_threshold.json"

    # --- R2 fallback flag (decided in pre-flight 05; injected at load) ---
    r2_runtime_mutable: bool = True    # overridden by pre-flight artifact

    # ---------------------------------------------------------------------
    # Constructors / serialization
    # ---------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> "Config":
        """Load a Config from YAML, applying overlays for r2 flag if present."""
        with open(path, "r") as f:
            d = yaml.safe_load(f) or {}
        # Allow nested 'payoffs' shorthand
        if "payoffs" in d:
            payoffs = d.pop("payoffs")
            d.setdefault("T", payoffs.get("T", 5.0))
            d.setdefault("R", payoffs.get("R", 3.0))
            d.setdefault("P", payoffs.get("P", 1.0))
            d.setdefault("S", payoffs.get("S", 0.0))
        # Allow tuple fields specified as lists in YAML
        for f_name in ("lora_target_modules", "wandb_tags"):
            if f_name in d and isinstance(d[f_name], list):
                d[f_name] = tuple(d[f_name])
        # Inject r2 fallback flag from preflight artifact if it exists.
        flag_path = Path(d.get("project_root", cls.project_root)) / "configs" / "r2_runtime_mutable.json"
        if flag_path.exists():
            with open(flag_path) as f:
                d["r2_runtime_mutable"] = bool(json.load(f)["r2_runtime_mutable"])
        return cls(**d)

    @property
    def payoffs(self) -> PayoffMatrix:
        return PayoffMatrix(T=self.T, R=self.R, P=self.P, S=self.S)

    def to_wandb_dict(self) -> dict[str, Any]:
        """Flatten for `wandb.config.update(...)`. Tuples become lists."""
        d = asdict(self)
        d["lora_target_modules"] = list(d["lora_target_modules"])
        d["wandb_tags"] = list(d["wandb_tags"])
        # Add D6 audit field — PRD v6.1 explicitly requires logging this False.
        d["fp32_lm_head"] = False
        return d

    # ---------------------------------------------------------------------
    # Path helpers
    # ---------------------------------------------------------------------

    def checkpoint_dir(self) -> Path:
        return Path(self.project_root) / self.checkpoint_root / self.run_name

    def snapshot_dir(self) -> Path:
        return Path(self.project_root) / self.snapshot_root / self.run_name

    def log_dir(self) -> Path:
        return Path(self.project_root) / self.log_root / self.run_name

    def buffer_state_path(self) -> Path:
        return self.checkpoint_dir() / "buffer_state.json"

    def threshold_path(self) -> Path:
        return Path(self.project_root) / self.threshold_json


# ---------------------------------------------------------------------------
# Threshold loader (Map §6)
# ---------------------------------------------------------------------------

class CalibratedThresholdMissingError(RuntimeError):
    """Raised at training start if calibrated_threshold.json is absent.

    Forces preflight task 0 (analysis/threshold_calibration.py) to run
    before any training. See Implementation Map §6.
    """


def load_calibrated_threshold(cfg: Config) -> float:
    """Read the advantage-collapse threshold computed in preflight task 0."""
    path = cfg.threshold_path()
    if not path.exists():
        raise CalibratedThresholdMissingError(
            f"{path} missing. Run analysis/threshold_calibration.py first."
        )
    with open(path) as f:
        record = json.load(f)
    threshold = float(record["threshold"])
    if not (0.0 < threshold < 1.0):
        raise ValueError(
            f"Suspect calibrated threshold {threshold} outside (0,1); "
            f"re-run calibration."
        )
    return threshold


__all__ = [
    "Config",
    "PayoffMatrix",
    "load_calibrated_threshold",
    "CalibratedThresholdMissingError",
]
