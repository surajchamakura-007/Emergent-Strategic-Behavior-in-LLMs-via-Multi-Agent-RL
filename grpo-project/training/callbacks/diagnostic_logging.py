"""Diagnostic per-step logging + the shared `RunState` object.

Authority: Implementation Map §2.5.

Why a shared RunState. The R2 callback needs to read cooperation_rate (not
typically logged by TRL), the format-warmup callback needs to read
format_violation_rate, and the trainer subclass writes opponent_diversity.
Plumbing all of these through TRL's `state` and `kwargs` would require many
TRL hook overrides; instead, every callback that needs cross-callback
visibility receives a shared `RunState` instance at construction time.

Concurrency note. Callbacks fire synchronously in the trainer's main thread;
no locking is needed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Sequence

try:
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover — pre-flight catches this
    class TrainerCallback:  # type: ignore[no-redef]
        pass

import numpy as np


# ---------------------------------------------------------------------------
# RunState — the cross-callback shared object
# ---------------------------------------------------------------------------

@dataclass
class RunState:
    """Per-run rolling diagnostics shared across callbacks.

    Updated by:
        - DiagnosticLoggingCallback: reads metrics from TRL state and writes
          rolling windows.
        - FrozenSnapshotGRPOTrainer: writes per-rollout opponent_diversity_buffer
          and per-step cooperation_rate.

    Read by:
        - TempBumpCallback (advantage_mean_abs + cooperation_rate windows).
        - FormatWarmupCallback (format_violation_rate window).
    """
    window_size: int = 20

    # Rolling per-step metric windows
    advantage_mean_abs_window: deque[float] = field(init=False)
    cooperation_rate_window:    deque[float] = field(init=False)
    format_violation_rate_window: deque[float] = field(init=False)

    # Per-rollout flags within the current step (cleared each step end)
    opponent_diversity_buffer: list[float] = field(default_factory=list)
    coop_actions_buffer:       list[int]   = field(default_factory=list)
    format_ok_buffer:          list[int]   = field(default_factory=list)

    # Mutable knobs
    sampling_temperature: float = 0.9

    # Trainer-set step counter (TRL state.global_step is also available, but
    # this is convenient for callbacks that don't get state passed in).
    last_step: int = 0

    def __post_init__(self) -> None:
        self.advantage_mean_abs_window = deque(maxlen=self.window_size)
        self.cooperation_rate_window = deque(maxlen=self.window_size)
        self.format_violation_rate_window = deque(maxlen=self.window_size)

    # -------- per-step ingestion (called by DiagnosticLoggingCallback) --------

    def update_from_step(
        self,
        *,
        advantage_mean_abs: float,
        cooperation_rate: float,
        format_violation_rate: float,
    ) -> None:
        self.advantage_mean_abs_window.append(advantage_mean_abs)
        self.cooperation_rate_window.append(cooperation_rate)
        self.format_violation_rate_window.append(format_violation_rate)

    # -------- per-rollout ingestion (called by trainer subclass) --------

    def record_rollout(
        self,
        *,
        opponent_from_buffer: bool,
        action_was_C: bool | None,
        format_ok: bool,
    ) -> None:
        self.opponent_diversity_buffer.append(1.0 if opponent_from_buffer else 0.0)
        self.format_ok_buffer.append(1 if format_ok else 0)
        if action_was_C is not None:
            self.coop_actions_buffer.append(1 if action_was_C else 0)

    # -------- step-end aggregations --------

    def step_end_aggregates(self) -> dict[str, float]:
        """Compute opponent_diversity, cooperation_rate, format_violation_rate
        for the current step. Called by DiagnosticLoggingCallback.on_step_end
        BEFORE clearing the buffers."""
        opp_div = (
            float(np.mean(self.opponent_diversity_buffer))
            if self.opponent_diversity_buffer else 0.0
        )
        coop = (
            float(np.mean(self.coop_actions_buffer))
            if self.coop_actions_buffer else 0.0
        )
        fmt_v = (
            1.0 - float(np.mean(self.format_ok_buffer))
            if self.format_ok_buffer else 0.0
        )
        return {
            "opponent_diversity": opp_div,
            "cooperation_rate":   coop,
            "format_violation_rate": fmt_v,
        }

    def reset_step_buffers(self) -> None:
        self.opponent_diversity_buffer.clear()
        self.coop_actions_buffer.clear()
        self.format_ok_buffer.clear()


# ---------------------------------------------------------------------------
# Helper — pull advantage stats from TRL's per-step logs
# ---------------------------------------------------------------------------

def _extract_advantage_stats(logs: dict[str, Any]) -> tuple[float, float]:
    """Best-effort extraction of (advantage_mean_abs, group_reward_std).

    TRL 1.0.0 logs `rewards/advantages_mean_abs` (or `advantages_abs_mean`),
    `rewards/std`, etc. Names vary between minor versions; we try a list.
    """
    adv_keys = (
        "advantages/mean_abs", "advantages_mean_abs",
        "rewards/advantages_mean_abs", "advantage_mean_abs",
    )
    std_keys = (
        "rewards/std", "group_reward_std", "rewards_std",
    )
    adv = next((float(logs[k]) for k in adv_keys if k in logs), float("nan"))
    std = next((float(logs[k]) for k in std_keys if k in logs), float("nan"))
    return adv, std


# ---------------------------------------------------------------------------
# DiagnosticLoggingCallback
# ---------------------------------------------------------------------------

class DiagnosticLoggingCallback(TrainerCallback):
    """Per-step W&B logging + RunState updates.

    Writes the metric set defined in PRD v6.1 §5.2:
        advantage_mean_abs, advantage_max_abs, group_reward_std,
        group_reward_range, pairwise_levenshtein_mean,
        unique_action_fraction, format_violation_rate, opponent_diversity,
        truncated_at_cap_fraction.

    Most of these are computed by the trainer subclass (rollout-level)
    and forwarded via RunState; this callback aggregates and logs.
    """

    def __init__(self, run_state: RunState, wandb_run=None) -> None:
        self.run_state = run_state
        self.wandb_run = wandb_run

    def on_log(self, args, state, control, logs=None, **kwargs):
        """TRL calls on_log with the same dict it logged — capture advantage stats."""
        if logs is None:
            return
        adv, std = _extract_advantage_stats(logs)
        # Stash for use in on_step_end (TRL's on_log fires before on_step_end)
        self._latest_adv_mean_abs = adv
        self._latest_group_reward_std = std

    def on_step_end(self, args, state, control, **kwargs):
        # Compute step-level aggregates from per-rollout buffers
        agg = self.run_state.step_end_aggregates()
        adv = getattr(self, "_latest_adv_mean_abs", float("nan"))

        # Update rolling windows
        if not (adv != adv):  # not NaN
            self.run_state.update_from_step(
                advantage_mean_abs=adv,
                cooperation_rate=agg["cooperation_rate"],
                format_violation_rate=agg["format_violation_rate"],
            )

        # Log to W&B
        if self.wandb_run is not None:
            payload = {
                "diag/advantage_mean_abs": adv,
                "diag/group_reward_std": getattr(self, "_latest_group_reward_std", float("nan")),
                "diag/cooperation_rate": agg["cooperation_rate"],
                "diag/format_violation_rate": agg["format_violation_rate"],
                "diag/opponent_diversity": agg["opponent_diversity"],
                "diag/sampling_temperature": self.run_state.sampling_temperature,
            }
            try:
                self.wandb_run.log(payload, step=state.global_step)
            except Exception:
                pass

        self.run_state.last_step = state.global_step
        self.run_state.reset_step_buffers()


__all__ = ["RunState", "DiagnosticLoggingCallback"]
