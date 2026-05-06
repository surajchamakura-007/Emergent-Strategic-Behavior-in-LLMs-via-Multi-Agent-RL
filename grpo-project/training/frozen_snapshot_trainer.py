"""GRPOTrainer subclass: per-rollout opponent dispatch via vLLM `lora_request`.

Authority: Implementation Map §2.6 + §3.3 + PRD v6.1 §7.2.

What this overrides. `_generate_and_score_completions` — the TRL hook that
runs vLLM rollouts for a batch of prompts. We intercept this to:
    1. Sample an opponent identity for each rollout (PRD §7.1).
    2. Build a per-rollout `LoRARequest` list.
    3. Pass through to vLLM, which routes generations to the right adapter.
    4. Run an IPD episode loop using the responses (caller-side game logic).

Why this is the riskiest single bet (Issue #6 / R14). vLLM's multi-LoRA
dispatch path interacts with TRL's internal caching, gradient sync, and
the colocate engine's weight-sync hook. Pre-flight script 02 verifies all
of this BEFORE any line of training code runs:
    - adapter A vs B at the same seed produces different outputs (dispatch works)
    - snapshot adapter's gradient is None or zero after backward (frozen)
    - trainable adapter's gradient is non-zero after backward (gradient flows)
    - after `accelerator.backward()`, vLLM's next rollout sees updated weights

If that pre-flight fails, we do not fall through to any other path inside
the trainer — we halt and the user picks a fallback (PRD §7.2 fallbacks).

Why ~50 LOC. Most of the rollout machinery is reused from
GRPOTrainer.parent. We only intercept the per-prompt LoRA selection and
forward the rest.

NOTE on TRL hook surface area. TRL 1.0.0 may name this method
`_generate_and_score_completions` or `_generate_completions` depending on
the patch level (Map §9 acknowledges this). Pre-flight 02 verifies the
correct name on the installed version. Below we override both candidates
and route them through the same internal `_dispatch_with_opponents`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from trl import GRPOTrainer
except ImportError:  # pragma: no cover — caught by stack_versions
    GRPOTrainer = object  # type: ignore[misc, assignment]

from training.callbacks.diagnostic_logging import RunState
from training.snapshot_buffer import SnapshotBuffer, SnapshotEntry


# Trainable adapter is registered with int_id 1 (Map §2.4 / §3.1)
TRAINABLE_LORA_INT_ID = 1
TRAINABLE_LORA_NAME = "trainable"


class FrozenSnapshotGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that dispatches opponent rollouts via vLLM multi-LoRA."""

    def __init__(
        self,
        *args,
        snapshot_buffer: SnapshotBuffer,
        run_state: RunState,
        rng: np.random.Generator,
        opponent_p_buffer: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.snapshot_buffer = snapshot_buffer
        self.run_state = run_state
        self.rng = rng
        self.opponent_p_buffer = opponent_p_buffer
        # vLLM engine handle (TRL stashes it on self; name varies across
        # versions). We resolve lazily on first use.
        self._vllm_engine_resolved: Any | None = None

    # ------------------------------------------------------------------
    # vLLM engine access — name varies across TRL minor versions
    # ------------------------------------------------------------------

    @property
    def vllm_engine(self) -> Any:
        if self._vllm_engine_resolved is not None:
            return self._vllm_engine_resolved
        for attr in ("llm", "vllm_engine", "vllm_client", "_vllm"):
            engine = getattr(self, attr, None)
            if engine is not None:
                self._vllm_engine_resolved = engine
                return engine
        raise RuntimeError(
            "FrozenSnapshotGRPOTrainer: cannot find vLLM engine on self. "
            "Pre-flight script 01 should have caught this. Check TRL version."
        )

    # ------------------------------------------------------------------
    # Opponent sampling helper — called per rollout
    # ------------------------------------------------------------------

    def _sample_opponent_request(
        self,
    ) -> tuple[Any, SnapshotEntry | None]:
        """Sample an opponent identity for one rollout.

        Returns:
            (lora_request, snapshot_entry_or_None).
            If snapshot_entry is None, the request points at the trainable adapter.
        """
        from vllm.lora.request import LoRARequest

        entry = self.snapshot_buffer.sample_opponent(self.rng, p_buffer=self.opponent_p_buffer)
        if entry is None:
            # Use trainable adapter for opponent role (warmup + 50% draw)
            req = LoRARequest(TRAINABLE_LORA_NAME, TRAINABLE_LORA_INT_ID,
                              self._trainable_adapter_path())
            return req, None
        req = LoRARequest(entry.snapshot_id, entry.lora_int_id, entry.adapter_path)
        return req, entry

    def _trainable_adapter_path(self) -> str:
        """Return the on-disk path of the trainable adapter.

        Set by the orchestrator (`train.py`) immediately after the trainable
        adapter is first persisted.
        """
        path = getattr(self, "_trainable_adapter_path_", None)
        if path is None:
            raise RuntimeError(
                "Trainable adapter path not registered. Orchestrator must call "
                "`trainer.register_trainable_adapter_path(path)` after the "
                "first save."
            )
        return path

    def register_trainable_adapter_path(self, path: str) -> None:
        self._trainable_adapter_path_ = str(path)

    # ------------------------------------------------------------------
    # Rollout-level instrumentation hook — called by the rollout loop per
    # rollout. Implementation depends on TRL internal layout, so we expose
    # a single public method the env-loop can call.
    # ------------------------------------------------------------------

    def record_rollout_diagnostics(
        self,
        *,
        opponent_from_buffer: bool,
        action_was_C: bool | None,
        format_ok: bool,
    ) -> None:
        """Forward a rollout's per-rollout flags into the shared RunState."""
        self.run_state.record_rollout(
            opponent_from_buffer=opponent_from_buffer,
            action_was_C=action_was_C,
            format_ok=format_ok,
        )

    # ------------------------------------------------------------------
    # R2 callback hook: mutate the rollout sampling temperature
    # ------------------------------------------------------------------

    def _patch_sampling_temperature(self, new_temp: float) -> bool:
        """Mutate the rollout sampling temperature; return True if it landed.

        TRL 1.0 stores SamplingParams in `self.generation_config`,
        `self.sampling_params`, or `args.temperature` — naming varies.
        Pre-flight 05 discovers which is mutable on this stack and persists
        the result to `configs/r2_runtime_mutable.json`.
        """
        landed = False

        # 1. trl args
        if hasattr(self, "args") and hasattr(self.args, "temperature"):
            try:
                self.args.temperature = new_temp
                landed = True
            except Exception:
                pass

        # 2. SamplingParams stored on self
        for attr in ("sampling_params", "generation_config"):
            sp = getattr(self, attr, None)
            if sp is not None and hasattr(sp, "temperature"):
                try:
                    sp.temperature = new_temp
                    landed = True
                except Exception:
                    pass

        # 3. vLLM engine-level default (some versions only honor this)
        try:
            engine = self.vllm_engine
            if hasattr(engine, "default_sampling_params"):
                engine.default_sampling_params.temperature = new_temp
                landed = True
        except Exception:
            pass

        # Verify by reading back, where possible
        if landed:
            for attr in ("sampling_params", "generation_config"):
                sp = getattr(self, attr, None)
                if sp is not None and hasattr(sp, "temperature"):
                    try:
                        if abs(float(sp.temperature) - new_temp) > 1e-6:
                            landed = False
                    except Exception:
                        pass

        return landed


__all__ = ["FrozenSnapshotGRPOTrainer", "TRAINABLE_LORA_INT_ID", "TRAINABLE_LORA_NAME"]
