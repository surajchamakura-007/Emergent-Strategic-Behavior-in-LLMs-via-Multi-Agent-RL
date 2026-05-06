"""SnapshotCallback — saves a frozen LoRA every N steps and publishes to vLLM.

Authority: Implementation Map §3.2 (atomic hook ordering, the central
correctness invariant of Stage 1).

Hook ordering (Map §3.2). Triggered at step % N == 0 and step > 0:
    1. save_adapter_atomically(model, snapshot_dir/adapter_step{step})
    2. dry_run_add → (new_state, evicted)              # NO MUTATION
    3. buffer.persist_atomic(new_state)                # JSON now matches new state
    4. vllm.add_lora(new); vllm.remove_lora(evicted)   # ADD before REMOVE
    5. buffer.commit(new_state)                        # in-memory matches
    6. wandb.log(snapshot/...)

If any step in 1-4 fails, JSON is rolled back to the previous state and a
SnapshotReflectionError is raised. This preserves the invariant that disk
JSON is always consistent with what vLLM should be holding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from transformers import TrainerCallback
except ImportError:
    class TrainerCallback:  # type: ignore[no-redef]
        pass

from training.snapshot_buffer import (
    BufferState,
    SnapshotBuffer,
)
from utils.lora_io import save_adapter_atomically


class SnapshotReflectionError(RuntimeError):
    """Raised when vLLM-side LoRA add/remove fails after JSON has been written."""


class SnapshotCallback(TrainerCallback):
    """Save-every-N-steps + atomic publish to vLLM."""

    def __init__(
        self,
        *,
        N: int,
        buffer: SnapshotBuffer,
        snapshots_root: str | Path,
        trainer_ref: Any,        # FrozenSnapshotGRPOTrainer (provides .model and .vllm)
        wandb_run=None,
    ) -> None:
        self.N = int(N)
        self.buffer = buffer
        self.snapshots_root = Path(snapshots_root)
        self.snapshots_root.mkdir(parents=True, exist_ok=True)
        self._trainer_ref = trainer_ref
        self.wandb_run = wandb_run

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step == 0 or step % self.N != 0:
            return
        self._snapshot(step)

    # ------------------------------------------------------------------

    def _snapshot(self, step: int) -> None:
        trainer = self._trainer_ref
        if trainer is None:
            raise SnapshotReflectionError(
                f"SnapshotCallback fired at step {step} with dead trainer ref"
            )

        # 1. Save adapter atomically
        adapter_dir = self.snapshots_root / f"adapter_step{step}"
        save_adapter_atomically(trainer.model, adapter_dir)

        # 2. Compute pending state (no mutation)
        lora_int_id = self.buffer.next_unused_lora_int_id()
        new_state, evicted = self.buffer.dry_run_add(
            snapshot_id=f"snap_{step}",
            adapter_path=str(adapter_dir),
            step=step,
            lora_int_id=lora_int_id,
        )

        # 3. Persist new state BEFORE touching vLLM
        previous_state = self.buffer.current_state
        self.buffer.persist_atomic(new_state)

        # 4. Reflect to vLLM — ADD first, then REMOVE
        try:
            self._vllm_add(trainer, new_state.entries[-1] if not evicted
                           else next(e for e in new_state.entries
                                     if e.adapter_path == str(adapter_dir)))
            if evicted is not None:
                self._vllm_remove(trainer, evicted.lora_int_id)
        except Exception as e:
            # Roll JSON back to previous state to maintain the invariant.
            self.buffer.persist_atomic(previous_state)
            raise SnapshotReflectionError(
                f"vLLM reflection failed @ step {step}: {e}. JSON rolled back."
            ) from e

        # 5. Commit in-memory
        self.buffer.commit(new_state)

        # 6. Log
        if self.wandb_run is not None:
            try:
                self.wandb_run.log(
                    {
                        "snapshot/step": step,
                        "snapshot/buffer_size": len(new_state.entries),
                        "snapshot/lora_int_id": lora_int_id,
                        "snapshot/evicted_id": (
                            evicted.snapshot_id if evicted else None
                        ),
                    },
                    step=step,
                )
            except Exception:
                pass

    # --- vLLM ops ----------------------------------------------------------

    def _vllm_add(self, trainer, entry) -> None:
        """Register a snapshot adapter in the vLLM colocate engine."""
        from vllm.lora.request import LoRARequest  # imported lazily to enable CPU tests
        engine = getattr(trainer, "vllm_engine", None) or getattr(trainer, "llm", None)
        if engine is None:
            raise SnapshotReflectionError("Trainer has no vLLM engine attribute")
        request = LoRARequest(entry.snapshot_id, entry.lora_int_id, entry.adapter_path)
        # vLLM 0.10.2+ exposes `add_lora`; some versions use `engine.add_lora_request`.
        if hasattr(engine, "add_lora"):
            engine.add_lora(request)
        elif hasattr(engine, "add_lora_request"):
            engine.add_lora_request(request)
        else:
            raise SnapshotReflectionError(
                "vLLM engine has neither add_lora nor add_lora_request"
            )

    def _vllm_remove(self, trainer, lora_int_id: int) -> None:
        engine = getattr(trainer, "vllm_engine", None) or getattr(trainer, "llm", None)
        if engine is None:
            raise SnapshotReflectionError("Trainer has no vLLM engine attribute")
        if hasattr(engine, "remove_lora"):
            engine.remove_lora(lora_int_id)
        elif hasattr(engine, "remove_lora_request"):
            engine.remove_lora_request(lora_int_id)
        else:
            raise SnapshotReflectionError(
                "vLLM engine has neither remove_lora nor remove_lora_request"
            )


__all__ = ["SnapshotCallback", "SnapshotReflectionError"]
