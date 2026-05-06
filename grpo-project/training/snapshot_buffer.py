"""Ring buffer of frozen LoRA snapshots, with atomic JSON persistence.

Authority: Implementation Map §2.4 (full public surface) + §3.2 (atomic
hook ordering invariant) + PRD v6.1 §7.1 (sampling rule + halt-on-missing).

Invariant (Map §3.2). `buffer_state.json` on disk is the source of truth
for what vLLM should be holding at any quiescent point. Anything in vLLM
not listed in `buffer_state.json` is recoverable garbage; anything listed
but not in vLLM is a bug.

Hook design (Map §3.2 SnapshotCallback.on_step_end). Mutation goes through
a strict order:
    1. Save adapter atomically (utils/lora_io.save_adapter_atomically).
    2. Compute pending state via dry_run_add — DOES NOT MUTATE.
    3. Persist pending state to JSON via persist_atomic.
    4. Reflect to vLLM (ADD first, then REMOVE).
    5. Commit in-memory.

This ordering means a crash anywhere is recoverable: either the JSON has
the old state (vLLM rebuilt from JSON on resume), or it has the new state
and the new adapter is on disk. The forbidden middle state — JSON updated,
adapter file missing — cannot occur because step 1 finishes before step 3.

Empty-buffer rule (PRD v6.1 §7.1). Steps 1–39 (|B|=0): opponent = current
policy w.p. 1.0. Step ≥ 40 (|B|≥1): 0.5 current / 0.5 uniform-from-buffer.
`sample_opponent` returns None for the "current policy" choice so the
trainer's dispatch code can branch on identity.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BufferStateMissingError(RuntimeError):
    """Raised on resume when buffer_state.json is absent from the ckpt dir.

    Per PRD v6.1 §7.1 — silent revert to shared-weights self-play after restart
    is a methodology-drift bug. We halt loudly instead.
    """


class BufferIntegrityError(RuntimeError):
    """Raised on resume when a listed adapter_path does not exist on disk."""


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SnapshotEntry:
    """One adapter living in the ring buffer."""
    snapshot_id: str        # e.g. "snap_40"
    adapter_path: str       # absolute path on disk
    step_at_save: int
    lora_int_id: int        # vLLM-side numeric ID; 1 reserved for trainable

    def to_json(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "adapter_path": self.adapter_path,
            "step_at_save": self.step_at_save,
            "lora_int_id": self.lora_int_id,
        }

    @classmethod
    def from_json(cls, d: dict) -> "SnapshotEntry":
        return cls(
            snapshot_id=str(d["snapshot_id"]),
            adapter_path=str(d["adapter_path"]),
            step_at_save=int(d["step_at_save"]),
            lora_int_id=int(d["lora_int_id"]),
        )


@dataclass(frozen=True)
class BufferState:
    """Immutable snapshot of the buffer at a point in time."""
    entries: tuple[SnapshotEntry, ...]
    eviction_pointer: int   # FIFO write index, mod capacity
    step_at_save: int
    capacity: int

    def to_json(self) -> dict[str, Any]:
        return {
            "snapshot_paths": [e.adapter_path for e in self.entries],
            "entries": [e.to_json() for e in self.entries],
            "eviction_pointer": self.eviction_pointer,
            "step_at_save": self.step_at_save,
            "buffer_capacity": self.capacity,
        }

    @classmethod
    def from_json(cls, d: dict) -> "BufferState":
        # Backward-compat: prefer the structured 'entries' field; fall back to
        # 'snapshot_paths' if older format is encountered.
        if "entries" in d:
            entries = tuple(SnapshotEntry.from_json(e) for e in d["entries"])
        else:
            entries = tuple(
                SnapshotEntry(
                    snapshot_id=Path(p).name,
                    adapter_path=p,
                    step_at_save=0,
                    lora_int_id=2 + i,  # arbitrary; trainable=1
                )
                for i, p in enumerate(d.get("snapshot_paths", []))
            )
        return cls(
            entries=entries,
            eviction_pointer=int(d.get("eviction_pointer", 0)),
            step_at_save=int(d.get("step_at_save", 0)),
            capacity=int(d.get("buffer_capacity", 8)),
        )


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------

class SnapshotBuffer:
    """FIFO ring buffer of frozen adapters with atomic JSON persistence.

    Construction does NOT touch disk. Callers must explicitly call
    `load_or_halt()` on resume, or just start using `dry_run_add → commit`
    on a fresh run.
    """

    _DEFAULT_LORA_INT_ID_START = 2  # int_id 1 reserved for trainable

    def __init__(self, capacity: int, persist_path: str | os.PathLike) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self.persist_path = Path(persist_path).resolve()
        # Empty initial state.
        self._state: BufferState = BufferState(
            entries=(),
            eviction_pointer=0,
            step_at_save=0,
            capacity=capacity,
        )

    # ------------------ accessors ------------------

    @property
    def current_state(self) -> BufferState:
        return self._state

    def __len__(self) -> int:
        return len(self._state.entries)

    @property
    def diversity_indicator(self) -> float:
        """Convenience: 0.0 if empty, expected ~0.5 once populated.

        Used by the diagnostic logger for the `opponent_diversity` floor
        check before the rollout-level metric is available.
        """
        return 0.0 if len(self._state.entries) == 0 else 0.5

    def used_lora_int_ids(self) -> set[int]:
        return {e.lora_int_id for e in self._state.entries}

    def next_unused_lora_int_id(self, reserved: set[int] | None = None) -> int:
        """Return the smallest int_id not in current entries or in `reserved`.

        Trainable adapter (int_id=1) is implicitly reserved.
        """
        used = self.used_lora_int_ids() | {1} | (reserved or set())
        i = self._DEFAULT_LORA_INT_ID_START
        while i in used:
            i += 1
        return i

    # ------------------ mutation: dry_run + commit pattern ------------------

    def dry_run_add(
        self,
        snapshot_id: str,
        adapter_path: str | os.PathLike,
        step: int,
        lora_int_id: int,
    ) -> tuple[BufferState, SnapshotEntry | None]:
        """Compute the new state IF this entry were added. Does NOT mutate.

        Returns:
            (new_state, evicted_entry)
            evicted_entry is None when the buffer was below capacity.
        """
        new_entry = SnapshotEntry(
            snapshot_id=snapshot_id,
            adapter_path=str(Path(adapter_path).resolve()),
            step_at_save=step,
            lora_int_id=lora_int_id,
        )
        entries = list(self._state.entries)
        evicted: SnapshotEntry | None = None

        if len(entries) < self.capacity:
            entries.append(new_entry)
            new_pointer = (self._state.eviction_pointer + 1) % self.capacity
        else:
            # FIFO eviction at the eviction_pointer position
            ptr = self._state.eviction_pointer
            evicted = entries[ptr]
            entries[ptr] = new_entry
            new_pointer = (ptr + 1) % self.capacity

        new_state = BufferState(
            entries=tuple(entries),
            eviction_pointer=new_pointer,
            step_at_save=step,
            capacity=self.capacity,
        )
        return new_state, evicted

    def commit(self, new_state: BufferState) -> None:
        """Replace in-memory state. Caller is responsible for vLLM reflection
        BEFORE calling commit (Map §3.2 step 4 precedes step 5)."""
        if new_state.capacity != self.capacity:
            raise ValueError(
                f"new_state.capacity {new_state.capacity} != "
                f"buffer.capacity {self.capacity}"
            )
        self._state = new_state

    # ------------------ persistence ------------------

    def persist_atomic(self, state: BufferState) -> None:
        """Write `state` to `self.persist_path` atomically (POSIX rename).

        Per Map §3.2: write tmp file, fsync the file, fsync the parent dir,
        atomic-rename, fsync the parent dir again.
        """
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.persist_path.with_suffix(self.persist_path.suffix + ".tmp")

        with open(tmp, "w") as f:
            json.dump(state.to_json(), f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, self.persist_path)

        # fsync the parent directory inode to publish the rename.
        dir_fd = os.open(str(self.persist_path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def load_or_halt(self) -> BufferState:
        """On resume: read persist_path, validate, return state.

        Raises:
            BufferStateMissingError: file absent.
            BufferIntegrityError:    a listed adapter_path doesn't exist on disk.
        """
        if not self.persist_path.exists():
            raise BufferStateMissingError(
                f"{self.persist_path} not found. PRD v6.1 §7.1 requires "
                f"halt-on-missing to prevent silent revert to shared-weights "
                f"self-play. If this is intentional (fresh run), do not call "
                f"load_or_halt()."
            )
        with open(self.persist_path) as f:
            d = json.load(f)
        state = BufferState.from_json(d)

        for entry in state.entries:
            if not Path(entry.adapter_path).is_dir():
                raise BufferIntegrityError(
                    f"Adapter directory missing on resume: {entry.adapter_path}\n"
                    f"buffer_state.json lists {len(state.entries)} entries; "
                    f"this one is gone. Fix or delete buffer_state.json to start fresh."
                )

        if state.capacity != self.capacity:
            raise BufferIntegrityError(
                f"Persisted buffer capacity {state.capacity} != "
                f"current configured capacity {self.capacity}. "
                f"Capacity mismatch on resume — refuse."
            )

        self._state = state
        return state

    # ------------------ sampling ------------------

    def sample_opponent(
        self,
        rng: np.random.Generator,
        p_buffer: float = 0.5,
    ) -> SnapshotEntry | None:
        """Sample an opponent per the PRD §7.1 rule.

        Args:
            rng: numpy RNG (deterministic given the run seed).
            p_buffer: P(opponent ∈ buffer | buffer non-empty). PRD default 0.5.

        Returns:
            None if the rollout should use the trainable (current) policy.
            SnapshotEntry if the rollout should use a frozen snapshot.
        """
        if len(self._state.entries) == 0:
            return None  # warmup: always current policy
        if rng.random() < p_buffer:
            idx = int(rng.integers(0, len(self._state.entries)))
            return self._state.entries[idx]
        return None


__all__ = [
    "SnapshotBuffer",
    "SnapshotEntry",
    "BufferState",
    "BufferStateMissingError",
    "BufferIntegrityError",
]
