"""SnapshotBuffer unit tests.

Authority: Implementation Map §2.4 (public surface) + §3.2 (atomic write).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from training.snapshot_buffer import (
    BufferIntegrityError,
    BufferState,
    BufferStateMissingError,
    SnapshotBuffer,
    SnapshotEntry,
)


def _mk_adapter_dir(tmp_path: Path, name: str) -> str:
    p = tmp_path / name
    p.mkdir(parents=True, exist_ok=True)
    (p / "adapter_config.json").write_text("{}")
    return str(p)


# ---------------------------------------------------------------------------
# Basic accessors
# ---------------------------------------------------------------------------

def test_empty_buffer_basics(tmp_path):
    b = SnapshotBuffer(capacity=8, persist_path=tmp_path / "buf.json")
    assert len(b) == 0
    assert b.diversity_indicator == 0.0
    assert b.next_unused_lora_int_id() == 2  # 1 reserved for trainable


def test_dry_run_does_not_mutate(tmp_path):
    b = SnapshotBuffer(capacity=8, persist_path=tmp_path / "buf.json")
    p = _mk_adapter_dir(tmp_path, "snap_40")
    b.dry_run_add("snap_40", p, step=40, lora_int_id=2)
    assert len(b) == 0  # not committed


def test_commit_after_dry_run(tmp_path):
    b = SnapshotBuffer(capacity=8, persist_path=tmp_path / "buf.json")
    p = _mk_adapter_dir(tmp_path, "snap_40")
    new_state, evicted = b.dry_run_add("snap_40", p, step=40, lora_int_id=2)
    assert evicted is None
    b.commit(new_state)
    assert len(b) == 1


# ---------------------------------------------------------------------------
# FIFO eviction
# ---------------------------------------------------------------------------

def test_fifo_eviction(tmp_path):
    b = SnapshotBuffer(capacity=3, persist_path=tmp_path / "buf.json")
    paths = [_mk_adapter_dir(tmp_path, f"snap_{i}") for i in range(40, 200, 40)]
    int_ids = [2, 3, 4, 5]
    # Fill to capacity
    for i in range(3):
        new_state, evicted = b.dry_run_add(f"snap_{(i+1)*40}", paths[i], step=(i+1)*40, lora_int_id=int_ids[i])
        assert evicted is None
        b.commit(new_state)
    # Next add evicts the oldest
    new_state, evicted = b.dry_run_add("snap_160", paths[3], step=160, lora_int_id=int_ids[3])
    assert evicted is not None
    assert evicted.snapshot_id == "snap_40"
    b.commit(new_state)
    assert len(b) == 3
    assert "snap_40" not in {e.snapshot_id for e in new_state.entries}
    assert "snap_160" in {e.snapshot_id for e in new_state.entries}


# ---------------------------------------------------------------------------
# next_unused_lora_int_id
# ---------------------------------------------------------------------------

def test_next_unused_int_id_skips_used(tmp_path):
    b = SnapshotBuffer(capacity=8, persist_path=tmp_path / "buf.json")
    p1 = _mk_adapter_dir(tmp_path, "s1")
    new_state, _ = b.dry_run_add("s1", p1, step=40, lora_int_id=2)
    b.commit(new_state)
    assert b.next_unused_lora_int_id() == 3  # 1 reserved, 2 used


# ---------------------------------------------------------------------------
# Sampling rule (PRD v6.1 §7.1)
# ---------------------------------------------------------------------------

def test_sample_opponent_empty_returns_none(tmp_path):
    b = SnapshotBuffer(capacity=8, persist_path=tmp_path / "buf.json")
    rng = np.random.default_rng(0)
    for _ in range(100):
        assert b.sample_opponent(rng) is None


def test_sample_opponent_nonempty_50_50(tmp_path):
    b = SnapshotBuffer(capacity=8, persist_path=tmp_path / "buf.json")
    p = _mk_adapter_dir(tmp_path, "s1")
    new_state, _ = b.dry_run_add("s1", p, step=40, lora_int_id=2)
    b.commit(new_state)

    rng = np.random.default_rng(0)
    n_buf = 0
    n_cur = 0
    N = 5000
    for _ in range(N):
        if b.sample_opponent(rng, p_buffer=0.5) is None:
            n_cur += 1
        else:
            n_buf += 1
    # Should be near 50/50, allow ~3% slop at N=5000
    assert abs(n_cur / N - 0.5) < 0.03
    assert abs(n_buf / N - 0.5) < 0.03


# ---------------------------------------------------------------------------
# Atomic persistence + load_or_halt (Map §3.2)
# ---------------------------------------------------------------------------

def test_persist_and_load_round_trip(tmp_path):
    persist = tmp_path / "buf.json"
    b = SnapshotBuffer(capacity=8, persist_path=persist)
    p1 = _mk_adapter_dir(tmp_path, "s1")
    p2 = _mk_adapter_dir(tmp_path, "s2")
    new_state, _ = b.dry_run_add("s1", p1, step=40, lora_int_id=2)
    b.persist_atomic(new_state)
    b.commit(new_state)
    new_state, _ = b.dry_run_add("s2", p2, step=80, lora_int_id=3)
    b.persist_atomic(new_state)
    b.commit(new_state)

    # Now reload from disk
    b2 = SnapshotBuffer(capacity=8, persist_path=persist)
    state = b2.load_or_halt()
    assert len(state.entries) == 2
    assert {e.snapshot_id for e in state.entries} == {"s1", "s2"}
    assert state.eviction_pointer in (0, 1, 2)


def test_load_missing_raises(tmp_path):
    b = SnapshotBuffer(capacity=8, persist_path=tmp_path / "missing.json")
    with pytest.raises(BufferStateMissingError):
        b.load_or_halt()


def test_load_missing_adapter_dir_raises(tmp_path):
    persist = tmp_path / "buf.json"
    b = SnapshotBuffer(capacity=8, persist_path=persist)
    p1 = _mk_adapter_dir(tmp_path, "s1")
    new_state, _ = b.dry_run_add("s1", p1, step=40, lora_int_id=2)
    b.persist_atomic(new_state)
    b.commit(new_state)

    # Now delete the adapter dir on disk
    import shutil
    shutil.rmtree(p1)

    b2 = SnapshotBuffer(capacity=8, persist_path=persist)
    with pytest.raises(BufferIntegrityError):
        b2.load_or_halt()


def test_persist_writes_valid_json(tmp_path):
    persist = tmp_path / "buf.json"
    b = SnapshotBuffer(capacity=8, persist_path=persist)
    p1 = _mk_adapter_dir(tmp_path, "s1")
    new_state, _ = b.dry_run_add("s1", p1, step=40, lora_int_id=2)
    b.persist_atomic(new_state)

    raw = json.loads(persist.read_text())
    assert "snapshot_paths" in raw
    assert "entries" in raw
    assert raw["buffer_capacity"] == 8
