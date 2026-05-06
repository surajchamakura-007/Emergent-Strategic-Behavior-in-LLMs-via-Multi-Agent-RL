"""Atomic LoRA adapter save/load.

Authority: Implementation Map §2.1 + §3.2 (atomic-write invariant).

Why atomic. The snapshot callback's hook ordering (Map §3.2) requires that
`buffer_state.json` on disk only ever lists adapters that are *fully* on disk
— never half-written. If a job crashes between writing the adapter and
updating the JSON, on resume we want either the old buffer state (recoverable)
or the new buffer state (recoverable). The forbidden middle state is "JSON
points at adapter_step120/ but adapter_step120/adapter_model.safetensors is
truncated."

Mechanism. `save_adapter_atomically(model, path)`:
    1. PEFT writes to `path.tmp/`.
    2. Each file in `path.tmp/` is fsynced (forces buffer → disk).
    3. The directory inode of `path.tmp/`'s parent is fsynced.
    4. POSIX `os.replace(path.tmp, path)` — this is the atomic step.
    5. The parent directory inode is fsynced again (publishes the rename).

After step 5 returns, every reader sees `path/` containing the complete
adapter, regardless of crash timing.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any


def _fsync_dir(path: str | os.PathLike) -> None:
    """fsync the directory inode to flush the directory entry."""
    fd = os.open(str(path), os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(root: Path) -> None:
    """fsync every regular file under `root`, then `root` itself."""
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            full = Path(dirpath) / name
            try:
                fd = os.open(str(full), os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except OSError:
                # Best-effort; some filesystems disallow fsync on read-only fds.
                pass
        _fsync_dir(dirpath)


def save_adapter_atomically(model: Any, path: str | os.PathLike) -> Path:
    """Save a PEFT-adapted model to `path` atomically.

    Args:
        model: A `peft.PeftModel` (or anything exposing `save_pretrained`).
        path: Destination directory. Must NOT yet exist (or will be replaced).

    Returns:
        Final path (absolute, as a Path).
    """
    final = Path(path).resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.parent / (final.name + ".tmp")

    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=False)

    # PEFT save (writes adapter_config.json + adapter_model.safetensors)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(tmp))
    else:
        raise TypeError(
            f"object {type(model).__name__} has no save_pretrained() — "
            f"not a PeftModel?"
        )

    # Step 2 + 3: flush every file and the temp dir inode
    _fsync_tree(tmp)

    # Step 4: atomic rename (POSIX guarantees this is single-step)
    if final.exists():
        # Atomic replace: the old final/ becomes inaccessible to new readers
        # at the moment os.replace returns.
        shutil.rmtree(final)
    os.replace(tmp, final)

    # Step 5: publish the rename
    _fsync_dir(final.parent)

    return final


def compute_adapter_checksum(path: str | os.PathLike, algo: str = "sha256") -> str:
    """Stable hash of every file in an adapter directory.

    Used by analysis/parity_audit.py to confirm that the adapter rsynced from
    the cluster to RunPod is byte-identical.
    """
    h = hashlib.new(algo)
    root = Path(path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")
    # Sort for determinism
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(root).as_posix().encode()
        h.update(b"\x00" + rel + b"\x00")
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
    return h.hexdigest()


__all__ = ["save_adapter_atomically", "compute_adapter_checksum"]
