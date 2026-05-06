"""W&B experiment logging helpers.

Authority: PRD v6.1 §3.4 ("W&B mode = offline; sync from login node post-job").

Why offline by default. Bridges-2 compute nodes are firewalled (no outbound
HTTPS). Online mode would silently fail to log on the cluster. Offline mode
writes to `$WANDB_DIR/wandb/run-*/` and the user runs `wandb sync` from the
login node. See manual_wandb_sync.sh for the tmux loop.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import wandb

from configs.config import Config


def init_wandb_offline(cfg: Config, run_name_suffix: str = "") -> wandb.sdk.wandb_run.Run:
    """Initialize W&B in offline mode and return the run handle.

    Sets `$WANDB_DIR` to `cfg.log_dir() / "wandb"` so multiple runs on the same
    cluster filesystem don't collide.
    """
    wandb_dir = cfg.log_dir() / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_dir)

    run_name = cfg.run_name + (f"_{run_name_suffix}" if run_name_suffix else "")
    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=run_name,
        config=cfg.to_wandb_dict(),
        tags=list(cfg.wandb_tags) + [f"T{int(cfg.T)}", f"seed{cfg.seed}"],
        mode=cfg.wandb_mode,  # default "offline"
        dir=str(wandb_dir),
        resume="allow",
    )
    return run


def manual_sync_helper(cfg: Config) -> str:
    """Print and return the exact `wandb sync` command for the login node."""
    wandb_dir = cfg.log_dir() / "wandb"
    cmd = f"wandb sync --include-offline {wandb_dir}"
    print(f"[experiment_logger] To sync from login node:\n  {cmd}")
    return cmd


def log_static_run_config(run: Any, extras: dict[str, Any] | None = None) -> None:
    """Update wandb.config with extras (e.g., resolved threshold, stack versions).

    Called after threshold calibration is loaded — the threshold needs to be
    in the run config but isn't part of the static Config dataclass.
    """
    if extras:
        run.config.update(extras, allow_val_change=True)


__all__ = [
    "init_wandb_offline",
    "manual_sync_helper",
    "log_static_run_config",
]
