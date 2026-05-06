"""Preflight 04 — Buffer + resume smoke (PRD v6.1 §7.1, S1-15 mitigation).

Rationale
---------
The frozen-snapshot ring buffer is checkpointed via `buffer_state.json`.
On resume we MUST reload the buffer exactly; if the file is missing or
corrupted, we halt loudly (not silently revert to shared-weights self-play
for ~40 steps and contaminate the run).

Test plan (PRD v6.1 §7.1 acceptance)
------------------------------------
    1. Run a smoke training to step 80 with N=40 (snapshots saved at 40, 80).
    2. Verify `buffer_state.json` exists and contains 2 entries with valid
       paths and checksums.
    3. Simulate a kill: tear down the trainer.
    4. Construct a fresh SnapshotBuffer + load_or_halt(); confirm it reloads
       both entries.
    5. Sample opponents from the loaded buffer 1000 times; coverage of both
       snapshots should be roughly uniform.
    6. Tamper with the JSON (delete one snapshot file from disk) and
       reconstruct the buffer; expect BufferIntegrityError.

Output
------
    JSON record at logs/preflight/04_buffer_resume_smoke.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import Config  # noqa: E402
from configs.stack_versions import assert_versions  # noqa: E402
from utils.seed import seed_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out",
                        default="logs/preflight/04_buffer_resume_smoke.json")
    args = parser.parse_args()

    out = {
        "passed": False,
        "buffer_state_exists": False,
        "buffer_entries_after_resume": 0,
        "sampled_uniform": False,
        "integrity_error_raised": False,
        "errors": [],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    try:
        assert_versions()
        cfg = Config.from_yaml(args.config)
        seed_all(cfg.seed)

        from training.train import build_smoke_trainer  # noqa: E402
        from training.snapshot_buffer import (  # noqa: E402
            SnapshotBuffer,
            BufferIntegrityError,
            BufferStateMissingError,
        )

        # Train to step 80 — snapshots fire at 40 and 80.
        trainer, _ = build_smoke_trainer(cfg, num_steps=80)
        trainer.train()
        del trainer

        state_path = cfg.buffer_state_path()
        out["buffer_state_exists"] = state_path.exists()
        if not out["buffer_state_exists"]:
            raise RuntimeError(f"buffer_state.json missing at {state_path}")

        # Fresh load.
        fresh = SnapshotBuffer(capacity=cfg.buffer_capacity,
                               persist_path=state_path)
        fresh.load_or_halt()
        out["buffer_entries_after_resume"] = len(fresh.state.entries)
        assert out["buffer_entries_after_resume"] == 2, \
            f"expected 2 entries, got {out['buffer_entries_after_resume']}"

        # Sample 1000 opponents.
        rng = random.Random(0)
        counts = {e.adapter_path: 0 for e in fresh.state.entries}
        snap_count = 0
        for _ in range(1000):
            opp = fresh.sample_opponent(rng, p_buffer=1.0)  # always-buffer
            if opp is not None:
                snap_count += 1
                counts[opp.adapter_path] = counts.get(opp.adapter_path, 0) + 1
        # Roughly 500/500 with tolerance.
        c1, c2 = sorted(counts.values())
        out["sampled_uniform"] = (c1 / max(1, c2)) > 0.7

        # Integrity check — corrupt by deleting one adapter dir on disk.
        first = fresh.state.entries[0]
        shutil.rmtree(first.adapter_path, ignore_errors=True)
        try:
            corrupted = SnapshotBuffer(capacity=cfg.buffer_capacity,
                                       persist_path=state_path)
            corrupted.load_or_halt()
        except BufferIntegrityError:
            out["integrity_error_raised"] = True
        except Exception as e:  # noqa: BLE001
            raise AssertionError(
                f"expected BufferIntegrityError, got {type(e).__name__}: {e}"
            )

        # Missing-file check.
        os.remove(state_path)
        try:
            ghost = SnapshotBuffer(capacity=cfg.buffer_capacity,
                                   persist_path=state_path)
            ghost.load_or_halt()
        except BufferStateMissingError:
            pass
        except Exception as e:  # noqa: BLE001
            raise AssertionError(
                f"expected BufferStateMissingError, got {type(e).__name__}: {e}"
            )

        out["passed"] = (
            out["buffer_state_exists"]
            and out["buffer_entries_after_resume"] == 2
            and out["sampled_uniform"]
            and out["integrity_error_raised"]
        )
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"{type(e).__name__}: {e}")
        out["errors"].append(traceback.format_exc())
    finally:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps({"passed": out["passed"]}, indent=2))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
