"""Cluster ↔ RunPod numerical parity verdict (R11, S1-11).

Authority: STAGE1_EXECUTION_PLAN_v3.1 §3.2.

Workflow
--------
    1. Run `preflight/09_parity_audit.py` on the cluster → produces
       `logs/preflight/09_parity_<v100_label>.json`.
    2. Run the same on RunPod → `09_parity_<4090_label>.json`.
    3. Run THIS script with both JSONs as input to compute per-step
       relative differences in reward, advantage_mean_abs, group_reward_std.
    4. Halt fan-out if any divergence > 5%.

Output
------
    JSON record with per-step diffs and a single `verdict` field:
        "pass" — all diffs within 5%
        "fail" — at least one metric exceeded threshold

The threshold can be relaxed via --tol; default 0.05 = 5% per the plan.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


METRICS = ("reward", "advantage_mean_abs", "group_reward_std")


def _index_by_step(records: list[dict]) -> dict[int, dict]:
    return {int(r["step"]): r for r in records if "step" in r}


def _rel_diff(a: float, b: float) -> float:
    """Symmetric relative difference: 2|a-b| / (|a|+|b|+eps)."""
    denom = abs(a) + abs(b)
    if denom < 1e-9:
        return 0.0
    return 2.0 * abs(a - b) / denom


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", required=True,
                        help="Path to preflight 09 JSON from the cluster (V100).")
    parser.add_argument("--runpod", required=True,
                        help="Path to preflight 09 JSON from RunPod (4090).")
    parser.add_argument("--tol", type=float, default=0.05,
                        help="Per-metric per-step relative-diff tolerance.")
    parser.add_argument("--out", default="logs/preflight/parity_verdict.json")
    args = parser.parse_args()

    cluster = json.loads(Path(args.cluster).read_text())
    runpod = json.loads(Path(args.runpod).read_text())

    c_recs = _index_by_step(cluster.get("step_records", []))
    r_recs = _index_by_step(runpod.get("step_records", []))
    common_steps = sorted(set(c_recs) & set(r_recs))

    diffs = []
    breaches: list[dict] = []
    for s in common_steps:
        row = {"step": s}
        for m in METRICS:
            if m in c_recs[s] and m in r_recs[s]:
                d = _rel_diff(c_recs[s][m], r_recs[s][m])
                row[f"{m}_rel_diff"] = d
                row[f"{m}_cluster"] = c_recs[s][m]
                row[f"{m}_runpod"] = r_recs[s][m]
                if d > args.tol:
                    breaches.append({"step": s, "metric": m,
                                     "rel_diff": d,
                                     "cluster": c_recs[s][m],
                                     "runpod": r_recs[s][m]})
        diffs.append(row)

    verdict = "pass" if not breaches else "fail"
    record = {
        "verdict": verdict,
        "tolerance": args.tol,
        "n_common_steps": len(common_steps),
        "cluster_platform": cluster.get("platform_label"),
        "runpod_platform": runpod.get("platform_label"),
        "cluster_torch": cluster.get("torch_version"),
        "runpod_torch": runpod.get("torch_version"),
        "diffs": diffs,
        "breaches": breaches,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)

    print(json.dumps({
        "verdict": verdict,
        "n_common_steps": len(common_steps),
        "n_breaches": len(breaches),
        "out": args.out,
    }, indent=2))
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
