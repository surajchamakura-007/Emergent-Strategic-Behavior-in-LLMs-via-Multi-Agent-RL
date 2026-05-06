"""Preflight 10 — Untrained Qwen trace baseline (PRD v6.1 §8.1, blocking dep).

Rationale
---------
Every faithfulness comparison (RFEval, REMUL probes) requires a "before
training" reference point. This preflight runs the untrained Qwen base model
through the IPD evaluation harness against the four canonical fixed
opponents (TfT, AlwaysCooperate, AlwaysDefect, Random50) and writes both:
    - eval results in the trained-model schema (per-opponent cooperation
      rate, avg reward, episode length distribution)
    - reasoning traces for downstream REMUL/RFEval analysis

Test plan
---------
    1. Boot a vLLM engine at the base model (no LoRA).
    2. For each fixed opponent in {TfT, AC, AD, R50}:
        - Run 20 episodes (matches Tier A spec).
        - Record per-round (action, reasoning, opp_action, payoff).
    3. Aggregate to the standard eval schema and write JSON.

Output
------
    logs/preflight/10_untrained_baseline.json plus a trace dump at
    logs/preflight/10_untrained_traces.jsonl (one episode per line).
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import Config  # noqa: E402
from configs.stack_versions import assert_versions  # noqa: E402
from utils.seed import seed_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Used to source model_path and game settings only.")
    parser.add_argument("--episodes-per-opponent", type=int, default=20)
    parser.add_argument("--out",
                        default="logs/preflight/10_untrained_baseline.json")
    parser.add_argument("--traces-out",
                        default="logs/preflight/10_untrained_traces.jsonl")
    args = parser.parse_args()

    out = {
        "passed": False,
        "model_path": None,
        "per_opponent": {},
        "errors": [],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    try:
        assert_versions()
        cfg = Config.from_yaml(args.config)
        seed_all(cfg.seed)
        out["model_path"] = cfg.model_path

        from evaluation.eval import run_match_vs_fixed_opponent  # noqa: E402
        from evaluation.opponents import (  # noqa: E402
            AlwaysCooperate,
            AlwaysDefect,
            Random50,
            TitForTat,
        )

        opponents = {
            "TfT": TitForTat(),
            "AlwaysCooperate": AlwaysCooperate(),
            "AlwaysDefect": AlwaysDefect(),
            "Random50": Random50(seed=0),
        }

        traces_handle = open(args.traces_out, "w")
        try:
            for name, opp in opponents.items():
                result = run_match_vs_fixed_opponent(
                    cfg=cfg,
                    adapter_path=None,  # untrained: load base model only
                    opponent=opp,
                    n_episodes=args.episodes_per_opponent,
                    write_traces_to=traces_handle,
                )
                out["per_opponent"][name] = {
                    "coop_rate": result.coop_rate,
                    "avg_reward_per_round": result.avg_reward_per_round,
                    "n_rounds": result.total_rounds,
                    "episode_length_mean": result.episode_length_mean,
                    "format_violation_rate": result.format_violation_rate,
                }
        finally:
            traces_handle.close()

        # Pass = at least all 4 opponents had episodes complete cleanly.
        out["passed"] = len(out["per_opponent"]) == 4
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"{type(e).__name__}: {e}")
        out["errors"].append(traceback.format_exc())
    finally:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps({"passed": out["passed"],
                          "n_opponents": len(out["per_opponent"])}, indent=2))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
