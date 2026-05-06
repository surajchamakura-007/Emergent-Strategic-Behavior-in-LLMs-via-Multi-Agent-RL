"""Trace evaluation — dump per-round reasoning + decision for REMUL/RFEval.

Authority: PRD v6.1 §8.2 + §8.3 (RFEval extension).

Differences from `evaluation/eval.py`:
    - No aggregation: this writes a JSONL where each line is one round,
      with the FULL completion text (reasoning + action) so downstream
      tools can probe legibility, backtracking, and counterfactual χ/κ.
    - Designed to be re-run on a fresh adapter without re-running the full
      Tier A eval (cheaper).
    - Always also runs the untrained-Qwen baseline trace for the same
      opponents — RFEval comparisons require trained-vs-untrained on the
      same prompt distribution.

Output schema (one JSONL line per round)
----------------------------------------
    {
      "run_name": str,
      "is_untrained_baseline": bool,
      "opponent": str,
      "episode": int,
      "round": int,
      "prompt": str,             # what the model saw
      "completion": str,         # full <reasoning>...</reasoning><action>X</action>
      "parsed_action": "C" | "D" | null,
      "opp_action": "C" | "D",
      "agent_payoff": float,
      "format_ok": bool,
      "T": float,
      "seed": int
    }

Re-runs are idempotent on the trace file path, so calling this on a
specific (adapter, T) combo overwrites the previous trace.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import Config  # noqa: E402
from evaluation.eval import run_match_vs_fixed_opponent  # noqa: E402
from evaluation.opponents import (  # noqa: E402
    AlwaysCooperate,
    AlwaysDefect,
    Opponent,
    Random50,
    TitForTat,
)


def collect_traces(
    cfg: Config,
    *,
    adapter_path: Optional[str],
    n_episodes: int,
    out_path: Path,
    seed: int = 0,
) -> int:
    """Run all 4 fixed-opponent matchups and write traces. Returns total rounds."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    opponents: dict[str, Opponent] = {
        "TfT": TitForTat(),
        "AlwaysCooperate": AlwaysCooperate(),
        "AlwaysDefect": AlwaysDefect(),
        "Random50": Random50(seed=seed),
    }
    rounds_total = 0
    with open(out_path, "w") as f:
        for name, opp in opponents.items():
            res = run_match_vs_fixed_opponent(
                cfg=cfg,
                adapter_path=adapter_path,
                opponent=opp,
                n_episodes=n_episodes,
                seed=seed,
                write_traces_to=f,
            )
            rounds_total += res.total_rounds
    return rounds_total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None,
                        help="LoRA adapter dir; omit for untrained baseline.")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Episodes per opponent. Default 10 — cheaper than Tier A.")
    parser.add_argument("--out", default=None,
                        help="Output JSONL path. Auto-named if omitted.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--also-untrained", action="store_true",
                        help="Also run untrained baseline trace alongside.")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    out = Path(args.out) if args.out else (
        Path("logs/eval") / cfg.run_name
        / ("traces.jsonl" if args.adapter else "untrained_traces.jsonl")
    )
    rounds = collect_traces(
        cfg=cfg, adapter_path=args.adapter,
        n_episodes=args.episodes, out_path=out, seed=args.seed,
    )
    print(json.dumps({"out": str(out), "rounds": rounds}, indent=2))

    if args.also_untrained and args.adapter is not None:
        out_unt = Path("logs/eval") / cfg.run_name / "untrained_traces.jsonl"
        rounds_unt = collect_traces(
            cfg=cfg, adapter_path=None,
            n_episodes=args.episodes, out_path=out_unt, seed=args.seed,
        )
        print(json.dumps({"untrained_out": str(out_unt),
                          "rounds": rounds_unt}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
