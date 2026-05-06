"""Tier A evaluation — fixed-opponent matchups on IPD.

Authority: PRD v6.1 §8.2 + STAGE1_EXECUTION_PLAN_v3.1 §6.

Per the execution plan §6.1, Tier A produces these per-(adapter, opponent)
metrics on IPD:
    - cooperation rate (fraction of agent rounds with action == 'C')
    - average reward per round (the agent's reward; the metric reviewers care about)
    - episode-length distribution (mean, std, truncation rate)
    - format-violation rate
    - per-round trace (optional; written to a JSONL stream for REMUL/RFEval)

Run-time invariant: this script is the SAME code that produces both
`logs/preflight/10_untrained_baseline.json` (untrained Qwen, no adapter)
and `logs/eval/<run>/eval_results.json` (post-training, adapter loaded).
The only difference is whether `adapter_path` is None.

Vs. the prior 250-step `eval.py`: this version
    - uses vLLM colocate for generation (matches training stack);
    - parses completions via the same FORMAT_RE as `training/reward.py`;
    - serializes history via `envs/history.serialize_history` (Map §4
      truncation policy) — eval and training see identical prompts.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import Config  # noqa: E402
from envs.history import Round  # noqa: E402
from envs.prisoners_dilemma import PrisonersDilemmaEnv  # noqa: E402
from evaluation.opponents import (  # noqa: E402
    AlwaysCooperate,
    AlwaysDefect,
    Opponent,
    Random50,
    TitForTat,
)
from training.prompt_builder import build_prompt  # noqa: E402
from training.reward import parse_completion  # noqa: E402


# ---------------------------------------------------------------------------
# Output records
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """Aggregated metrics for one (adapter, opponent) match."""
    opponent_name: str
    adapter_path: Optional[str]
    n_episodes: int
    total_rounds: int
    coop_rate: float
    avg_reward_per_round: float
    avg_reward_per_episode: float
    episode_length_mean: float
    episode_length_std: float
    truncation_rate: float
    format_violation_rate: float
    seed: int
    elapsed_s: float


# ---------------------------------------------------------------------------
# vLLM helper
# ---------------------------------------------------------------------------

class _GenerationBackend:
    """Thin wrapper around vLLM colocate for eval-only generation.

    Holds a base-model engine and a list of registered adapters. `generate`
    routes through the requested adapter ID (or the base model if `lora_request`
    is None).

    This is deliberately simpler than the training trainer's engine: no
    rollout loop, no advantage scoring — just text in, text out.
    """

    def __init__(self, cfg: Config, adapter_path: Optional[str]) -> None:
        from vllm import LLM, SamplingParams  # noqa: F401

        self._cfg = cfg
        self._llm = LLM(
            model=cfg.model_path,
            dtype=cfg.vllm_dtype,
            gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
            max_lora_rank=cfg.lora_rank,
            enable_lora=adapter_path is not None,
            max_loras=2,
            max_model_len=cfg.max_seq_length,
        )
        self._adapter_path = adapter_path
        self._adapter_id = None

        if adapter_path is not None:
            from vllm.lora.request import LoRARequest
            self._adapter_id = 1
            self._lora_request = LoRARequest("eval_adapter", 1, adapter_path)
            if hasattr(self._llm, "add_lora"):
                self._llm.add_lora(self._lora_request)
            elif hasattr(self._llm, "add_lora_request"):
                self._llm.add_lora_request(self._lora_request)
        else:
            self._lora_request = None

    @property
    def tokenizer(self):
        return self._llm.get_tokenizer()

    def generate(self, prompt: str, *, temperature: float, max_tokens: int,
                 seed: int) -> str:
        from vllm import SamplingParams
        sp = SamplingParams(
            temperature=temperature,
            top_p=self._cfg.sampling_top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
        out = self._llm.generate([prompt], sampling_params=sp,
                                 lora_request=self._lora_request)
        return out[0].outputs[0].text


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------

def run_match_vs_fixed_opponent(
    cfg: Config,
    adapter_path: Optional[str],
    opponent: Opponent,
    n_episodes: int,
    *,
    seed: int = 0,
    write_traces_to: Optional[IO[str]] = None,
) -> MatchResult:
    """Play `n_episodes` against a fixed opponent; return aggregate metrics."""
    backend = _GenerationBackend(cfg, adapter_path=adapter_path)
    tokenizer = backend.tokenizer

    coop_count = 0
    total_rounds = 0
    total_reward = 0.0
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    truncation_count = 0
    format_violations = 0

    t0 = time.perf_counter()
    for ep in range(n_episodes):
        env_seed = seed * 10_000 + ep
        opponent.reset(seed=env_seed)
        env = PrisonersDilemmaEnv(
            payoffs=cfg.payoffs,
            p=cfg.geometric_p,
            cap=cfg.episode_cap,
            seed=env_seed,
        )
        history: list[Round] = []
        ep_reward = 0.0

        while True:
            prompt, _ = build_prompt(
                history=history,
                round_idx=len(history) + 1,
                payoffs=cfg.payoffs,
                p=cfg.geometric_p,
                tokenizer=tokenizer,
                history_token_budget=cfg.history_token_budget,
                keep_last_k=cfg.keep_last_k,
            )
            text = backend.generate(
                prompt,
                temperature=cfg.sampling_temp_default,
                max_tokens=cfg.max_completion_length,
                seed=env_seed * 31 + len(history),
            )
            parsed = parse_completion(text)
            if parsed is None:
                format_violations += 1
                # Format-violation: agent gets 0 reward this round, env still
                # advances with a placeholder action 'D' (worst case for agent).
                agent_action = "D"
                fmt_ok = False
            else:
                _, agent_action = parsed
                fmt_ok = True

            opp_action = opponent.act(history)
            step = env.step(agent_action, opp_action)

            # Reward: 0 on format violation, else env payoff.
            r = step.my_payoff if fmt_ok else 0.0
            ep_reward += r
            total_reward += r
            total_rounds += 1
            if agent_action == "C" and fmt_ok:
                coop_count += 1

            history.append(Round(
                my_action=agent_action,
                opp_action=opp_action,
                my_payoff=step.my_payoff,
            ))

            if write_traces_to is not None:
                write_traces_to.write(json.dumps({
                    "opponent": opponent.name,
                    "episode": ep,
                    "round": len(history),
                    "prompt": prompt,
                    "completion": text,
                    "agent_action": agent_action,
                    "opp_action": opp_action,
                    "agent_payoff": step.my_payoff,
                    "format_ok": fmt_ok,
                }) + "\n")

            if step.done:
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(len(history))
        if len(history) >= cfg.episode_cap:
            truncation_count += 1

    elapsed = time.perf_counter() - t0

    return MatchResult(
        opponent_name=opponent.name,
        adapter_path=adapter_path,
        n_episodes=n_episodes,
        total_rounds=total_rounds,
        coop_rate=coop_count / max(1, total_rounds),
        avg_reward_per_round=total_reward / max(1, total_rounds),
        avg_reward_per_episode=statistics.fmean(episode_rewards) if episode_rewards else 0.0,
        episode_length_mean=statistics.fmean(episode_lengths) if episode_lengths else 0.0,
        episode_length_std=statistics.pstdev(episode_lengths) if len(episode_lengths) > 1 else 0.0,
        truncation_rate=truncation_count / max(1, n_episodes),
        format_violation_rate=format_violations / max(1, total_rounds),
        seed=seed,
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_tier_a_eval(
    cfg: Config,
    adapter_path: Optional[str],
    *,
    n_episodes: int = 20,
    out_dir: str = "logs/eval",
    include_gpt4o: bool = False,
    seed: int = 0,
) -> dict:
    """Run all Tier A matchups and aggregate to one JSON dict."""
    out_dir_p = Path(out_dir) / cfg.run_name
    out_dir_p.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir_p / "traces.jsonl"
    results_path = out_dir_p / "eval_results.json"

    opponents: dict[str, Opponent] = {
        "AlwaysCooperate": AlwaysCooperate(),
        "AlwaysDefect": AlwaysDefect(),
        "TfT": TitForTat(),
        "Random50": Random50(seed=seed),
    }
    if include_gpt4o:
        from evaluation.opponents import GPT4oMiniOpponent
        opponents["GPT-4o-mini"] = GPT4oMiniOpponent()

    aggregate: dict[str, dict] = {}
    with open(traces_path, "w") as traces_f:
        for name, opp in opponents.items():
            res = run_match_vs_fixed_opponent(
                cfg=cfg,
                adapter_path=adapter_path,
                opponent=opp,
                n_episodes=n_episodes,
                seed=seed,
                write_traces_to=traces_f,
            )
            aggregate[name] = asdict(res)

    with open(results_path, "w") as f:
        json.dump({
            "run_name": cfg.run_name,
            "adapter_path": adapter_path,
            "n_episodes_per_match": n_episodes,
            "T": cfg.T,
            "seed": seed,
            "matchups": aggregate,
        }, f, indent=2)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None,
                        help="Path to LoRA adapter dir (omit for untrained Qwen).")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out-dir", default="logs/eval")
    parser.add_argument("--include-gpt4o", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    aggregate = run_tier_a_eval(
        cfg=cfg,
        adapter_path=args.adapter,
        n_episodes=args.episodes,
        out_dir=args.out_dir,
        include_gpt4o=args.include_gpt4o,
        seed=args.seed,
    )
    print(json.dumps({k: {"coop": v["coop_rate"],
                          "avg_r": v["avg_reward_per_round"]}
                      for k, v in aggregate.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
