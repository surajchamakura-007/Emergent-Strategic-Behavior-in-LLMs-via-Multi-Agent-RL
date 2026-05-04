# Emergent Strategic Behavior in LLMs via Multi-Agent RL

Training [Qwen-2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) with **GRPO self-play** on canonical social dilemmas (Iterated Prisoner's Dilemma, Stag Hunt, Public Goods) and asking three questions:

1. Does payoff-only reinforcement learning push a small instruction-tuned model toward recognizable strategic behavior?
2. How does that behavior shift as the temptation payoff `T` is swept from 4 to 10?
3. Is the model's chain-of-thought actually causally driving its actions, or is it post-hoc decoration?

UC Berkeley Deep Learning & RL course project. Single RTX 4090. 8-week timeline.

> **Framing note.** After a 2025 prior-art audit (`docs/PRIOR_ART_AUDIT.md`), this is positioned as an **empirical replication-and-characterization study**, not a methodological first. The closest prior work is Piche et al., "Learning Robust Social Strategies with Large Language Models" (arXiv:2511.19405, Nov 2025). The novelty contribution is the RFEval-style reasoning-faithfulness extension applied to strategic settings.

---

## What's in here

| Path | What it is |
|---|---|
| `configs/config.py` | Central config — model paths, payoff matrices, LoRA settings, GRPO hyperparameters |
| `envs/prisoners_dilemma.py` | Iterated Prisoner's Dilemma environment (stochastic horizon, geometric continuation) |
| `envs/stag_hunt.py` | Stag Hunt — used for zero-shot transfer evaluation |
| `envs/public_goods.py` | N-player Public Goods (Phase 3 stretch) |
| `training/train.py` | GRPO self-play training loop, built on HuggingFace TRL |
| `training/reward.py` | Reward function — multiplicative format gate × payoff (no additive bonus) |
| `evaluation/eval.py` | Baseline tournament: AlwaysDefect, AlwaysCooperate, TitForTat, Random50, GPT-4o-mini, untrained Qwen |
| `evaluation/trace_eval.py` | Reasoning-trace harvester for CoT analysis |
| `evaluation/rfeval/` | Counterfactual-intervention pipeline (χ, κ, RF metrics) |
| `utils/experiment_logger.py` | W&B logging + checkpoint snapshot ring buffer |

---

## Stack

- **Model:** Qwen-2.5-7B-Instruct, 4-bit QLoRA (r=16, target = all 7 attention + MLP projections)
- **Algorithm:** GRPO via [HuggingFace TRL](https://github.com/huggingface/trl) (`scale_rewards` toggle for Dr. GRPO ablation)
- **Inference:** vLLM ≥ 0.10.2 for fast rollouts
- **Environments:** IPD / Stag Hunt / Public Goods + TextArena (negotiation, stretch goal)
- **Tracking:** Weights & Biases — project `grpo-social-dilemmas`
- **Compute:** RunPod RTX 4090 (24GB), ~$0.59/hr

VRAM footprint: ~18–22 GB at `max_seq_length=1024`, `num_generations=8`, `max_completion_length=400`.

---

## Method at a glance

Each training step samples a game state, prompts the model 8 times to produce
`<reasoning>...</reasoning><action>C|D</action>`,
plays each completion against an opponent (current policy or a frozen snapshot from a ring buffer), scores by the IPD payoff matrix, and computes group-relative advantage:

```
A_i = R_i_episode − mean_j R_j_episode      (Dr. GRPO — no std normalization)
A_i = (R_i_episode − mean_j R_j_episode) / std_j R_j_episode    (vanilla GRPO)
```

Reward is a **multiplicative format gate** times the payoff — malformed completions get exactly zero, well-formed completions get the raw game payoff with no additive bonuses. This removes one well-known reward-hacking surface (format/length padding for free reward).

---

## Quick start

```bash
# 1. Spin up a pod with an RTX 4090 (or any 24GB+ card) and clone
git clone https://github.com/<you>/grpo-social-dilemmas.git
cd grpo-social-dilemmas

# 2. Install deps to a network volume so they survive pod restarts
export PIP_TARGET=/workspace/pip-packages
export PYTHONPATH=/workspace/pip-packages:$PYTHONPATH
pip install -r requirements.txt

# 3. Pull the base model
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir /workspace/models/qwen2.5-7b-instruct

# 4. Configure W&B
wandb login

# 5. Smoke test — 10 GRPO steps at T=5
python -u training/train.py --config configs/smoke.yaml

# 6. Full training run (~9–18 hrs depending on vLLM version)
python -u training/train.py --config configs/drgrpo_T5_seed1.yaml
```

Checkpoints (LoRA adapters only, ~80 MB each) land in `/workspace/grpo-project/checkpoints/`.

---

## Evaluation

Three tiers, all running over saved checkpoints:

**Tier A — Strategy & performance.** Round-robin against AlwaysDefect, AlwaysCooperate, TitForTat, Random50, GPT-4o-mini, and untrained Qwen. 20 episodes per matchup. Reports cooperation rate, mean score, and win/loss/draw.

**Tier B — Reasoning traces.** `trace_eval.py` harvests full `<reasoning>` blocks across rollouts. Used for qualitative strategy identification (current run: trained model exhibits "slow generous tit-for-tat" against AllD — defects R1, cooperates R2–4 hoping for reciprocation, locks into defection R5–10).

**Tier C — Reasoning faithfulness (RFEval extension).** Counterfactual-intervention pipeline. Generate `r'` traces that argue for the *opposite* action, prefill them inside the `<think>` block, measure whether the action flips. Reports stance consistency χ, causal influence κ, and combined faithfulness `RF = χ ∧ κ`. Compared trained vs. untrained checkpoint.

```bash
python evaluation/eval.py        --checkpoint checkpoints/final-T5.0
python evaluation/trace_eval.py  --checkpoint checkpoints/final-T5.0 --opponent always_defect
python evaluation/rfeval/run.py  --checkpoint checkpoints/final-T5.0 --opponent tit_for_tat
```

---

## Current results (T=5.0, single seed, 250 GRPO steps)

| Opponent | Coop rate | Score (model vs. opp) | Outcome |
|---|---|---|---|
| AlwaysDefect | 0.30 | 7 vs. 22 | loss |
| AlwaysCooperate | 0.80 | 34 vs. 24 | win |
| TitForTat | 0.20 | 20 vs. 15 | win |
| Random50 | — | 23.85 vs. 19.35 | win (60%) |
| Untrained Qwen | — | 31 vs. 16 | win (100%) |

Stag Hunt zero-shot transfer: cooperation rate 1.0.

**T-sweep anomaly to investigate:** at `T=4–8` the model wins with low cooperation; at `T=9–10` it switches to full cooperation and draws. This is the *opposite* of game-theoretic prediction. Working hypothesis: artifact of GRPO's std-normalization of advantages collapsing under low reward variance — testable via a Dr. GRPO ablation (`scale_rewards=False`).

---

## Known issues & active work

The full audit lives in `docs/MODEL_AND_LIMITATIONS.md`. Headline items:

- **Shared-weights "self-play" isn't really self-play.** Mitigation: frozen-snapshot ring buffer (save every 25–40 steps, sample 50/50 between current and snapshot opponents).
- **Fixed 10-round horizon enables backward-induction unraveling.** Mitigation: stochastic continuation `p=0.9`.
- **GRPO broadcasts a single scalar advantage uniformly across all ~400 tokens.** Format, reasoning, and action tokens are indistinguishable to the optimizer. This is the unifying lens behind format hacking, the T=9–10 anomaly, and why CoT faithfulness needs eval-time measurement (RFEval) rather than training-time fixes alone.
- **Baselines (AllD/AllC/TfT/Random50) are weak for a 7B model.** Stronger comparison would be Advantage-Alignment-trained agents from Piche et al.

---

## Repository layout

```
grpo-social-dilemmas/
├── configs/
│   ├── config.py                  # Python config defaults
│   ├── smoke.yaml                 # 10-step smoke test
│   ├── drgrpo_T5_seed1.yaml       # one of 10 main run configs
│   └── ...
├── envs/
│   ├── prisoners_dilemma.py
│   ├── stag_hunt.py
│   └── public_goods.py
├── training/
│   ├── train.py
│   └── reward.py
├── evaluation/
│   ├── eval.py
│   ├── trace_eval.py
│   └── rfeval/
│       ├── run.py
│       ├── flaw_toolkit.py        # counterfactual r' generation
│       └── chi_kappa_scorer.py
├── utils/
│   └── experiment_logger.py
├── docs/
│   ├── PRIOR_ART_AUDIT.md
│   ├── MODEL_AND_LIMITATIONS.md
│   ├── PROJECT_CONCEPTS.md
│   └── RFEVAL_EXTENSION.md
├── logs/                          # gitignored
├── checkpoints/                   # gitignored
├── requirements.txt
├── LICENSE
└── README.md
```

---

## References

- **Piche et al. (2025)** — Learning Robust Social Strategies with Large Language Models. arXiv:2511.19405. *(closest prior art)*
- **Han et al. (2026)** — RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in LRMs. ICLR 2026.
- **Chen et al. (2025)** — Reasoning Models Don't Always Say What They Think. Anthropic Alignment Science.
- **Liu et al. (2025)** — SPIRAL: Self-play with Role-conditioned Advantage. arXiv:2506.24119.
- **Yuan et al. (2025)** — MARS / MARSHAL. arXiv:2510.15414.
- **Duque et al. (2025)** — Advantage Alignment Algorithms. ICLR 2025.
- **Lambert** — RLHF Book, Chapters 6, 7, 14, 15.

Full reading list in `docs/PROJECT_CONCEPTS.md` and `docs/PRIOR_ART_AUDIT.md`.

---

## License

MIT. See `LICENSE`.

## Citation

If you reference this project, please cite the underlying methods (Piche et al. for the GRPO+LoRA+IPD setup, Han et al. for RFEval) rather than this repo, which is a course-project replication-and-extension.
