"""Evaluation layer — Stage 1 Tier A/B/C/D.

Authority: PRD v6.1 §8 + STAGE1_EXECUTION_PLAN_v3.1 §3.5.

Modules:
    opponents     — fixed-strategy opponents (TfT, AC, AD, R50) +
                    GPT-4o-mini wrapper for Tier A.
    eval          — Tier A (fixed-opponent matchups) on IPD.
    trace_eval    — Per-round reasoning trace dumper for REMUL/RFEval.
    elo           — Tier B Elo tournament across the 4 final adapters
                    + GPT-4o-mini.
    transfer      — Tier C transfer eval to Stag Hunt + Public Goods.
"""
