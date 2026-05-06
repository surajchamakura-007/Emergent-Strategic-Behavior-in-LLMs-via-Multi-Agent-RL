"""Stack version assertions, called at the top of train.py.

Authority: PRD v6.1 changelog (stack version pin policy → ranges, not exact),
Implementation Map §2.1 ("called from train.py line 1").

Why ranges, not pins. TRL 1.0 stable as of April 2026; ranges allow
security/bug patches without re-pinning per run. But fan-out training over 4
runs across days can silently cross a minor-version boundary (e.g., a security
patch landing during a queue wait), so we assert at every job start.

What this catches.
    - vllm < 0.10.2: TIS temperature/logprob mismatch (PRD §5.1.2).
    - trl < 1.0:     advantage-normalization flag absent.
    - vllm >= 0.12:  multi-LoRA API may have moved.
    - transformers < 4.57: PEFT integration regressions (Qwen2.5 path).
"""

from __future__ import annotations

import importlib
import re
import sys


# Pinned ranges from PRD v6.1 §1 changelog "Stack version pin policy"
# Format: (pkg, min_inclusive, max_exclusive_or_None)
_REQUIRED_RANGES: tuple[tuple[str, str, str | None], ...] = (
    ("trl",          "1.0.0",  "1.1.0"),
    ("vllm",         "0.10.2", "0.12.0"),
    ("transformers", "4.57.0", None),
    ("peft",         "0.18.0", None),
    ("bitsandbytes", "0.43.0", None),
    ("torch",        "2.4.0",  None),
)


class StackVersionError(EnvironmentError):
    """Raised when any installed package falls outside the required range."""


def _parse_version(v: str) -> tuple[int, ...]:
    """Loose semver parse; treats '0.10.2.post1' as (0,10,2)."""
    parts = re.findall(r"\d+", v)[:3]
    if not parts:
        raise ValueError(f"Cannot parse version string: {v!r}")
    return tuple(int(p) for p in parts) + (0,) * (3 - len(parts))


def _satisfies(actual: str, lo: str, hi: str | None) -> bool:
    a = _parse_version(actual)
    if a < _parse_version(lo):
        return False
    if hi is not None and a >= _parse_version(hi):
        return False
    return True


def assert_versions(verbose: bool = True) -> dict[str, str]:
    """Verify all stack packages are within the required ranges.

    Returns:
        Dict of {pkg: installed_version} (for W&B logging).

    Raises:
        StackVersionError: any package missing or out of range.
    """
    installed: dict[str, str] = {}
    failures: list[str] = []

    for pkg, lo, hi in _REQUIRED_RANGES:
        try:
            mod = importlib.import_module(pkg)
        except ImportError as e:
            failures.append(f"{pkg}: NOT INSTALLED ({e})")
            continue

        version = getattr(mod, "__version__", None)
        if version is None:
            failures.append(f"{pkg}: __version__ attribute missing")
            continue

        installed[pkg] = version
        if not _satisfies(version, lo, hi):
            bound = f">={lo}" + (f",<{hi}" if hi else "")
            failures.append(f"{pkg}=={version} (require {bound})")

    if failures:
        msg = (
            "Stack version check failed (PRD v6.1):\n  - "
            + "\n  - ".join(failures)
            + "\nFix the environment before launching training."
        )
        raise StackVersionError(msg)

    if verbose:
        print("[stack_versions] OK:", ", ".join(f"{k}={v}" for k, v in installed.items()),
              file=sys.stderr)
    return installed


__all__ = ["assert_versions", "StackVersionError"]
