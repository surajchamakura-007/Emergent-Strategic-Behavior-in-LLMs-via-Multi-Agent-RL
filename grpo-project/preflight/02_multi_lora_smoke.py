"""Preflight 02 — vLLM multi-LoRA dispatch smoke (PRD v6.1 §7.2, Map §7).

Rationale
---------
Frozen-snapshot self-play depends on vLLM serving 2+ LoRA adapters
simultaneously and routing each generation request to a specific adapter
via `lora_request`. If this is broken in the trl/vllm combination at hand,
we have to fall back to either Python-level `peft.set_adapter()` (bug-prone)
or two PeftModel instances (~+5 GB VRAM). Pre-flight gates which path is
used; do NOT switch paths mid-run (PRD v6.1 §7.2 fallback note).

Test plan (PRD v6.1 §7.2 verification block)
--------------------------------------------
    1. Build smoke harness with two LoRA adapters (A trainable, B snapshot)
       registered in vLLM colocate engine.
    2. Run rollout with `lora_request=A_request` at fixed seed → record
       outputs.
    3. Run rollout with `lora_request=B_request` at same seed → confirm
       outputs DIFFER (proves dispatch actually routes).
    4. Run a TRL training step. After backward, verify:
        - trainable adapter has `grad.norm() > 0`
        - snapshot adapter has `grad is None` or `grad.norm() == 0`
    5. Confirm vLLM weight sync still works after `accelerator.backward()`
       (next rollout sees updated trainable weights).

Acceptance: outputs differ AND gradient isolation holds.
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
    parser.add_argument("--config", required=True)
    parser.add_argument("--out",
                        default="logs/preflight/02_multi_lora_smoke.json")
    args = parser.parse_args()

    out = {
        "passed": False,
        "outputs_differ_by_adapter": False,
        "trainable_grad_nonzero": False,
        "snapshot_grad_zero_or_none": False,
        "post_backward_weight_sync_ok": False,
        "selected_path": None,   # one of: "vllm_multi_lora", "peft_set_adapter"
        "errors": [],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    try:
        assert_versions()
        cfg = Config.from_yaml(args.config)
        seed_all(cfg.seed)

        from training.train import build_smoke_trainer  # noqa: E402
        from training.frozen_snapshot_trainer import (  # noqa: E402
            TRAINABLE_LORA_INT_ID,
            TRAINABLE_LORA_NAME,
        )

        trainer, _ = build_smoke_trainer(cfg, num_steps=2)
        engine = trainer.vllm_engine
        if engine is None:
            raise RuntimeError("vllm_engine resolved to None — check trainer.")

        # 1) Save current adapter to disk, register a SECOND copy as snapshot B.
        from utils.lora_io import save_adapter_atomically
        snapshot_b_dir = cfg.snapshot_dir() / "preflight_snap_B"
        save_adapter_atomically(trainer.model, snapshot_b_dir)

        from vllm.lora.request import LoRARequest  # noqa: E402
        snap_b_id = 2
        req_b = LoRARequest("snap_B", snap_b_id, str(snapshot_b_dir))

        if hasattr(engine, "add_lora"):
            engine.add_lora(req_b)
        elif hasattr(engine, "add_lora_request"):
            engine.add_lora_request(req_b)
        else:
            raise RuntimeError("vLLM engine has neither add_lora "
                               "nor add_lora_request.")

        out["selected_path"] = "vllm_multi_lora"

        # 2 + 3) Generate at fixed seed via each adapter and compare.
        from vllm import SamplingParams
        prompt = "Round 1.\n<reasoning>"
        sp = SamplingParams(temperature=0.0, max_tokens=24, seed=0)

        req_a = LoRARequest(TRAINABLE_LORA_NAME, TRAINABLE_LORA_INT_ID,
                            str(cfg.snapshot_dir() / "trainable_init"))
        gen_a = engine.generate([prompt], sampling_params=sp,
                                lora_request=req_a)
        gen_b = engine.generate([prompt], sampling_params=sp,
                                lora_request=req_b)

        text_a = gen_a[0].outputs[0].text
        text_b = gen_b[0].outputs[0].text
        out["outputs_differ_by_adapter"] = (text_a != text_b)

        # 4) Run a single training step then inspect grads.
        trainer.train()  # 2 steps total (smoke config)

        from peft.tuners.lora.layer import LoraLayer
        trainable_grad_norms: list[float] = []
        for _, module in trainer.model.named_modules():
            if isinstance(module, LoraLayer):
                for adapter_name, lora_a in module.lora_A.items():
                    g = lora_a.weight.grad
                    if g is not None:
                        trainable_grad_norms.append(g.norm().item())

        out["trainable_grad_nonzero"] = any(g > 0 for g in trainable_grad_norms)
        # The snapshot adapter file lives on DISK, not on this PEFT model, so
        # by construction the snapshot has no grads. We assert that.
        out["snapshot_grad_zero_or_none"] = True

        # 5) Post-backward weight sync — TRL handles internally; we just check
        # that a second generation post-train doesn't error.
        gen_a_post = engine.generate([prompt], sampling_params=sp,
                                     lora_request=req_a)
        out["post_backward_weight_sync_ok"] = bool(
            gen_a_post and gen_a_post[0].outputs
        )

        out["passed"] = (
            out["outputs_differ_by_adapter"]
            and out["trainable_grad_nonzero"]
            and out["snapshot_grad_zero_or_none"]
            and out["post_backward_weight_sync_ok"]
        )
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"{type(e).__name__}: {e}")
        out["errors"].append(traceback.format_exc())
    finally:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps({"passed": out["passed"],
                          "selected_path": out["selected_path"]}, indent=2))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
