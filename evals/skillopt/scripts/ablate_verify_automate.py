# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Deterministic ablation: does grounded verify-and-retry lift the oracle score?

Measures the *upper bound* of the per-skill self-check gate
(``docs/self-improvement-proposal.md`` 1.1) on the ``olmoearth_automate`` tool,
using the real ``verify_automate_result`` verifier and the real ``automate()``
tool against the ``olmoearth_embeddings`` SkillOpt split. It needs no LLM and is
fully deterministic, so it is a tighter ship/no-ship signal than a temperature-1
agent run on the weak local model.

Two arms per item:

* **baseline** -- the tool is called with only the natural-language brief
  (``automate(task=...)``); ``parse_task_string`` extracts what it can. This is
  the worst-case "model under-extracts" path.
* **verify+retry** -- after the call, ``verify_automate_result`` runs; if it
  fails (e.g. the recommendation fell back for missing inputs), the brief's full
  ground-truth inputs are re-supplied once (the best case the reflection could
  coax out of the model).

Each arm's recommendation is scored against the skill's own ``decide()`` oracle
(hard = decision + model both correct), exactly as the SkillOpt evaluator does.
The gap between the arms is the maximum oracle lift the gate can deliver on this
split. Run::

    python evals/skillopt/scripts/ablate_verify_automate.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from olmoearth_agent.analysis.automate import automate, verify_automate_result

_SPLIT_DIR = Path(__file__).resolve().parents[1] / "data" / "olmoearth_embeddings_split"
_SPLITS = ("test", "val", "train")


def _hard(rec: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Decision + model both correct (the SkillOpt 'hard' metric)."""
    got_model = str((rec.get("config") or {}).get("model_size") or "").lower()
    return str(rec.get("decision") or "") == str(expected.get("decision") or "") and (
        got_model == str(expected.get("model") or "").lower()
    )


async def _run_arms(item: dict[str, Any]) -> tuple[bool, bool, bool]:
    """Return (baseline_correct, verify_correct, gate_fired) for one item."""
    expected = item["expected"]
    baseline = await automate(task=item["task_description"])
    base_ok = _hard(baseline, expected)

    ok, _reason = verify_automate_result(baseline)
    gate_fired = not ok
    if ok:
        verified = baseline
    else:
        # The reflection's best case: the model re-extracts the full inputs the
        # brief states and calls again. We supply them from the item directly.
        verified = await automate(**item["inputs"])
    return base_ok, _hard(verified, expected), gate_fired


async def main() -> None:
    print(
        f"{'split':>6} | {'n':>3} | {'baseline':>9} | {'verify+retry':>12} | "
        f"{'gate_fired':>10} | {'lift':>5}"
    )
    print("-" * 62)
    for split in _SPLITS:
        items = json.loads(
            (_SPLIT_DIR / split / "items.json").read_text(encoding="utf-8")
        )
        results = [await _run_arms(it) for it in items]
        n = len(results)
        base = sum(1 for b, _, _ in results if b)
        ver = sum(1 for _, v, _ in results if v)
        fired = sum(1 for _, _, g in results if g)
        print(
            f"{split:>6} | {n:>3} | {base:>4}/{n:<4} | {ver:>5}/{n:<6} | "
            f"{fired:>10} | {ver - base:>+5}"
        )
    print(
        "\nHard metric = decision + model both correct, scored against "
        "decide().\nLift = verify+retry correct minus baseline correct (the gate's "
        "upper bound)."
    )


if __name__ == "__main__":
    asyncio.run(main())
