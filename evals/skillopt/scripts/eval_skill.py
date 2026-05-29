"""Evaluate one skill .md on a split of the OlmoEarth job-config benchmark.

Self-contained (doesn't use eval_only.py's CLI, which lacks a qwen backend
choice). Configures the local Qwen as the target, runs the env's rollout, and
prints hard (exact-config) + soft (field-average) accuracy plus a per-field
breakdown. Used for the smoke test, the baseline, and the final comparison.

    python olmoearth_local/eval_skill.py --skill <path.md> --split test [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

os.environ.setdefault(
    "SKILLOPT_OPENAI_COMPAT_EXTRA_BODY",
    '{"chat_template_kwargs": {"enable_thinking": false}}',
)

from skillopt.model.backend_config import set_target_backend, set_optimizer_backend
from skillopt.model import (
    configure_azure_openai,
    configure_qwen_chat,
    set_target_deployment,
    set_optimizer_deployment,
    set_reasoning_effort,
)
import importlib

MODEL = "unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS"
ENDPOINT = "http://localhost:8000/v1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup(temperature: float = 0.0) -> None:
    set_optimizer_backend("openai_chat")
    set_target_backend("qwen_chat")
    configure_azure_openai(endpoint=ENDPOINT, auth_mode="openai_compatible")
    configure_qwen_chat(
        base_url=ENDPOINT, max_tokens=2048, enable_thinking=False,
        temperature=temperature, timeout_seconds=300,
    )
    set_target_deployment(MODEL)
    set_optimizer_deployment(MODEL)
    set_reasoning_effort(None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="olmoearth_jobconfig")
    ap.add_argument("--skill", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    setup(a.temperature)
    run_batch = importlib.import_module(f"skillopt.envs.{a.env}.rollout").run_batch
    with open(a.skill, encoding="utf-8") as f:
        skill = f.read()
    items_path = os.path.join(ROOT, "data", f"{a.env}_split", a.split, "items.json")
    with open(items_path, encoding="utf-8") as f:
        items = json.load(f)
    if a.limit:
        items = items[: a.limit]

    tag = os.path.basename(a.skill).replace(".md", "")
    out = a.out or os.path.join(ROOT, "outputs", f"eval_{a.split}_{tag}_n{len(items)}")
    os.makedirs(out, exist_ok=True)

    res = run_batch(
        items=items, out_root=out, skill_content=skill,
        workers=a.workers, exec_timeout=240, max_completion_tokens=2048,
    )
    n = len(res)
    hard = sum(r["hard"] for r in res) / n if n else 0.0
    soft = sum(r["soft"] for r in res) / n if n else 0.0

    fc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in res:
        for k, v in (r.get("field_results") or {}).items():
            fc[k][1] += 1
            fc[k][0] += 1 if v else 0

    print(f"\n=== {a.split}  n={n}  skill={a.skill} ===")
    print(f"hard (exact wizard config): {hard:.3f}")
    print(f"soft (field-average):       {soft:.3f}")
    print("per-field correct:")
    for k, (c, t) in sorted(fc.items()):
        print(f"  {k:24s} {c}/{t}")

    summary = {"split": a.split, "skill": a.skill, "n": n, "hard": hard, "soft": soft,
               "per_field": {k: {"correct": c, "total": t} for k, (c, t) in fc.items()}}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({**summary, "results": res}, f, ensure_ascii=False, indent=2)
    print(f"summary -> {os.path.join(out, 'summary.json')}")


if __name__ == "__main__":
    main()
