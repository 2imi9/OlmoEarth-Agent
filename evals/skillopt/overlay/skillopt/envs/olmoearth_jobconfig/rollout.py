"""OlmoEarth job-config rollout — single-turn config generation + scoring.

The target model receives the skill document as its system prompt and a
plain-English task as the user message, and returns a Studio wizard config as
JSON. We score it against the expected preset and persist the trajectory for
the reflect/analyst stage.

Public API
----------
- :func:`process_one` — run + score one task
- :func:`run_batch`   — parallel, resume-aware execution of a list of tasks
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from skillopt.envs.olmoearth_jobconfig.evaluator import evaluate
from skillopt.model import chat_target
from skillopt.prompts import load_prompt


def _build_system(skill_content: str) -> str:
    if skill_content.strip():
        skill_section = f"## Skill\n{skill_content.strip()}\n\n"
    else:
        skill_section = ""
    return load_prompt("rollout_system", env="olmoearth_jobconfig").format(
        skill_section=skill_section
    )


def _build_user(task_description: str) -> str:
    return f"## Task\n{task_description.strip()}"


def _eval_detail(item: dict, eval_result: dict) -> str:
    return (
        "[EVALUATION RESULT]\n"
        f"Task: {item.get('task_description', '')}\n"
        f"Predicted config: {json.dumps(eval_result.get('predicted', {}), ensure_ascii=False)}\n"
        f"Expected config: {json.dumps(item.get('expected', {}), ensure_ascii=False)}\n"
        f"Per-field correct: {json.dumps(eval_result.get('field_results', {}), ensure_ascii=False)}\n"
        f"hard={eval_result['hard']}  soft={eval_result['soft']:.3f}\n"
        f"{eval_result.get('fail_reason', '')}"
    )


def process_one(
    item: dict,
    out_root: str,
    skill_content: str,
    max_turns: int = 1,
    exec_timeout: int = 120,
    max_completion_tokens: int = 2048,
) -> dict:
    """Run one job-config task: prompt the target, parse + score its config."""
    item_id = str(item["id"])
    task_description = item.get("task_description", "")
    expected = item.get("expected", {})

    result = {
        "id": item_id,
        "task_description": task_description,
        "task_type": item.get("task_type", expected.get("output_type", "unknown")),
        "hard": 0,
        "soft": 0.0,
        "parsed_ok": False,
        "predicted": {},
        "expected": expected,
        "field_results": {},
        "response": "",
        "fail_reason": "",
        "agent_ok": False,
        "n_turns": 0,
    }

    try:
        pred_dir = os.path.join(out_root, "predictions", item_id)
        os.makedirs(pred_dir, exist_ok=True)

        system = _build_system(skill_content)
        user = _build_user(task_description)

        response, _ = chat_target(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=5,
            stage="rollout",
            timeout=exec_timeout,
        )

        result["response"] = response
        result["agent_ok"] = True
        result["n_turns"] = 1

        eval_result = evaluate(response, expected)
        result["hard"] = eval_result["hard"]
        result["soft"] = eval_result["soft"]
        result["parsed_ok"] = eval_result["parsed_ok"]
        result["predicted"] = eval_result["predicted"]
        result["field_results"] = eval_result["field_results"]
        result["fail_reason"] = eval_result["fail_reason"]

        conversation = [
            {"type": "message", "turn": 1, "content": response},
            {"role": "system", "content": _eval_detail(item, eval_result)},
        ]
        with open(os.path.join(pred_dir, "target_system_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(system)
        with open(os.path.join(pred_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(user)
        with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)

    except Exception as e:  # noqa: BLE001
        result["fail_reason"] = f"error: {e}"

    return result


def run_batch(
    items: list[dict],
    out_root: str,
    skill_content: str,
    max_turns: int = 1,
    exec_timeout: int = 120,
    workers: int = 3,
    max_completion_tokens: int = 2048,
    task_timeout: int = 600,
    **_kwargs,
) -> list[dict]:
    """Run all tasks with a thread pool. Resume-aware via results.jsonl."""
    task_timeout = max(int(task_timeout), int(exec_timeout) + 60)
    results_path = os.path.join(out_root, "results.jsonl")
    os.makedirs(out_root, exist_ok=True)

    done_ids: set[str] = set()
    existing: list[dict] = []
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_ids.add(str(r["id"]))
                    existing.append(r)
                except Exception:  # noqa: BLE001
                    pass

    pending = [it for it in items if str(it["id"]) not in done_ids]
    if not pending:
        return existing

    total = len(existing) + len(pending)
    completed = len(existing)
    correct = sum(1 for r in existing if r.get("hard", 0))
    results = list(existing)

    def _timeout_result(item: dict, reason: str) -> dict:
        return {
            "id": str(item["id"]),
            "task_description": item.get("task_description", ""),
            "task_type": item.get("task_type", "unknown"),
            "hard": 0,
            "soft": 0.0,
            "parsed_ok": False,
            "predicted": {},
            "expected": item.get("expected", {}),
            "field_results": {},
            "response": "",
            "fail_reason": reason,
            "agent_ok": False,
            "n_turns": 0,
        }

    started_at: dict[str, float] = {}

    def _run_one(item: dict) -> dict:
        started_at[str(item["id"])] = time.time()
        return process_one(
            item, out_root, skill_content, max_turns, exec_timeout, max_completion_tokens
        )

    with open(results_path, "a", encoding="utf-8") as outf:
        ex = ThreadPoolExecutor(max_workers=workers)
        try:
            futs = {ex.submit(_run_one, it): it for it in pending}
            pending_futs = set(futs)
            while pending_futs:
                done, _ = wait(pending_futs, timeout=5, return_when=FIRST_COMPLETED)
                now = time.time()
                timed_out = [
                    fut for fut in pending_futs - done
                    if str(futs[fut]["id"]) in started_at
                    and now - started_at[str(futs[fut]["id"])] >= task_timeout
                ]
                for fut in done:
                    pending_futs.remove(fut)
                    item = futs[fut]
                    try:
                        res = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        res = _timeout_result(item, f"unexpected: {type(exc).__name__}: {exc}")
                    results.append(res)
                    completed += 1
                    if res.get("hard", 0):
                        correct += 1
                    acc = correct / completed if completed else 0
                    print(
                        f"    [rollout] {completed}/{total} (acc={acc:.3f}) "
                        f"id={res['id']} hard={res.get('hard', '?')} soft={res.get('soft', 0):.2f}",
                        flush=True,
                    )
                    outf.write(json.dumps(res, ensure_ascii=False) + "\n")
                    outf.flush()
                for fut in timed_out:
                    pending_futs.remove(fut)
                    fut.cancel()
                    res = _timeout_result(futs[fut], f"task-timeout-{task_timeout}s")
                    results.append(res)
                    completed += 1
                    print(
                        f"    [rollout] {completed}/{total} id={res['id']} TIMEOUT",
                        flush=True,
                    )
                    outf.write(json.dumps(res, ensure_ascii=False) + "\n")
                    outf.flush()
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    return results
