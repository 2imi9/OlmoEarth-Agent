"""Regression gate: compare a new eval summary against the stored baseline.

Shippy-style rule: a build that regresses on the eval suite doesn't ship.
``eval_skill.py`` writes a ``summary.json`` per run (hard/soft + per-field);
this script diffs one of those against the committed baseline for the same
env/split and exits non-zero on regression, printing a score-change report.

    python scripts/regression_gate.py --new outputs/eval_test_SKILL_n14/summary.json \
        --baseline baselines/olmoearth_jobconfig_test.json [--epsilon 0.0]

    # after an accepted improvement, promote the run to be the new baseline:
    python scripts/regression_gate.py --new .../summary.json \
        --baseline baselines/olmoearth_jobconfig_test.json --update-baseline

Runs are temperature-0/deterministic, so the default epsilon is 0: any drop
in ``hard`` or ``soft`` fails the gate. Exit codes: 0 = no regression,
1 = regression, 2 = usage/IO error.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

METRICS = ("hard", "soft")


def compare(new: dict, baseline: dict, epsilon: float) -> tuple[bool, list[str]]:
    """Diff shared metrics; returns (passed, report_lines)."""
    lines: list[str] = []
    passed = True
    if new.get("n") is not None and baseline.get("n") is not None and new["n"] != baseline["n"]:
        lines.append(
            f"WARNING: item counts differ (baseline n={baseline['n']}, new n={new['n']}) — "
            "scores may not be comparable; regenerate on the full split."
        )
    for metric in METRICS:
        if metric not in new or metric not in baseline:
            continue
        old_v, new_v = float(baseline[metric]), float(new[metric])
        delta = new_v - old_v
        verdict = "REGRESSION" if delta < -epsilon else ("improved" if delta > 0 else "unchanged")
        if delta < -epsilon:
            passed = False
        lines.append(f"{metric:6s} {old_v:.3f} -> {new_v:.3f}  ({delta:+.3f})  {verdict}")
    if not any(m in new and m in baseline for m in METRICS):
        passed = False
        lines.append("ERROR: no shared metrics between the new summary and the baseline.")
    return passed, lines


def main() -> int:
    """CLI entry: load the two summaries, report deltas, gate on regression."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", required=True, help="summary.json from eval_skill.py")
    ap.add_argument("--baseline", required=True, help="committed baseline json")
    ap.add_argument("--epsilon", type=float, default=0.0,
                    help="tolerated drop before failing (default 0: any drop fails)")
    ap.add_argument("--update-baseline", action="store_true",
                    help="write the new scores into the baseline file after comparing")
    a = ap.parse_args()

    try:
        with open(a.new, encoding="utf-8") as f:
            new = json.load(f)
        with open(a.baseline, encoding="utf-8") as f:
            baseline = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot load summaries: {exc}", file=sys.stderr)
        return 2

    passed, lines = compare(new, baseline, a.epsilon)
    print(f"=== regression gate: {a.new} vs {a.baseline} (epsilon={a.epsilon}) ===")
    for line in lines:
        print(line)
    print("PASS" if passed else "FAIL: score regressed — do not ship this skill/model change.")

    if a.update_baseline and passed:
        updated = {
            **{k: baseline[k] for k in baseline if k not in METRICS},
            **{m: new[m] for m in METRICS if m in new},
            "n": new.get("n", baseline.get("n")),
            "source": f"promoted from {a.new}",
            "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }
        with open(a.baseline, "w", encoding="utf-8") as f:
            json.dump(updated, f, indent=2)
            f.write("\n")
        print(f"baseline updated -> {a.baseline}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
