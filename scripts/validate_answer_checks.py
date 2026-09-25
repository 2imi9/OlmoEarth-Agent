# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Run the harness's answer checks offline over recorded answers, and score them.

Reads a trial's recorded runs (``<rounds>/<round>/runs/<brief>/<setup>/<run>/``
with ``stdout.txt``, ``events.jsonl`` and ``brief.txt``, as exp86's driver
writes them), rebuilds each run's evidence from its tool calls and results,
runs every check of :mod:`olmoearth_agent.harness.checks` on the answer that
was shown, and reports how many answers each check flags per round. With an
audit (exp86's blind audit of rounds 6 and 7, ``exp86_audit_rounds_6_7.json``)
it lists every flag on the audited rounds and matches it to the audit's
confirmed findings: a flag on a sentence the audit confirmed false is a true
flag, any other is a false alarm (or an error the audit did not record).

The recorded tool results predate the tools' ``facts``, ``must_state`` and
``forbidden_claims``. So that the checks have what they read, this script
adds them to each result from what the result (or the run's workspace
files) already holds, marked ``simulated``:

- ``olmoearth_compare_review``: ``dominant_change`` (with the reverse
  direction's count and whether it ties) and ``more_confident_side`` (the
  sides' shares, from counts) from the run's recorded scores files, or,
  without them, from ``class_changes`` and
  ``a_more_confident_share_of_differing``; ``concentration`` from the two
  files' classes (``top_band_share`` of the northmost of four row bands,
  and ``max_band`` over four row and four column bands, with the grid and
  the count); forbidden ``winner_without_labels``, and
  ``another_date_settles_it`` when the dates differ; must-state: the maps
  describe different times.
- ``olmoearth_review_set`` and ``olmoearth_review_set_from_result``:
  ``margin_ratio`` (the median margin over the listed margins, and over the
  whole review set's, from the lowest margin to the one at the budget cut);
  forbidden ``error_rate_without_labels``; must-state: the set is not a
  sample.
- ``olmoearth_plan_label_sample``: ``unused_labels`` when the plan holds
  fewer labels than the run's first plan call asked for; forbidden
  ``certify_from_nonrandom_design`` for a non-random design and
  ``subset_labelling_sufficient``.
- ``olmoearth_estimate_map_error``: forbidden
  ``certify_from_nonrandom_design`` for a non-random design.
- ``olmoearth_certify_zone``: forbidden ``post_hoc_alpha``, and
  ``rule_switch_after_failure`` when nothing was certified; must-state: no
  zone is certified.
- ``olmoearth_compare_results`` of regression bands: forbidden
  ``error_rate_for_unthresholded_regression``, and
  ``combined_statistic_across_properties`` when the properties differ.

The facts follow the contract the tools and the harness share (field names
as in ``harness/checks.py``). These are estimates of what the checks would
catch once the tools emit the keys; the tools' own facts and sentences may
differ.

Usage::

    uv run python scripts/validate_answer_checks.py \\
        --rounds ../olmoearth_inferenceX/exp/out/exp86_trial/rounds \\
        --audit ../olmoearth_inferenceX/exp/out/exp86_audit_rounds_6_7.json \\
        --out /tmp/answer_checks.json
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from olmoearth_agent.analysis.review_set import margins, predicted_classes
from olmoearth_agent.harness.checks import CHECKS, RunEvidence, ToolRecord, run_checks

#: The audit's label for each round (its key).
AUDIT_ROUNDS = {"X": "6", "Y": "7"}


# --------------------------------------------------------------------------- loading


def run_dirs(rounds: Path) -> Iterator[tuple[str, str, str, Path]]:
    """(round, configuration, run, directory) of every recorded run, in order."""
    for d in sorted(rounds.glob("*/runs/*/*/*"), key=lambda p: p.parts[-5:]):
        if (d / "events.jsonl").is_file():
            rnd, _, brief, setup, run = d.parts[-5:]
            yield rnd, f"{brief}/{setup}", run, d


def load_run(d: Path) -> tuple[str | None, list[ToolRecord], str]:
    """The answer shown, the tool calls with their results, and the brief."""
    pending: dict[Any, dict[str, Any]] = {}
    records: list[ToolRecord] = []
    final = None
    for line in (d / "events.jsonl").read_text(encoding="utf-8").splitlines():
        ev = json.loads(line)
        if ev.get("type") == "tool_call":
            pending[(ev.get("turn"), ev.get("id"))] = ev
        elif ev.get("type") == "tool_result":
            call = pending.pop((ev.get("turn"), ev.get("id")), {})
            args = call.get("arguments")
            records.append(
                ToolRecord(
                    ev.get("name", ""),
                    args if isinstance(args, dict) else {},
                    ev["result"],
                )
            )
        elif ev.get("type") == "final":
            final = ev.get("content")
    stdout = d / "stdout.txt"
    text = stdout.read_text(encoding="utf-8") if stdout.is_file() else ""
    answer = text.strip() or (final.strip() if isinstance(final, str) else None)
    brief = d / "brief.txt"
    return (
        answer,
        records,
        brief.read_text(encoding="utf-8").strip() if brief.is_file() else "",
    )


# --------------------------------------------------------------------------- the simulated contract


def _scores_file(d: Path, path: str) -> dict[str, Any] | None:
    """A recorded scores file, found under the run's workspace by its tail."""
    parts = [p for p in path.replace("\\", "/").split("/") if p not in ("", ".")]
    for k in range(len(parts), 0, -1):
        cand = d / "workspace" / Path(*parts[-k:])
        if cand.is_file():
            data = json.loads(cand.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {"scores": data}
    return None


def _band_shares(indices: list[int], rows: int, cols: int) -> dict[str, Any]:
    """``top_band_share`` and ``max_band`` over four row and four column bands."""
    n = len(indices) or 1
    counts = {("rows", k): 0 for k in range(4)} | {("cols", k): 0 for k in range(4)}
    for i in indices:
        r, c = divmod(i, cols)
        counts[("rows", min(3, r * 4 // rows))] += 1
        counts[("cols", min(3, c * 4 // cols))] += 1
    (axis, band), top = max(counts.items(), key=lambda kv: kv[1])
    return {
        "top_band_share": round(counts[("rows", 0)] / n, 6),
        "max_band": {
            "axis": axis,
            "band": band,
            "of_grid": f"{25 * band}-{25 * (band + 1)}%",
            "share": round(top / n, 6),
        },
    }


def _compare_contract(
    d: Path, record: ToolRecord, result: dict[str, Any]
) -> dict[str, list[Any]]:
    facts: list[dict[str, Any]] = []
    changes = result.get("class_changes")
    pairs: dict[tuple[Any, Any], int] = {}
    n_differing = result.get("n_differing")
    sides: tuple[int, int, int] | None = None
    concentration = None
    a = _scores_file(d, str(record.arguments.get("scores_path_a", "")))
    b = _scores_file(d, str(record.arguments.get("scores_path_b", "")))
    if a and b and len(a["scores"]) == len(b["scores"]):
        ca, cb = predicted_classes(a["scores"]), predicted_classes(b["scores"])
        ma, mb = margins(a["scores"]), margins(b["scores"])
        differing = [i for i in range(len(ca)) if ca[i] != cb[i]]
        n_differing = len(differing)
        for i in differing:
            pairs[(ca[i], cb[i])] = pairs.get((ca[i], cb[i]), 0) + 1
        more_a = sum(ma[i] > mb[i] for i in differing)
        more_b = sum(mb[i] > ma[i] for i in differing)
        sides = (more_a, more_b, len(differing) - more_a - more_b)
        windows = a.get("windows") or list(range(len(ca)))
        diff = [int(windows[i]) for i in differing]
        grid = a.get("grid")
        if diff and grid:
            rows, cols = int(grid[0]), int(grid[1])
            bands = _band_shares(diff, rows, cols)
            concentration = {
                "id": "concentration",
                "grid": [rows, cols],
                "n_differing": len(diff),
                **bands,
                "sentence": f"{bands['top_band_share']:.1%} of the {len(diff)} "
                "differing windows are in the northmost quarter of the rows; the "
                f"most are in the {bands['max_band']['axis']} "
                f"{bands['max_band']['of_grid']} "
                f"({bands['max_band']['share']:.1%})",
                "simulated": True,
            }
    elif changes:
        pairs = {(c["class_a"], c["class_b"]): c["n"] for c in changes}
    if pairs and n_differing:
        ranked = sorted(pairs.items(), key=lambda kv: (-kv[1], str(kv[0])))
        (src, dst), n = ranked[0]
        reverse_n = pairs.get((dst, src), 0)
        ties = [p for p, k in ranked[1:] if k == n]
        sentence = (
            f"the largest class change is {src} -> {dst}, {n} of the "
            f"{n_differing} differing windows ({n / n_differing:.1%})"
        )
        if ties:
            sentence += "; tied with " + ", ".join(f"{x} -> {y}" for x, y in ties)
        facts.append(
            {
                "id": "dominant_change",
                "from_class": src,
                "to_class": dst,
                "n": n,
                "share": round(n / n_differing, 6),
                "reverse_n": reverse_n,
                "reverse_share": round(reverse_n / n_differing, 6),
                "tied_with_reverse": reverse_n == n,
                "sentence": sentence,
                "simulated": True,
            }
        )
    share_a = result.get("a_more_confident_share_of_differing")
    if sides is None and isinstance(share_a, int | float) and n_differing:
        more_a = round(share_a * n_differing)
        sides = (more_a, n_differing - more_a, 0)
    if sides and sum(sides):
        more_a, more_b, equal = sides
        total = sum(sides)
        side = "A" if more_a > more_b else "B" if more_b > more_a else "neither"
        facts.append(
            {
                "id": "more_confident_side",
                "side": side,
                "share_a": round(more_a / total, 6),
                "share_b": round(more_b / total, 6),
                "share_equal": round(equal / total, 6),
                "sentence": f"side A is the more confident on {more_a / total:.1%} "
                f"of the differing windows and side B on {more_b / total:.1%}",
                "simulated": True,
            }
        )
    if concentration:
        facts.append(concentration)
    dates = result.get("dates") if isinstance(result.get("dates"), dict) else {}
    forbidden = [{"id": "winner_without_labels"}]
    must = []
    if dates.get("status") in ("different_time", "overlapping_time"):
        forbidden.append({"id": "another_date_settles_it"})
        must.append(
            "The maps describe different times, so a window where they differ "
            "may have changed on the ground."
        )
    return {"facts": facts, "forbidden_claims": forbidden, "must_state": must}


def _review_contract(result: dict[str, Any]) -> dict[str, list[Any]]:
    facts = []
    summary = result.get("margin_summary")
    if isinstance(summary, dict):
        median = summary.get("median_margin")
        listed = (summary.get("listed") or {}).get("margin_range")
        lowest, cut = summary.get("lowest_margin"), summary.get("margin_at_budget_cut")
        if isinstance(median, int | float) and listed and min(listed) > 0:
            fact = {
                "id": "margin_ratio",
                "listed_low": round(median / max(listed), 2),
                "listed_high": round(median / min(listed), 2),
                "versus": "median",
                "simulated": True,
            }
            if isinstance(lowest, int | float) and isinstance(cut, int | float):
                if lowest > 0 and cut > 0:
                    fact["review_set_low"] = round(median / cut, 2)
                    fact["review_set_high"] = round(median / lowest, 2)
            fact["sentence"] = (
                f"the median margin is {fact['listed_low']:g} to "
                f"{fact['listed_high']:g} times the listed windows' margins"
            )
            facts.append(fact)
    return {
        "facts": facts,
        "forbidden_claims": [{"id": "error_rate_without_labels"}],
        "must_state": [
            "The review set ranks suspicion; it is not a sample, so its error "
            "rate is not the map's."
        ],
    }


def adapt(d: Path, records: list[ToolRecord]) -> list[ToolRecord]:
    """The records with a simulated contract added to each result that has none."""
    first_budget = next(
        (
            r.arguments.get("budget")
            for r in records
            if r.name == "olmoearth_plan_label_sample"
            and isinstance(r.arguments.get("budget"), int | float)
        ),
        None,
    )
    out = []
    for record in records:
        envelope = copy.deepcopy(record.envelope)
        result = envelope.get("result") if isinstance(envelope, dict) else None
        if not record.ok or not isinstance(result, dict) or "facts" in result:
            out.append(record)
            continue
        extra: dict[str, list[Any]] = {}
        if record.name == "olmoearth_compare_review":
            extra = _compare_contract(d, record, result)
        elif record.name in (
            "olmoearth_review_set",
            "olmoearth_review_set_from_result",
        ):
            extra = _review_contract(result)
        elif record.name == "olmoearth_plan_label_sample":
            facts = []
            planned = sum(result.get("allocation") or []) or result.get("budget")
            if isinstance(first_budget, int | float) and isinstance(
                planned, int | float
            ):
                if first_budget > planned:
                    unused = int(first_budget - planned)
                    facts.append(
                        {
                            "id": "unused_labels",
                            "requested": int(first_budget),
                            "planned": int(planned),
                            "n": unused,
                            "sentence": f"the plan places {int(planned)} of the "
                            f"{int(first_budget)} labels asked for; {unused} are "
                            "unused",
                            "simulated": True,
                        }
                    )
            forbidden = [{"id": "subset_labelling_sufficient"}]
            if result.get("design") != "random":
                forbidden.append({"id": "certify_from_nonrandom_design"})
            extra = {"facts": facts, "forbidden_claims": forbidden}
        elif record.name == "olmoearth_estimate_map_error":
            if result.get("design") != "random":
                extra = {"forbidden_claims": [{"id": "certify_from_nonrandom_design"}]}
        elif record.name == "olmoearth_certify_zone":
            forbidden = [{"id": "post_hoc_alpha"}]
            must = []
            if result.get("certified") is False:
                forbidden.append({"id": "rule_switch_after_failure"})
                must.append("No zone is certified at this alpha.")
            extra = {"forbidden_claims": forbidden, "must_state": must}
        elif record.name == "olmoearth_compare_results":
            forbidden = []
            if result.get("value_type") == "regression" and not result.get("threshold"):
                forbidden.append({"id": "error_rate_for_unthresholded_regression"})
            if result.get("statistics_left_out") or "different properties" in str(
                result.get("warning", "")
            ):
                forbidden.append({"id": "combined_statistic_across_properties"})
            extra = {"forbidden_claims": forbidden}
        for key, items in extra.items():
            if items:
                result[key] = items
        out.append(ToolRecord(record.name, record.arguments, envelope))
    return out


# --------------------------------------------------------------------------- the audit


def _norm(text: str) -> str:
    text = re.sub(r"[*_`]+", "", text.lower())
    text = text.replace("→", "->").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def quote_in(quote: str, sentence: str) -> bool:
    """Whether an audit quote is in a flagged sentence (or the other way round)."""
    q, s = _norm(quote), _norm(sentence)
    parts = [p.strip(" .,;:") for p in re.split(r"\.\.\.|…", q) if len(p.strip()) > 3]
    if parts and all(p[:80] in s for p in parts):
        return True
    if s and len(s) > 12 and s.strip(" .,;:|-") in q:
        return True
    qw, sw = set(re.findall(r"\w+", q)), set(re.findall(r"\w+", s))
    return bool(qw) and len(qw & sw) / len(qw) >= 0.8


def load_audit(path: Path) -> list[dict[str, Any]]:
    """Every finding of the audit, with its round and whether it was confirmed."""
    data = json.loads(path.read_text(encoding="utf-8"))

    def finding(v: dict[str, Any], cls: Any, status: str) -> dict[str, Any]:
        return {
            "round": AUDIT_ROUNDS[v["label"]],
            "configuration": v["configuration"],
            "run": str(v["run"]),
            "cls": cls,
            "quote": v["quote"],
            "status": status,
        }

    out = []
    for group in data["groups"]:
        out.extend(
            finding(
                v,
                v.get("cls_final") or v.get("cls"),
                "refuted" if v["verdict"] == "refuted" else "confirmed",
            )
            for v in group["verify"]["verdicts"]
        )
        out.extend(
            finding(v, v.get("cls"), "added by the verifier, unverified")
            for v in group["verify"].get("missed", [])
        )
    return out


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    """Run the checks over every recorded run; print the report, write the JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--rounds", type=Path, required=True, help="the trial's rounds directory"
    )
    ap.add_argument(
        "--audit", type=Path, help="the blind audit of rounds 6 and 7 (JSON)"
    )
    ap.add_argument("--out", type=Path, help="where to write every flag as JSON")
    ap.add_argument(
        "--no-simulate",
        action="store_true",
        help="use the recorded results as they are, without the simulated contract",
    )
    args = ap.parse_args(argv)

    audit = load_audit(args.audit) if args.audit else []
    audited_rounds = {f["round"] for f in audit}
    per_round: dict[str, Counter[str]] = defaultdict(Counter)
    answers: Counter[str] = Counter()
    flags: list[dict[str, Any]] = []
    for rnd, config, run, d in run_dirs(args.rounds):
        answer, records, brief = load_run(d)
        if not answer:
            continue
        answers[rnd] += 1
        if not args.no_simulate:
            records = adapt(d, records)
        evidence = RunEvidence(tools=records, user_messages=[brief], surface="cli")
        found = run_checks(answer, evidence)
        for name, violations in found.items():
            per_round[rnd][name] += 1
            flags.extend(
                {"round": rnd, "configuration": config, "run": run, **v}
                for v in violations
            )

    names = list(CHECKS)
    print("Answers flagged per check and round (an answer counts once per check)\n")
    print("| round | answers | " + " | ".join(names) + " | any |")
    print("|---|---|" + "---|" * (len(names) + 1))
    for rnd in sorted(answers, key=int):
        flagged_any = {
            (f["configuration"], f["run"]) for f in flags if f["round"] == rnd
        }
        cells = " | ".join(str(per_round[rnd][n]) for n in names)
        print(f"| {rnd} | {answers[rnd]} | {cells} | {len(flagged_any)} |")

    report: list[dict[str, Any]] = []
    if audit:
        print("\nFlags on the audited rounds, against the audit's findings\n")
        for f in flags:
            if f["round"] not in audited_rounds:
                continue
            same_run = [
                a
                for a in audit
                if (a["round"], a["configuration"], a["run"])
                == (f["round"], f["configuration"], f["run"])
            ]
            matches = [a for a in same_run if quote_in(a["quote"], f["text"])]
            # A confirmed finding first, then one the verifier added, then a refuted one.
            order = {
                "confirmed": 0,
                "added by the verifier, unverified": 1,
                "refuted": 2,
            }
            match = min(matches, key=lambda a: order[a["status"]]) if matches else None
            verdict = (
                "false alarm (no audit finding)"
                if match is None
                else (
                    "true flag"
                    if match["status"] == "confirmed"
                    else f"matches a finding {match['status']}"
                )
            )
            report.append({**f, "verdict": verdict, "audit": match})
            print(
                f"- r{f['round']} {f['configuration']}/{f['run']} [{f['check']}] "
                f"{verdict}{' (' + match['cls'] + ')' if match else ''}: "
                f"{f['text'][:160]!r} -- {f['detail'][:160]}"
            )
        print("\nPer check on the audited rounds\n")
        print(
            "| check | flags | true | matches an unverified or refuted finding | false alarms |"
        )
        print("|---|---|---|---|---|")
        for n in names:
            mine = [r for r in report if r["check"] == n]
            true = sum(r["verdict"] == "true flag" for r in mine)
            other = sum(r["verdict"].startswith("matches") for r in mine)
            print(
                f"| {n} | {len(mine)} | {true} | {other} | {len(mine) - true - other} |"
            )
        confirmed = [a for a in audit if a["status"] == "confirmed"]
        caught = [
            a
            for a in confirmed
            if any(r["verdict"] == "true flag" and r["audit"] is a for r in report)
        ]
        by_cls = Counter(a["cls"] for a in confirmed)
        caught_cls = Counter(a["cls"] for a in caught)
        print("\nConfirmed findings caught, by the audit's class\n")
        print("| class | confirmed | caught |")
        print("|---|---|---|")
        for cls in sorted(by_cls):
            print(f"| {cls} | {by_cls[cls]} | {caught_cls[cls]} |")
        print(f"| all | {len(confirmed)} | {len(caught)} |")
    if args.out:
        args.out.write_text(
            json.dumps({"flags": flags, "audited": report}, indent=1, default=str),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
