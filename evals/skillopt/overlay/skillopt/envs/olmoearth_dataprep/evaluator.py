"""Scoring for the OlmoEarth data-prep pitfall-diagnosis benchmark.

Each task describes a data-prep situation/error; the model must name which of
the 8 documented pitfalls it is (`pitfall_id`, 1-8) and the corrective
`action`. Ground truth is hand-authored from the skill's own pitfalls table.

``hard`` (0/1) = pitfall_id correct AND action family correct.
``soft`` (0-1) = mean of (pitfall_id, action family).
"""
from __future__ import annotations

import re
from typing import Any

from skillopt.utils import extract_json

# action family -> keywords that identify it in a free-text fix
_ACTION_KEYWORDS: dict[str, list[str]] = {
    "fix_schema": ["schema", "rename", "oe_labels", "es_label", "sample_category",
                   "field name", "field-name", "property name"],
    "real_aoi": ["watershed", "basin", "nldi", "huc", "wbd", "polygon", "real aoi", "catchment"],
    "per_metric_file": ["per metric", "per-metric", "one file per", "separate file",
                        "split metric", "one import per"],
    "emit_json": [".json", "json extension", "octet-stream", "mime", "both extensions"],
    "equal_frequency": ["equal-frequency", "equal frequency", "equal_frequency"],
    "spatial_split": ["spatial", "spatial-block", "spatial block", "block cv", "geographic split"],
    "shard": ["shard", "chunk", "split into", "10000", "10,000", "10k", "batch upload"],
    "negative_class": ["negative", "background", "other class", "non-target", "stable class"],
}


def _action_family(text: Any) -> str:
    s = str(text or "").lower()
    if not s:
        return ""
    for family, kws in _ACTION_KEYWORDS.items():
        if any(k in s for k in kws):
            return family
    return s.strip()[:24]


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    m = re.search(r"\d+", str(value or ""))
    return int(m.group()) if m else None


def evaluate(prediction_text: str, expected: dict) -> dict:
    """Score one predicted pitfall diagnosis against the oracle."""
    pred = extract_json(prediction_text or "") or {}
    parsed_ok = bool(pred)

    exp_id = _to_int(expected.get("pitfall_id"))
    exp_action = expected.get("action")  # already a family key in the dataset

    id_ok = _to_int(pred.get("pitfall_id")) == exp_id
    action_ok = _action_family(pred.get("action")) == exp_action

    checks = {"pitfall_id": id_ok, "action": action_ok}
    soft = sum(1 for v in checks.values() if v) / len(checks)
    hard = 1 if (id_ok and action_ok) else 0
    wrong = [k for k, v in checks.items() if not v]

    if hard:
        fail_reason = ""
    elif not parsed_ok:
        fail_reason = "no JSON diagnosis found in the response"
    else:
        fail_reason = (
            f"got pitfall_id={pred.get('pitfall_id')} action={pred.get('action')!r}; "
            f"expected {exp_id}/{exp_action} (wrong: {', '.join(wrong)})"
        )

    return {
        "hard": hard,
        "soft": round(soft, 4),
        "parsed_ok": parsed_ok,
        "predicted": pred,
        "field_results": {k: bool(v) for k, v in checks.items()},
        "fail_reason": fail_reason,
    }
