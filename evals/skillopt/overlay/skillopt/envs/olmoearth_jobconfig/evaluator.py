"""Scoring for the OlmoEarth Studio job-config benchmark.

The target model, guided by the skill, must turn a plain-English EO task into
a filled Studio "new job" wizard config. We score that config against the
verified preset (the skill's own ``recommend.py`` oracle).

``hard`` (0/1) = all five core wizard decisions correct:
    output_type, foundation_model, time_frame.mode, patch_size_m,
    imagery_sources (as a set).
``soft`` (0-1) = fraction of all checked fields correct, including the
    mode-specific time sub-fields — a smoother signal so the optimizer still
    sees progress when only some fields are right.
"""
from __future__ import annotations

from typing import Any

from skillopt.utils import extract_json

CORE_FIELDS = (
    "output_type",
    "foundation_model",
    "time_frame_mode",
    "patch_size_m",
    "imagery_sources",
)


def _norm_str(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm_sources(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    return tuple(sorted(_norm_str(s) for s in (value or []) if _norm_str(s)))


def _time_subfield_checks(pred_tf: dict, exp_tf: dict, mode: str) -> list[tuple[str, bool]]:
    """Mode-specific time_frame sub-field checks (partial-credit signal)."""
    if mode == "period":
        return [
            ("period_months", _to_int(pred_tf.get("period_months")) == _to_int(exp_tf.get("period_months"))),
            ("start_months", set(pred_tf.get("start_months") or []) == set(exp_tf.get("start_months") or [])),
        ]
    if mode == "single_moment_with_context":
        return [
            ("before_months", (_to_int(pred_tf.get("before_months")) or 0) == (_to_int(exp_tf.get("before_months")) or 0)),
            ("after_months", (_to_int(pred_tf.get("after_months")) or 0) == (_to_int(exp_tf.get("after_months")) or 0)),
        ]
    if mode == "single_moment":
        return [
            ("observation_window_hours",
             _to_num(pred_tf.get("observation_window_hours")) == _to_num(exp_tf.get("observation_window_hours"))),
        ]
    return []


def evaluate(prediction_text: str, expected: dict) -> dict:
    """Score one predicted wizard config against the expected preset."""
    pred = extract_json(prediction_text or "") or {}
    parsed_ok = bool(pred)

    exp_tf = expected.get("time_frame") or {}
    pred_tf = pred.get("time_frame") or {}
    if not isinstance(pred_tf, dict):
        pred_tf = {}
    exp_mode = _norm_str(exp_tf.get("mode"))
    pred_mode = _norm_str(pred_tf.get("mode"))

    core = {
        "output_type": _norm_str(pred.get("output_type")) == _norm_str(expected.get("output_type")),
        "foundation_model": _norm_str(pred.get("foundation_model")) == _norm_str(expected.get("foundation_model")),
        "time_frame_mode": pred_mode == exp_mode and bool(exp_mode),
        "patch_size_m": _to_int(pred.get("patch_size_m")) == _to_int(expected.get("patch_size_m")),
        "imagery_sources": _norm_sources(pred.get("imagery_sources")) == _norm_sources(expected.get("imagery_sources")),
    }

    # Time sub-fields only count when the expected mode is known. They are
    # correct only if the predicted mode matches AND the sub-values match.
    sub: list[tuple[str, bool]] = []
    if exp_mode:
        if pred_mode == exp_mode:
            sub = _time_subfield_checks(pred_tf, exp_tf, exp_mode)
        else:
            sub = [(name, False) for name, _ in _time_subfield_checks(exp_tf, exp_tf, exp_mode)]

    all_checks = list(core.items()) + sub
    soft = sum(1 for _, ok in all_checks if ok) / len(all_checks) if all_checks else 0.0
    hard = 1 if all(core.values()) else 0
    wrong = [name for name, ok in all_checks if not ok]

    if hard:
        fail_reason = ""
    elif not parsed_ok:
        fail_reason = "no JSON wizard config found in the response"
    else:
        fail_reason = "incorrect fields: " + ", ".join(wrong)

    return {
        "hard": hard,
        "soft": round(soft, 4),
        "parsed_ok": parsed_ok,
        "predicted": pred,
        "field_results": {**{k: bool(v) for k, v in core.items()}, **{n: bool(ok) for n, ok in sub}},
        "fail_reason": fail_reason,
    }
