"""Scoring for the OlmoEarth embeddings-vs-fine-tune benchmark.

The target model, guided by the skill, reads a task's (samples, classes,
compute, goal) and returns a JSON recommendation. We score it against the
skill's own ``recommend.py`` oracle (`decide()`).

``hard`` (0/1) = the two decisive outputs both correct: `decision` and `model`.
``soft`` (0-1) = mean of decision, model, and (when applicable) classifier
    family — a smoother gradient.
"""
from __future__ import annotations

from typing import Any

from skillopt.utils import extract_json

_DECISIONS = {
    "embeddings": "embeddings",
    "fine_tune": "fine_tune",
    "finetune": "fine_tune",
    "fine_tuning": "fine_tune",
    "embeddings_then_fine_tune": "embeddings_then_fine_tune",
    "embeddings_then_finetune": "embeddings_then_fine_tune",
}


def _norm_decision(value: Any) -> str:
    s = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return _DECISIONS.get(s, s)


def _norm_model(value: Any) -> str:
    return str(value or "").strip().lower()


def _classifier_family(value: Any) -> str:
    """Map a free-form classifier string to a coarse family."""
    s = str(value or "").strip().lower()
    if not s or s in {"none", "n/a", "null"}:
        return "none"
    if "knn" in s or "k-nn" in s or "nearest" in s:
        return "knn"
    if "linear" in s or "logistic" in s or "lp" == s or "linear_probe" in s:
        return "linear_probe"
    if "mlp" in s or "head" in s:
        return "mlp"
    if "means" in s or "cluster" in s or "hdbscan" in s or "dbscan" in s:
        return "clustering"
    return s


def evaluate(prediction_text: str, expected: dict) -> dict:
    """Score one predicted embeddings recommendation against the oracle."""
    pred = extract_json(prediction_text or "") or {}
    parsed_ok = bool(pred)

    exp_decision = _norm_decision(expected.get("decision"))
    exp_model = _norm_model(expected.get("model"))
    exp_clf = _classifier_family(expected.get("classifier"))

    checks: dict[str, bool] = {
        "decision": _norm_decision(pred.get("decision")) == exp_decision,
        "model": _norm_model(pred.get("model")) == exp_model,
    }
    # Classifier only counts when the oracle specifies one (fine_tune has none).
    if exp_clf != "none":
        checks["classifier"] = _classifier_family(pred.get("classifier")) == exp_clf

    soft = sum(1 for v in checks.values() if v) / len(checks) if checks else 0.0
    hard = 1 if (checks["decision"] and checks["model"]) else 0
    wrong = [k for k, v in checks.items() if not v]

    if hard:
        fail_reason = ""
    elif not parsed_ok:
        fail_reason = "no JSON recommendation found in the response"
    else:
        fail_reason = "incorrect fields: " + ", ".join(wrong)

    return {
        "hard": hard,
        "soft": round(soft, 4),
        "parsed_ok": parsed_ok,
        "predicted": pred,
        "field_results": {k: bool(v) for k, v in checks.items()},
        "fail_reason": fail_reason,
    }
