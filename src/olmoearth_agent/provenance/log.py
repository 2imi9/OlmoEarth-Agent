# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Append-only provenance log for an agent run.

One :class:`~olmoearth_agent.types.ProvenanceManifest` entry per tool
call. Records the tool name, a hash of the arguments, and an
id-only summary of the result (no raw geometry, per operational rule
§3.1). Emits a JSON manifest and a runnable replay script.

The verification tools' calls (:data:`VERIFICATION_TOOLS`: the label plan,
the error estimate, the certified zone, and the certificate written and
checked, through olmoearth-inferencex) are also kept whole, as
:class:`VerificationCall` (``ProvenanceLog.verification``): the stakeholder
report (skill #6) states their numbers as the tools returned them, and the
provenance summary (skill #5) lists each call's design and seed, from which a
reader can draw the same sample again (a certificate's seed is the one its
plan took from the map).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from olmoearth_agent.types import ProvenanceManifest

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from olmoearth_agent.security.egress import EgressDecision

# Keys we lift out of a tool result into the manifest's typed fields.
_ID_KEYS = ("prediction_id", "id", "model_id")

#: The verification tools, whose calls the log keeps whole (with their
#: result): the plan of which windows to label, the error estimate (and the
#: which-map grading), the certified zone, and the certificate a certified
#: zone is packed into and its check.
VERIFICATION_TOOLS = (
    "olmoearth_plan_label_sample",
    "olmoearth_estimate_map_error",
    "olmoearth_certify_zone",
    "olmoearth_write_certificate",
    "olmoearth_verify_certificate",
)

#: What repeats a verification call's sample, in this order: the design, its
#: budget and seed (and whether the seed was taken from the map), the maps it
#: reads, the labels it grades and the zone's settings. Inline scores and
#: labels are counted, never copied.
_REPEAT_KEYS = (
    "design",
    "budget",
    "seed",
    "seed_from_map",
    "result_id",
    "result_id_b",
    "scores_path",
    "scores_path_b",
    "property_name",
    "threshold",
    "grid",
    "min_per_class",
    "date_a",
    "date_b",
    "design_path",
    "labels_path",
    "alpha",
    "delta",
    "rule",
    "date",
    "certificate_path",
    "map_path",
    "window",
    "verified",
)
#: Inline arrays a verification call may take: given by their length only.
_INLINE_KEYS = ("scores", "wrong", "reference", "window_indices")
#: The inline labels among them, also given by a digest of their values, so a
#: reader sees when one design file was graded with other labels.
_INLINE_LABEL_KEYS = ("wrong", "reference")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def labels_digest(values: list[Any]) -> str:
    """The first 12 hex digits of the sha256 of inline labels, as JSON.

    Examples
    --------
    >>> labels_digest([0, 1, 0]) == labels_digest([0, 1, 0])
    True
    >>> labels_digest([0, 1, 0]) == labels_digest([0, 0, 1])
    False
    """
    blob = json.dumps(values, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _hash_args(arguments: dict[str, Any]) -> str:
    """Stable sha256 of the call arguments (sorted keys)."""
    blob = json.dumps(arguments, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Reduce a tool result to ids / status / counts: never raw geometry."""
    payload = result.get("result", result)
    if not isinstance(payload, dict):
        return {"ok": result.get("ok")}
    keep = {}
    for key in ("id", "name", "status", "total", "project_count", "model_id"):
        if key in payload:
            keep[key] = payload[key]
    keep["ok"] = result.get("ok")
    return keep


@dataclass(frozen=True)
class VerificationCall:
    """One call of a verification tool, kept whole for the report and the summary.

    ``arguments`` are the call's arguments as dispatched; ``result`` is the
    tool's own output (the registry envelope's ``result``) when the call
    succeeded, else ``None``; ``request_hash`` is its manifest entry's.
    """

    tool: str
    timestamp: str
    request_hash: str
    arguments: dict[str, Any] = field(default_factory=dict)
    ok: bool = False
    result: dict[str, Any] | None = None

    def _value(self, key: str) -> Any:
        """``key`` from the result, else from the arguments, else ``None``."""
        if isinstance(self.result, Mapping) and self.result.get(key) is not None:
            return self.result[key]
        return self.arguments.get(key)

    @property
    def seed(self) -> int | None:
        """The seed of the sample this call drew or read, as the tool returned it.

        A plan returns its own; an estimate or a zone returns the seed of the
        design file it read. ``None`` when neither the result nor the
        arguments give one (windows labelled without a design).
        """
        seed = self._value("seed")
        if isinstance(seed, int) and not isinstance(seed, bool):
            return seed
        return None

    @property
    def design(self) -> str | None:
        """The sample's design, by the plan's name for it (``random``, ``class``...)."""
        if isinstance(self.result, Mapping) and self.result.get("plan_design"):
            return str(self.result["plan_design"])
        design = self._value("design")
        return None if design is None else str(design)

    def repeat(self) -> dict[str, Any]:
        """What repeats this call: the tool, whether it ran, design, seed, inputs.

        Returns
        -------
        dict
            ``tool``, ``ok``, then each of the design, budget, seed, maps,
            labels file and zone settings the call has (result first, then
            arguments), inline arrays as ``"inline, N rows"`` (inline labels
            also with their digest, ``"inline, N rows, sha256 <12 hex>"``:
            :func:`labels_digest`), and the result's ``n_labelled``.

        Examples
        --------
        >>> call = VerificationCall(
        ...     "olmoearth_plan_label_sample", "t", "h",
        ...     {"scores": [[0.1, 0.9]], "budget": 1, "seed": 4}, True,
        ...     {"design": "random", "seed": 4, "budget": 1},
        ... )
        >>> call.repeat()["seed"], call.repeat()["scores"]
        (4, 'inline, 1 rows')
        """
        out: dict[str, Any] = {"tool": self.tool, "ok": self.ok}
        for key in _REPEAT_KEYS:
            if key == "design":
                value: Any = self.design
            elif key == "seed":
                value = self.seed
            elif key == "seed_from_map":
                # The argument only: the plan's result holds a block of that
                # name. With it, the seed was taken from the map, and the plan
                # is drawn again with seed_from_map, not with the seed.
                value = self.arguments.get("seed_from_map") or None
            else:
                value = self._value(key)
            if value is not None:
                out[key] = value
        for key in _INLINE_KEYS:
            values = self.arguments.get(key)
            if not isinstance(values, list):
                continue
            out[key] = f"inline, {len(values)} rows"
            if key in _INLINE_LABEL_KEYS:
                out[key] += f", sha256 {labels_digest(values)}"
        labelled = self.result.get("n_labelled") if self.result else None
        if labelled is not None:
            out["n_labelled"] = labelled
        return out


class ProvenanceLog:
    """Collects manifest entries for one agent run."""

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or str(uuid.uuid4())
        self.entries: list[ProvenanceManifest] = []
        #: Outbound-endpoint decisions (host + capability + verdict), appended
        #: by the egress guard. Host only -- never a full URL -- so a key in a
        #: query string can never land here. See ``security/egress.py``.
        self.egress: list[dict[str, Any]] = []
        #: The verification tools' calls, whole and in order
        #: (:data:`VERIFICATION_TOOLS`), for the report and the summary.
        self.verification: list[VerificationCall] = []

    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> ProvenanceManifest:
        """Append one entry for a dispatched tool call.

        A verification tool's call is also kept whole
        (:attr:`verification`): its arguments, and its result when it
        succeeded.
        """
        payload = result.get("result", {})
        prediction_id = None
        model_id = None
        if isinstance(payload, dict):
            prediction_id = payload.get("prediction_id") or (
                payload.get("id") if tool_name.endswith("prediction") else None
            )
            model_id = payload.get("model_id")
        entry = ProvenanceManifest(
            run_id=self.run_id,
            timestamp=_utc_now(),
            api_call=tool_name,
            request_hash=_hash_args(arguments),
            response_summary=_summarize_result(result),
            prediction_id=prediction_id,
            model_id=model_id,
        )
        self.entries.append(entry)
        if tool_name in VERIFICATION_TOOLS:
            ok = bool(result.get("ok")) and isinstance(payload, dict)
            self.verification.append(
                VerificationCall(
                    tool=tool_name,
                    timestamp=entry.timestamp,
                    request_hash=entry.request_hash,
                    arguments=dict(arguments),
                    ok=ok,
                    result=payload if ok else None,
                )
            )
        return entry

    def verification_calls(self) -> list[dict[str, Any]]:
        """Each verification call with what repeats it (``VerificationCall.repeat``).

        Returns
        -------
        list of dict
            In call order: the tool, whether it ran, its design, budget and
            seed, the maps and labels it read, and the zone's settings.
        """
        return [call.repeat() for call in self.verification]

    def record_egress(self, decision: "EgressDecision") -> dict[str, Any]:
        """Append one outbound-endpoint decision to the audit trail.

        Records the capability, the bare host, the allow/deny verdict, and the
        reason -- enough to answer "what external hosts did this run contact,
        and were any off the allowlist?" without storing the URL itself.
        """
        record = {
            "timestamp": _utc_now(),
            "capability": decision.capability,
            "host": decision.host,
            "allowed": decision.allowed,
            "reason": decision.reason,
        }
        self.egress.append(record)
        return record

    def to_dict(self) -> dict[str, Any]:
        """Full manifest as a JSON-able dict."""
        return {
            "run_id": self.run_id,
            "entry_count": len(self.entries),
            "entries": [asdict(e) for e in self.entries],
            "egress_count": len(self.egress),
            "egress": list(self.egress),
            "verification": self.verification_calls(),
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Full manifest as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def replay_script(self) -> str:
        """Emit a runnable Python script that re-issues the recorded calls.

        The script re-dispatches each recorded tool call (by name + the
        original arguments are NOT stored verbatim (only their hash) so
        the script is a skeleton that documents the sequence and leaves
        argument reconstruction to the operator). This is enough to audit
        *what* the run did and in what order. The verification calls follow,
        each with its design and seed, so the same sample can be drawn again.
        """
        lines = [
            "# Auto-generated replay skeleton: OlmoEarth Agent provenance",
            f"# run_id: {self.run_id}",
            f"# {len(self.entries)} tool call(s)",
            "#",
            "# Each step lists the tool, the argument hash, and the result",
            "# summary. Re-run by dispatching these tools in order through a",
            "# fresh ToolRegistry (see olmoearth_agent.harness.LeadAgent).",
            "",
        ]
        for i, e in enumerate(self.entries, 1):
            ids = e.prediction_id or e.model_id or "-"
            lines.append(
                f"# {i:>2}. {_one_line(e.api_call)}  "
                f"args_sha256={e.request_hash[:12]}  "
                f"result={e.response_summary}  ids={_one_line(ids)}"
            )
        if self.verification:
            lines += [
                "",
                "# Verification calls: the same design, budget and seed (and the",
                "# same inputs) draw the same sample again; a plan with",
                "# seed_from_map=True took its seed from the map, and is drawn",
                "# again with seed_from_map and its alpha, delta and rule.",
            ]
            for i, call in enumerate(self.verification_calls(), 1):
                settings = "  ".join(
                    f"{k}={_one_line(v)}"
                    for k, v in call.items()
                    if k not in ("tool", "ok")
                )
                state = "" if call["ok"] else "  (failed)"
                lines.append(f"# {i:>2}. {_one_line(call['tool'])}  {settings}{state}")
        return "\n".join(lines) + "\n"


def _one_line(value: Any) -> str:
    """``value`` as one comment line of the replay script.

    A value given as a call's argument (a date, a path) can hold a line
    break, which would end the ``#`` comment and leave the rest of the value
    as code in a script that says it is runnable. Such a value is written as
    a JSON string instead, its breaks escaped (``ensure_ascii`` escapes the
    Unicode line separators too); any other value is written as it is.

    Examples
    --------
    >>> _one_line("2024-06-01")
    '2024-06-01'
    >>> _one_line("x\\nimport os")
    '"x\\\\nimport os"'
    """
    text = str(value)
    if not text or text.splitlines() == [text]:
        return text
    return json.dumps(text)
