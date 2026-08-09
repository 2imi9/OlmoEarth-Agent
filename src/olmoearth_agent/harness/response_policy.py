# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Runtime response budgets for concise, actionable agent answers.

Prompt prose alone is not a reliable length control.  This policy adds a
run-specific response contract, bounds every agent-model completion, and tells
the harness when an overlong final answer needs one isolated editing pass.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal, cast

ResponseStyle = Literal["auto", "concise", "detailed"]
EffectiveStyle = Literal["concise", "detailed"]

_STYLE_ENV = "OLMOEARTH_RESPONSE_STYLE"
_MAX_WORDS_ENV = "OLMOEARTH_RESPONSE_MAX_WORDS"
_MAX_TOKENS_ENV = "OLMOEARTH_AGENT_MAX_OUTPUT_TOKENS"

_DETAILED_REQUEST = re.compile(
    r"\b(?:detailed|in detail|comprehensive|exhaustive|full report|"
    r"(?:complete|full) (?:list|table)|"
    r"full (?:output|configuration|config|code)|step[- ]by[- ]step|"
    r"explain thoroughly|show all)\b|"
    r"\b(?:list|show|return|include)\s+(?:me\s+)?(?:all|every)\b|"
    r"\b(?:write|generate|produce|show|return)\s+(?:the\s+|a\s+)?"
    r"(?:full\s+)?(?:code|script|yaml|json|configuration|config|notebook)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"\b[\w./:@+-]+\b")
_LIST_MARKER = re.compile(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+")
_REQUIRED_MARKER = re.compile(
    r"\b(?:ok|completed|failed|error|pending|queued|running|cancelled|"
    r"canceled|succeeded|warning|unknown)\b|"
    r"https?://[^\s)>\]]+|"
    r"(?:[\w.-]+[\\/])+[^\s`]+|"
    r"\b(?=[A-Za-z0-9_.:-]*[A-Za-z])(?=[A-Za-z0-9_.:-]*\d)"
    r"[A-Za-z0-9_.:-]{3,}\b|"
    r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?",
    re.IGNORECASE,
)


def _bounded_env_int(name: str, default: int, low: int, high: int) -> int:
    """Read one integer env setting, falling back and clamping safely."""
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value))


@dataclass(frozen=True)
class ResponsePolicy:
    """Concision contract and token limits for one lead agent.

    ``auto`` is concise unless the user's brief explicitly asks for a detailed
    treatment.  ``concise`` and ``detailed`` let an operator pin the behavior.
    The larger detailed values are derived from the concise configuration so
    the public configuration surface stays small.
    """

    style: ResponseStyle = "auto"
    concise_max_words: int = 120
    concise_max_tokens: int = 4096
    repair_max_tokens: int = 768

    @classmethod
    def from_env(cls) -> ResponsePolicy:
        """Build the policy from optional ``OLMOEARTH_RESPONSE_*`` settings."""
        raw_style = os.environ.get(_STYLE_ENV, "auto").strip().lower()
        style = (
            cast(ResponseStyle, raw_style)
            if raw_style in {"auto", "concise", "detailed"}
            else "auto"
        )
        return cls(
            style=style,
            concise_max_words=_bounded_env_int(_MAX_WORDS_ENV, 120, 40, 1000),
            concise_max_tokens=_bounded_env_int(_MAX_TOKENS_ENV, 4096, 512, 32768),
        )

    def effective_style(self, brief: str) -> EffectiveStyle:
        """Resolve ``auto`` from the user's explicit request for detail."""
        if self.style == "detailed":
            return "detailed"
        if self.style == "auto" and _DETAILED_REQUEST.search(brief):
            return "detailed"
        return "concise"

    def max_words(self, brief: str) -> int:
        """Maximum requested final-answer words for this brief."""
        if self.effective_style(brief) == "detailed":
            return max(400, self.concise_max_words * 4)
        return self.concise_max_words

    def max_tokens(self, brief: str) -> int:
        """Per-model-call output ceiling, including reasoning and tool calls."""
        if self.effective_style(brief) == "detailed":
            return min(32768, max(8192, self.concise_max_tokens * 2))
        return self.concise_max_tokens

    def contract(self, brief: str) -> str:
        """Return the run-specific system-prompt response contract."""
        limit = self.max_words(brief)
        if self.effective_style(brief) == "detailed":
            return (
                "RESPONSE CONTRACT (detailed, because the user requested it):\n"
                f"- Stay under {limit} words unless correctness requires code or data.\n"
                "- Lead with the answer, then organize only the evidence and steps needed.\n"
                "- Preserve exact ids, statuses, warnings, paths, and next actions.\n"
                "- Do not recap the conversation or narrate tool calls."
            )
        return (
            "RESPONSE CONTRACT (concise default):\n"
            f"- Stay under {limit} words. Answer first, with no preamble.\n"
            "- Use one short paragraph plus at most 5 bullets, OR one table with "
            "at most 5 data rows.\n"
            "- Include only the outcome, necessary ids/paths, one critical caveat, "
            "and the next action when one exists.\n"
            "- Do not recap the brief, narrate tool calls, or add a generic offer "
            "to help further."
        )

    def word_count(self, content: str) -> int:
        """Count prose/code tokens approximately for deterministic gating."""
        return len(_WORD.findall(content))

    def required_markers(self, content: str) -> set[str]:
        """Extract literals an editorial rewrite must not lose.

        List ordinals are removed before extraction; identifiers, paths, URLs,
        and numeric values remain. Numeric percent/comma formatting is
        normalized so harmless ``7 percent`` / ``7%`` changes still pass.
        """
        without_ordinals = _LIST_MARKER.sub("", content)
        markers: set[str] = set()
        for raw in _REQUIRED_MARKER.findall(without_ordinals):
            marker = raw.rstrip(".,;:").lower().replace(",", "").removesuffix("%")
            markers.add(marker)
        return markers

    def preserves_required_markers(self, draft: str, candidate: str) -> bool:
        """Whether a rewrite retains literals without inventing new ones."""
        return self.required_markers(draft) == self.required_markers(candidate)

    def needs_compaction(self, brief: str, content: str | None) -> bool:
        """Whether a concise final answer exceeded its requested word budget."""
        return bool(
            content
            and "```" not in content
            and self.effective_style(brief) == "concise"
            and self.word_count(content) > self.max_words(brief)
        )

    def repair_prompt(self, draft: str, brief: str) -> str:
        """Prompt for a no-tools, lossless rewrite of an overlong final answer."""
        return (
            f"Rewrite the draft below in at most {self.max_words(brief)} words. "
            "Preserve every actionable identifier, status, number, warning, file "
            "path, and URL. Do not add facts or actions. Lead with the outcome; "
            "use at most five bullets or five table rows. Return only the rewrite.\n\n"
            f"DRAFT:\n{draft}"
        )
