# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Model routing: send simple lookups to the small local model.

Shippy's roadmap: "Not every question needs a frontier model" — route simple
lookups to smaller, faster ones and save the full-weight model for complex
investigations. Our deployment always has the free local model and may have a
BYO-key hosted backend; when the user opts in (webui "auto-route" toggle →
``X-LLM-Route: auto`` header), :func:`classify_brief` decides per request
whether the brief is a *simple lookup* the local model handles fine, and the
bridge then demotes the run from the hosted backend to the local one. Routing
only ever demotes hosted→local (never invents a hosted client — that needs
the user's key), so a wrong "simple" verdict costs quality on one lookup,
never money.

Deterministic and dumb on purpose: a word-list heuristic is auditable,
testable, and free — an LLM-based router would spend the tokens it is trying
to save. Tuned conservative: anything that smells like mutation, analysis, or
a multi-step workflow is "complex".
"""

from __future__ import annotations

import re

#: Unmistakable mutation / analysis / workflow verbs. Verb-form bounded on
#: purpose: ``train\b`` flags "train a classifier" but not the status noun in
#: "is it finished training"; a bare noun like "prediction" or "run" never
#: flags on its own — a *lookup about* a prediction is still a lookup.
_MUTATION_MARKERS = re.compile(
    r"\b("
    r"create|configure|set\s*up|build|train|fine[- ]?tune|"
    r"submit|launch|predict|"
    r"compare|analy[sz]e|evaluate|assess|investigate|"
    r"detect|classify|segment|"
    r"export|download|generate|"
    r"draw|delete|remove|update|rename|modify"
    r")\b",
    re.IGNORECASE,
)

#: Openers typical of a bounded read-only lookup. A brief must BOTH open like
#: a lookup AND carry no mutation verb to route local.
_SIMPLE_OPENERS = re.compile(
    r"^\s*(list|show|what|which|who|how\s+many|when|where|is|are|do(es)?|"
    r"get|give\s+me|status|check)\b",
    re.IGNORECASE,
)

#: Above this the brief is a paragraph of intent, not a lookup.
_MAX_SIMPLE_WORDS = 24


def classify_brief(brief: str, *, forced_skill: str = "") -> str:
    """Classify one brief as ``"simple"`` or ``"complex"``.

    ``simple`` requires ALL of: no user-forced skill (a forced skill means
    the user wants a full workflow), a short brief, at most two sentences, a
    lookup opener ("list...", "what's...", "check..."), and no mutation /
    analysis verb anywhere. Everything else is ``complex``. Ties break toward
    ``complex``: the cost of over-routing to the strong model is money, but
    the cost of under-routing is a wrong answer, and correctness wins.
    """
    if forced_skill:
        return "complex"
    text = brief.strip()
    if not text:
        return "complex"
    if len(text.split()) > _MAX_SIMPLE_WORDS:
        return "complex"
    # More than two sentences usually means task framing plus instructions.
    if len([s for s in re.split(r"[.?!;\n]+", text) if s.strip()]) > 2:
        return "complex"
    if _MUTATION_MARKERS.search(text):
        return "complex"
    if _SIMPLE_OPENERS.match(text):
        return "simple"
    return "complex"
