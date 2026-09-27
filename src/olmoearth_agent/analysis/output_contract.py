# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The limits of the shared output contract, and the helpers that keep a result inside them.

A tool result may carry three keys the harness checks an answer against:
``facts`` (sentences computed by code), ``must_state`` (limits the answer
must convey) and ``forbidden_claims`` (``{"id", "why"}``, claims it must not
make). Two limits hold for every result:

- ``must_state`` holds at most :data:`MUST_STATE_MAX` sentences of at most
  :data:`MUST_STATE_MAX_WORDS` words each, each stating a limit. Long scope
  detail stays in ``evidence_scope`` or a file; a sentence never tells the
  agent to do what no agent tool can (exp87 review: "pass form='top1'"
  reached ``must_state`` though no agent tool takes a ``form``).
- a ``forbidden_claims`` id appears once per result, however many parts of a
  tool add it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

#: Most sentences one result's ``must_state`` holds.
MUST_STATE_MAX = 3
#: Most words (split on whitespace) one ``must_state`` sentence holds.
MUST_STATE_MAX_WORDS = 25


def word_count(sentence: str) -> int:
    """Words in a sentence, split on whitespace (the contract's measure)."""
    return len(sentence.split())


def add_must_state(out: dict[str, Any], sentences: Iterable[str]) -> None:
    """Append sentences to ``out['must_state']``: once each, at most :data:`MUST_STATE_MAX`.

    Every sentence is written by code, so one over :data:`MUST_STATE_MAX_WORDS`
    words is a programming error and raises. A sentence past the cap is left
    out and logged: the cap is the contract's, and the first sentences are
    the ones a tool adds first (the evidence scope, then the case's own).

    Raises
    ------
    ValueError
        For a sentence longer than :data:`MUST_STATE_MAX_WORDS` words.
    """
    must = out.setdefault("must_state", [])
    for sentence in sentences:
        n = word_count(sentence)
        if n > MUST_STATE_MAX_WORDS:
            msg = (
                f"a must_state sentence has {n} words, more than "
                f"{MUST_STATE_MAX_WORDS}: {sentence!r}"
            )
            raise ValueError(msg)
        if sentence in must:
            continue
        if len(must) >= MUST_STATE_MAX:
            logger.warning(
                "must_state already holds %d sentences; left out: %s",
                MUST_STATE_MAX,
                sentence,
            )
            continue
        must.append(sentence)


def add_forbidden(out: dict[str, Any], claims: Iterable[Mapping[str, str]]) -> None:
    """Append claims to ``out['forbidden_claims']``, each id once."""
    listed = out.setdefault("forbidden_claims", [])
    seen = {c.get("id") for c in listed}
    for claim in claims:
        if claim["id"] in seen:
            continue
        listed.append(dict(claim))
        seen.add(claim["id"])
