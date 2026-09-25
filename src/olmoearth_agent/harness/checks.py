# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Checks of the final answer against the run, before it is shown.

Each check takes the answer and the run's evidence (:class:`RunEvidence`: the
tool calls with their full results, the user's messages and the surface the
answer is shown on) and returns its violations, each a dict
``{"check": name, "text": sentence, "detail": why}``. The harness runs all of
them on the final answer, asks once for a rewrite whose note lists every
violation, runs them again, and marks each sentence still flagged with
``[unverified: <check>]`` (see ``LeadAgent.run_stream``).

The checks read what the tools computed; they never compute a finding
themselves. Besides numbers, they read three keys a tool's result may carry
(the contract between the tools and the harness):

- ``facts``: ``[{"id", "sentence", ...fields}]``, a fact the answer may
  repeat. The direction check knows ``dominant_change`` (``from_class``,
  ``to_class``, ``n``, ``share``; map A's class to map B's),
  ``more_confident_side`` (``side`` "A" or "B", ``share``),
  ``concentration`` (``where``, ``share``, ``top_band_share``, and optionally
  ``bottom_band_share``, ``left_band_share``, ``right_band_share``, ``grid``
  and ``n_differing``), ``margin_ratio`` (``low``, ``high``, ``versus``) and
  ``unused_labels`` (``n``, ``requested``, ``planned``). Other ids are shown,
  never checked.
- ``must_state``: short sentences the answer must convey whenever it reports
  that result.
- ``forbidden_claims``: ``[{"id", "why"}]``, claims the answer must not make
  about that result; :data:`FORBIDDEN_DETECTORS` holds a detector per id.

The detectors are regular expressions over one sentence at a time, and they
are conservative on purpose: a false alarm costs a rewrite and, when it
persists, a marked sentence, so a sentence that negates, hedges or offers an
example is left alone, and a rule fires only on the wording exp86's audits
found. What a check cannot read (a paraphrase, a claim spread over two
sentences) it misses; ``scripts/validate_answer_checks.py`` measures both
kinds of error on exp86's recorded answers.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property, partial
from typing import Any

from olmoearth_agent.harness.grounding import NumberPool, tokenize

logger = logging.getLogger(__name__)

#: One violation: ``{"check": name, "text": sentence, "detail": why}``.
Violation = dict[str, Any]
#: A check: the answer and the run's evidence in, its violations out.
Check = Callable[[str, "RunEvidence"], list[Violation]]

#: Where the answer is shown. On ``"web"`` the tool results are on screen
#: beside it, so "listed above" may point at them; on ``"cli"`` only the
#: answer's text is printed.
SURFACES = ("cli", "web")

NUMBERS = "numbers"
DIRECTION = "direction"
ACTIONS = "actions"
FORBIDDEN = "forbidden_claims"
MUST_STATE = "must_state"

#: The marker appended to a sentence a check still flags after the rewrite.
MARK = "[unverified: {check}]"

#: A concentration claim about a band contradicts the facts when that band
#: holds less than this share of the differing windows.
BAND_SHARE_MAX = 0.10
#: A magnitude is outside a ``margin_ratio`` fact's range when it is more
#: than this factor beyond either end.
RATIO_SLACK = 1.5
#: The share of a ``must_state`` sentence's key terms the answer must hold.
MUST_STATE_MIN_OVERLAP = 0.5


# --------------------------------------------------------------------------
# The run's evidence


@dataclass(frozen=True)
class ToolRecord:
    """One dispatched tool call: its name, arguments and full result envelope."""

    name: str
    arguments: Mapping[str, Any]
    envelope: Any

    @property
    def ok(self) -> bool:
        """Whether the tool ran without error."""
        return isinstance(self.envelope, Mapping) and bool(self.envelope.get("ok"))

    @property
    def result(self) -> Any:
        """The tool's own output inside the registry's ``{"ok", "result"}``."""
        if isinstance(self.envelope, Mapping) and "result" in self.envelope:
            return self.envelope["result"]
        return self.envelope


@dataclass
class RunEvidence:
    """What the answer may rest on: the run's tool calls and the user's words.

    ``user_messages`` are the brief and the user's turns of the history,
    never the system prompt, the saved preferences or an assistant message.
    """

    tools: list[ToolRecord] = field(default_factory=list)
    user_messages: list[str] = field(default_factory=list)
    surface: str = "cli"

    @cached_property
    def pool(self) -> NumberPool:
        """The numbers of the tool results and the user's messages."""
        return NumberPool([*(t.envelope for t in self.tools), *self.user_messages])

    def contract(self, key: str) -> Iterator[tuple[ToolRecord, Any]]:
        """Every item of the list a successful tool result holds under ``key``."""
        for record in self.tools:
            result = record.result
            if not record.ok or not isinstance(result, Mapping):
                continue
            items = result.get(key)
            if isinstance(items, list):
                for item in items:
                    yield record, item

    def facts(self, fact_id: str) -> list[tuple[ToolRecord, Mapping[str, Any]]]:
        """The run's facts with id ``fact_id``, with the tool that computed each."""
        return [
            (record, item)
            for record, item in self.contract("facts")
            if isinstance(item, Mapping) and item.get("id") == fact_id
        ]

    @cached_property
    def class_names(self) -> dict[str, str]:
        """Class id (as a string) to name, from any result's ``classes`` map."""
        names: dict[str, str] = {}
        for record in self.tools:
            result = record.result
            classes = result.get("classes") if isinstance(result, Mapping) else None
            if isinstance(classes, Mapping):
                for key, value in classes.items():
                    if isinstance(value, str) and str(key).isdigit():
                        names.setdefault(str(key), value)
        return names

    @cached_property
    def written_files(self) -> list[tuple[str, str]]:
        """``(key, path)`` of every file path a tool reported and was not given.

        A path in a tool's result that is not among its call's arguments is
        one the tool produced (``scores_path``, ``labels_csv_path``); a path
        it only echoes (``estimate``'s ``design_path``) is an input.
        """
        out: list[tuple[str, str]] = []
        for record in self.tools:
            if not record.ok:
                continue
            given = {value for _, value in _strings(record.arguments)}
            for key, value in _strings(record.result):
                if (
                    value not in given
                    and "://" not in value  # a URL (a tile, a page) is no file
                    and _FILE_RE.search(value)
                    and all(value != p for _, p in out)
                ):
                    out.append((key, value))
        return out


def _strings(obj: Any, key: str = "", depth: int = 0) -> Iterator[tuple[str, str]]:
    """``(key, string)`` for every string in ``obj``, with its nearest key."""
    if depth > 6:
        return
    if isinstance(obj, str):
        yield key, obj
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            yield from _strings(v, str(k), depth + 1)
    elif isinstance(obj, list | tuple):
        for v in obj:
            yield from _strings(v, key, depth + 1)


# --------------------------------------------------------------------------
# Sentences


@dataclass(frozen=True)
class Sentence:
    """A sentence of the answer: its span and its text (stripped)."""

    start: int
    end: int
    text: str


_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "eg.",
    "ie.",
    "vs.",
    "etc.",
    "cf.",
    "approx.",
    "incl.",
    "no.",
    "fig.",
    "al.",
    "resp.",
    "min.",
    "max.",
)
_BREAK_RE = re.compile(r"(?<=[.!?;])\s+")
_TABLE_RULE_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$")


def sentences(text: str) -> list[Sentence]:
    """The answer's sentences: each line, split after ``.``, ``!``, ``?``, ``;``.

    A line is a bullet, a table row or a paragraph; a table's rule line
    (``|---|``) is none. A full stop splits only where no lower-case letter
    follows and no abbreviation (``e.g.``) precedes; a semicolon always does.
    """
    out: list[Sentence] = []
    for line in re.finditer(r"[^\n]+", text):
        body, base = line.group(), line.start()
        if not body.strip() or _TABLE_RULE_RE.match(body):
            continue
        cut = 0
        for m in _BREAK_RE.finditer(body):
            head = body[cut : m.start()]
            nxt = body[m.end() : m.end() + 1]
            if body[m.start() - 1] != ";" and (
                head.lower().endswith(_ABBREVIATIONS) or nxt.islower()
            ):
                continue
            out.extend(_sentence(text, base + cut, base + m.start()))
            cut = m.end()
        out.extend(_sentence(text, base + cut, base + len(body)))
    return out


def _sentence(text: str, start: int, end: int) -> list[Sentence]:
    raw = text[start:end]
    if not raw.strip():
        return []
    lead = len(raw) - len(raw.lstrip())
    trail = len(raw) - len(raw.rstrip())
    return [Sentence(start + lead, end - trail, raw.strip())]


def _plain(text: str) -> str:
    """The sentence without emphasis and code marks, keeping ``snake_case``."""
    text = re.sub(r"\*+|`+", "", text).replace(" ", " ")
    return re.sub(r"(?<![A-Za-z0-9])_+|_+(?![A-Za-z0-9])", "", text)


_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|none|nothing|cannot|can't|cant|don't|doesn't|didn't|"
    r"won't|wouldn't|shouldn't|isn't|aren't|wasn't|weren't|neither|nor|"
    r"without|avoid|instead\s+of|rather\s+than|refuses?|refused|invalid|"
    r"forbid\w*|unsupported|unrelated|(?:wrong|misleading|incorrect)\s+to)\b",
    re.I,
)
_EXAMPLE_RE = re.compile(
    r"\b(?:e\.g\.|eg\.|for\s+example|for\s+instance|such\s+as|including)", re.I
)


#: Where a clause ends: a semicolon, a colon, a bracket, a dash between
#: spaces, or a contrasting conjunction after a comma.
_CLAUSE_BREAK_RE = re.compile(
    r"[;:]|\s[-\u2013\u2014]\s|\u2014"
    r"|,\s*(?:but|although|though|while|whereas|yet|so|and\s+so)\b|\bbut\b",
    re.I,
)
_BRACKETS = {"(": ")", "[": "]"}


def _negated(text: str) -> bool:
    return bool(_NEGATION_RE.search(text))


def _brackets(s: str) -> list[tuple[int, int]]:
    """The spans ``(open, close)`` of the outermost brackets of ``s``."""
    spans: list[tuple[int, int]] = []
    stack: list[tuple[str, int]] = []
    for i, ch in enumerate(s):
        if ch in _BRACKETS:
            stack.append((ch, i))
        elif stack and ch == _BRACKETS[stack[-1][0]]:
            _, at = stack.pop()
            if not stack:
                spans.append((at, i))
    return spans


def _clause_span(s: str, start: int, end: int) -> tuple[int, int]:
    """The span of the clause of ``s`` around ``s[start:end]``.

    A bracket is a clause of its own: a claim inside one is bounded by it, and
    a claim outside one reads past it ("A looser alpha (e.g. 0.15) would
    certify ..., but ..." is one clause up to the "but").
    """
    lo, hi = 0, len(s)
    masked = s
    for a, b in _brackets(s):
        if a < start and end <= b + 1:
            lo, hi = a + 1, b  # the claim is inside this bracket
        elif not (a < start < b):
            masked = masked[:a] + " " * (b + 1 - a) + masked[b + 1 :]
    for m in _CLAUSE_BREAK_RE.finditer(masked, lo, start):
        lo = m.end()
    after = _CLAUSE_BREAK_RE.search(masked, end, hi)
    return lo, after.start() if after else hi


def _clause(s: str, start: int, end: int) -> str:
    """The clause of ``s`` around ``s[start:end]``, its other brackets blanked."""
    lo, hi = _clause_span(s, start, end)
    text = s[lo:hi]
    for a, b in _brackets(text):
        if not (a <= start - lo < b + 1):
            text = text[:a] + " " * (b + 1 - a) + text[b + 1 :]
    return text


def _negated_at(s: str, m: re.Match[str]) -> bool:
    """Whether the clause a claim is in negates it ("... is not saved").

    A hedge in another clause of a long sentence ("..., but it does not pick
    a winner") leaves the claim standing, and so does a negation inside the
    claim's own words (the class "not water").
    """
    lo, _ = _clause_span(s, m.start(), m.end())
    clause = _clause(s, m.start(), m.end())
    at, to = m.start() - lo, m.end() - lo
    return _negated(clause[:at] + " " + clause[to:])


def _example_before(s: str, start: int) -> bool:
    """Whether ``s[start:]`` is part of an example ("e.g. ...", "such as ...").

    An example runs from its cue to the end of the sentence or a semicolon,
    or to the end of the bracket the cue opens in.
    """
    cues = list(_EXAMPLE_RE.finditer(s, 0, start))
    if not cues:
        return False
    cue = cues[-1].start()
    if ";" in s[cue:start]:
        return False
    return not any(a < cue < b < start for a, b in _brackets(s))


# --------------------------------------------------------------------------
# numbers


def unsupported_of(violations: Iterable[Violation]) -> list[str]:
    """The numbers the number check's violations name, once each, in order."""
    out: list[str] = []
    for v in violations:
        if v.get("detail") not in out:
            out.append(v["detail"])
    return out


def check_numbers(answer: str, run: RunEvidence) -> list[Violation]:
    """Numbers no tool result and no user message supports (``grounding.py``).

    ``detail`` is the number as written; ``text`` the sentence it is in.
    """
    return [
        {"check": NUMBERS, "text": s.text, "detail": number}
        for s in sentences(answer)
        for number in run.pool.unsupported(s.text)
    ]


# --------------------------------------------------------------------------
# direction: the answer against the tools' facts

_ARROW = r"(?:-{1,2}>|→|⟶|⇒|=>|➔|➜|↦)"
_DOMINANT_RE = re.compile(
    r"\b(?:most|mostly|majority|dominant|dominat\w*|predominant\w*|main|mainly|"
    r"primar\w*|chiefly|largest|biggest|principal|bulk|overwhelming\w*|"
    r"usually|typically|generally)\b",
    re.I,
)
#: The stronger cue for naming a class pair the dominant one.
_DOMINANT_PAIR_RE = re.compile(
    r"\b(?:dominant|dominat\w*|predominant\w*|most\s+common|most\s+frequent|"
    r"largest|biggest|main|primary|principal)\b",
    re.I,
)
_UNDIRECTED = r"\s*(?:vs\.?|versus|↔|<->|<=>|/|\s+and\s+|\s+or\s+)\s*"


def _is_int_class(value: Any) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, str) and value.strip().isdigit()
    )


def _name_pattern(name: str, others: Sequence[str]) -> str:
    """A regex for a class name: its words joined by ``_``, ``-``, ``/`` or space.

    The first word alone stands for the name when no other class shares it
    (``grassland`` for ``grassland_barren``); a name that ends another name
    (``water`` of ``not water``) is not read inside it.
    """
    words = [w for w in re.split(r"[\s_\-/]+", name.strip()) if w]
    if not words:
        return r"(?!)"
    alts = [r"[\s_\-/]*".join(re.escape(w) for w in words)]
    first = words[0].lower()
    shared = any(
        re.split(r"[\s_\-/]+", o.strip())[0].lower() == first
        for o in others
        if o.strip().lower() != name.strip().lower()
    )
    if len(words) > 1 and len(first) >= 4 and not shared:
        alts.append(re.escape(words[0]))
    lookbehinds = "".join(
        rf"(?<!{re.escape(o[: len(o) - len(name)])})"
        for o in others
        if o.lower().endswith(name.lower()) and len(o) > len(name)
    )
    return rf"{lookbehinds}(?<![A-Za-z0-9])(?:{'|'.join(alts)})(?![A-Za-z0-9])"


def _class_pattern(value: Any, names: Mapping[str, str]) -> str:
    """A regex for a class as an answer writes it: its name, or ``class 1``."""
    alts: list[str] = []
    others = list(names.values())
    if _is_int_class(value):
        c = int(value)
        # Not inside a longer number: "0.5", "10", "1,570".
        alts.append(
            rf"(?:class(?:es)?\s+)?(?:\b[AB]\s*[=:]\s*)?(?<![\w.,]){c}"
            r"(?![\w%]|[.,]\d)"
        )
        name = names.get(str(c))
        if name:
            alts.append(_name_pattern(name, others))
    elif isinstance(value, str) and value.strip():
        alts.append(_name_pattern(value, others))
    return "(?:" + "|".join(alts) + ")" if alts else r"(?!)"


def _directed(x: str, y: str) -> re.Pattern[str]:
    """A change from ``x`` to ``y`` written with an arrow or "from ... to"."""
    return re.compile(
        rf"{x}\W{{0,3}}\s*{_ARROW}\s*\W{{0,3}}{y}"
        rf"|\bfrom\s+{x}\s+(?:to|into)\s+{y}"
        rf"|{x}\s+(?:flips?|flipped|changes?|changed|turns?|turned|"
        rf"switch(?:es|ed)?)\s+(?:to|into)\s+{y}",
        re.I,
    )


def _fact_sentence(record: ToolRecord, fact: Mapping[str, Any]) -> str:
    sentence = fact.get("sentence")
    return f"{record.name}: {sentence}" if isinstance(sentence, str) else ""


def _dominant_change(
    s: str, *, record: ToolRecord, fact: Mapping[str, Any], names: Mapping[str, str]
) -> str | None:
    src, dst = fact.get("from_class"), fact.get("to_class")
    if src is None or dst is None:
        return None
    local = dict(names)
    for key, cls in (("from_name", src), ("to_name", dst)):
        if isinstance(fact.get(key), str) and _is_int_class(cls):
            local[str(int(cls))] = fact[key]
    x, y = _class_pattern(src, local), _class_pattern(dst, local)
    forward, reverse = _directed(x, y).search(s), _directed(y, x).search(s)
    about = _fact_sentence(record, fact) or "the tool counts it from map A to map B"
    if (
        reverse
        and not forward
        and _DOMINANT_RE.search(s)
        and not _negated_at(s, reverse)
    ):
        return f"states the dominant change in the opposite direction; {about}"
    if forward or reverse or not local or not _DOMINANT_PAIR_RE.search(s):
        return None
    if re.search(x, s, re.I) and re.search(y, s, re.I):
        return None
    # A class pair named the dominant one that is not the dominant pair, by
    # name only (bare digits joined by "/" or "and" are too often not classes).
    others = list(local.values())
    pats = {k: _name_pattern(v, others) for k, v in local.items()}
    for a, pa in pats.items():
        for b, pb in pats.items():
            if a < b and re.search(
                rf"{pa}{_UNDIRECTED}{pb}|{pb}{_UNDIRECTED}{pa}", s, re.I
            ):
                return f"names {local[a]} / {local[b]} the dominant change; {about}"
    return None


_CONFIDENT_VERB = (
    r"(?:\s+(?:map|side|run|model|prediction|one))?"
    r"(?:\s*\([^)]{0,40}\))?\s+(?:is|was|appears|seems|looks|remains|"
    r"tends\s+to\s+be)\s+(?:(?:usually|generally|typically|mostly|often|"
    r"consistently|overall|clearly|much|far|slightly|the)\s+)*more[\s-]+confident"
)
_MAJORITY_RE = re.compile(
    r"\b(?:usually|mostly|most|majority|more\s+often|typically|generally|"
    r"in\s+most)\b",
    re.I,
)
_MEAN_RE = re.compile(r"\b(?:mean|average|on\s+average|median)\b", re.I)
_PATH_WORDS = frozenset(
    {"json", "csv", "tif", "scores", "score", "run", "file", "path", "workspace"}
)


def _label_words(value: str) -> set[str]:
    return {
        w
        for w in re.split(r"[^A-Za-z0-9]+", value)
        if 2 <= len(w) <= 16 and w.lower() not in _PATH_WORDS
    }


def _side_labels(
    run: RunEvidence, record: ToolRecord, fact: Mapping[str, Any]
) -> dict[str, list[str]]:
    """The words the answer may call each side by: A/B, and what tells them apart.

    The fact's own ``labels``/``label_a``/``label_b``, then the words that
    differ between the call's ``*_a`` and ``*_b`` arguments and the
    arguments of the calls whose results named those (``C1`` and ``C2``,
    ``pre`` and ``post``, ``2023`` and ``2022``).
    """
    labels: dict[str, list[str]] = {"A": ["A"], "B": ["B"]}
    given = fact.get("labels")
    for side in ("A", "B"):
        extra = [
            fact.get(f"label_{side.lower()}"),
            given.get(side) if isinstance(given, Mapping) else None,
        ]
        labels[side].extend(e for e in extra if isinstance(e, str) and e.strip())
    words: dict[str, set[str]] = {"A": set(), "B": set()}
    for key, value in record.arguments.items():
        side = "A" if key.endswith("_a") else "B" if key.endswith("_b") else ""
        if not side or not isinstance(value, str):
            continue
        words[side] |= _label_words(value)
        for other in run.tools:
            if other is not record and any(
                value == v for _, v in _strings(other.result)
            ):
                for _, argument in _strings(other.arguments):
                    words[side] |= _label_words(argument)
    for side, rest in (("A", "B"), ("B", "A")):
        labels[side].extend(sorted(words[side] - words[rest]))
    return labels


def _alias(label: str) -> str:
    escaped = re.escape(label)
    if len(label) == 1:
        return rf"(?:\b(?:map|side|model|run|prediction)\s+{escaped}\b|\b{escaped}\b)"
    return rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"


def _more_confident_side(
    s: str, *, run: RunEvidence, record: ToolRecord, fact: Mapping[str, Any]
) -> str | None:
    side = str(fact.get("side", "")).strip().upper()
    if side not in ("A", "B"):
        return None
    other = "B" if side == "A" else "A"
    labels = _side_labels(run, record, fact)
    alias = {k: "(?:" + "|".join(_alias(v) for v in labels[k]) + ")" for k in labels}

    def claims(k: str) -> Iterator[re.Match[str]]:
        """Clauses naming side ``k`` the more confident, as a majority claim."""
        for pattern in (
            rf"{alias[k]}{_CONFIDENT_VERB}",
            rf"more[\s-]+confident\s+(?:side|map|one|run)?\s*(?:is|was|:|=)\s*"
            rf"(?:map\s+|side\s+)?{alias[k]}",
        ):
            for m in re.finditer(pattern, s):
                clause = _clause(s, m.start(), m.end())
                # Not a negated clause, nor a minority share ("more confident
                # on 42.2% of them"), nor a mean without a majority word.
                if not (
                    _negated(clause)
                    or any(t.percent and t.value < 50 for t in tokenize(clause))
                    or (_MEAN_RE.search(clause) and not _MAJORITY_RE.search(clause))
                ):
                    yield m

    if next(claims(side), None) is not None or next(claims(other), None) is None:
        return None
    about = _fact_sentence(record, fact) or f"the tool finds side {side} more confident"
    return f"names side {other} the more confident; {about}"


_PLACES = {
    "top": r"north(?:ern)?(?:\s+(?:edge|rows?|strip|part|side|border|margin|end|"
    r"half|band))?|top\s+(?:rows?|edge|band|of\s+the\s+(?:grid|map|area|image|"
    r"scene))|upper\s+(?:rows?|edge|part|half|band)|first\s+rows",
    "bottom": r"south(?:ern)?(?:\s+(?:edge|rows?|strip|part|side|border|margin|end|"
    r"half|band))?|bottom\s+(?:rows?|edge|band|of\s+the\s+(?:grid|map|area|image|"
    r"scene))|lower\s+(?:rows?|edge|part|half|band)|last\s+rows",
    "left": r"west(?:ern)?\s+(?:edge|columns?|strip|part|side|border|margin|half|"
    r"band)|left\s+(?:edge|columns?|side|band)",
    "right": r"east(?:ern)?\s+(?:edge|columns?|strip|part|side|border|margin|half|"
    r"band)|right\s+(?:edge|columns?|side|band)",
}
#: A claim that most of the differences are somewhere (not "a cluster at").
_CONCENTRATION_RE = re.compile(
    r"\b(?:mostly|most\s+of|most\s+(?:[\w-]+\s+){0,2}?(?:differ\w*|changes?|flips?|"
    r"windows|disagree\w*|cells|pixels)|most\s+[\w-]+\s+(?:zone|area|region|part|"
    r"strip|block|band|cluster)|concentrat\w*|dominat\w*|mainly|largely|primarily|"
    r"majority|bulk|predominant\w*)\b",
    re.I,
)
_ROWS_RE = re.compile(
    r"\brows?\s*~?(?P<r0>\d+)(?:\s*(?:[-–—]|to|and)\s*~?(?P<r1>\d+))?" r"(?!\d|[.,]\d)",
    re.I,
)


def _as_grid(value: Any) -> tuple[int, int] | None:
    if (
        isinstance(value, list | tuple)
        and len(value) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in value)
    ):
        return value[0], value[1]
    return None


def _grid(
    run: RunEvidence, record: ToolRecord, fact: Mapping[str, Any]
) -> tuple[int, int] | None:
    """The window grid ``(rows, cols)`` of the fact, its result, or the run.

    From the run, only a grid whose size is the result's ``n_windows`` (or,
    without one, the run's only grid).
    """
    result = record.result if isinstance(record.result, Mapping) else {}
    for value in (fact.get("grid"), result.get("grid")):
        if grid := _as_grid(value):
            return grid
    grids = {
        grid
        for other in run.tools
        if isinstance(other.result, Mapping)
        and (grid := _as_grid(other.result.get("grid")))
    }
    n = result.get("n_windows")
    if isinstance(n, int):
        grids = {g for g in grids if g[0] * g[1] == n}
    return grids.pop() if len(grids) == 1 else None


def _concentration(
    s: str, *, run: RunEvidence, record: ToolRecord, fact: Mapping[str, Any]
) -> str | None:
    if not _CONCENTRATION_RE.search(s):
        return None
    about = (
        _fact_sentence(record, fact) or f"the tool finds them at {fact.get('where')}"
    )
    for band, pattern in _PLACES.items():
        share = fact.get(f"{band}_band_share")
        if not isinstance(share, int | float) or share >= BAND_SHARE_MAX:
            continue
        for m in re.finditer(rf"\b(?:{pattern})\b", s, re.I):
            if not _example_before(s, m.start()) and not _negated_at(s, m):
                return (
                    f"places the differences at the {band} ({m.group()}), which "
                    f"holds {share:.1%} of them; {about}"
                )
    # A strip of rows too narrow to hold most of the differing windows.
    grid = _grid(run, record, fact)
    result = record.result if isinstance(record.result, Mapping) else {}
    n = fact.get("n_differing", result.get("n_differing"))
    if not grid or not isinstance(n, int | float) or n <= 0:
        return None
    strips = [
        m
        for m in _ROWS_RE.finditer(s)
        if not (
            _example_before(s, m.start())
            or _negated_at(s, m)
            or re.match(r"\s*=", s[m.end() :])  # a legend: "row 0 = north"
        )
    ]
    if not strips:
        return None
    rows = sum(
        abs(int(m.group("r1") or m.group("r0")) - int(m.group("r0"))) + 1
        for m in strips
    )
    capacity = rows * grid[1]
    if capacity < 0.5 * n:
        where = ", ".join(m.group().strip() for m in strips)
        return (
            f"places most differences in {where}, which hold at most {capacity} "
            f"windows of the {int(n)} that differ; {about}"
        )
    return None


_MAGNITUDE_CONTEXT_RE = re.compile(
    r"\b(?:margin\w*|confiden\w*|uncertain\w*|certain\w*|decisive\w*|median|"
    r"typical\w*|sure)\b",
    re.I,
)
_ORDERS_RE = re.compile(
    r"(?:(?P<o0>\d+(?:\.\d+)?)(?:\s*(?:[-–—]|to)\s*(?P<o1>\d+(?:\.\d+)?))?"
    r"|(?P<one>an|one))\s+orders?\s+of\s+magnitude|(?P<bare>orders\s+of\s+magnitude)",
    re.I,
)
_TIMES_RE = re.compile(
    r"(?P<t0>\d+(?:\.\d+)?)(?:\s*(?:[-–—]|to)\s*(?P<t1>\d+(?:\.\d+)?))?"
    r"\s*(?:×(?!\s*\d)|x\b(?!\s*\d)|times\b|-fold\b|fold\b)",
    re.I,
)


def _margin_ratio(s: str, *, record: ToolRecord, fact: Mapping[str, Any]) -> str | None:
    low, high = fact.get("low"), fact.get("high")
    if not isinstance(low, int | float) or not isinstance(high, int | float):
        return None
    if not _MAGNITUDE_CONTEXT_RE.search(s):
        return None
    stated: list[tuple[float, float, str]] = []
    for m in _ORDERS_RE.finditer(s):
        if m.group("bare"):
            stated.append((100.0, math.inf, m.group()))
        elif m.group("one"):
            stated.append((10.0, 10.0, m.group()))
        else:
            o0 = float(m.group("o0"))
            stated.append((10**o0, 10 ** float(m.group("o1") or o0), m.group()))
    for m in _TIMES_RE.finditer(s):
        t0 = float(m.group("t0"))
        stated.append((t0, float(m.group("t1") or t0), m.group()))
    for lo, hi, text in stated:
        if lo > high * RATIO_SLACK or hi < low / RATIO_SLACK:
            versus = fact.get("versus", "median")
            return (
                f"states a magnitude of {text.strip()}; the tool's ratio to the "
                f"{versus} is {low:g} to {high:g}"
            )
    return None


_UNUSED_RE = re.compile(
    r"(?P<n>\d[\d,]*)\s+(?:\w+\s+){0,3}?labels?\s+(?:\w+\s+){0,4}?"
    r"(?:unused|left\s+over|leftover|spare|unallocated|not\s+(?:be\s+)?used|"
    r"(?:have|has)\s+nowhere)"
    r"|(?:unused|leftover|spare|unallocated|remaining|extra)\s+(?:\w+\s+){0,2}?"
    r"(?P<m>\d[\d,]*)\s+labels?",
    re.I,
)


def _unused_labels(
    s: str, *, record: ToolRecord, fact: Mapping[str, Any]
) -> str | None:
    n, requested = fact.get("n"), fact.get("requested")
    if not isinstance(n, int | float):
        return None
    about = _fact_sentence(record, fact) or f"the tool leaves {n:g} labels unused"
    if n > 0 and isinstance(requested, int | float):
        req = f"{int(requested):,}".replace(",", ",?")
        if all_used := re.search(
            rf"\b(?:all|every|the\s+full|full)\s+(?:of\s+the\s+)?{req}\s+labels?\b"
            r"[^.;]{0,40}\b(?:planned|allocated|used|placed|drawn|spent|go|goes|"
            r"sampled|in\s+the\s+(?:plan|design|sample))\b",
            s,
            re.I,
        ):
            if not _negated_at(s, all_used):
                return f"says all {int(requested)} labels are used; {about}"
    for m in _UNUSED_RE.finditer(s):
        stated = float((m.group("n") or m.group("m")).replace(",", ""))
        if abs(stated - float(n)) > 0.5:
            return f"states {stated:g} unused labels; {about}"
    return None


def check_direction(answer: str, run: RunEvidence) -> list[Violation]:
    """Sentences that contradict a tool's fact.

    A dominant change stated in the opposite direction (or another class
    pair named the dominant one), the other side named the more confident, a
    place of concentration where the facts put under
    :data:`BAND_SHARE_MAX` of the differences (or a strip of rows too narrow
    to hold most of them), a magnitude outside a ``margin_ratio``'s range,
    and a count of unused labels other than the tool's.
    """
    rules: list[Callable[[str], str | None]] = []
    for record, fact in run.facts("dominant_change"):
        rules.append(
            partial(_dominant_change, record=record, fact=fact, names=run.class_names)
        )
    for record, fact in run.facts("more_confident_side"):
        rules.append(partial(_more_confident_side, run=run, record=record, fact=fact))
    for record, fact in run.facts("concentration"):
        rules.append(partial(_concentration, run=run, record=record, fact=fact))
    detectors: tuple[tuple[str, Callable[..., str | None]], ...] = (
        ("margin_ratio", _margin_ratio),
        ("unused_labels", _unused_labels),
    )
    for fact_id, detector in detectors:
        for record, fact in run.facts(fact_id):
            rules.append(partial(detector, record=record, fact=fact))
    out: list[Violation] = []
    for sent in sentences(answer) if rules else []:
        plain = _plain(sent.text)
        for rule in rules:
            detail = rule(plain)
            if detail:
                out.append({"check": DIRECTION, "text": sent.text, "detail": detail})
                break
    return out


# --------------------------------------------------------------------------
# actions: what the answer says was saved or listed

_FILE_RE = re.compile(
    r"[\w.\-~…]+\.(?:json|csv|tsv|tif|tiff|geojson|gpkg|txt|md|png|jpe?g|"
    r"qlr|xml|parquet|npz|npy|zip|shp|html|ya?ml|log|nc)\b",
    re.I,
)
_SAVE_RE = re.compile(
    r"\b(?:i|we)(?:'ve|\s+have)?\s+(?:also\s+)?(?:saved|exported|written|wrote|"
    r"attached|stored|uploaded)\b"
    r"|\b(?:was|were|is|are|has\s+been|have\s+been|been)\s+(?:also\s+)?"
    r"(?:saved|exported|written|attached|stored|uploaded)\b"
    r"|\b(?:saved|exported|written|stored)\s+(?:to|at|in|as|under|into)\b"
    r"|\bthe\s+saved\s+\w+",
    re.I,
)
_HEDGE_RE = re.compile(
    r"\b(?:can|could|will|would|shall|should|may|might|if|want\s+me|once|after|"
    r"when\s+you|to\s+be)\b",
    re.I,
)
_LIST_CLAIM_RE = re.compile(
    r"\b(?:list|listing|ranking|ranked\s+windows|review\s+set|review\s+list)\b", re.I
)
#: The words of a result key that say the file it names is a list.
_LIST_FILE_WORDS = frozenset(
    {
        "list",
        "listing",
        "lists",
        "ranking",
        "ranked",
        "review",
        "differing",
        "windows",
        "rows",
        "table",
        "labels",
        "label",
        "design",
        "sample",
    }
)
_LISTED_RE = re.compile(
    r"\b(?:listed|shown|given|included|displayed|see)\s+(?:only\s+)?"
    r"(?:the\s+first\s+(?:few|\w+)\s+|in\s+the\s+(?:list|table)\s+)?"
    r"(?:above|below)\b"
    r"|\b(?:the\s+)?(?:list|table)\s+(?:above|below)\b",
    re.I,
)
_FIRST_N_RE = re.compile(
    r"\b(?:first|top)\s+(?P<n>\d+)\b[^.;:]{0,40}?\b(?:are|is|were|was)\s+"
    r"(?:listed|shown|given|included)\b(?:\s+(?:above|below))?",
    re.I,
)
_ITEM_RE = re.compile(r"^\s*(?:[-*+•]\s+|\d+[.)]\s+)")


def _items(lines: Iterable[str]) -> int:
    """List items in ``lines``: bullets, numbered items and table body rows."""
    n = 0
    in_table = False
    for line in lines:
        if line.strip() and _TABLE_RULE_RE.match(line):
            in_table = True
        elif line.lstrip().startswith("|"):
            n += 1 if in_table else 0
        else:
            in_table = False
            n += 1 if _ITEM_RE.match(line) else 0
    return n


def _file_matches(named: str, path: str) -> bool:
    """Whether a file the answer names is ``path``, allowing ``...`` elisions."""
    parts = [p for p in re.split(r"\.\.\.|…", named) if p.strip("/")]
    at = 0
    for part in parts:
        found = path.find(part, at)
        if found < 0:
            return False
        at = found + len(part)
    return bool(parts)


def _is_list_file(key: str, path: str) -> bool:
    words = {w.lower() for w in re.split(r"[^A-Za-z]+", key) if w}
    stem = os.path.basename(path).lower()
    return (
        bool(words & _LIST_FILE_WORDS)
        or stem.endswith((".csv", ".tsv"))
        or any(w in stem for w in ("list", "review", "differing", "to_label"))
    )


def _save_violation(s: str, text: str, run: RunEvidence) -> str | None:
    m = _SAVE_RE.search(s)
    if not m:
        return None
    lo, _ = _clause_span(s, m.start(), m.end())
    if _negated(s[lo : m.end()]) or _HEDGE_RE.search(s[lo : m.start()]):
        return None
    written = run.written_files
    named = [f.group() for f in _FILE_RE.finditer(text)]
    # What was saved: the word after "the saved", else the clause's subject.
    if m.group().lower().startswith("the saved"):
        listy = bool(_LIST_CLAIM_RE.match(m.group().split()[-1]))
    else:
        listy = bool(_LIST_CLAIM_RE.search(s[max(lo, m.start() - 40) : m.start()]))
    pool = [p for k, p in written if not listy or _is_list_file(k, p)]
    if named:
        if any(_file_matches(f, p) for f in named for p in pool):
            return None
        return (
            f"says {named[0]} holds what was saved; no tool of this run reported "
            f"writing it{' as a list' if listy else ''}"
        )
    if pool:
        return None
    what = "a list" if listy and written else "a file"
    return f"says something was saved; no tool of this run reported writing {what}"


def _listing_violation(s: str, lines: list[str], line_no: int) -> str | None:
    first = _FIRST_N_RE.search(s)
    claim = first or _LISTED_RE.search(s)
    if claim is None or _negated_at(s, claim) or re.search(r"\btool", s, re.I):
        return None  # "listed in the tool output" is about the tool, not the answer
    words = claim.group().lower()
    where = "above" if "above" in words else "below" if "below" in words else ""
    above, below = lines[:line_no], lines[line_no + 1 :]
    have = _items({"above": above, "below": below}.get(where, above + below))
    need = int(first.group("n")) if first else 1
    if have >= need:
        return None
    what = f"the first {need}" if first else "items"
    return f"says {what} are listed {where or 'in the answer'}; the answer lists {have} there"


def check_actions(answer: str, run: RunEvidence) -> list[Violation]:
    """Claims of something saved, written, attached or listed that the run did not do.

    A save claim is backed by a file a tool of this run reported writing: the
    file it names (``...`` elisions allowed), or, when it names none, any
    file; a claim that a *list* was saved needs a file a tool reported as a
    list (by its key or name, or a CSV). A claim that items are listed above
    or below, or that the first N are, is backed by that many list items in
    the answer on that side of it; on the ``"web"`` surface, where tool
    results are shown beside the answer, it is not checked.
    """
    out: list[Violation] = []
    lines = answer.split("\n")
    for sent in sentences(answer):
        s = _plain(sent.text)
        detail = _save_violation(s, sent.text, run)
        if detail is None and run.surface == "cli":
            detail = _listing_violation(s, lines, answer.count("\n", 0, sent.start))
        if detail:
            out.append({"check": ACTIONS, "text": sent.text, "detail": detail})
    return out


# --------------------------------------------------------------------------
# forbidden claims

_OFFER_RE = re.compile(
    r"\?|\b(?:want\s+me|shall\s+i|should\s+i|i\s+can|i\s+could|i'll|i\s+will|"
    r"we\s+can|we\s+could|let\s+me|would\s+you\s+like|could|would|might|try|"
    r"re-?run|re-?test|redo|next)\b",
    re.I,
)
_ALPHA_WORD = r"(?:α|alpha)"
#: An alpha said to be fixed in advance is the rule, not a post-hoc choice.
_IN_ADVANCE_RE = re.compile(
    r"\b(?:before|in\s+advance|beforehand|ahead\s+of|pre-?regist\w*|"
    r"pre-?specif\w*|fixed)\b",
    re.I,
)


def _unnegated(pattern: str, s: str, flags: int = re.I) -> Iterator[re.Match[str]]:
    """The matches of ``pattern`` in ``s`` whose own clause does not negate them."""
    for m in re.finditer(pattern, s, flags):
        if not _negated_at(s, m):
            yield m


def _alpha_used(record: ToolRecord) -> float | None:
    for source in (record.arguments, record.result):
        if isinstance(source, Mapping) and isinstance(source.get("alpha"), int | float):
            return float(source["alpha"])
    return None


def _post_hoc_alpha(s: str, record: ToolRecord) -> bool:
    if not (_OFFER_RE.search(s) or re.search(r"\bcertif\w*", s, re.I)):
        return False
    used = _alpha_used(record)
    for m in _unnegated(
        r"\b(?:looser|different|another|higher|larger|relaxed|less\s+strict|laxer|"
        rf"other|new|bigger)\s+{_ALPHA_WORD}"
        rf"|{_ALPHA_WORD}\s*(?:\(\s*(?:e\.g\.|say)\s*)?(?:=|of|at|to|:|≈)?\s*"
        r"(?P<v>0?\.\d+|\d+(?:\.\d+)?\s*%)",
        s,
    ):
        if _IN_ADVANCE_RE.search(_clause(s, m.start(), m.end())):
            continue
        if m.group("v"):
            raw = m.group("v").replace(" ", "")
            value = float(raw[:-1]) / 100 if raw.endswith("%") else float(raw)
            if used is not None and abs(value - used) < 1e-9:
                continue
        return True
    return False


def _rule_switch(s: str, record: ToolRecord) -> bool:
    rules = (
        r"(?:bonferroni|holm|fixed[\s-]sequence|prefix|another\s+rule|"
        r"a\s+different\s+rule|other\s+rules?|stricter\s+rule|looser\s+rule)"
    )
    return bool(_OFFER_RE.search(s)) and any(
        _unnegated(
            r"\b(?:re-?run|run|try|re-?test|redo|switch\w*|use|apply|repeat)\b"
            rf"[^.;]{{0,50}}\b{rules}",
            s,
        )
    )


def _certify_nonrandom(s: str, record: ToolRecord) -> bool:
    if re.search(r"\brandom\b", s, re.I) or not _OFFER_RE.search(s):
        return False
    return any(_unnegated(r"\bcertif\w*", s))


def _error_rate_without_labels(s: str, record: ToolRecord) -> bool:
    if _negated(s) or re.search(
        r"\b(?:would|if|once|after|need|needs|requires?|estimate|overstate\w*|"
        r"can|could|will|label\w*)\b",
        s,
        re.I,
    ):
        return False
    return bool(
        re.search(
            r"\b(?:error\s+rate|accuracy|misclassification\s+rate|share\s+wrong)\b"
            r"[^.;]{0,30}?(?:is|of|at|about|around|~|≈|=|:)\s*~?\d",
            s,
            re.I,
        )
        or re.search(
            r"\d+(?:\.\d+)?\s*%\s+(?:of\s+(?:the\s+)?(?:windows|map|pixels|cells)\s+)?"
            r"(?:are\s+|is\s+)?(?:wrong|misclassified|in\s+error)\b",
            s,
            re.I,
        )
    )


def _error_rate_regression(s: str, record: ToolRecord) -> bool:
    if re.search(r"\bthreshold", s, re.I) or not re.search(
        r"\b(?:i\s+can|i'll|i\s+will|then|would|could|to\s+estimate|estimate|"
        r"score|compute|show|run|measure)\b",
        s,
        re.I,
    ):
        return False
    return any(
        _unnegated(
            r"\b(?:error\s+rates?|accuracy|classification\s+metrics|confusion\s+matrix|"
            r"precision|recall|f1|misclassif\w*|how\s+wrong|"
            r"olmoearth_classification_metrics|olmoearth_estimate_map_error)\b",
            s,
        )
    )


def _subset_sufficient(s: str, record: ToolRecord) -> bool:
    return any(
        _unnegated(
            r"\b(?:first|top|only|just)\s+(?:the\s+)?(?:\d+|few|some|handful)\b"
            r"[^.;]{0,50}\b(?:enough|suffic\w*|would\s+do|is\s+fine|are\s+fine)\b"
            r"|\b(?:is|are)\s+(?:\d+|these|those|that)\s+(?:windows\s+|labels\s+)?"
            r"enough\b",
            s,
        )
    )


_WINNER_SUBJECT = (
    r"(?:\b(?:[Mm]ap|[Ss]ide)\s+[AB]\b|\b[AB]\b|\bC\d+\b|"
    r"\b[Tt]he\s+(?:first|second|newer|older|later|earlier|pre|post|\d{4})"
    r"[\w-]*\s+map\b)"
)


def _winner(s: str, record: ToolRecord) -> bool:
    if _negated(s) or re.search(
        r"\b(?:if|would|once|after|with\s+labels|only|rate|often|half|times|"
        r"whether|which)\b|%",
        s,
        re.I,
    ):
        return False
    return bool(
        re.search(
            rf"{_WINNER_SUBJECT}(?:\s*\([^)]{{0,40}}\))?\s+(?:is|was|looks|seems|"
            r"appears)\s+(?:probably\s+|likely\s+|clearly\s+)?(?:the\s+)?"
            r"(?:right|correct|more\s+accurate|closer\s+to\s+(?:the\s+)?truth|"
            r"the\s+winner|more\s+reliable|the\s+better\s+(?:map|one))\b",
            s,
        )
    )


def _combined_statistic(s: str, record: ToolRecord) -> bool:
    if not re.search(r"\d", s) or re.search(
        r"\b(?:meaningful|left\s+out|omitted)\b", s, re.I
    ):
        return False
    return any(
        _unnegated(
            r"\b(?:rmse|mean\s+(?:absolute\s+)?diff\w*|bias|mae|agreement(?:\s+fraction|"
            r"\s+within)?|max(?:imum)?\s+(?:absolute\s+)?diff\w*|differ\s+by\s+~?\d)",
            s,
        )
    )


def _another_date(s: str, record: ToolRecord) -> bool:
    if not re.search(
        r"\b(?:settle\w*|check\w*|trajectory|show|tell|resolve\w*|decide|which|"
        r"verdict|confirm\w*|reversal\w*|separate|disambiguat\w*)\b",
        s,
        re.I,
    ):
        return False
    return any(
        # A reference at the maps' own dates ("another acquisition per date")
        # is what separates change from error; another date is not.
        not re.match(
            r"\s*(?:per|for\s+each|at\s+each|on\s+the\s+same|at\s+the\s+same)\s+"
            r"(?:date|period|year)",
            s[m.end() :],
            re.I,
        )
        for m in _unnegated(
            r"\b(?:third|another|additional|extra|later|earlier|intermediate)\s+"
            r"(?:dated\s+)?(?:date|map|image|acquisition|scene|time\s*(?:point|step)|"
            r"snapshot|epoch)s?\b",
            s,
        )
    )


#: A detector per ``forbidden_claims`` id: does one sentence make that claim?
FORBIDDEN_DETECTORS: dict[str, Callable[[str, ToolRecord], bool]] = {
    "post_hoc_alpha": _post_hoc_alpha,
    "rule_switch_after_failure": _rule_switch,
    "certify_from_nonrandom_design": _certify_nonrandom,
    "error_rate_without_labels": _error_rate_without_labels,
    "error_rate_for_unthresholded_regression": _error_rate_regression,
    "subset_labelling_sufficient": _subset_sufficient,
    "winner_without_labels": _winner,
    "combined_statistic_across_properties": _combined_statistic,
    "another_date_settles_it": _another_date,
}


def check_forbidden_claims(answer: str, run: RunEvidence) -> list[Violation]:
    """Sentences making a claim a tool of this run named in ``forbidden_claims``."""
    wanted: list[tuple[ToolRecord, str, str]] = []
    for record, item in run.contract("forbidden_claims"):
        claim_id = item.get("id") if isinstance(item, Mapping) else item
        why = item.get("why", "") if isinstance(item, Mapping) else ""
        if (
            isinstance(claim_id, str)
            and claim_id in FORBIDDEN_DETECTORS
            and all(claim_id != w[1] for w in wanted)
        ):
            wanted.append((record, claim_id, str(why)))
    out: list[Violation] = []
    for sent in sentences(answer) if wanted else []:
        s = _plain(sent.text)
        for record, claim_id, why in wanted:
            if FORBIDDEN_DETECTORS[claim_id](s, record):
                detail = f"{claim_id} ({record.name})" + (f": {why}" if why else "")
                out.append({"check": FORBIDDEN, "text": sent.text, "detail": detail})
                break
    return out


# --------------------------------------------------------------------------
# must_state

_STOPWORDS = frozenset(
    """a an the and or but if then than that this these those it its it's is are
    was were be been being to of in on at by for from with as into onto over
    under about can could would should will may might must do does did not no
    so such any each every all some more most less least only also just very
    what which who whom whose when where why how there here they them their
    you your we our he she his her one two per via vs""".split()
)


def _terms(text: str) -> set[str]:
    """Key terms: words other than stopwords, on their first five letters, and numbers."""
    words = re.findall(r"[a-z][a-z0-9']*", text.lower())
    terms = {w[:5] for w in words if w not in _STOPWORDS}
    terms |= {f"#{t.value * t.scale:g}" for t in tokenize(text) if not t.exempt}
    return terms


def _reports(answer: str, record: ToolRecord) -> bool:
    """Whether the answer reports a result: it names the tool or one of its numbers."""
    if record.name in answer:
        return True
    pool = NumberPool([record.envelope])
    return any(
        not t.exempt and (t.value > 8 or t.decimals >= 2) and pool.supports(t)
        for t in tokenize(answer)
    )


def check_must_state(answer: str, run: RunEvidence) -> list[Violation]:
    """``must_state`` sentences of a reported result the answer does not convey.

    Conveyed means the answer holds at least :data:`MUST_STATE_MIN_OVERLAP`
    of the sentence's key terms (words other than stopwords, compared on
    their first five letters, and its numbers).
    """
    out: list[Violation] = []
    have = _terms(answer)
    seen: set[str] = set()
    for record, item in run.contract("must_state"):
        sentence = item.get("sentence") if isinstance(item, Mapping) else item
        if not isinstance(sentence, str) or not sentence.strip() or sentence in seen:
            continue
        seen.add(sentence)
        terms = _terms(sentence)
        if not terms or not _reports(answer, record):
            continue
        found = len(terms & have)
        if found / len(terms) < MUST_STATE_MIN_OVERLAP:
            out.append(
                {
                    "check": MUST_STATE,
                    "text": sentence.strip(),
                    "detail": f"{record.name} requires it; the answer holds "
                    f"{found} of its {len(terms)} key terms",
                }
            )
    return out


# --------------------------------------------------------------------------
# All of them

#: The checks, in the order they run and are reported.
CHECKS: dict[str, Check] = {
    NUMBERS: check_numbers,
    DIRECTION: check_direction,
    ACTIONS: check_actions,
    FORBIDDEN: check_forbidden_claims,
    MUST_STATE: check_must_state,
}


def run_checks(
    answer: str, run: RunEvidence, names: Iterable[str] = tuple(CHECKS)
) -> dict[str, list[Violation]]:
    """The violations of each check in ``names`` that found any, in order.

    A check that fails with an error is logged and skipped: a fault in a
    checker must not cost the user the answer.
    """
    found: dict[str, list[Violation]] = {}
    for name in names:
        try:
            violations = CHECKS[name](answer, run)
        except Exception:  # noqa: BLE001 - logged; the answer is still shown
            logger.warning("answer check %r failed; skipped", name, exc_info=True)
            continue
        if violations:
            found[name] = violations
    return found


def grounding_prompt(unsupported: list[str]) -> str:
    """The harness's request to rewrite an answer without ``unsupported``."""
    return (
        "Harness note: these numbers in your answer are in no tool result of "
        f"this run and no user message: {', '.join(unsupported)}. Every number "
        "in the answer must be one a tool returned, as the tool returned it "
        "(rounding is fine). Rewrite the answer: remove each listed number, or "
        "replace it with the figure a tool returned. Call no tools, and do not "
        "mention this note."
    )


_SECTION_HEAD = {
    DIRECTION: "Sentences that contradict what a tool computed. State the "
    "tool's figure or direction instead, or remove the sentence:",
    ACTIONS: "Sentences that say something was saved, written or listed that "
    "this run did not do. Remove the claim, or name the file a tool reported "
    "writing:",
    FORBIDDEN: "Sentences that make a claim a tool says must not be made about "
    "its result. Remove them:",
    MUST_STATE: "Statements a tool requires whenever its result is reported, "
    "which the answer does not convey. Include each, in your own words:",
}


def revision_prompt(found: Mapping[str, list[Violation]]) -> str:
    """The one rewrite request that lists every violation of every check.

    When only the number check fired it is :func:`grounding_prompt`, the
    request exp86 used.
    """
    if list(found) == [NUMBERS]:
        return grounding_prompt(unsupported_of(found[NUMBERS]))
    parts = [
        "Harness note: checks of your answer against this run's tool results "
        "found the problems below. Rewrite the answer to fix every one, and "
        "keep the rest of it. Call no tools, and do not mention this note."
    ]
    for name, violations in found.items():
        if name == NUMBERS:
            parts.append(
                "Numbers in no tool result of this run and no user message: "
                + ", ".join(unsupported_of(violations))
                + ". Every number in the answer must be one a tool returned, as "
                "the tool returned it (rounding is fine): remove each listed "
                "number, or replace it with the figure a tool returned."
            )
            continue
        lines = [_SECTION_HEAD.get(name, f"Problems the {name} check found:")]
        lines += [f'- "{v["text"]}" ({v["detail"]})' for v in violations]
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _mark_at(answer: str, sent: Sentence) -> int:
    """Where a sentence's marker goes: after it, or inside a table row's last cell."""
    line_start = answer.rfind("\n", 0, sent.start) + 1
    line_end = answer.find("\n", sent.end)
    line = answer[line_start : len(answer) if line_end < 0 else line_end]
    is_row = line.lstrip().startswith("|") and line.rstrip().endswith("|")
    if is_row and answer[sent.end - 1] == "|":
        at = sent.end - 1
        while at > line_start and answer[at - 1] == " ":
            at -= 1
        return at
    return sent.end


def mark_answer(answer: str, found: Mapping[str, list[Violation]]) -> str:
    """The answer with ``[unverified: <check>]`` after each sentence still flagged.

    Nothing is deleted. A ``must_state`` sentence the answer lacks is added at
    the end, after its marker, as the tool states it.
    """
    by_text: dict[str, list[str]] = {}
    missing: list[str] = []
    for name, violations in found.items():
        for v in violations:
            if name == MUST_STATE:
                if v["text"] not in missing:
                    missing.append(v["text"])
                continue
            names = by_text.setdefault(v["text"], [])
            if name not in names:
                names.append(name)
    out = answer
    placed: set[str] = set()
    for sent in reversed(sentences(answer)):
        names = by_text.get(sent.text, [])
        if names:
            placed.add(sent.text)
            at = _mark_at(answer, sent)
            marker = " " + " ".join(MARK.format(check=n) for n in names)
            out = out[:at] + marker + out[at:]
    tail = [
        " ".join(MARK.format(check=n) for n in names) + f" {text}"
        for text, names in by_text.items()
        if text not in placed
    ]
    tail += [
        f"{MARK.format(check=MUST_STATE)} A tool also states: {t}" for t in missing
    ]
    if tail:
        out = out.rstrip() + "\n\n" + "\n".join(tail)
    return out
