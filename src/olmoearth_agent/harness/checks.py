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
  ``to_class``, ``n``, ``share``, ``reverse_n``, ``reverse_share``,
  ``tied_with_reverse``; map A's class to map B's), ``more_confident_side``
  (``side`` "A", "B" or "neither", ``share_a``, ``share_b``,
  ``share_equal``), ``concentration`` (``grid`` ``[rows, cols]``,
  ``n_differing``, ``top_band_share``, the share of the differing windows in
  the northmost of four row bands, and ``max_band`` ``{axis, band, of_grid,
  share}``), ``margin_ratio`` (``listed_low``, ``listed_high``,
  ``review_set_low``, ``review_set_high``, ``versus``) and ``unused_labels``
  (``requested``, ``planned``, ``n``). ``whole_map_estimate`` and any other
  id are shown, never checked.
- ``must_state``: at most :data:`MUST_STATE_MAX` short sentences (at most
  :data:`MUST_STATE_MAX_WORDS` words each) the answer must convey whenever
  it reports that result.
- ``forbidden_claims``: ``[{"id", "why"}]``, claims the answer must not make
  about that result; :data:`FORBIDDEN_DETECTORS` holds a detector per id,
  and an id without one is ignored.

The detectors are regular expressions over one sentence at a time, and they
are conservative on purpose: a false alarm costs a rewrite and, when it
persists, a marked sentence, so a sentence that negates, hedges or offers an
example is left alone, and a rule fires only on the wording exp86's audits
found. What a check cannot read (a paraphrase, a claim spread over two
sentences) it misses; ``scripts/validate_answer_checks.py`` measures both
kinds of error on exp86's recorded answers. Each sentence is read once into
its brackets, clauses and word spans (:class:`_Parse`), so the checks take
time linear in its length, a listing of hundreds of windows in one line
included.
"""

from __future__ import annotations

import bisect
import logging
import math
import os
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property, lru_cache, partial
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
#: The ``must_state`` sentences read per result, and the longest read: the
#: contract's limits; a longer sentence is scope detail, not a limit to state.
MUST_STATE_MAX = 3
MUST_STATE_MAX_WORDS = 25


# --------------------------------------------------------------------------
# The run's evidence


@dataclass(frozen=True)
class ToolRecord:
    """One dispatched tool call: its name, arguments and full result envelope.

    ``spilled_to`` is the file the harness wrote the full result to when it
    was too large for the model's context (``harness/spill.py``), or None.
    """

    name: str
    arguments: Mapping[str, Any]
    envelope: Any
    spilled_to: str | None = None

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
    ``assistant_messages`` are the assistant's earlier turns of the history:
    on the ``"web"`` surface they were checked when first shown, so a number
    they state may be restated; on ``"cli"`` they are no source.
    """

    tools: list[ToolRecord] = field(default_factory=list)
    user_messages: list[str] = field(default_factory=list)
    surface: str = "cli"
    assistant_messages: list[str] = field(default_factory=list)

    @cached_property
    def pool(self) -> NumberPool:
        """The numbers of the tool results and the user's messages.

        Also the paths the harness spilled results to (the model is told
        them), and, on the ``"web"`` surface, the earlier assistant turns.
        """
        earlier = self.assistant_messages if self.surface == "web" else []
        return NumberPool(
            [
                *(t.envelope for t in self.tools),
                *(t.spilled_to for t in self.tools if t.spilled_to),
                *self.user_messages,
                *earlier,
            ]
        )

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
        """``(key, path)`` of every file a tool's result reports writing.

        A file path in a tool's result is one the tool wrote (``scores_path``,
        ``labels_csv_path``), unless it echoes an input: the same path under
        the same key in the call's arguments (``estimate``'s ``design_path``).
        A key that names an output (``out_path``, ``saved_to``) reports a
        write even when the caller chose the path, and so does a path under
        another key than the argument's (``folds_path`` for ``output_path``).
        The harness's own spill of a result too large for the model's context
        (``saved_to``) is a written file too.
        """
        out: list[tuple[str, str]] = []

        def add(key: str, value: str) -> None:
            if all(value != p for _, p in out):
                out.append((key, value))

        for record in self.tools:
            if record.spilled_to:
                add("saved_to", record.spilled_to)
            if not record.ok:
                continue
            given: dict[str, set[str]] = {}
            for key, value in _strings(record.arguments):
                given.setdefault(value, set()).add(key)
            for key, value in _strings(record.result):
                if "://" in value or not _FILE_RE.search(value):
                    continue  # a URL (a tile, a page) is no file
                echoed = key in given.get(value, ()) and not _names_output(key)
                if not echoed:
                    add(key, value)
        return out


#: Words of a result's key that say the path under it was written.
_OUTPUT_KEY_WORDS = frozenset(
    {
        "out",
        "output",
        "outputs",
        "saved",
        "save",
        "written",
        "write",
        "wrote",
        "export",
        "exported",
        "exports",
        "dest",
        "destination",
        "spill",
        "spilled",
    }
)


def _names_output(key: str) -> bool:
    """Whether a result's key says its path was written (``out_path``, ``saved_to``)."""
    words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", key)
    return any(w.lower() in _OUTPUT_KEY_WORDS for w in words)


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
    r"[;:]|\s[-–—]\s|—"
    r"|,\s*(?:but|although|though|while|whereas|yet|so|and\s+so)\b|\bbut\b",
    re.I,
)
#: Every clause break, overlapping ones included (", but" and its "but"), so
#: a break found once serves every lookup.
_CLAUSE_BREAK_AT_RE = re.compile(rf"(?=({_CLAUSE_BREAK_RE.pattern}))", re.I)
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


def _blank(text: str, spans: Iterable[tuple[int, int]]) -> str:
    """``text`` with each ``(open, close)`` span replaced by spaces."""
    pieces: list[str] = []
    at = 0
    for a, b in spans:
        pieces += [text[at:a], " " * (b + 1 - a)]
        at = b + 1
    return "".join(pieces) + text[at:]


def _breaks(text: str, lo: int, hi: int) -> tuple[list[int], list[int]]:
    """The starts and the (sorted) ends of the clause breaks of ``text[lo:hi]``."""
    found = [(m.start(1), m.end(1)) for m in _CLAUSE_BREAK_AT_RE.finditer(text, lo, hi)]
    return [a for a, _ in found], sorted(b for _, b in found)


class _Parse:
    """One sentence, read once: its brackets, clause breaks and word spans.

    Every lookup a detector makes per match (the clause around it, a word in
    that clause, an example cue before it) is a bisection over what is read
    here, so a sentence is checked in time linear in its length: exp86's
    answers list hundreds of windows in one line.

    Outside brackets the clauses are read on ``masked``, the sentence with its
    outermost brackets blanked; inside one, on the sentence itself, bounded
    by that bracket.
    """

    def __init__(self, s: str) -> None:
        self.s = s
        self.brackets = _brackets(s)
        self._opens = [a for a, _ in self.brackets]
        self.masked = _blank(s, self.brackets)
        self._breaks = _breaks(self.masked, 0, len(s))
        self._inner: dict[int, tuple[list[int], list[int]]] = {}
        self._found: dict[tuple[re.Pattern[str], bool], list[tuple[int, int]]] = {}
        cues = [(m.start(), m.end()) for m in _EXAMPLE_RE.finditer(s)]
        self._cue_starts = [a for a, _ in cues]
        self._cue_ends = [b for _, b in cues]
        self._semicolons = [i for i, ch in enumerate(s) if ch == ";"]

    def bracket_at(self, start: int, end: int) -> tuple[int, int] | None:
        """The outermost bracket ``(open, close)`` that holds ``s[start:end]``."""
        i = bisect.bisect_left(self._opens, start) - 1
        if i >= 0:
            a, b = self.brackets[i]
            if start <= b and end <= b + 1:
                return a, b
        return None

    def span(self, start: int, end: int) -> tuple[int, int]:
        """The span of the clause around ``s[start:end]`` (see :func:`_clause_span`)."""
        inside = self.bracket_at(start, end)
        if inside:
            a, b = inside
            if a not in self._inner:
                self._inner[a] = _breaks(self.s, a + 1, b)
            (starts, ends), lo, hi = self._inner[a], a + 1, b
        else:
            (starts, ends), lo, hi = self._breaks, 0, len(self.s)
        i = bisect.bisect_right(ends, start) - 1
        if i >= 0:
            lo = max(lo, ends[i])
        j = bisect.bisect_left(starts, end)
        if j < len(starts):
            hi = min(hi, starts[j])
        return lo, hi

    def region(self, start: int, end: int) -> tuple[int, int, tuple[int, int] | None]:
        """The clause around ``s[start:end]``, read with the bracket it is in.

        A claim inside a bracket belongs to the clause the bracket elaborates
        ("dominated by a strip in the north edge (row 0, cols 3-48)"): the
        span of the clause around that bracket, read on ``masked``, and the
        bracket, read on the sentence (None outside brackets).
        """
        inside = self.bracket_at(start, end)
        if inside is None:
            return (*self.span(start, end), None)
        return (*self.span(inside[0], inside[1] + 1), inside)

    def clause(self, start: int, end: int) -> str:
        """The clause around ``s[start:end]``, its other brackets blanked."""
        lo, hi = self.span(start, end)
        if self.bracket_at(start, end) is None:
            return self.masked[lo:hi]
        text = self.s[lo:hi]
        return _blank(
            text, [(a, b) for a, b in _brackets(text) if not a <= start - lo <= b]
        )

    def _spans(self, pattern: re.Pattern[str], masked: bool) -> list[tuple[int, int]]:
        key = (pattern, masked)
        if key not in self._found:
            text = self.masked if masked else self.s
            self._found[key] = [(m.start(), m.end()) for m in pattern.finditer(text)]
        return self._found[key]

    def find(
        self,
        pattern: re.Pattern[str],
        lo: int,
        hi: int,
        *,
        masked: bool = True,
        skip: tuple[int, int] | None = None,
    ) -> tuple[int, int] | None:
        """The first match of ``pattern`` within ``[lo, hi)`` that does not
        overlap ``skip``, read on ``masked`` or on the sentence."""
        spans = self._spans(pattern, masked)
        i = bisect.bisect_left(spans, (lo, -1))
        while i < len(spans) and spans[i][0] < hi:
            a, b = spans[i]
            if b <= hi and not (skip and a < skip[1] and b > skip[0]):
                return a, b
            i += 1
        return None

    def in_clause(
        self, pattern: re.Pattern[str], start: int, end: int, *, own: bool = False
    ) -> bool:
        """Whether ``pattern`` matches in the clause around ``s[start:end]``.

        The clause is the bracket's when ``own`` (as :meth:`span` reads it),
        else the one the bracket elaborates (:meth:`region`); a match that
        overlaps ``s[start:end]`` does not count.
        """
        skip = (start, end)
        if own:
            lo, hi = self.span(start, end)
            masked = self.bracket_at(start, end) is None
            return self.find(pattern, lo, hi, masked=masked, skip=skip) is not None
        lo, hi, inside = self.region(start, end)
        if self.find(pattern, lo, hi, skip=skip):
            return True
        # The bracket is blank on ``masked``: read it on the sentence.
        return bool(
            inside
            and self.find(pattern, inside[0], inside[1] + 1, masked=False, skip=skip)
        )

    def in_clause_or_label(
        self, pattern: re.Pattern[str], start: int, end: int
    ) -> bool:
        """:meth:`in_clause`, or in the label a colon ends just before that clause.

        "**The main change:** water -> not water" states its cue in a label.
        """
        if self.in_clause(pattern, start, end):
            return True
        at, _, _ = self.region(start, end)
        at -= 1
        while at >= 0 and self.masked[at].isspace():
            at -= 1
        if at < 0 or self.masked[at] != ":":
            return False
        label_lo, _ = self.span(at, at + 1)
        return self.find(pattern, label_lo, at) is not None

    def negated(self, start: int, end: int) -> bool:
        """Whether the clause of ``s[start:end]`` negates it (:func:`_negated_at`)."""
        return self.in_clause(_NEGATION_RE, start, end, own=True)

    def example_before(self, start: int) -> bool:
        """Whether ``s[start:]`` is part of an example (:func:`_example_before`)."""
        i = bisect.bisect_right(self._cue_ends, start) - 1
        if i < 0:
            return False
        cue = self._cue_starts[i]
        j = bisect.bisect_left(self._semicolons, cue)
        if j < len(self._semicolons) and self._semicolons[j] < start:
            return False
        k = bisect.bisect_left(self._opens, cue) - 1
        return not (k >= 0 and cue < self.brackets[k][1] < start)


@lru_cache(maxsize=256)
def _parse(s: str) -> _Parse:
    """The sentence ``s`` read once; the checks look it up by its text."""
    return _Parse(s)


def _clause_span(s: str, start: int, end: int) -> tuple[int, int]:
    """The span of the clause of ``s`` around ``s[start:end]``.

    A bracket is a clause of its own: a claim inside one is bounded by it, and
    a claim outside one reads past it ("A looser alpha (e.g. 0.15) would
    certify ..., but ..." is one clause up to the "but").
    """
    return _parse(s).span(start, end)


def _clause(s: str, start: int, end: int) -> str:
    """The clause of ``s`` around ``s[start:end]``, its other brackets blanked."""
    return _parse(s).clause(start, end)


def _negated_at(s: str, m: re.Match[str]) -> bool:
    """Whether the clause a claim is in negates it ("... is not saved").

    A hedge in another clause of a long sentence ("..., but it does not pick
    a winner") leaves the claim standing, and so does a negation inside the
    claim's own words (the class "not water").
    """
    return _parse(s).negated(m.start(), m.end())


def _example_before(s: str, start: int) -> bool:
    """Whether ``s[start:]`` is part of an example ("e.g. ...", "such as ...").

    An example runs from its cue to the end of the sentence or a semicolon,
    or to the end of the bracket the cue opens in.
    """
    return _parse(s).example_before(start)


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
#: A pair ranked after the dominant one ("the second largest change").
_SECONDARY_RE = re.compile(
    r"\b(?:second|third|fourth|fifth|next)[\s-]+(?:most\s+)?(?:largest|biggest|"
    r"main|dominant|primary|common|frequent)\b",
    re.I,
)
#: A dominance cue governing a change noun ("the dominant contrast", "the main
#: transition"): a pair named so is a claim about change, not composition.
_DOMINANT_CHANGE_NOUN_RE = re.compile(
    r"\b(?:dominant|predominant|main|largest|biggest|primary|principal|"
    r"most\s+common|most\s+frequent)\s+(?:\w+\s+)?(?:change|flip|transition|"
    r"contrast|swap|disagreement|difference)s?\b",
    re.I,
)
#: A sentence about change between the maps, not about what they are made of
#: ("both maps are dominated by X and Y" is composition).
_CHANGE_RE = re.compile(
    r"\b(?:flip\w*|chang\w*|turn(?:s|ed|ing)?|becom\w*|became|convert\w*|"
    r"switch\w*|transition\w*|contrast\w*|swap\w*)\b|" + _ARROW,
    re.I,
)
_CHANGE_VERB = (
    r"(?:flip(?:s|ped|ping)?|chang(?:e|es|ed|ing)|turn(?:s|ed|ing)?|"
    r"switch(?:es|ed|ing)?|convert(?:s|ed|ing)?|shift(?:s|ed|ing)?|go(?:es)?|"
    r"went|mov(?:e|es|ed|ing))"
)
#: What joins two class names into a pair: "X vs Y", "X/Y", "X and Y",
#: "X -> Y", "X to Y".
_PAIR_JOIN_RE = re.compile(
    rf"\s*(?:vs\.?|versus|↔|<->|<=>|/|&|and|or|to|into|\W{{0,3}}{_ARROW}\W{{0,3}})\s*",
    re.I,
)
#: A fact's sentence that says the top count is tied.
_TIE_RE = re.compile(r"\b(?:tie|ties|tied|equal(?:ly)?)\b", re.I)


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


def _digit(c: int) -> str:
    """Class ``c`` written as digits: not inside a longer number ("0.5", "10")."""
    return rf"(?<![\w.,]){c}(?![\w%]|[.,]\d)"


def _class_pattern(value: Any, names: Mapping[str, str]) -> str:
    """A regex for a class as an answer names it: its name, or ``class 1``.

    A bare digit is never read as a class ("2 to 4 rows" is no change from
    class 2 to class 4); a digit is one after ``class`` or a side's label
    (``A=1``).
    """
    alts: list[str] = []
    others = list(names.values())
    if _is_int_class(value):
        c = int(value)
        alts.append(rf"(?:\bclass(?:es)?\s+#?|\b[AB]\s*[=:]\s*){_digit(c)}")
        name = names.get(str(c))
        if name:
            alts.append(_name_pattern(name, others))
    elif isinstance(value, str) and value.strip():
        alts.append(_name_pattern(value, others))
    return "(?:" + "|".join(alts) + ")" if alts else r"(?!)"


@lru_cache(maxsize=256)
def _directed(
    src: Any, dst: Any, names: tuple[tuple[str, str], ...]
) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """A change from class ``src`` to ``dst``: written unambiguously, and as "X to Y".

    The first pattern is an arrow, "from X to Y", a change verb ("X flips to
    Y") or "X becomes Y"; with "class" once for both ends ("class 1 -> 0").
    The second is a bare "X to Y" or "X into Y", a change only in a clause
    that speaks of one.
    """
    table = dict(names)
    x, y = _class_pattern(src, table), _class_pattern(dst, table)
    alts = [
        rf"{x}\W{{0,3}}\s*{_ARROW}\s*\W{{0,3}}{y}",
        rf"\bfrom\s+{x}\s+(?:to|into)\s+{y}",
        rf"{x}\s+{_CHANGE_VERB}\s+(?:to|into)\s+{y}",
        rf"{x}\s+(?:becomes?|became|becoming)\s+{y}",
    ]
    if _is_int_class(src) and _is_int_class(dst):
        a, b = _digit(int(src)), _digit(int(dst))
        alts += [
            rf"\bclass(?:es)?\s+#?{a}\s*{_ARROW}\s*{b}",
            rf"\bfrom\s+class(?:es)?\s+#?{a}\s+(?:to|into)\s+{b}",
        ]
    return (
        re.compile("|".join(alts), re.I),
        re.compile(rf"{x}\s+(?:to|into)\s+{y}", re.I),
    )


def _change(
    p: _Parse, patterns: tuple[re.Pattern[str], re.Pattern[str]]
) -> tuple[int, int] | None:
    """Where the sentence states the change ``patterns`` read (see :func:`_directed`)."""
    strong, weak = patterns
    if m := strong.search(p.s):
        return m.start(), m.end()
    for m in weak.finditer(p.s):
        if p.in_clause(_CHANGE_RE, m.start(), m.end()):
            return m.start(), m.end()
    return None


@lru_cache(maxsize=64)
def _mention_re(names: tuple[tuple[str, str], ...]) -> re.Pattern[str]:
    """One regex for every class name, a group ``c<i>`` per class."""
    others = [n for _, n in names]
    return re.compile(
        "|".join(
            rf"(?P<c{i}>{_name_pattern(name, others)})"
            for i, (_, name) in enumerate(names)
        ),
        re.I,
    )


def _fact_sentence(record: ToolRecord, fact: Mapping[str, Any]) -> str:
    sentence = fact.get("sentence")
    return f"{record.name}: {sentence}" if isinstance(sentence, str) else ""


def _class_key(value: Any, names: Mapping[str, str]) -> str:
    if _is_int_class(value):
        return str(int(value))
    for key, name in names.items():
        if isinstance(value, str) and name.strip().lower() == value.strip().lower():
            return key
    return str(value)


def _dominant_change(
    s: str, *, record: ToolRecord, fact: Mapping[str, Any], names: Mapping[str, str]
) -> str | None:
    """The dominant change stated backwards, or another pair named the dominant one.

    Only a sentence about change between the maps is read: an arrow, "from X
    to Y", a change verb, or a pair in a clause that speaks of flips,
    changes, transitions or contrasts. What the maps are made of ("both maps
    are dominated by X and Y") is not a change, and a change placed in a
    region ("in the south") is not the whole comparison's. A reverse
    direction tied with the top count (``tied_with_reverse``) is no
    contradiction, nor is another pair when the fact's sentence says the
    top count is tied.
    """
    src, dst = fact.get("from_class"), fact.get("to_class")
    if not all(
        isinstance(c, int | str) and not isinstance(c, bool) for c in (src, dst)
    ):
        return None
    p = _parse(s)
    table = tuple(sorted(names.items()))
    about = _fact_sentence(record, fact) or "the tool counts it from map A to map B"
    forward = _change(p, _directed(src, dst, table))
    reverse = _change(p, _directed(dst, src, table))
    n, reverse_n = fact.get("n"), fact.get("reverse_n")
    tied = fact.get("tied_with_reverse") is True or (
        isinstance(n, int | float)
        and isinstance(reverse_n, int | float)
        and n == reverse_n
    )
    if (
        reverse
        and not forward
        and not tied
        and p.in_clause_or_label(_DOMINANT_RE, *reverse)
        and not p.negated(*reverse)
        and not p.example_before(reverse[0])
        and not p.in_clause(_REGION_RE, *reverse)
    ):
        return f"states the dominant change in the opposite direction; {about}"
    if forward or reverse or not names or not _DOMINANT_PAIR_RE.search(s):
        return None
    if _TIE_RE.search(str(fact.get("sentence") or "")):
        return None
    x, y = _class_pattern(src, names), _class_pattern(dst, names)
    if re.search(x, s, re.I) and re.search(y, s, re.I):
        return None
    # Another pair of classes, by name (a bare digit is never a class), named
    # the dominant change.
    keys = [key for key, _ in table]
    top = {_class_key(src, names), _class_key(dst, names)}
    mentions = [
        (m.start(), m.end(), keys[int((m.lastgroup or "c0")[1:])])
        for m in _mention_re(table).finditer(s)
    ]
    for (a0, a1, ka), (b0, b1, kb) in zip(mentions, mentions[1:], strict=False):
        gap = s[a1:b0]
        if ka == kb or {ka, kb} == top or not _PAIR_JOIN_RE.fullmatch(gap):
            continue
        if p.example_before(a0) or p.negated(a0, b1) or p.in_clause(_REGION_RE, a0, b1):
            continue
        joined_by_change = bool(re.search(_ARROW, gap)) or gap.strip().lower() in (
            "to",
            "into",
        )
        if (
            p.in_clause_or_label(_DOMINANT_PAIR_RE, a0, b1)
            and not p.in_clause(_SECONDARY_RE, a0, b1)
            and (
                joined_by_change
                or p.in_clause_or_label(_DOMINANT_CHANGE_NOUN_RE, a0, b1)
            )
        ):
            return f"names {names[ka]} / {names[kb]} the dominant change; {about}"
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
#: A share under half: "more confident on 42.2% of them" is the minority.
_MINORITY_PCT_RE = re.compile(
    r"(?<![\d.,])(?:[1-4]?\d)(?:\.\d+)?\s*(?:%|percent\b|per\s?cent\b|pct\b)", re.I
)
#: A place: "more confident in the north" is about a region, not the side.
_REGION_RE = re.compile(
    r"\b(?:along|near|around|beside)\s+(?:the\s+|a\s+|its\s+)?"
    # the articles too: skipping the optional article must not make "the" a place
    r"(?!(?:the|a|its|median|mean|average|typical|half|most|all|about|roughly|every)\b)"
    r"[A-Za-z]"
    r"|\b(?:in|at|on|inside|towards?|for)\s+"
    r"(?:the\s+|a\s+|its\s+|some\s+|this\s+|that\s+)?(?:(?:far|extreme)\s+)?"
    r"(?:north|south|east|west|northern|southern|eastern|western|"
    r"north-?east|north-?west|south-?east|south-?west|top|bottom|left|right|"
    r"upper|lower|edges?|cent(?:re|er)|middle|corners?|borders?|margins\s+of|"
    r"rows?|columns?|cols?|bands?|strips?|regions?|areas?|zones?|parts?|"
    r"halves|half|clusters?|blocks?|patch(?:es)?|quadrants?)\b",
    re.I,
)
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
    """The side the tool did not find more confident, named the more confident.

    Read as a claim about the whole comparison: a clause that negates it,
    gives it a minority share ("on 42.2% of them"), rests on a mean without a
    majority word, or places it in a region ("A is more confident in the
    north") is not one. With ``side`` "neither" (the two sides more confident
    on as many windows), either side named is flagged.
    """
    side = str(fact.get("side", "")).strip().upper()
    if side not in ("A", "B", "NEITHER"):
        return None
    p = _parse(s)
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
                at = (m.start(), m.end())

                def has(rx: re.Pattern[str], at: tuple[int, int] = at) -> bool:
                    return p.in_clause(rx, *at, own=True)

                if not (
                    has(_NEGATION_RE)
                    or has(_MINORITY_PCT_RE)
                    or (has(_MEAN_RE) and not has(_MAJORITY_RE))
                    or has(_REGION_RE)
                ):
                    yield m

    about = _fact_sentence(record, fact)
    if side == "NEITHER":
        for k in ("A", "B"):
            if next(claims(k), None) is not None:
                return f"names side {k} the more confident; " + (
                    about or "the tool finds neither side more confident"
                )
        return None
    other = "B" if side == "A" else "A"
    if next(claims(side), None) is not None or next(claims(other), None) is None:
        return None
    return f"names side {other} the more confident; " + (
        about or f"the tool finds side {side} more confident"
    )


#: A place a claim may put the differences, by the band of four it names:
#: ``(axis, band)``. A half ("the north", "the northern part") is none.
_PLACES: dict[str, tuple[str, int, str]] = {
    "top": (
        "rows",
        0,
        r"north(?:ern|ernmost|most)?\s+(?:edge|rows?|strip|border|margin|end|band|"
        r"boundary)|top\s+(?:rows?|edge|band|strip|of\s+the\s+(?:grid|map|area|"
        r"image|scene))|upper\s+(?:rows?|edge|band|strip)|first\s+(?:few\s+)?rows",
    ),
    "bottom": (
        "rows",
        3,
        r"south(?:ern|ernmost|most)?\s+(?:edge|rows?|strip|border|margin|end|band|"
        r"boundary)|bottom\s+(?:rows?|edge|band|strip|of\s+the\s+(?:grid|map|area|"
        r"image|scene))|lower\s+(?:rows?|edge|band|strip)|last\s+(?:few\s+)?rows",
    ),
    "left": (
        "cols",
        0,
        r"west(?:ern|ernmost|most)?\s+(?:edge|columns?|cols|strip|border|margin|"
        r"band|boundary)|left\s+(?:edge|columns?|cols|band|strip)",
    ),
    "right": (
        "cols",
        3,
        r"east(?:ern|ernmost|most)?\s+(?:edge|columns?|cols|strip|border|margin|"
        r"band|boundary)|right\s+(?:edge|columns?|cols|band|strip)",
    ),
}
_PLACE_RES = {
    band: re.compile(rf"\b(?:{pattern})\b", re.I)
    for band, (_, _, pattern) in _PLACES.items()
}
#: A clause that places the bulk of the differences: "mostly", "most of",
#: "most differing windows", "concentrated", "clustered", "dominated", a
#: strip, "along". "The most suspect" (a superlative) is none.
_BULK_RE = re.compile(
    r"\b(?:mostly|(?<!the\s)most\s+of|(?<!the\s)most\s+(?:[\w-]+\s+){0,2}?"
    r"(?:differ\w*|changes?|flips?|windows|disagree\w*|cells|pixels)|"
    r"most\s+[\w-]+\s+(?:zone|area|region|part|strip|block|band)|concentrat\w*|"
    r"clustered|dominat\w*|mainly|largely|primarily|majority|bulk|"
    r"predominant\w*|strip|along)\b",
    re.I,
)
#: A clause about a minor part: "a small cluster", "only", "under 50%".
_MINOR_RE = re.compile(
    r"\b(?:only|few|small|tight|single|isolated|occasional|minor|handful)\b"
    r"|\b(?:a|one)\s+(?:cluster|group|patch|block)\b"
    r"|(?<![\d.,])(?:[1-4]?\d)(?:\.\d+)?\s*%",
    re.I,
)
#: A count of some of them: "51 of the 3,807", "51 of them".
_COUNT_OF_RE = re.compile(
    r"(?<![\d.,])(?P<n>\d[\d,]*)\s+of\s+(?:the\s+|all\s+)?"
    r"(?:(?P<m>\d[\d,]*)(?![\d.,]\d)|them\b|these\b|those\b)",
    re.I,
)
#: A clause about one window ("the highest-ranked window is at row 12").
_ONE_WINDOW_RE = re.compile(r"\bwindow\b", re.I)
_MANY_RE = re.compile(
    r"\b(?:windows|differences|flips|changes|disagreements|cells|pixels)\b", re.I
)
_ROWS_RE = re.compile(
    r"\brows?\s*~?(?P<r0>\d+)(?:\s*(?:[-–—]|to|and)\s*~?(?P<r1>\d+))?"
    r"(?![\d%]|[.,]\d|\s*%|\s*(?:[-–—]|to)\s*~?\d)",
    re.I,
)
#: A window's coordinates: "row 12, col 40", not a strip ("row 0, cols 3-48").
_ONE_COLUMN_RE = re.compile(
    r"\s*,?\s*(?:col(?:umn)?|c)\s*~?\d+(?!\d|\s*(?:[-–—]|to)\s*~?\d)", re.I
)
_LEGEND_BEFORE_RE = re.compile(r"[=:]\s*$")
_LEGEND_AFTER_RE = re.compile(r"\s*=")


def _as_grid(value: Any) -> tuple[int, int] | None:
    if (
        isinstance(value, list | tuple)
        and len(value) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in value)
    ):
        return value[0], value[1]
    return None


def _minor_count(p: _Parse, start: int, end: int, n_differing: Any) -> bool:
    """Whether the clause counts under half of the differing windows ("51 of them")."""
    lo, hi, inside = p.region(start, end)
    spans = [(lo, hi, True)] + ([(inside[0], inside[1] + 1, False)] if inside else [])
    for a, b, masked in spans:
        text = p.masked if masked else p.s
        at = a
        while found := p.find(_COUNT_OF_RE, at, b, masked=masked):
            m = _COUNT_OF_RE.match(text, found[0])
            at = found[1]
            if m is None:
                continue
            n = float(m.group("n").replace(",", ""))
            total = n_differing if isinstance(n_differing, int | float) else None
            if total is None and m.group("m"):
                total = float(m.group("m").replace(",", ""))
            if total and n < 0.5 * total:
                return True
    return False


def _bulk_place(p: _Parse, start: int, end: int, n_differing: Any = None) -> bool:
    """Whether ``s[start:end]``, a place, is where a clause puts the bulk.

    The clause (the one a bracket elaborates, for a place in brackets) must
    place the bulk (:data:`_BULK_RE`), and not be a negation, an example, a
    legend ("row 0 = north"), a minor part (a count under half of the
    ``n_differing`` windows among them) or one window.
    """
    s = p.s
    if (
        p.example_before(start)
        or p.negated(start, end)
        or _LEGEND_BEFORE_RE.search(s[max(0, start - 3) : start])
        or _LEGEND_AFTER_RE.match(s, end)
    ):
        return False
    if not p.in_clause(_BULK_RE, start, end) or p.in_clause(_MINOR_RE, start, end):
        return False
    if _minor_count(p, start, end, n_differing):
        return False
    return not (
        p.in_clause(_ONE_WINDOW_RE, start, end)
        and not p.in_clause(_MANY_RE, start, end)
    )


def _concentration(
    s: str, *, run: RunEvidence, record: ToolRecord, fact: Mapping[str, Any]
) -> str | None:
    """A clause that puts most differences where the ``concentration`` fact says
    they are not.

    At the north edge or the top rows when the northmost band holds under
    :data:`BAND_SHARE_MAX` of them (``top_band_share``); at another edge when
    the band holding the most (``max_band``) is another band on that axis;
    in rows too few to hold half of them (``grid`` and ``n_differing``). A
    window's own coordinates ("the highest-ranked window is at row 12, col
    40") are no such claim.
    """
    p = _parse(s)
    about = _fact_sentence(record, fact) or "the tool counts where they are"
    top = fact.get("top_band_share")
    band_fact = fact.get("max_band")
    max_band: Mapping[str, Any] = band_fact if isinstance(band_fact, Mapping) else {}
    for place, (axis, band, _) in _PLACES.items():
        if place == "top":
            if not isinstance(top, int | float) or top >= BAND_SHARE_MAX:
                continue
            why = f"where the northmost band holds {top:.1%} of them"
        else:
            if max_band.get("axis") != axis or max_band.get("band") in (band, None):
                continue
            where = max_band.get("of_grid") or f"band {max_band.get('band')}"
            why = f"while most of them are in the {axis} {where}"
        for m in _PLACE_RES[place].finditer(s):
            if _bulk_place(p, m.start(), m.end(), fact.get("n_differing")):
                return f"places the differences at the {place} ({m.group()}), {why}; {about}"
    # A strip of rows too narrow to hold most of the differing windows.
    grid = _as_grid(fact.get("grid"))
    n = fact.get("n_differing")
    if not grid or not isinstance(n, int | float) or n <= 0:
        return None
    strips = [
        m
        for m in _ROWS_RE.finditer(s)
        if not _ONE_COLUMN_RE.match(s, m.end())
        and _bulk_place(p, m.start(), m.end(), n)
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


_CONFIDENCE_RE = re.compile(
    r"\b(?:margin\w*|confiden\w*|uncertain\w*|certain\w*|decisive\w*|sure|"
    r"separation)\b",
    re.I,
)
#: A comparison with the typical window: "than typical windows", "below the
#: median".
_VERSUS_TYPICAL_RE = re.compile(
    r"\b(?:than|below|under|beneath|above|vs\.?|versus|relative\s+to|"
    r"compared\s+(?:to|with))\s+(?:the\s+|a\s+|an\s+|all\s+)?(?:median|typical|"
    r"average|usual|ordinary|other|remaining|rest)\b",
    re.I,
)
#: Right after a magnitude, a comparison of confidence: "20x less confident",
#: "orders of magnitude more uncertain", "10x below the median".
_RATIO_OF_CONFIDENCE_RE = re.compile(
    r"\s*(?:(?:more|less|smaller|lower|larger|higher|greater|further|closer)\s+"
    r"(?:[\w-]+\s+){0,2}?(?:confiden\w*|certain\w*|uncertain\w*|margins?|"
    r"decisive\w*|sure)|(?:below|under|beneath|above|(?:smaller|lower|larger|"
    r"higher)\s+than)\s+(?:the\s+|a\s+)?(?:median|typical|average|usual))",
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
    """A magnitude of the review windows' confidence against the typical
    window's outside the tool's ratio.

    Only a magnitude that compares confidence ("20x less confident", "orders
    of magnitude more uncertain than typical windows") is read, never any
    "N times" of a sentence that mentions confidence. The range is the
    listed windows' and the whole review set's together.
    """
    bounds = [
        v
        for k in ("listed_low", "listed_high", "review_set_low", "review_set_high")
        if isinstance(v := fact.get(k), int | float) and not isinstance(v, bool)
    ]
    if not bounds:
        return None
    low, high = min(bounds), max(bounds)
    p = _parse(s)
    stated: list[tuple[float, float, re.Match[str]]] = []
    for m in _ORDERS_RE.finditer(s):
        if m.group("bare"):
            stated.append((100.0, math.inf, m))
        elif m.group("one"):
            stated.append((10.0, 10.0, m))
        else:
            o0 = float(m.group("o0"))
            stated.append((10**o0, 10 ** float(m.group("o1") or o0), m))
    for m in _TIMES_RE.finditer(s):
        t0 = float(m.group("t0"))
        stated.append((t0, float(m.group("t1") or t0), m))
    for lo, hi, m in stated:
        about_confidence = _RATIO_OF_CONFIDENCE_RE.match(s, m.end()) or (
            p.in_clause(_CONFIDENCE_RE, m.start(), m.end(), own=True)
            and p.in_clause(_VERSUS_TYPICAL_RE, m.start(), m.end(), own=True)
        )
        if not about_confidence or p.negated(m.start(), m.end()):
            continue
        if lo > high * RATIO_SLACK or hi < low / RATIO_SLACK:
            versus = fact.get("versus", "median")
            return (
                f"states a magnitude of {m.group().strip()}; the tool's ratio to "
                f"the {versus} is {low:g} to {high:g}"
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
    place of concentration the bands contradict (or a strip of rows too
    narrow to hold most of the differences), a magnitude of confidence
    outside a ``margin_ratio``'s range, and a count of unused labels other
    than the tool's.
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
#: A claim that items are listed above or below: "listed above", "the table
#: below", "shown the first few above", "the windows shown below". "As shown
#: above" or "see above" points at prose, and is none.
_LISTED_RE = re.compile(
    r"\b(?:listed|included|enumerated|tabulated)\s+(?:only\s+)?"
    r"(?:the\s+first\s+(?:few|\w+)\s+(?:\w+\s+)?|in\s+the\s+(?:list|table)\s+)?"
    r"(?:above|below)\b"
    r"|\b(?:shown|given|displayed)\s+(?:only\s+)?(?:the\s+first\s+(?:few|\w+)\s+"
    r"(?:\w+\s+)?|in\s+the\s+(?:list|table)\s+)(?:above|below)\b"
    r"|\b(?:windows|rows|items|entries|pairs|labels|cells|ones)\s+(?:are\s+|is\s+)?"
    r"(?:shown|given|displayed)\s+(?:above|below)\b"
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


def _listing_violation(
    s: str, lines: list[str], line_no: int, surface: str
) -> str | None:
    """A claim that items are listed, against the list items of the answer.

    On the ``"web"`` surface the tool results are shown above the answer, so
    a claim of items "above" may point at them and is not checked; a claim of
    items "below", or of the first N listed with no side named, still is.
    """
    first = _FIRST_N_RE.search(s)
    claim = first or _LISTED_RE.search(s)
    if claim is None or _negated_at(s, claim) or re.search(r"\btool", s, re.I):
        return None  # "listed in the tool output" is about the tool, not the answer
    words = claim.group().lower()
    where = "above" if "above" in words else "below" if "below" in words else ""
    if surface == "web" and where == "above":
        return None
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
    results are shown above the answer, a claim of items above is not
    checked.
    """
    out: list[Violation] = []
    lines = answer.split("\n")
    for sent in sentences(answer):
        s = _plain(sent.text)
        detail = _save_violation(s, sent.text, run)
        if detail is None:
            line_no = answer.count("\n", 0, sent.start)
            detail = _listing_violation(s, lines, line_no, run.surface)
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


#: An alpha as written: "alpha = 0.05", "α of 5%", "alpha (e.g. 0.15)".
_ALPHA_VALUE = (
    rf"{_ALPHA_WORD}\s*(?:\(\s*(?:e\.g\.|say)\s*)?(?:=|of|at|to|:|≈|is)?\s*"
    r"(?P<v>0?\.\d+|\d+(?:\.\d+)?\s*%)"
)


def _alpha_used(record: ToolRecord) -> float | None:
    for source in (record.arguments, record.result):
        if isinstance(source, Mapping) and isinstance(source.get("alpha"), int | float):
            return float(source["alpha"])
    return None


def _alpha_value(raw: str) -> tuple[float, float]:
    """An alpha as written, and half a unit of its last decimal."""
    raw = raw.replace(" ", "")
    percent = raw.endswith("%")
    digits = raw[:-1] if percent else raw
    tol = 0.5 * 10.0 ** -len(digits.partition(".")[2])
    return (float(digits) / 100, tol / 100) if percent else (float(digits), tol)


def _alphas_asked(run: RunEvidence) -> list[float]:
    """The alphas the user's messages name ("certify at alpha 0.1")."""
    return [
        _alpha_value(m.group("v"))[0]
        for text in run.user_messages
        for m in re.finditer(_ALPHA_VALUE, text, re.I)
    ]


#: A sentence about an alpha split over levels (Bonferroni's per-level alpha).
_PER_LEVEL_RE = re.compile(
    r"\b(?:bonferroni|holm|per[\s-]level|each|per|split|divided|levels?)\b|/", re.I
)


def _allowed_alpha(
    value: float, tol: float, alphas: Iterable[float], per_level: bool
) -> bool:
    """Whether ``value`` is an alpha in force, or, ``per_level``, one split over k.

    ``alpha / k`` for a whole k of at least 2, at the precision written, in a
    sentence about levels, is the per-level alpha of a Bonferroni rule (0.01
    for 0.05 over 5 levels): an explanation of the rule in force, not
    another alpha.
    """
    for alpha in alphas:
        # an alpha in force matches exactly: at a written "0.1", half a unit of
        # the last decimal (0.05) would pass the canonical looser alpha
        if abs(value - alpha) <= 1e-9:
            return True
        if per_level and 0 < value < alpha:
            k = round(alpha / value)
            if any(
                j >= 2 and abs(alpha / j - value) <= tol + 1e-12
                for j in (k - 1, k, k + 1)
            ):
                return True
    return False


def _post_hoc_alpha(s: str, record: ToolRecord, run: RunEvidence) -> bool:
    """Another alpha offered or applied after the bounds were seen.

    The alpha the tool ran at and the alphas the user asked for are allowed,
    and so is either split over k levels (a Bonferroni explanation); an
    alpha said to be fixed in advance is the rule, not a choice.
    """
    if not (_OFFER_RE.search(s) or re.search(r"\bcertif\w*", s, re.I)):
        return False
    used = _alpha_used(record)
    alphas = ([] if used is None else [used]) + _alphas_asked(run)
    for m in _unnegated(
        r"\b(?:looser|different|another|higher|larger|relaxed|less\s+strict|laxer|"
        rf"other|new|bigger)\s+{_ALPHA_WORD}|{_ALPHA_VALUE}",
        s,
    ):
        if _IN_ADVANCE_RE.search(_clause(s, m.start(), m.end())):
            continue
        if m.group("v") and _allowed_alpha(
            *_alpha_value(m.group("v")), alphas, bool(_PER_LEVEL_RE.search(s))
        ):
            continue
        return True
    return False


def _rule_switch(s: str, record: ToolRecord, run: RunEvidence) -> bool:
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


#: Advice to draw a probability sample, which is what a certified zone needs.
_PROBABILITY_SAMPLE_RE = re.compile(
    r"\brandom(?:ly)?\b|\bprobability[\s-]+(?:based\s+)?(?:samples?|sampling|design)\b",
    re.I,
)


def _certify_nonrandom(s: str, record: ToolRecord, run: RunEvidence) -> bool:
    """An offer to certify a zone from labels of a design that is not random.

    Advice to draw a random or probability sample ("a simple random
    sample", "a probability sample") for it is the correct advice.
    """
    if _PROBABILITY_SAMPLE_RE.search(s) or not _OFFER_RE.search(s):
        return False
    return any(_unnegated(r"\bcertif\w*", s))


def _error_rate_without_labels(s: str, record: ToolRecord, run: RunEvidence) -> bool:
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


def _error_rate_regression(s: str, record: ToolRecord, run: RunEvidence) -> bool:
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


def _subset_sufficient(s: str, record: ToolRecord, run: RunEvidence) -> bool:
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


def _winner(s: str, record: ToolRecord, run: RunEvidence) -> bool:
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


def _combined_statistic(s: str, record: ToolRecord, run: RunEvidence) -> bool:
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


#: "the earlier map", "the later image": map A or B, not another date.
_THE_MAPS_BEFORE_RE = re.compile(
    r"\b(?:the|this|that|each|either|both|its|their|same|your|these|those)\s+$",
    re.I,
)


def _another_date(s: str, record: ToolRecord, run: RunEvidence) -> bool:
    """Another date offered as what settles which map is right.

    "The earlier map" or "the later image" names map A or B, not a new date;
    a reference at the maps' own dates is what does settle it.
    """
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
        and not (
            m.group(1).lower() in ("later", "earlier")
            and _THE_MAPS_BEFORE_RE.search(s[max(0, m.start() - 12) : m.start()])
        )
        for m in _unnegated(
            r"\b(third|another|additional|extra|later|earlier|intermediate)\s+"
            r"(?:dated\s+)?(?:date|map|image|acquisition|scene|time\s*(?:point|step)|"
            r"snapshot|epoch)s?\b",
            s,
        )
    )


#: A detector per ``forbidden_claims`` id: does one sentence make that claim
#: about the result of that tool, in this run? An id without one (the
#: contract's ``simple_random_interval_for_stratified_design``: the package's
#: own interval is a Wilson interval on the design's effective sample size,
#: which no wording tells from a naive one) is ignored.
FORBIDDEN_DETECTORS: dict[str, Callable[[str, ToolRecord, RunEvidence], bool]] = {
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
            if FORBIDDEN_DETECTORS[claim_id](s, record, run):
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
    their first five letters, and its numbers). Per result, the first
    :data:`MUST_STATE_MAX` sentences are read, and one longer than
    :data:`MUST_STATE_MAX_WORDS` words is not (the contract keeps scope detail
    out of ``must_state``; it is logged).
    """
    out: list[Violation] = []
    have = _terms(answer)
    seen: set[str] = set()
    read: dict[int, int] = {}
    for record, item in run.contract("must_state"):
        sentence = item.get("sentence") if isinstance(item, Mapping) else item
        if not isinstance(sentence, str) or not sentence.strip():
            continue
        if len(sentence.split()) > MUST_STATE_MAX_WORDS:
            logger.warning(
                "%s: a must_state sentence of %d words is not checked (at most %d)",
                record.name,
                len(sentence.split()),
                MUST_STATE_MAX_WORDS,
            )
            continue
        read[id(record)] = read.get(id(record), 0) + 1
        if read[id(record)] > MUST_STATE_MAX or sentence in seen:
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
