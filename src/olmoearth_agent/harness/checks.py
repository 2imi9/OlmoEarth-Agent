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
  row band 0 of four, ``max_band`` ``{axis, band, of_grid, share}`` and
  ``row_order``: row band 0 is the northmost only for ``"north_to_south"``,
  and a fact without the key is read as north-up), ``margin_ratio``
  (``listed_low``, ``listed_high``, ``review_set_low``, ``review_set_high``,
  ``versus``) and ``unused_labels`` (``requested``, ``planned``, ``n``).
  ``whole_map_estimate`` and any other id are shown, never checked.
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
#: The claim check (``harness/claim_check.py``): the agent's own model reads
#: the answer against the run. Not in :data:`CHECKS`, which are the rules: it
#: is a model call, which the answer-check middleware makes itself.
CLAIMS = "claims"

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
#: How a required statement the answer lacks is appended to it.
TOOL_NOTE = "Note from the tool:"
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
    def case_words(self) -> frozenset[str]:
        """The words of the run other than its tools' evidence.

        The user's messages, every argument, and every result key but an
        ``evidence*`` one (``evidence_scope`` describes the experiments a
        tool's claim rests on, not this case): a name only the evidence
        holds is the evidence's (``evidence_outside_its_scope``).
        """
        texts = [*self.user_messages]
        for record in self.tools:
            for source in (record.arguments, record.result):
                texts.extend(
                    v for k, v in _strings(source) if not k.startswith("evidence")
                )
        return frozenset(w for t in texts for w in re.findall(r"[A-Za-z0-9]+", t))

    @cached_property
    def review_bands(self) -> tuple[re.Pattern[str] | None, re.Pattern[str] | None]:
        """The names of the bands a review set is forbidden for, and of the others.

        The bands are the ones every ``review_set_for_unthresholded_regression``
        reason of the run names (``'sample_number' (declared range ...)``);
        the others, every other property the run's results name. Each is
        matched by its property name and by the names of the maps that carry
        it (a prediction's "KarstNumber--01-01-2025--..." reads as
        "KarstNumber"), a name both carry by neither. ``(None, None)`` when
        no reason names a band or the run holds no other property: an offer
        is then read whatever it names.
        """
        bands: set[str] = set()
        for _record, item in self.contract("forbidden_claims"):
            if isinstance(item, Mapping) and item.get("id") == (
                "review_set_for_unthresholded_regression"
            ):
                bands.update(
                    re.findall(r"'([^']+)' \(declared range", str(item.get("why", "")))
                )
        if not bands:
            return None, None
        aliases: dict[str, set[str]] = {}
        for record in self.tools:
            if record.ok:
                for props, names in _property_names(record.result):
                    for prop in props:
                        aliases.setdefault(prop, {prop}).update(names)
        mine = set().union(*(aliases.get(b, {b}) for b in bands))
        theirs = (
            set().union(
                *(names for prop, names in aliases.items() if prop not in bands)
            )
            - mine
        )
        if not theirs:
            return None, None
        return _names_re(mine), _names_re(theirs)

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
        (``saved_to``) is a written file too. Only a string that is a path
        counts (:func:`_is_path`), never a note that names a file.
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
                if not _is_path(value):
                    continue  # a URL (a tile, a page) or a note naming a file
                echoed = key in given.get(value, ()) and not _names_output(key)
                if not echoed:
                    add(key, value)
        return out

    @cached_property
    def list_files(self) -> frozenset[str]:
        """The paths a tool of this run names as holding a list (:func:`_holds_list`).

        Any path in a successful result, an echoed input included (the design
        a tool read), under a key that names a list; and the harness's spill
        of a result, which holds that result's listings whole. A note that
        names a file is no path (:func:`_is_path`): the review tools'
        ``listing_note`` says "review_set_evidence.json (evidence_detail_path)
        holds evidence text only, no windows" under a key that names a
        listing, and read as a path it made the evidence file a list's.
        """
        out: set[str] = set()
        for record in self.tools:
            named = [("saved_to", record.spilled_to)] if record.spilled_to else []
            if record.ok:
                named += [(k, v) for k, v in _strings(record.result) if _is_path(v)]
            out.update(path for key, path in named if _holds_list(key, path))
        return frozenset(out)


def _property_names(obj: Any, depth: int = 0) -> Iterator[tuple[list[str], set[str]]]:
    """``(properties, names)`` of every record in a result that names its property.

    A record with ``property_names`` (or ``property_name``) and a map's name
    (``prediction_name``, ``name``): the name and its leading word
    ("KarstNumber" of "KarstNumber--01-01-2025--12-31-2025--PA regular"), and
    the property with its underscores read as spaces.
    """
    if depth > 6:
        return
    if isinstance(obj, Mapping):
        found = obj.get("property_names")
        props = (
            [p for p in found if isinstance(p, str)]
            if isinstance(found, list)
            else (
                [obj["property_name"]]
                if isinstance(obj.get("property_name"), str)
                else []
            )
        )
        if props:
            names: set[str] = set()
            for key in ("prediction_name", "name"):
                value = obj.get(key)
                if isinstance(value, str) and value.strip():
                    names.add(value.strip())
                    lead = re.match(r"[A-Za-z][A-Za-z0-9]{3,}", value.strip())
                    if lead:
                        names.add(lead.group())
            for prop in props:
                names |= {prop, prop.replace("_", " ")}
            yield props, names
        for value in obj.values():
            yield from _property_names(value, depth + 1)
    elif isinstance(obj, list):
        for value in obj:
            yield from _property_names(value, depth + 1)


def _names_re(names: Iterable[str]) -> re.Pattern[str] | None:
    """One pattern for any of ``names`` as whole words, longest first."""
    kept = sorted({n for n in names if len(n) >= 3}, key=len, reverse=True)
    if not kept:
        return None
    return re.compile(
        r"(?<![A-Za-z0-9_])(?:"
        + "|".join(re.escape(n) for n in kept)
        + r")(?![A-Za-z0-9_])",
        re.I,
    )


def _is_path(value: str) -> bool:
    """Whether a result's string is a file path, not a URL or a note naming a file.

    A path holds no whitespace ("/w/review_list_ab12.csv"), or is rooted and
    ends in the file ("/Users/me/My Data/scores.json"); prose that mentions a
    file ("review_set_evidence.json (evidence_detail_path) holds evidence text
    only") is neither.
    """
    text = value.strip()
    if not text or "://" in text or not _FILE_RE.search(text):
        return False
    if not re.search(r"\s", text):
        return True
    return bool(_ROOTED_PATH_RE.fullmatch(text))


#: A rooted path that may hold spaces, ending in a file's extension.
_ROOTED_PATH_RE = re.compile(
    r"(?:[A-Za-z]:)?[/\\~][^\n;:,()\[\]]*?\.(?:json|csv|tsv|tif|tiff|geojson|gpkg|txt|"
    r"md|png|jpe?g|qlr|xml|parquet|npz|npy|zip|shp|html|ya?ml|log|nc)",
    re.I,
)

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
        self._straight = [i for i, ch in enumerate(s) if ch == '"']
        self._left = [i for i, ch in enumerate(s) if ch == "“"]
        self._right = [i for i, ch in enumerate(s) if ch == "”"]

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

    def has(
        self,
        pattern: re.Pattern[str],
        lo: int,
        hi: int,
        *,
        skip: tuple[int, int] | None = None,
    ) -> bool:
        """Whether ``pattern`` matches in ``s[lo:hi]`` (read on the sentence).

        A bisection over the pattern's matches in the whole sentence, found
        once: a detector that asks it per match of a long listing stays
        linear, where searching a slice per match does not.
        """
        return self.find(pattern, lo, hi, masked=False, skip=skip) is not None

    def last_end(self, pattern: re.Pattern[str], lo: int, hi: int) -> int:
        """Where the last match of ``pattern`` in ``s[lo:hi]`` ends, or ``lo``."""
        spans = self._spans(pattern, False)
        i = bisect.bisect_left(spans, (hi, -1)) - 1
        while i >= 0 and spans[i][0] >= lo:
            if spans[i][1] <= hi:
                return spans[i][1]
            i -= 1
        return lo

    def quoted(self, start: int) -> bool:
        """Whether ``s[start:]`` sits inside double quotes (:func:`_quoted`)."""
        straight = bisect.bisect_left(self._straight, start)
        left = bisect.bisect_left(self._left, start)
        right = bisect.bisect_left(self._right, start)
        return straight % 2 == 1 or left > right

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


def _label_start(p: _Parse, lo: int) -> int:
    """Where the label a colon ends just before ``lo`` starts, or ``lo``.

    "**Where they differ:** effectively everywhere" and "Full list of the
    differing windows: <file>" state their subject in the label.
    """
    at = lo - 1
    while at >= 0 and p.masked[at].isspace():
        at -= 1
    return p.span(at, at + 1)[0] if at >= 0 and p.masked[at] == ":" else lo


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
#: A place named by the compass, which a grid's row order may turn around.
_COMPASS_RE = re.compile(r"(?:north|south)", re.I)
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
    # A fact from before row_order was read as north-up; one that carries it
    # names row band 0 north only for "north_to_south" (the fix-r8 review: a
    # rewrite said "northmost" of a grid of chips with no georeference).
    order = fact.get("row_order", "north_to_south")
    for place, (axis, band, _) in _PLACES.items():
        if place == "top":
            if not isinstance(top, int | float) or top >= BAND_SHARE_MAX:
                continue
            if order == "north_to_south":
                why = f"where the northmost band holds {top:.1%} of them"
            elif order == "south_to_north":
                why = (
                    f"where row band 0 (the grid's first rows, the south edge "
                    f"here) holds {top:.1%} of them"
                )
            else:
                why = (
                    f"where row band 0 (the grid's first rows) holds {top:.1%} of "
                    "them, and the scores carry no georeference, so no band is "
                    "north or south"
                )
        else:
            if max_band.get("axis") != axis or max_band.get("band") in (band, None):
                continue
            where = max_band.get("of_grid") or f"band {max_band.get('band')}"
            why = f"while most of them are in the {axis} {where}"
        for m in _PLACE_RES[place].finditer(s):
            if order == "south_to_north" and _COMPASS_RE.match(m.group()):
                continue  # north is the last row band here, south the first
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


#: Words of a result's key that say only where a file is, not what it holds.
_PLACE_KEY_WORDS = _OUTPUT_KEY_WORDS | {
    "path",
    "paths",
    "file",
    "files",
    "filename",
    "name",
    "dir",
    "directory",
    "location",
    "to",
    "at",
}


def _holds_list(key: str, path: str) -> bool:
    """Whether a result names ``path`` as holding a list, by the key it is under.

    A key that names its content says what the file holds, and it is a list
    only when that content is one (:data:`_LIST_FILE_WORDS`: ``differing_path``,
    ``labels_csv_path``, a ``review_list_path``). ``evidence_detail_path``
    names evidence and ``scores_path`` scores, whatever the file is called:
    exp86 round 8 (B8/cluster runs 2 and 3) said the 819-window review list
    was saved in ``review_set_evidence.json``, which holds evidence text only,
    and the check read "review" in its name as a list. Only a key that names
    no content (``out_path``, the harness's ``saved_to``) leaves the file's
    name to tell.
    """
    words = {w.lower() for w in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", key)}
    content = words - _PLACE_KEY_WORDS
    if content:
        return bool(content & _LIST_FILE_WORDS)
    stem = os.path.basename(path).lower()
    return stem.endswith((".csv", ".tsv")) or any(
        w in stem for w in ("list", "review", "differing", "to_label")
    )


#: A list, a ranking or the windows, as what a claim puts in a file.
_LIST_NOUN_RE = re.compile(
    r"\b(?:lists?|listing|ranking|ranked\s+(?:windows|list)|review\s+(?:set|list))\b",
    re.I,
)
_WINDOWS_RE = re.compile(r"\bwindows\b", re.I)
#: A comma between clauses, not one inside a number ("3,807").
_COMMA_RE = re.compile(r",(?!\d)")
_SCORES_WORD_RE = re.compile(r"\bscores?\b", re.I)


#: A count of windows, not a list of them: "the count of wrong windows in
#: <file>" (exp86 round 9, B6/files run 2, of the user's own labels file).
_COUNT_OF_WINDOWS_RE = re.compile(
    r"\b(?:count|number|share|fraction|proportion|rate|percentage|how\s+many)\s+of\b",
    re.I,
)


def _is_listy(subject: str) -> bool:
    """Whether a claim's subject is a list: a list or ranking, or windows
    ("all 819 windows") that are not scores ("scores for all windows") and
    not a count ("the count of wrong windows")."""
    return bool(_LIST_NOUN_RE.search(subject)) or bool(
        _WINDOWS_RE.search(subject)
        and not _SCORES_WORD_RE.search(subject)
        and not _COUNT_OF_WINDOWS_RE.search(subject)
    )


def _subject(s: str, lo: int, at: int) -> str:
    """What a claim at ``at`` says of: its clause before it, from the last comma
    (brackets kept: "Full list (819 windows) and per-window scores are saved")."""
    commas = [m.end() for m in _COMMA_RE.finditer(_parse(s).masked, lo, at)]
    return s[(commas[-1] if commas else lo) : at]


def _not_a_list_file(named: list[str], run: RunEvidence) -> str:
    """Why the files an answer says hold a list do not: what the run says they are."""
    for f in named:
        for key, path in run.written_files:
            if _file_matches(f, path):
                return (
                    f"says {f} holds the list; the tool that wrote it names it "
                    f"{key}, not a list"
                    + (
                        f" (a list is in {', '.join(sorted(os.path.basename(p) for p in run.list_files))})"
                        if run.list_files
                        else ""
                    )
                )
    return f"says {named[0]} holds the list; no tool of this run names it as holding a list"


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
        listy = bool(_LIST_NOUN_RE.match(m.group().split()[-1]))
    else:
        listy = _is_listy(_subject(s, lo, m.start()))
    pool = [p for _, p in written if not listy or p in run.list_files]
    if named:
        if any(_file_matches(f, p) for f in named for p in pool):
            return None
        if listy:
            return _not_a_list_file(named, run)
        return (
            f"says {named[0]} holds what was saved; no tool of this run reported "
            "writing it"
        )
    if pool:
        return None
    what = "a list" if listy and written else "a file"
    return f"says something was saved; no tool of this run reported writing {what}"


#: A claim that a list sits in a file, without a save verb: "are listed in",
#: "full list in", "Full list of the differing windows: <file>".
_LISTED_IN_RE = re.compile(
    r"\b(?:listed|enumerated|tabulated|ranked|kept|held|available|included|"
    r"recorded)\s+(?:\w+\s+){0,3}?(?:in|at|under|inside|within)\b",
    re.I,
)
#: Right before the file: "in", "at", "under" or a label's colon, with the
#: path's leading directories ("in .../workspace/").
_AT_FILE_RE = re.compile(
    r"(?:\b(?:in|at|under|inside|within)\b|:)\s*[(\[]?\s*(?:the\s+(?:file|json|"
    r"csv)\s+)?\S*$",
    re.I,
)


def _list_location_violation(s: str, run: RunEvidence) -> str | None:
    """A list, a ranking or the windows said to be in a named file that holds none.

    The file must be one a tool of the run names as holding a list
    (:attr:`RunEvidence.list_files`): a file the run wrote that its tool
    names as something else (``evidence_detail_path``, ``scores_path``) is
    a violation, and so is a file no tool named. A list named in a label
    ("Full list of the differing windows: <file>") is read with its label.
    """
    p = _parse(s)
    for f in _FILE_RE.finditer(s):
        lo = _label_start(p, p.region(f.start(), f.end())[0])
        pre = s[lo : f.start()]
        claim = _LISTED_IN_RE.search(pre)
        # "listed in", or a list (or the windows) right before the file ("the
        # full ranked list is in <file>", where "ranked" is no verb; "all 3,807
        # differing windows are in <file>")
        verb = claim is not None and _is_listy(_subject(s, lo, lo + claim.start()))
        noun = bool(
            (_LIST_NOUN_RE.search(pre) or _WINDOWS_RE.search(pre))
            and _AT_FILE_RE.search(pre)
            and _is_listy(_subject(s, lo, lo + len(pre)))
        )
        if not (verb or noun):
            continue
        if _negated(pre) or _HEDGE_RE.search(pre):
            continue
        if any(_file_matches(f.group(), path) for path in run.list_files):
            continue
        # a file the user named holds what the user says it does ("labelled in
        # F3/labels_random_300_s0.csv"): no tool has to name it
        if any(os.path.basename(f.group()) in text for text in run.user_messages):
            continue
        return _not_a_list_file([f.group()], run)
    return None


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
    file; a claim that a *list* (a ranking, the windows) was saved, or is
    listed in a named file, needs a file a tool names as holding a list: by
    its key, or by its name only under a key that names no content
    (:func:`_holds_list`). A claim that items are listed above
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
        if detail is None and not _SAVE_RE.search(s):
            detail = _list_location_violation(s, run)
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


#: A change of alpha or rule stated as what the guarantee does not cover, the
#: tool's own next step: "a looser alpha on these same labels would fall
#: outside what the guarantee covers", "... would void the guarantee".
_OUTSIDE_GUARANTEE_RE = re.compile(
    r"\b(?:void\w*|invalidat\w*|break\w*|forfeit\w*|los(?:e|es|ing)|undermin\w*)\s+"
    r"(?:\w+\s+){0,2}?guarantee"
    r"|\b(?:fall\w*|lies?|is|are|be)\s+outside\s+(?:\w+\s+){0,4}?guarantee"
    r"|\bguarantee\s+(?:\w+\s+){0,3}?(?:does\s*n[o']t|would\s*n[o']t|no\s+longer)\s+"
    r"(?:cover|hold|apply)"
    r"|\bnot\s+covered\s+by\s+(?:\w+\s+){0,2}?guarantee",
    re.I,
)


def _post_hoc_alpha(s: str, record: ToolRecord, run: RunEvidence) -> bool:
    """Another alpha offered or applied after the bounds were seen.

    The alpha the tool ran at and the alphas the user asked for are allowed,
    and so is either split over k levels (a Bonferroni explanation); an
    alpha said to be fixed in advance is the rule, not a choice.
    """
    if not (_OFFER_RE.search(s) or re.search(r"\bcertif\w*", s, re.I)):
        return False
    if _OUTSIDE_GUARANTEE_RE.search(s):
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
    return (
        bool(_OFFER_RE.search(s))
        and not _OUTSIDE_GUARANTEE_RE.search(s)
        and any(
            _unnegated(
                r"\b(?:re-?run|run|try|re-?test|redo|switch\w*|use|apply|repeat)\b"
                rf"[^.;]{{0,50}}\b{rules}",
                s,
            )
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


#: A negation that governs a claim from after it: "labels for one date
#: cannot settle it", "... is not enough".
_NEGATED_AFTER_RE = re.compile(
    r"\b(?:cannot|can't|can\s+not|won't|will\s+not|would\s*n[o']t|does\s*n[o']t|"
    r"do\s*n[o']t|is\s*n[o']t|are\s*n[o']t|is\s+not|are\s+not|never|not\s+enough|"
    r"(?:should|must|may|might|could|need)\s*(?:n[o']t|not)|"
    r"insufficient|blocked|impossible|not\s+possible|"
    r"only\s+(?:tells?|shows?|measures?|says?))\b",
    re.I,
)
#: The words after a claim in which a negation still governs it.
_NEGATION_REACH = 6


def _negated_close(
    p: _Parse, start: int, end: int, *, besides: re.Pattern[str] | None = None
) -> bool:
    """Whether a negation governs ``s[start:end]``: before it in its clause, or
    within :data:`_NEGATION_REACH` words after it.

    Clause-level negation (:meth:`_Parse.negated`) lets a hedge far along a
    long clause exempt a claim: exp86 round 7 (B7/files run 1), "To settle
    it, provide labels for one or both dates, or treat this as a
    change-detection layer rather than a contest", read as negated by its
    last words. A negation inside a match of ``besides`` governs something
    else ("without a threshold, I can still build a review set").
    """
    lo, hi = p.span(start, end)
    inside = p.bracket_at(start, end) is not None
    at = lo
    while found := p.find(_NEGATION_RE, at, start, masked=not inside):
        own = besides and p.find(besides, found[0], start, masked=not inside)
        if not own or own[0] != found[0]:
            return True
        at = found[1]
    text = p.s if inside else p.masked
    # read in place (pos, endpos): a slice per match of a long listing is quadratic
    after = _REACH_RE.match(text, end, hi)
    return bool(after and _NEGATED_AFTER_RE.search(after.group()))


#: The words after a claim that :func:`_negated_close` reads.
_REACH_RE = re.compile(rf"(?:\W*\w+){{0,{_NEGATION_REACH}}}")


#: A labelling sample drawn from the least confident windows only: "a
#: targeted labelling sample", "from the lower-confidence windows", "the most
#: uncertain windows", a review set. A confidence-stratified design samples
#: every stratum and weights them, which :data:`_WEIGHTED_DESIGN_RE` exempts.
_LOW_CONFIDENCE_DRAW_RE = re.compile(
    r"\btargeted\b"
    r"|\b(?:lower|low|least|lowest|less)[\s-]+(?:confidence|confident|certain|"
    r"margin)\b(?:[\s-]+first)?"
    r"|\b(?:most|more)[\s-]+(?:uncertain|ambiguous|suspect|doubtful)\b"
    r"|\breview[\s_-]+(?:set|list)\b",
    re.I,
)
#: A promise that an error rate is sound: "defensible error rates", "an
#: honest estimate", "the estimate stays unbiased".
_SOUND_RATE_RE = re.compile(
    r"\b(?:defensible|honest|unbiased|design-unbiased|valid|trustworthy|"
    r"representative|reliable)\s+(?:\w+[\s-]+){0,2}?(?:error\s+rates?|error|"
    r"estimates?|rates?|accuracy)\b"
    r"|\b(?:error\s+rates?|estimates?|accuracy)\s+(?:\w+\s+){0,2}?(?:stays?|remains?|"
    r"is|are|will\s+be|would\s+be)\s+(?:\w+\s+)?(?:unbiased|honest|defensible|valid)\b",
    re.I,
)
#: A design that samples every stratum and weights it: its estimate is sound.
_WEIGHTED_DESIGN_RE = re.compile(
    r"\bstratif\w*|\bweight\w*|\bevery\s+stratum\b|\ball\s+(?:the\s+)?strata\b|"
    r"\boversampl\w*|\bNeyman\b",
    re.I,
)


#: A sound rate said to come from such a draw: "... from labelling the
#: review set", "... by labelling the most uncertain windows".
_FROM_DRAW_RE = re.compile(
    r"\W*(?:\w+\W+){0,4}?(?:from|by|via|through|with)\s+(?:labell?ing\s+)?"
    r"(?:only\s+)?(?:a\s+|the\s+)?(?:\w+\s+)?(?:targeted|review[\s_-]+set|"
    r"(?:most|more)[\s-]+(?:uncertain|ambiguous|suspect)|(?:lower|low|least|"
    r"lowest)[\s-]+(?:confidence|margin))",
    re.I,
)
#: Where one offer ends and another begins inside a clause: ", or run ...".
#: A step of the same offer (", you label them, and I compute ...") is none.
_NEXT_OFFER_RE = re.compile(
    r",\s*or\s+|\bor\s+(?:else\s+)?(?:run|build|use|try)\b", re.I
)


def _low_confidence_sample_sound(s: str) -> bool:
    """A sample of the least confident windows promised a sound error rate.

    exp86 round 8 (B3/studio run 3) offered to "design a targeted labeling
    sample from the lower-confidence windows of each map ... and compute
    defensible error rates per map"; round 6 (B4/cluster run 2) said "because
    this is a targeted (low-confidence-first) design, the estimate stays
    unbiased". A rate over the least confident windows describes those
    windows, not the map. The draw must be described before the promise in
    its clause (brackets kept), or named as its source right after it
    ("... by labelling the most uncertain windows"); another offer between
    them ("an honest error estimate, or run a per-class review set") is
    another claim. A stratified or weighted design named anywhere in the
    sentence is what gives the map's rate, and passes (round 5, B4/cluster
    run 2: "a confidence-stratified sample: ... the most suspect windows get
    labelled first while the estimate stays unbiased").
    """
    if _WEIGHTED_DESIGN_RE.search(s):
        return False
    p = _parse(s)
    for m in _SOUND_RATE_RE.finditer(s):
        if _negated_close(p, m.start(), m.end()) or p.example_before(m.start()):
            continue
        lo, hi, _ = p.region(m.start(), m.end())
        # the draw is read after the last other offer before the promise, in
        # place: slicing the clause per match is quadratic on a long listing
        before = p.last_end(_NEXT_OFFER_RE, lo, m.start())
        if p.has(_LOW_CONFIDENCE_DRAW_RE, before, m.start()) or _FROM_DRAW_RE.match(
            p.s, m.end(), hi
        ):
            return True
    return False


def _subset_sufficient(s: str, record: ToolRecord, run: RunEvidence) -> bool:
    """A part of the sample offered as enough, or a sample of the least
    confident windows offered as a sound error rate.

    "Not enough" is no offer: "the more confident side is right only 51-70%
    of the time, not enough to pick a winner" (round 3, B3/cluster run 3).
    """
    return _low_confidence_sample_sound(s) or any(
        _unnegated(
            r"\b(?:first|top|only|just)\s+(?:the\s+)?(?:\d+|few|some|handful)\b"
            r"[^.;]{0,50}\b(?<!not\s)(?:enough|suffic\w*|would\s+do|is\s+fine|"
            r"are\s+fine)\b"
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


def _quoted(s: str, start: int) -> bool:
    """Whether ``s[start:]`` sits inside double quotes: a phrase cited, not claimed.

    Counted by bisection over the quotes found once per sentence: counting
    from the sentence's start per match was quadratic on a long listing.
    """
    return _parse(s).quoted(start)


#: A statement about what one correlation can say, not a reading of it:
#: "only whether they rise and fall together is meaningful", "it cannot say
#: where they agree", "the interval is too wide to tell".
_CORRELATION_LIMIT_RE = re.compile(
    r"\b(?:whether|if|how|cannot|can't|cant|could\s*n[o']t|unclear|uncertain\w*|"
    r"too\s+(?:few|wide|small)|not\s+enough|detect\w*|no\s+basis|"
    r"no\s+location|has\s+no\s+(?:place|location)|not\s+(?:known|established)|"
    r"(?:does|do|did)\s*n[o']t\s+(?:say|show|tell|establish|prove|locate|mean)|"
    r"says?\s+only|only\s+(?:says?|tells?|shows?|measures?|reports?)|(?:wrong|"
    r"misleading|"
    r"incorrect|invalid)\s+to)\b",
    re.I,
)
#: A statement that the correlation cannot establish what follows it.
_DOES_NOT_SHOW_THAT_RE = re.compile(
    r"\b(?:does|do|did|would|can|could)\s*(?:n[o']t|not)\s+(?:by\s+itself\s+)?"
    r"(?:show|prove|mean|imply|establish|demonstrate|tell\s+you)\s+(?:that\b|whether\b)?",
    re.I,
)
#: Co-variation predicated of the two maps, either way: "they do not rise and
#: fall together", "essentially no relationship", "their values rise and
#: fall independently", "one going high says nothing about the other",
#: "they strongly agree", "a moderate degree of overlap".
_COVARIATION_CLAIM_RE = re.compile(
    r"\b(?:do|does|did)\s*(?:not|n't)\s+(?:\w+\s+){0,2}?(?:agree|co-?vary|"
    r"correlate|track|relate|move\s+together|go\s+together|"
    r"rise\s*(?:and|&|/)\s*fall)\b"
    r"|\b(?:vary|varies|move|moves|rise\s+and\s+fall|rise/fall|behave|behaves|"
    r"fluctuate|fluctuates|change|changes)\s+independently\b"
    r"|\b(?:are|'re|is|look|looks|seem|seems|appear|appears|essentially|"
    r"effectively|largely|basically|statistically|mutually)\s+(?:\w+\s+)?"
    r"(?:independent|unrelated|decoupled|disconnected)\b"
    r"|\b(?:uncorrelated|anti-?correlated)\b"
    r"|\b(?:are|'re|is)\s+not\s+(?:\w+\s+)?(?:correlated|related|associated|"
    r"linked|connected)\b"
    r"|\bnothing\s+in\s+common\b"
    r"|\b(?:no|little|zero|essentially\s+no|virtually\s+no|hardly\s+any|almost\s+no)"
    r"\s+(?:real\s+|meaningful\s+|linear\s+|spatial\s+|clear\s+)?(?:relationship|"
    r"relation|association|connection|link|co-?variation|correspondence|"
    r"coupling)\b"
    r"|\b(?:says?|tells?|reveals?|predicts?)\s+(?:you\s+|us\s+)?(?:essentially\s+|"
    r"almost\s+|virtually\s+|little\s+or\s+)?nothing\s+about\s+the\s+other\b"
    r"|\b(?:strongly|closely|highly|well|largely|broadly|moderately|weakly|"
    r"tightly|clearly|barely|hardly|scarcely|poorly)\s+(?:agree|agrees|correlated|"
    r"co-?vary|track|match|related|aligned|coupled)\b"
    r"|\b(?:agree|agrees|disagree|disagrees|track|tracks|match|matches)\s+"
    r"(?:well|closely|strongly|poorly|weakly|completely|entirely|totally|"
    r"substantially|largely|broadly)\b"
    r"|\b(?:completely|entirely|totally|substantially|mostly)\s+(?:agree|"
    r"disagree)\w*"
    r"|\b(?:strong|moderate|weak|high|good|close|substantial|partial|poor|clear)\s+"
    r"(?:degree\s+of\s+)?(?:agreement|overlap|co-?variation|correspondence|"
    r"association|relationship|alignment|coupling)\b"
    r"|\b(?:they|maps|models|results|surfaces|values|layers|bands|both)\s+"
    r"(?:\w+\s+){0,2}?(?:rise\s+and\s+fall|rise/fall|move|go\s+up)\s+together\b"
    r"|\bcorrelation\b[^.;:]{0,40}?\b(?:suggests?|shows?|means|indicates?|implies|"
    r"confirms?|reflects?)\s+(?:that\s+)?(?:they|the\s+(?:two\s+)?(?:maps|models|"
    r"results|surfaces|bands|layers)|both)\b",
    re.I,
)


#: A map against its reference, not against the other map: "does not agree
#: with the ground truth" is a labelling instruction.
_AGAINST_REFERENCE_RE = re.compile(
    r"\W*(?:\w+\W+){0,2}?(?:with|to)\s+(?:the\s+|its\s+|a\s+)?(?:ground|"
    r"reference|truth|labels?|field)",
    re.I,
)


#: The sample said to be unable to say whether the maps co-vary, anywhere in
#: the sentence: "so this sample cannot say whether the maps co-vary" (the
#: tool's fact and must_state), "it is unclear whether they do". A bare
#: "whether" is not it: "only whether they rise and fall together is
#: comparable — and ... they do not agree" (exp86 round 8, B3/studio run 1)
#: states what is comparable, then a reading.
_CANNOT_SAY_WHETHER_RE = re.compile(
    r"\b(?:cannot|can't|cant|can\s+not|could\s*n[o']t|does\s*n[o']t|do\s*n[o']t|"
    r"is\s*n[o']t\s+able\s+to|unable\s+to|too\s+(?:few|wide|small|uncertain)\s+to)"
    r"\s+(?:\w+\s+){0,2}?(?:say|tell|show|establish|determine|decide|settle|know|"
    r"distinguish)\b[^.;]{0,40}?\bwhether\b"
    r"|\b(?:unknown|unclear|not\s+known|undetermined|uncertain|open)\s+whether\b",
    re.I,
)
#: An interval said to hold a value, not a reading of the maps: "the interval
#: holds both no relation and a moderate one" (the tool's fact).
_INTERVAL_HOLDS_RE = re.compile(
    r"\binterval\b(?:[^.;:]|\.(?=\d)){0,60}?\b(?:holds?|holding|includes?|including|spans?|"
    r"spanning|covers?|covering|contains?|containing|allows?|admits?|"
    r"consistent\s+with|runs?\s+from|ranges?\s+from|reaches?)\b",
    re.I,
)
#: A stated interval: "95% interval 0.51 to 0.71", "CI [0.51, 0.71]".
_STATED_INTERVAL_RE = re.compile(
    r"\b(?:interval|CI)\b[^\d\n]{0,24}?(?P<lo>[-−+]?\d*\.?\d+)\s*(?:to|,|–|—|\.\.)\s*"
    r"(?P<hi>[-−+]?\d*\.?\d+)",
    re.I,
)
_R_VALUE_RE = re.compile(r"(?<![\w.])[-−]?\d*\.\d+")


def _number(text: str) -> float:
    return float(text.replace("−", "-").replace("+", ""))


def _signed_correlation_stated(s: str, run: RunEvidence) -> bool:
    """Whether the sentence states a correlation whose sign is known.

    An interval that excludes 0 ("95% interval 0.51 to 0.71", the tool's own
    signed fact), or the r of a ``correlation`` fact whose interval excludes
    0 (a group's certain pair beside an uncertain one): saying those maps
    rise and fall together is the tool's reading, not this claim.
    """
    for m in _STATED_INTERVAL_RE.finditer(s):
        lo, hi = _number(m.group("lo")), _number(m.group("hi"))
        if lo * hi > 0:
            return True
    stated = [m.group() for m in _R_VALUE_RE.finditer(s)]
    if not stated:
        return False
    known: list[float] = []
    unknown: list[float] = []
    for _record, fact in run.facts("correlation"):
        r = fact.get("r")
        if isinstance(r, int | float) and not isinstance(r, bool):
            (unknown if fact.get("co_varies") == "unknown" else known).append(float(r))

    def said(value: float) -> bool:
        for text in stated:
            decimals = len(text.partition(".")[2])
            if abs(_number(text) - value) <= 0.5 * 10.0**-decimals + 1e-12:
                return True
        return False

    return any(said(r) for r in known) and not any(said(r) for r in unknown)


def _held_by_an_interval(p: _Parse, lo: int, start: int) -> bool:
    """Whether the claim at ``start`` is what an interval holds ("the interval
    holds both no relation and a moderate one"): the interval's verb ends at
    most three words before it, in its clause. "The interval spans zero and
    the maps are unrelated" is a reading beside the interval, not its object.
    """
    end = p.last_end(_INTERVAL_HOLDS_RE, lo, start)
    if end <= lo or start - end > 40:
        return False
    return len(p.s[end:start].split()) <= 3


def _agreement_uncertain(s: str, record: ToolRecord, run: RunEvidence) -> bool:
    """That the maps do, or do not, co-vary, read from a correlation that cannot say.

    Emitted when the correlation's 95% interval holds both no relation and a
    moderate one: exp86 round 8 (B3/studio) read r = -0.02 over 25 cells
    (interval -0.41 to 0.38) as "they do not agree spatially at all", "the
    two maps do not rise and fall together", "one going high says
    essentially nothing about the other", where a 12 x 12 grid of the same
    pair gives 0.50. The claim is often a negation, so a negation does not
    exempt it; a statement of what the correlation can say does, in the
    claim's clause ("only whether they rise and fall together is
    meaningful", "only how they move together is meaningful", "it cannot
    tell") or anywhere in the sentence when it says the sample cannot say
    whether they co-vary (the tool's own fact: "the interval holds both no
    relation and a moderate one, so this sample cannot say whether the maps
    co-vary"); so does what an interval holds, a sentence about the number
    alone ("the correlation is -0.017, essentially zero over 25 cells"), and
    a correlation stated with a sign the tool found (an interval that
    excludes 0: "the maps' values tend to rise and fall together"). The
    interval stated beside a reading does not withdraw it.
    """
    if _CANNOT_SAY_WHETHER_RE.search(s) or _signed_correlation_stated(s, run):
        return False
    p = _parse(s)
    for m in _COVARIATION_CLAIM_RE.finditer(s):
        lo, _hi = p.span(m.start(), m.end())
        if (
            # the claim is what the correlation does not show: "a correlation
            # near zero does not show that the maps are independent"
            _DOES_NOT_SHOW_THAT_RE.search(s, 0, m.end())
            or p.in_clause(_CORRELATION_LIMIT_RE, m.start(), m.end(), own=True)
            or _held_by_an_interval(p, lo, m.start())
            or _AGAINST_REFERENCE_RE.match(s, m.end())
            or p.example_before(m.start())
            or p.quoted(m.start())
        ):
            continue
        return True
    return False


#: Where, or in what pattern, the two maps agree or differ. "Agree spatially"
#: is how much they co-vary, not where (``agreement_from_uncertain_correlation``
#: reads it): round 3 put "the two maps agree spatially only moderately" beside
#: r = 0.50 over 91 cells.
_PLACE_OF_AGREEMENT_RE = re.compile(
    r"\b(?:anywhere|nowhere|everywhere|somewhere|locally|in\s+places)\b"
    r"|\b(?:at|in)\s+the\s+same\s+(?:locations?|places?|spots?|areas?|cells?|"
    r"pixels?|parts?|sites?)\b"
    r"|\b(?:large|most|many|some|much|big|other)\s+(?:\w+\s+)?(?:parts?|areas?|"
    r"regions?|portions?|patches|stretches)\b"
    # "part of" a place is a place; "a large part of the disagreement" or "of
    # the 2%" is a share of the differences
    r"(?!\s+of\s+(?!(?:the\s+|this\s+|your\s+|their\s+)?(?:\w+\s+)?(?:area|aoi|"
    r"region|map|extent|grid|scene|image|landscape|country|state|county|basin|"
    r"watershed)s?\b))"
    r"|\bthe\s+rest\s+(?:of\s+the\s+(?:area|aoi|region|map|extent)\s+)?"
    r"(?:disagree|differ|diverge)\w*"
    r"|\bin\s+(?:the\s+)?(?:\w+\s+)?patterns?\b|\b(?:similar|same|different)\s+"
    r"(?:\w+\s+)?patterns?\b"
    r"|\bwhere\s+the\s+other\b|\bvice\s+versa\b"
    r"|\bwhere\s+(?:the\s+)?(?:\w+\s+){0,2}?(?:is|are|reads?|runs?|goes|sits?)\s+"
    r"(?:\w+\s+)?(?:high|higher|low|lower|elevated|strong|weak)\b"
    r"|\b(?:coincid\w*|co-?locat\w*|line\s+up|lines\s+up)\b"
    r"|\bin\s+the\s+(?:far\s+)?(?:north|south|east|west|centre|center|middle|"
    r"interior|edges?|margins)\w*",
    re.I,
)
#: A relation between the two maps: agreement, co-variation, difference.
_RELATION_WORD_RE = re.compile(
    r"\b(?:agree\w*|disagree\w*|co-?var\w*|correlat\w*|track\w*|match\w*|"
    r"similar\w*|diverg\w*|differ\w*|overlap\w*|together|relat\w*|indifferent|"
    r"coincid\w*|opposite\w*|align\w*|contrast\w*)\b|\brise\s+and\s+fall\b|rise/fall",
    re.I,
)
#: A level of one map ("high", "low"): a relation only beside the other map.
_LEVEL_WORD_RE = re.compile(
    r"\b(?:high\w*|low\w*|elevated|strong\w*|weak\w*|rise\w*|fall\w*)\b", re.I
)
#: The two maps together: "both", "they", "the other", "vice versa".
_BOTH_MAPS_RE = re.compile(
    r"\b(?:both|they|them|their|the\s+two|each\s+other|one\s+another|the\s+other|"
    r"vice\s+versa|between)\b",
    re.I,
)
#: An offer to show the maps, not a claim about where they agree.
_SHOW_OFFER_RE = re.compile(
    r"\b(?:if\s+you|i\s+can|i\s+could|i'll|want\s+me|would\s+you|to\s+see|"
    r"overlay\w*|visuali[sz]\w*|layer\s+packs?)\b",
    re.I,
)
#: A breakdown a tool computed by place (rows, bands, differing windows): a
#: location read from it is the tool's, not one correlation's.
_COMPUTED_PLACE_RE = re.compile(
    r"\b(?:rows?|columns?|cols?|bands?|quarters?|quadrants?)\b|\bdiffering\s+windows\b",
    re.I,
)


def _spatial_from_correlation(s: str, record: ToolRecord, run: RunEvidence) -> bool:
    """Where the maps agree or differ, or a pattern of it, read from one correlation.

    One pooled correlation has no location. exp86 round 8 (B3/studio) read
    one r as "they do not rise and fall together anywhere in the sampled
    region" and "one is high where the other is indifferent, and vice
    versa"; round 7 as "large parts of the AOI show similar relative
    patterns while much of the rest disagrees". A place is read only in a
    clause about the two maps together, its label included ("Where they
    differ: effectively everywhere", round 1): a relation (agree, differ,
    co-vary, together), or a level beside the other map ("high where the
    other is low", "where A is high, B is low"). One map's level in a place
    ("KarstBinary is low almost everywhere") is no such claim (the fix-r8
    review). What the correlation cannot say ("it has no location, so it
    cannot say where they differ"), a cited phrase and an offer to overlay
    the maps are no claim, and neither is a place in the rows or bands of a
    breakdown a tool computed by place ("differences cluster in the middle
    rows") or a share of the differences ("a large part of the disagreement
    sits at class edges").
    """
    p = _parse(s)
    for m in _PLACE_OF_AGREEMENT_RE.finditer(s):
        lo, hi, _ = p.region(m.start(), m.end())
        lo = _label_start(p, lo)
        own = (m.start(), m.end())
        related = p.has(_RELATION_WORD_RE, lo, hi, skip=own) or (
            p.has(_LEVEL_WORD_RE, lo, hi, skip=own)
            and (
                p.has(_BOTH_MAPS_RE, lo, hi)
                # "where A is high, B ...": the place is one map's level
                or m.group().lower().startswith("where")
            )
        )
        if not related:
            continue
        if (
            p.has(_CORRELATION_LIMIT_RE, lo, hi)
            or p.has(_SHOW_OFFER_RE, lo, hi)
            or p.has(_COMPUTED_PLACE_RE, lo, hi)
            or p.example_before(m.start())
            or p.quoted(m.start())
        ):
            continue
        return True
    return False


#: What a review set ranks by: margins, the least decided windows, a
#: per-class review, or the review-set tools by name.
_REVIEW_RANKING_RE = re.compile(
    r"\breview[\s_-]+(?:set|list)s?\b|\bolmoearth_review_set\w*"
    r"|\bper[\s-]+class\s+review\b"
    r"|\bmost[\s-]+(?:ambiguous|uncertain|undecided|suspect)\b"
    r"|\b(?:least|lowest|low|lower|less)[\s-]+(?:certain|confident|confidence|"
    r"decided|decisive|margin)\b"
    r"|\bmargins?\b|\bundecided\b|\b(?:nearest|closest)\s+to\s+(?:0?\.5|the\s+"
    r"decision)",
    re.I,
)
_THRESHOLD_RE = re.compile(r"\bthreshold\w*|\bcut-?\s?off\b|\bcut\s+point\b", re.I)
#: A threshold stated as what the review needs, not merely mentioned: "the
#: review set needs a threshold", "once a threshold is named", "name a
#: threshold first", "with a threshold of 0.5". "Without a threshold, I can
#: still build a review set" and "there is no decision threshold, but I can
#: rank by margin" mention one and offer the review anyway (the fix-r8
#: review).
_THRESHOLD_REQUIRED_RE = re.compile(
    r"\b(?:need|needs|needed|needing|require|requires|required|requiring|name|"
    r"names|named|naming|give|gives|given|giving|set|pick|choose|provide|"
    r"provided|supply|specify|specified|pass|with|using|at|once|after|until|"
    r"unless|if|first)\s+(?:me\s+|us\s+)?(?:(?:a|an|the|its|your|one|each|some|"
    r"this|that|their)\s+)?(?:(?:decision|valid|specific|numeric|chosen)\s+)?"
    r"(?:threshold\w*|cut-?\s?off|cut\s+point)"
    r"|\b(?:threshold\w*|cut-?\s?off|cut\s+point)\s+(?:is|are|was|has\s+been|"
    r"have\s+been|gets|get)\s+(?:\w+\s+)?(?:named|given|set|chosen|picked|"
    r"provided|specified|fixed|passed|known)\b"
    r"|\b(?:threshold\w*|cut-?\s?off|cut\s+point)\s*(?:of|=|at|:)\s*[-−]?\d"
    r"|\b(?:threshold\w*|cut-?\s?off|cut\s+point)\s+(?:\w+\s+)?first\b",
    re.I,
)
#: A threshold said to be missing: the negation in it is the threshold's,
#: not the review's ("without a threshold, I can still build a review set").
_NO_THRESHOLD_RE = re.compile(
    r"\b(?:no|without|not\s+(?:a|any)|lacks?|lacking)\s+(?:(?:a|an|any|its|the)\s+)?"
    r"(?:(?:decision|valid|clear|natural)\s+)?(?:threshold\w*|cut-?\s?off|cut\s+point)",
    re.I,
)
#: An offer or a recommendation: what the answer proposes to do next. A
#: statement of what a review set would do ("a review set would overstate
#: the error") is none, nor is a table's header row ("| Rank | Margin |")
#: or a label ("Run: C1/awf").
_PROPOSAL_RE = re.compile(
    r"\?|\b(?:want\s+me|shall\s+i|should|i\s+can|i\s+could|i'll|i\s+will|we\s+can|"
    r"we\s+could|let\s+me|would\s+you\s+like|you\s+(?:can|could|might|may)|"
    r"happy\s+to|recommend\w*|suggest\w*|path\s+forward|next\s+step|"
    r"alternatively|option|try|next)\b"
    r"|(?:[:—–]\s*|\s-\s)(?:[-*+•]\s+|\d+[.)]\s+)?(?:run|build|"
    r"use|flag|open|check|plan|start)\b(?!\s*:)",
    re.I,
)
#: The same proposal verbs opening a clause ("Run olmoearth_review_set_from_result
#: on each result", a bullet's "- Build ...").
_PROPOSAL_START_RE = re.compile(
    r"(?:[-*+•]\s+|\d+[.)]\s+)?(?:run|build|use|flag|open|check|plan|start)\b"
    r"(?!\s*:)",
    re.I,
)
#: A [0, 1] score's decision stated: "a 0-1 score", "[0, 1]", "nearest 0.5".
_UNIT_SCORE_RE = re.compile(
    r"(?<![\d.])(?:\[\s*0(?:\.0)?\s*,\s*1(?:\.0)?\s*\]|0(?:\.0)?\s*(?:[-–—]|to)\s*"
    r"1(?:\.0)?(?![\d.])|0?\.5(?![\d]))",
)


#: Every map of the run: "each map", "both maps", "either result".
_EVERY_MAP_RE = re.compile(
    r"\b(?:each|both|either|every|all)\s+(?:of\s+the\s+)?(?:two\s+)?(?:maps?|results?|"
    r"bands?|models?|layers?|predictions?)\b|\bboth\b|\bone\s+or\s+both\b",
    re.I,
)


def _proposed_in(p: _Parse, lo: int, hi: int) -> bool:
    """Whether the clause ``s[lo:hi]`` offers or recommends (:data:`_PROPOSAL_RE`).

    Read by bisection over the sentence's matches, never on a slice: a
    listing of hundreds of windows in one clause asks it once per window.
    """
    while lo < hi and p.s[lo].isspace():
        lo += 1
    return p.has(_PROPOSAL_RE, lo, hi) or bool(_PROPOSAL_START_RE.match(p.s, lo, hi))


def _review_set_regression(s: str, record: ToolRecord, run: RunEvidence) -> bool:
    """A review set, margins or the least decided windows offered for a band
    with no threshold.

    A regression value has no decision to be near, so it has no margin and
    no least-decided window until a threshold is named. exp86 round 8
    (B3/studio) offered "a per-class review set to flag where each map
    looks uncertain" and to "flag its own most-ambiguous (lowest-margin)
    windows" of two unthresholded regression bands; round 6 "a human review
    set (lowest-confidence windows first)". Only an offer or a
    recommendation in the review's own clause is read (the id is a review
    set *offered* for the band): a report of a review set a tool ran is
    another band's, and so is a table row. A sentence that states a
    threshold as what the review needs ("the review set needs a threshold
    for this band", "once you name a threshold") or negates the review ("no
    review set applies") is none; one that only mentions a threshold
    ("without a threshold, I can still build a review set") is read. The
    claim is about the bands its reason names: an offer for another band of
    the run (a [0, 1] score the tools decide at 0.5, named by its map or its
    property) is not it, while an offer that names no band ("each map") is.
    """
    if _THRESHOLD_REQUIRED_RE.search(s) or s.lstrip().startswith("|"):
        return False
    p = _parse(s)
    bands, others = run.review_bands
    for m in _REVIEW_RANKING_RE.finditer(s):
        if (
            _negated_close(p, m.start(), m.end(), besides=_NO_THRESHOLD_RE)
            or p.example_before(m.start())
            or p.quoted(m.start())
        ):
            continue
        lo, hi, _ = p.region(m.start(), m.end())
        if not _proposed_in(p, lo, hi):
            continue
        if others is not None and not _about_the_band(p, lo, hi, bands, others):
            continue
        return True
    return False


def _about_the_band(
    p: _Parse, lo: int, hi: int, bands: re.Pattern[str] | None, others: re.Pattern[str]
) -> bool:
    """Whether an offer in ``s[lo:hi]`` may be about an unthresholded band.

    Not when its clause, or else its sentence, names only another band of the
    run (by its map's or its property's name, or as a [0, 1] score) and not
    every map ("each map", "both").
    """
    if p.has(_EVERY_MAP_RE, lo, hi):
        return True
    for a, b in ((lo, hi), (0, len(p.s))):
        names_band = bool(bands and p.has(bands, a, b))
        names_other = p.has(others, a, b) or p.has(_UNIT_SCORE_RE, a, b)
        if names_band:
            return True
        if names_other:
            return False
    return True


_DATE_NOUN = r"(?:dates?|years?|periods?|times?|epochs?|acquisitions?)"
#: A reference for one of the two dates offered where each needs its own:
#: "labels for either date", "one of the two years", "one or both dates",
#: "at least one", "(ideally one set per year)".
_ONE_DATE_RE = re.compile(
    rf"\beither\s+(?:of\s+the\s+(?:two\s+)?)?{_DATE_NOUN}\b"
    rf"|\bone\s+of\s+(?:the\s+)?(?:two\s+|those\s+|these\s+|both\s+)?{_DATE_NOUN}\b"
    rf"|\bone\s+or\s+(?:both|the\s+other)(?:\s+(?:of\s+the\s+)?(?:\w+\s+)?"
    rf"{_DATE_NOUN})?\b"
    r"|\bat\s+least\s+one\b"
    rf"|\b(?:a\s+single|only\s+one|just\s+one)\s+(?:\w+\s+)?{_DATE_NOUN}\b"
    r"|\bdated\s+to\s+(?:either|one)\b"
    r"|\b(?:for|from|at|of)\s+(?:only\s+|just\s+)?(?:one|a\s+single)\s+(?:\w+\s+)?"
    rf"{_DATE_NOUN}\b"
    r"|\bideally\s+(?:also|both|one\s+(?:set\s+)?(?:per|for\s+each))\b",
    re.I,
)
#: One reference plus another model run: "plus a date-matched second inference".
_PLUS_RUN_RE = re.compile(
    r"\b(?:plus|with|and)\s+(?:a|an|another|one)?\s*(?:(?:date-?matched|second|"
    r"third|new|additional|extra|matching)\s+)+(?:inference|model\s+run|run|"
    r"prediction|map)s?\b",
    re.I,
)
#: What a reference would decide: which map is right, change from error.
#: Grading one map is not a verdict between them ("labels from April 2018
#: can grade map B, but not map A" is the rule): only grading both is.
_SETTLE_RE = re.compile(
    r"\b(?:settle\w*|verdict|decide\w*|resolve\w*|separat\w*|"
    r"grad\w*\s+(?:\w+\s+){0,2}?(?:both|each|either|them|the\s+two|all)\b|"
    r"disambiguat\w*|tell\s+(?:you\s+)?which|which\s+(?:map|one|side)|"
    r"right|correct|errs|winner|pick)\b",
    re.I,
)
_REFERENCE_RE = re.compile(
    r"\b(?:labels?|labell?ed|references?|ground[\s-]+truth|truth|annotations?)\b",
    re.I,
)
_DATE_CONTEXT_RE = re.compile(rf"\b{_DATE_NOUN}\b|\b(?:19|20)\d\d\b|\bdated\b", re.I)


#: What labels at one date do say, which the tool states: "labels dated
#: 2023 would measure which map matches the ground at that date, counting
#: the other wrong wherever the ground changed".
_ONE_DATE_CAVEAT_RE = re.compile(
    r"\b(?:at\s+that\s+date|counting\s+the\s+other|wherever\s+the\s+ground|"
    r"matches\s+the\s+ground)\b",
    re.I,
)
#: What labels for one date grade, stated as the rule: "labels for one date
#: grade only that date's map" (the tools' dates must_state), "labels dated
#: 2023-06-01 grade only a map of that date" (which_side_is_right), "can grade
#: only the map of that date", "would grade the 2023 map only".
_GRADES_ONLY_ITS_DATE_RE = re.compile(
    r"\bgrad\w*\s+only\b|\bonly\s+(?:\w+\s+){0,2}?grad\w*"
    r"|\b(?:that|the\s+same|its|their)\s+(?:own\s+)?(?:date|year|period|time)'?s?"
    r"\s+maps?\b"
    r"|\bmaps?\s+of\s+(?:that|the\s+same|its|their)\s+(?:own\s+)?(?:date|year|"
    r"period|time)\b"
    r"|\bmaps?\s+only\b"
    r"|\bonly\s+(?:for|at|of)?\s*(?:that|this|its|the\s+same)\s+(?:date|year|"
    r"period|time)\b",
    re.I,
)
#: A reference for each map, the rule: "each", "both", "per date".
#: "Both" or "each" of the dates or maps' own references is the rule; "grade
#: both maps" against one date's labels is not.
_EACH_DATE_RE = re.compile(
    r"\b(?:each|both|every|per)\s+(?:of\s+the\s+(?:two\s+)?)?(?:\w+\s+)?"
    r"(?:dates?|years?|periods?|times?|epochs?)\b"
    r"|\beach\s+map'?s?\s+(?:own\s+)?(?:date|year|period|reference)"
    r"|\b(?:its|their)\s+own\s+(?:date|year|period)",
    re.I,
)
_YEAR_RE = re.compile(r"(?<![\d.])(?:19|20)\d\d(?![\d.])")


def _map_years(record: ToolRecord) -> tuple[set[str], set[str]]:
    """The years of the two maps the emitting comparison compared.

    From its result's ``dates`` (``{"a", "b"}``) or its ``date_a`` and
    ``date_b`` arguments; two empty sets when it gives neither, or when the
    maps share a year.
    """
    result = record.result if isinstance(record.result, Mapping) else {}
    found = result.get("dates")
    dates: Mapping[str, Any] = found if isinstance(found, Mapping) else {}
    a = str(dates.get("a") or record.arguments.get("date_a") or "")
    b = str(dates.get("b") or record.arguments.get("date_b") or "")
    ya, yb = set(_YEAR_RE.findall(a)), set(_YEAR_RE.findall(b))
    return (ya, yb) if ya and yb and not ya & yb else (set(), set())


def _one_map_year(s: str, record: ToolRecord) -> Iterator[re.Match[str]]:
    """A year of one map named alone as the reference's ("a reference for 2023")."""
    ya, yb = _map_years(record)
    if not ya or _EACH_DATE_RE.search(s):
        return
    years = list(_YEAR_RE.finditer(s))
    named = {m.group() for m in years}
    if bool(named & ya) != bool(named & yb):
        yield from (m for m in years if m.group() in ya | yb)


def _one_reference(s: str, record: ToolRecord, run: RunEvidence) -> bool:
    """A reference for one of two dated maps, or one plus another run, offered
    to say which map is right.

    Where two maps of different dates differ, the ground may have changed:
    labels at one date count the other map wrong wherever it did, and another
    unlabelled run is no reference. exp86 round 8 offered to "grade which map
    is right" from "reference labels for either date" (B3/cluster run 1),
    and references "for both periods (or for at least one, plus a
    date-matched second inference)" (B7/files run 1); round 7 "labels for
    one or both dates". A year of one map named alone ("a reference for
    2023 would settle it") is read against the comparison's own dates.
    "Labels dated to each map's date" and "only a reference dated to each
    map would separate them" are the rule, and pass; so does a clause that
    negates the one-date reference ("labels for one date cannot settle it",
    "even with labels for only one date, grading would be blocked"),
    what one date's labels do measure, as the tool states it ("which map
    matches the ground at that date"), and a clause that restricts them to
    their own date's map, as the tools' must_state and which_side_is_right
    do ("labels for one date grade only that date's map", "labels dated 2023
    would grade the 2023 map only").
    """
    if not _REFERENCE_RE.search(s) or _ONE_DATE_CAVEAT_RE.search(s):
        return False
    p = _parse(s)
    # a verdict the sentence denies is the rule: "a 2023 reference would tell
    # you how accurate the 2023 map is, but not which map is right"
    if not any(
        not _negated_close(p, m.start(), m.end()) for m in _SETTLE_RE.finditer(s)
    ):
        return False
    cues = [*_PLUS_RUN_RE.finditer(s), *_one_map_year(s, record)]
    if _DATE_CONTEXT_RE.search(s):
        # "at least one", "one or both": a date is what they count
        cues += list(_ONE_DATE_RE.finditer(s))
    return any(
        not (
            _negated_close(p, m.start(), m.end())
            or p.in_clause(_GRADES_ONLY_ITS_DATE_RE, m.start(), m.end())
            or p.example_before(m.start())
            or p.quoted(m.start())
        )
        for m in cues
    )


#: Evidence from recorded experiments. A bare "evidence" or "measured" is
#: often the run's own ("the pixel-level evidence can't separate them",
#: "here measured by the mean margin"), so each needs its experimental sense.
_EVIDENCE_RE = re.compile(
    r"\b(?:upstream|experiments?|stud(?:y|ies)|suite|benchmarks?|exp\d+|"
    r"published|literature)\b"
    r"|\b(?:measured|recorded|prior|experimental|empirical)\s+(?:evidence|"
    r"results?|measurements?|stud(?:y|ies)|experiments?|record)\b"
    r"|\bevidence\s+(?:shows?|says?|puts?|suggests?|indicates?|found|finds|from|"
    r"behind|on)\b"
    r"|\bmeasured\s+to\b"
    r"|\b(?:previous|prior|earlier|past)\s+(?:\w+\s+)?(?:experiments?|stud(?:y|ies)|"
    r"results?|work|runs|tests|trials)\b|\bresearch\b"
    r"|\b(?:has|have|was|were)\s+been\s+shown\b|\b(?:was|were)\s+shown\s+to\b",
    re.I,
)
#: The evidence carried over to this case: "in comparable cases", "of such
#: windows", "behind this rule", "trust in the signal", "why these: ...".
_APPLIED_HERE_RE = re.compile(
    r"\b(?:comparable|similar|analogous|equivalent)\s+(?:\w+\s+)?(?:cases?|pairs?|"
    r"maps?|settings?|situations?|windows?|comparisons?|tasks?|data|sets?)\b"
    r"|\bsuch\s+(?:\w+\s+)?(?:windows|cases|pairs|maps|sets|comparisons|"
    r"disagreements|differences)\b"
    r"|\b(?:cases?|pairs?|maps?|situations?)\s+like\s+(?:this|these|yours|ours)\b"
    r"|\b(?:behind|supports?|supporting|justif\w*|backs?|backing|validat\w*|"
    r"trust\s+in)\s+(?:this|these|the|your)\s+(?:rule|ranking|ordering|signal|"
    r"ranker|choice|method|approach|pair|maps?|case|result|comparison)\b"
    r"|\b(?:appl(?:y|ies)|holds?|carr(?:y|ies)\s+over|transfers?|"
    r"generali[sz]es?)\s+(?:here|to\s+(?:this|these|your))\b"
    r"|\bwhy\s+(?:these|this)\b"
    r"|\b(?:so|means|thus|hence|therefore)\b[^.;:]{0,60}?\b(?:is|are)\s+(?:the\s+)?"
    r"(?:best|incumbent|most\s+reliable|proven|right\s+(?:ranker|signal|choice))\b",
    re.I,
)
#: The evidence's scope stated: it does not cover this case, or only in part.
_SCOPE_STATED_RE = re.compile(
    r"\b(?:does|do|did)\s*n[o']t\s+(?:\w+\s+)?(?:cover|apply|extend|transfer|"
    r"grade|include)\w*|\bnot\s+(?:\w+\s+)?(?:cover\w*|apply|applicable)\b"
    r"|\bno\s+recorded\b|\bin\s+part\b|\boutside\s+(?:its|their|the)\b"
    r"|\bnot\s+(?:this|these|your)\s+(?:pair|case|maps?)\b",
    re.I,
)


#: A name of the kind a dataset or an experiment has: letters and digits
#: ("Sen1Floods11", "exp58").
_EVIDENCE_NAME_RE = re.compile(r"\b(?=[A-Za-z0-9]*\d)[A-Za-z][A-Za-z0-9]{3,}\b")


def _evidence_only_names(record: ToolRecord, run: RunEvidence) -> set[str]:
    """Names that only the emitting tool's evidence carries.

    Read from the result's top-level ``evidence*`` keys (``evidence_scope``);
    a name the user wrote, or that any other key of any result or argument of
    the run holds (a fact's "(exp58)", a file name), is the case's own
    (:attr:`RunEvidence.case_words`).
    """
    result = record.result if isinstance(record.result, Mapping) else {}
    evidence = [
        v
        for key, value in result.items()
        if str(key).startswith("evidence")
        for _, v in _strings(value)
    ]
    names = {m.group() for text in evidence for m in _EVIDENCE_NAME_RE.finditer(text)}
    return names - run.case_words if names else set()


def _evidence_outside_scope(s: str, record: ToolRecord, run: RunEvidence) -> bool:
    """An experiment's result carried over to a case the tool says it does not cover.

    exp86 round 8 (B3/cluster run 3): "upstream evidence shows the more
    confident side was only right 51-70% of the time in comparable cases",
    where the tool's evidence was flood maps of one date and this pair two
    land-cover maps a year apart; round 7 "upstream evidence shows ... of
    such windows" and "measured evidence behind this rule". Round 8 (B7/files
    run 2) also named this pair "the two Sen1Floods11 flood maps", a name
    only the tool's evidence held: a name of the evidence's own used without
    its scope is the evidence put in the case's place. Citing the evidence
    with its scope ("that study does not cover this pair", "no recorded
    experiment grades this case", "covers this case only in part") is what
    the tool asks, and passes.
    """
    if _SCOPE_STATED_RE.search(s):
        return False
    names = _evidence_only_names(record, run)
    if any(re.search(rf"\b{re.escape(n)}\b", s) for n in names):
        return True
    if not _EVIDENCE_RE.search(s):
        return False
    p = _parse(s)
    return any(
        not (_negated_close(p, m.start(), m.end()) or p.quoted(m.start()))
        for m in _APPLIED_HERE_RE.finditer(s)
    )


#: A plan, design or sample size promised to certify: "a guaranteed-
#: certifiable region", "enough labels to certify", "would certify".
_CERTIFY_PROMISE_RE = re.compile(
    r"\bguaranteed?[\s-]+(?:to\s+)?(?:be\s+)?(?:a\s+|an\s+|the\s+)?(?:\w+\s+)?"
    r"certif\w*"
    r"|(?<!the\s)(?<!its\s)(?<!that\s)\bguarantees?\s+(?:that\s+)?(?:a|an|the|you|"
    r"your|some|certif\w*)\b[^.;]{0,30}?\bcertif\w*"
    r"|\bcertif\w*\s+(?:is\s+|are\s+|will\s+be\s+)?guaranteed\b"
    r"|\b(?:ensures?|ensuring|assures?|make\s+sure|makes\s+sure|secures?)\b"
    r"[^.;]{0,40}?\bcertif\w*"
    r"|\b(?:enough|sufficient)\s+(?:\w+\s+){0,3}?to\s+(?:certify|get\s+a\s+"
    r"certified)\b"
    r"|\b(?:certain|sure|bound)\s+to\s+(?:certify|be\s+certified)\b"
    r"|\bdefinitely\s+(?:be\s+)?certif\w*"
    r"|\b(?:will|would|shall)\s+(?:\w+\s+){0,2}?(?:certify|be\s+certified|be\s+"
    r"certifiable|yield\s+(?:you\s+)?a\s+certified|give\s+(?:you\s+)?a\s+"
    r"certified|get\s+(?:you\s+)?a\s+certified)\b",
    re.I,
)
#: What the promise is made of: a design, a sample, labels, a plan.
_DESIGN_WORD_RE = re.compile(
    r"\b(?:design\w*|sample\w*|plan\w*|labels?|budget|random|draw)\b", re.I
)
#: What the certification test itself guarantees: a certified zone's error
#: bound at alpha, except with probability delta, for the alpha fixed before
#: the labels ("the guarantee covers only that alpha").
_GUARANTEE_OF_THE_TEST_RE = re.compile(
    r"guarant\w*\s+(?:that\s+)?(?:(?:the|a|any|each|every|its)\s+)?(?:certif\w*\s+"
    r"(?:\w+\s+)?(?:zone|region|area|level)(?:'s|s')\s+(?:\w+\s+)?errors?\b|errors?\s+"
    r"(?:rate\s+)?(?:of|in|inside|within)\s+(?:the|a|any|each|every|that)\s+certif\w*)"
    r"|guarant\w*\s+(?:\w+\s+){0,2}?(?:covers?|holds?|applies)\s+only\b"
    r"|guarant\w*\s+only\s+(?:covers?|holds?|applies)\b",
    re.I,
)
#: A requirement, not a promise: "a design would need to certify ...".
_REQUIREMENT_RE = re.compile(
    r"\b(?:need|needs|needed|require|requires|required|take|takes)\b", re.I
)
#: A promise kept conditional on what the labels will show: "if its errors
#: are few", "provided the zone holds up".
_CONDITIONAL_RE = re.compile(
    r"\b(?:if|unless|provided|providing|as\s+long\s+as|only\s+when|whether|"
    r"could|might|may|possibl\w*|plausibl\w*|probabl\w*|perhaps|potentially|"
    r"chance)\b",
    re.I,
)


def _certification_guaranteed(s: str, record: ToolRecord, run: RunEvidence) -> bool:
    """A design, sample size or plan promised to certify a zone.

    A random design makes certification possible; whether a zone certifies
    depends on the errors its labels show. exp86 round 8 (B5/files run 2)
    offered "a guaranteed-certifiable region". "Certification needs a random
    design", "you would need more labels to certify" (a requirement) and "a
    zone could be certified if its errors are few" pass, as does a hedged
    offer ("how many labels would plausibly certify a zone"); so does "the
    guarantee holds only for an alpha fixed before", the test's own
    guarantee. The design, sample or plan must be in the promise's own
    clause: "a looser alpha would certify the top zone, but the guarantee
    covers only an alpha fixed before seeing the labels" promises nothing of
    a design (``post_hoc_alpha`` reads it).
    """
    p = _parse(s)
    for m in _CERTIFY_PROMISE_RE.finditer(s):
        if (
            _negated_close(p, m.start(), m.end())
            or p.example_before(m.start())
            or p.quoted(m.start())
            or _REQUIREMENT_RE.search(m.group())
        ):
            continue
        lo, hi, _ = p.region(m.start(), m.end())
        if not p.has(_DESIGN_WORD_RE, lo, hi):
            continue
        # the test's own guarantee: a certified zone's error bound, or the
        # alpha it covers ("guarantees that the certified zone's error is at
        # most alpha", "the guarantee covers only that alpha")
        if _GUARANTEE_OF_THE_TEST_RE.match(s, m.start()):
            continue
        # a hedge inside the promise ("would plausibly certify") or its clause
        if "guarant" not in m.group().lower() and (
            _CONDITIONAL_RE.search(m.group())
            or p.in_clause(_CONDITIONAL_RE, m.start(), m.end(), own=True)
        ):
            continue
        return True
    return False


#: How likely something is: "most likely", "likeliest", "probably", "a high
#: chance". "Probability" alone is none: "one minus the top probability ranked
#: errors better" (the tools' must_state) names a score.
_LIKELY = (
    r"(?:(?:most|more|very|highly|quite)\s+)?likely|likeliest|likelier|"
    r"probabl[ey]|in\s+all\s+likelihood|chances\s+are|almost\s+certainly|"
    r"high(?:er|est)?\s+(?:chance|odds|likelihood|risk)\s+of"
)
#: Wrong: an error, a wrong call or label, mislabelled. An error rate, bound or
#: estimate is a measure, not a window's error (error_rate_without_labels
#: reads those).
_WRONG = (
    r"wrong|mis-?label+ed|mis-?classified|misclassifications?|incorrect(?:ly)?|"
    r"erroneous|mistakes?|mistaken|in\s+error|"
    r"errors?(?![\s-]+(?:rates?|bounds?|estimates?|intervals?|bars?|probabilit))"
)
#: A window, or the review set's windows, called likely wrong: "most likely
#: mislabeled", "the likeliest spots for a wrong call", "the most likely
#: places for a wrong label", "probably errors", "errors are most likely
#: here", and the margin called a probability of error.
_ERROR_LIKELIHOOD_RE = re.compile(
    rf"\b(?P<lik>{_LIKELY})\b(?:[\s,]+(?:to\s+)?(?:be\s+|being\s+|hold\s+|have\s+|"
    rf"contain\s+|find\s+)?(?:[\w'-]+\s+){{0,3}}?)?(?:a\s+|an\s+|the\s+)?"
    rf"\b(?:{_WRONG})\b"
    r"|\b(?:errors?|mistakes?|misclassifications?)\s+(?:are|is|sit|lie)\s+"
    r"(?P<lik2>(?:most\s+|more\s+)?(?:likely|likeliest|probable))\b"
    r"|\b(?P<prob>(?:probabilit(?:y|ies)|likelihood|chance|odds)\s+of\s+(?:being\s+)?"
    r"(?:an?\s+)?(?:error|wrong|mis-?label\w*|misclassif\w*)|error[\s-]+"
    r"probabilit(?:y|ies))\b",
    re.I,
)
#: A copula right before "most likely": "they're most likely mislabeled" says
#: they are probably wrong, not that they are the likeliest of the windows.
_COPULA_BEFORE_RE = re.compile(
    r"(?:\b(?:is|are|was|were|be|been|being)|'re|'s|’re|’s)\s*$", re.I
)
#: "More likely wrong than right" is a probability over one half.
_THAN_RIGHT_RE = re.compile(r"\W*(?:\w+\W+){0,4}?than\s+(?:right|correct|not)\b", re.I)
#: The values of ``evidence_covers_this_case`` under which the ranking
#: evidence measured that the margin ranks errors (analysis.review_set.COVERS).
_ORDER_COVERED = ("yes", "in part")


def _an_order(p: _Parse, m: re.Match[str]) -> bool:
    """Whether a likelihood claim ranks the windows rather than calls them wrong.

    "The likeliest spots", "the most likely places for a wrong label", "the
    windows most likely to be wrong", "where it is most likely wrong", "more
    likely to be misclassified" and "errors are most likely here" order the
    windows; "they're most likely mislabeled", "probably errors", "likely
    wrong", "more likely wrong than right" and a probability of error say how
    likely they are wrong.
    """
    if m.group("prob"):
        return False
    lik = (m.group("lik") or m.group("lik2") or "").lower()
    if "likeliest" in lik or "likelier" in lik:
        return True
    if m.group("lik2"):
        return lik.startswith(("most", "more"))
    if lik.startswith("more"):
        return not _THAN_RIGHT_RE.match(p.s, m.end())
    if lik.startswith("most"):
        lo, _ = p.span(m.start(), m.end())
        before = p.s[lo : m.start()]
        return bool(re.search(r"\bwhere\b", before, re.I)) or not (
            _COPULA_BEFORE_RE.search(before)
        )
    return False


def _margin_as_error_probability(s: str, record: ToolRecord, run: RunEvidence) -> bool:
    """A review set's windows called likely wrong, or its margin a probability of error.

    exp86 round 9: "the model was weakest there, so they're most likely
    mislabeled" and "when the top two class logits are nearly tied, the
    argmax label is most likely wrong there" (B8/cluster runs 2 and 3), "the
    likeliest spots for a wrong call" (B2/studio run 1). An order of the
    windows by likelihood ("the most likely places for a wrong label", round 8
    B8/cluster run 1, which the blind audit of rounds 7 and 8 refuted as a
    finding) is read only where the emitting tool says its ranking evidence
    does not, or may not, cover the case (``evidence_covers_this_case``):
    where it covers it, the evidence measured that the margin ranks errors.
    "Least decided", "most uncertain", "checked first" and "not a
    probability of error" pass, as do the tools' own sentences, and a
    negated, quoted or example claim ('not "likely wrong"') is none.
    """
    result = record.result if isinstance(record.result, Mapping) else {}
    covered = result.get("evidence_covers_this_case") in _ORDER_COVERED
    p = _parse(s)
    for m in _ERROR_LIKELIHOOD_RE.finditer(s):
        if (
            _negated_close(p, m.start(), m.end())
            or p.example_before(m.start())
            or p.quoted(m.start())
            or (covered and _an_order(p, m))
        ):
            continue
        return True
    return False


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
    "spatial_pattern_from_one_correlation": _spatial_from_correlation,
    "agreement_from_uncertain_correlation": _agreement_uncertain,
    "review_set_for_unthresholded_regression": _review_set_regression,
    "one_reference_settles_two_dates": _one_reference,
    "evidence_outside_its_scope": _evidence_outside_scope,
    "certification_guaranteed": _certification_guaranteed,
    "margin_as_error_probability": _margin_as_error_probability,
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
    CLAIMS: "Sentences that a reading of this run's record (the tool outputs, "
    "the user's messages and what each tool can do) does not support. A "
    "statement no tool output supports: remove it, or state what a tool "
    "returned. One that contradicts a tool output: state the tool's reading. "
    "An offer no tool can carry out, or whose inputs the run lacks: offer "
    "instead a step a tool can take, and say what it needs:",
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
    the end as the tool states it, as a note from the tool (it is a limit the
    tool states, not an unverified claim of the answer).
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
    tail += [f"{TOOL_NOTE} {t}" for t in missing]
    if tail:
        out = out.rstrip() + "\n\n" + "\n".join(tail)
    return out
