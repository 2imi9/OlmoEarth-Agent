# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Which numbers of an answer no tool result and no user message supports.

The harness checks the final answer with this before it is shown (see
``LeadAgent.run_stream``). exp86 found, in each of three rounds, a number
the model stated that no tool had returned ("delta/18", "5,565+ windows"),
although the soul says to state numbers exactly as the tools returned them.

The sources are the run's full tool results and the user's messages (the
brief and the user's turns of any history), never the system prompt, the
saved preferences, the tool descriptions or an earlier assistant message: a
figure quoted from a description, or one the model derived in an earlier
answer, is not evidence.

Reading rules. The same tokenizer reads the answer and every string in the
sources, so an id or a date splits into the same numbers on both sides:

- A number is digits with optional thousands separators (``16,384``) and
  decimals, then an optional ``%`` (or "percent"), or a ``k``/``K``/``M``
  directly after that no letter or digit follows (``6.4M``), or an exponent
  with a sign or after decimals (``1.2e-5``). Digit runs glued to letters
  count too: ``128x128`` gives 128 and 128, ``EMSR279-11`` gives 279 and 11,
  ``2023-01-01`` gives 2023, 1 and 1. In a range whose upper end has a
  percent sign (``32-59%``), the lower end is a percent too.
- Values are compared in absolute value: a ``-`` before a number is as often
  a hyphen, and a sign is not what this check is for. U+2212 is read as a
  minus sign like ``-``.
- An integer is supported by a source value equal to it, or by a source value
  that is not an integer and rounds to it (``10,091`` from 10091.4); a count
  off by one is not supported. Failing that, it is tried as a percent of a
  fraction, exactly (``85`` against 0.85).
- A decimal is supported by a source value within half a unit of its last
  decimal, as written or as a percent of a fraction (``23.0`` against
  0.2298; ``-0.017`` against -0.0172). A suffixed number is compared in its
  unit at the same rounding (``6.4M`` against 6,412,345), or as written.
- A percent is supported at the same rounding only by a share: any value
  in [0, 1], of any numeric type and in any string, read as a percent
  (``23%`` against 0.2298, ``0%`` against 0, ``100%`` against 1.0), a
  percent written in a source string (``42.2%``), or a value above 1 under a
  key that names a percent, share or rate (``error_pct: 23.2``; the key's
  words include ``percent``, ``pct``, ``share``, ``rate``, ``fraction`` or
  ``proportion``). A count above 1 under any other key never supports it:
  exp86's "~23%" (69/300, derived) passed against an unrelated
  ``n_wrong_inside`` of 23 in rounds 1 to 6.
- A thousands-separated integer written in brackets is also supported when
  each of its parts is (a window written "(24,108)"). Unbracketed, it is one
  number: "1,000" is a thousand, never the parts 1 and 0 (exp86 round 5: "e.g.
  1,000+ labels" passed that way).
- A number inside a word of letters and digits of at least 4 characters (a
  shortened id, ``a7c40be9``, ``419c``), or inside any word of at least 3
  next to an ellipsis (``5aafb53d...704``), is supported when that word
  occurs in a source string.
- Integers from 0 to 8 without ``%`` are never reported (list positions and
  small counts; exp64's threshold). Numbers inside a URL are not read.
"""

from __future__ import annotations

import bisect
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

#: Integers up to this absolute value, written without ``%``, are not checked.
SMALL_INT_EXEMPT = 8

#: The shortest word of letters and digits looked up whole in the sources.
MIN_ID_WORD = 4

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>()\[\]\"'`]+", re.IGNORECASE)
#: An optional sign (ASCII hyphen-minus or U+2212 MINUS SIGN), then a number.
_NUM_RE = re.compile(
    r"(?P<sign>[-\u2212])?"
    r"(?P<num>[0-9]{1,3}(?:,[0-9]{3})+(?![0-9])(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)"
)
_EXP_RE = re.compile(r"[eE](?P<esign>[-+\u2212]?)[0-9]+(?![A-Za-z0-9])")
#: A percent sign, after at most a space, a no-break or a narrow no-break space.
_PCT_RE = re.compile(r"[ \u00a0\u202f]?%|\s+(?:percent|per\s?cent|pct)\b", re.I)
_SUFFIX_RE = re.compile(r"[kKM](?![A-Za-z0-9])")
_SUFFIXES = {"k": 1e3, "K": 1e3, "M": 1e6}
#: Between the two ends of a range: a hyphen, an en or em dash, or "to".
_RANGE_RE = re.compile(r"\s*(?:[-\u2013\u2014]|to)\s*")
_ELLIPSES = ("...", "\u2026")
_EPS = 1e-9

#: Words of a field's key that name a share: a value above 1 under it
#: supports a percent as written (any value in [0, 1] supports one anyway).
SHARE_KEY_WORDS = frozenset(
    {
        "share",
        "shares",
        "percent",
        "percentage",
        "pct",
        "rate",
        "rates",
        "fraction",
        "frac",
        "proportion",
    }
)
_KEY_WORD_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+")


def is_share_key(key: Any) -> bool:
    """Whether a field's key names a share, percent or rate (``error_pct``)."""
    if not isinstance(key, str):
        return False
    return any(w.lower() in SHARE_KEY_WORDS for w in _KEY_WORD_RE.findall(key))


def _is_word_char(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


@dataclass(frozen=True)
class NumberToken:
    """One number read from a text.

    ``text`` is the number as written (with its sign, percent sign or
    suffix); ``value`` its absolute value before any ``%``, suffix or
    exponent; ``decimals`` its count of decimal places; ``scale`` the
    suffix's factor (1 without one); ``exponent`` the power of ten of a
    scientific form; ``parts`` the values between thousands separators;
    ``word`` the word around it to look up whole in the sources (a possible
    id, see :func:`tokenize`), or ``""``; ``bracketed`` whether it stands
    alone inside brackets, as a window "(24,108)" does.
    """

    text: str
    value: float
    decimals: int
    percent: bool = False
    scale: float = 1.0
    exponent: int = 0
    parts: tuple[int, ...] = ()
    word: str = ""
    bracketed: bool = False

    @property
    def is_plain_integer(self) -> bool:
        """An integer written with no decimals, ``%``, suffix or exponent."""
        return (
            self.decimals == 0
            and not self.percent
            and self.scale == 1.0
            and self.exponent == 0
        )

    @property
    def exempt(self) -> bool:
        """A small integer, never reported."""
        return self.is_plain_integer and self.value <= SMALL_INT_EXEMPT


def _id_word(text: str, start: int, end: int) -> str:
    """The word around ``text[start:end]`` when it may be an id, else ``""``.

    A word of ASCII letters and digits with a letter and at least
    :data:`MIN_ID_WORD` characters (``a7c40be9``), or any word of at least 3
    next to an ellipsis, where an id was shortened (``5aafb53d...704``).
    """
    lo, hi = start, end
    while lo > 0 and _is_word_char(text[lo - 1]):
        lo -= 1
    while hi < len(text) and _is_word_char(text[hi]):
        hi += 1
    word = text[lo:hi]
    if len(word) >= MIN_ID_WORD and any(c.isalpha() for c in word):
        return word
    elided = text[:lo].endswith(_ELLIPSES) or text[hi:].startswith(_ELLIPSES)
    return word if elided and len(word) >= 3 else ""


def tokenize(text: str, *, skip_urls: bool = True) -> list[NumberToken]:
    """Every number in ``text``, by the module's reading rules."""
    if skip_urls:
        text = _URL_RE.sub(" ", text)
    tokens: list[NumberToken] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    while m := _NUM_RE.search(text, pos):
        num = m.group("num")
        start, end = m.start("num"), m.end("num")
        # A '-' or U+2212 is a sign only where no letter or digit precedes it
        # ("EMSR279-11", "2023-01-01"); the value is absolute either way.
        sign = m.group("sign") or ""
        if sign and m.start() > 0 and _is_word_char(text[m.start() - 1]):
            sign = ""
        digits = num.replace(",", "")
        decimals = len(digits.partition(".")[2])
        parts: tuple[int, ...] = ()
        if "," in num:
            parts = tuple(int(p) for p in num.split(".")[0].split(","))
        percent, scale, exponent = False, 1.0, 0
        glued = start > 0 and text[start - 1].isascii() and text[start - 1].isalpha()
        em = None if glued else _EXP_RE.match(text, end)
        # "1.2e-5" or "3e+06"; a bare "7e91" is as often a hex id's head.
        if em and (decimals or em.group("esign")):
            exponent = int(em.group()[1:].replace("\u2212", "-"))
            end = em.end()
        elif pm := _PCT_RE.match(text, end):
            percent = True
            end = pm.end()
        elif sm := _SUFFIX_RE.match(text, end):
            scale = _SUFFIXES[sm.group()]
            end = sm.end()
        tokens.append(
            NumberToken(
                text=(sign + text[start:end]).strip(),
                value=abs(float(digits)),
                decimals=decimals,
                percent=percent,
                scale=scale,
                exponent=exponent,
                parts=parts,
                word=_id_word(text, start, m.end("num")),
                bracketed=start > 0
                and text[start - 1] in "(["
                and text[end : end + 1] in (")", "]"),
            )
        )
        spans.append((start, end))
        pos = end
    # "32-59%": a range's percent sign is its lower end's too.
    for i in range(len(tokens) - 1):
        lower, upper = tokens[i], tokens[i + 1]
        plain = not (lower.percent or lower.scale != 1.0 or lower.exponent)
        between = text[spans[i][1] : spans[i + 1][0]]
        if upper.percent and plain and _RANGE_RE.fullmatch(between):
            tokens[i] = replace(lower, percent=True)
    return tokens


class NumberPool:
    """The numbers and strings of a run's sources, to check answers against.

    ``sources`` are walked recursively: dict keys and values, list, tuple and
    set items, numbers (booleans excluded), and every string, read by
    :func:`tokenize` (a string's percent adds its fraction, and a
    thousands-separated number's parts are added beside its value). The
    shares a percent may be read against are kept apart (see the module
    docstring): fractions, and percents written or keyed as such.
    """

    def __init__(self, sources: Iterable[Any]) -> None:
        values: set[float] = set()
        fractions: set[float] = set()
        percents: set[float] = set()
        self._texts: list[str] = []
        self._walk(list(sources), values, (fractions, percents), set(), False)
        self._values = sorted(values)
        self._ints = {int(v) for v in values if v.is_integer()}
        self._fractional = [v for v in self._values if not v.is_integer()]
        self._fractions = sorted(fractions)
        self._percents = sorted(percents)

    def _walk(
        self,
        obj: Any,
        values: set[float],
        shares: tuple[set[float], set[float]],
        seen: set[int],
        share_key: bool,
    ) -> None:
        fractions, percents = shares
        if isinstance(obj, bool) or obj is None:
            return
        if isinstance(obj, int | float):
            v = abs(float(obj))
            if math.isfinite(v):
                values.add(v)
                if v <= 1:
                    fractions.add(v)
                elif share_key:
                    percents.add(v)
            return
        if isinstance(obj, str):
            if obj:
                self._texts.append(obj.lower())
            for tok in tokenize(obj, skip_urls=False):
                value = tok.value * tok.scale * 10.0**tok.exponent
                values.add(value)
                if tok.percent:
                    values.add(tok.value / 100)
                    fractions.add(tok.value / 100)
                    percents.add(tok.value)
                elif value <= 1:
                    fractions.add(value)
                elif share_key:
                    percents.add(value)
                values.update(float(p) for p in tok.parts)
            return
        if isinstance(obj, dict | list | tuple | set | frozenset):
            if id(obj) in seen:
                return
            seen.add(id(obj))
            if isinstance(obj, dict):
                for key, child in obj.items():
                    self._walk(key, values, shares, seen, False)
                    self._walk(child, values, shares, seen, is_share_key(key))
                return
            for child in obj:
                self._walk(child, values, shares, seen, share_key)

    @staticmethod
    def _within(values: list[float], x: float, tol: float) -> bool:
        tol = tol * (1 + _EPS) + 1e-12
        i = bisect.bisect_left(values, x - tol)
        return i < len(values) and values[i] <= x + tol

    def _near(self, x: float, tol: float) -> bool:
        return self._within(self._values, x, tol)

    def supports(self, tok: NumberToken) -> bool:
        """Whether a source supports ``tok``; an exempt integer always is."""
        if tok.exempt:
            return True
        if tok.word and any(tok.word.lower() in t for t in self._texts):
            return True
        x = tok.value
        if tok.is_plain_integer:
            return (
                int(x) in self._ints
                or self._within(self._fractional, x, 0.5)
                or self._near(x / 100, 0.0)
                or tok.bracketed
                and bool(tok.parts)
                and all(p <= SMALL_INT_EXEMPT or p in self._ints for p in tok.parts)
            )
        tol = 0.5 * 10.0 ** (-tok.decimals)
        if tok.exponent:
            return self._near(x * 10.0**tok.exponent, tol * 10.0**tok.exponent)
        if tok.scale != 1.0:
            return self._near(x * tok.scale, tol * tok.scale) or self._near(x, tol)
        if tok.percent:
            # A percent: only a share supports it, never a count that happens
            # to have its digits.
            return self._within(self._fractions, x / 100, tol / 100) or self._within(
                self._percents, x, tol
            )
        # A decimal: as written, or as a percent of a fraction.
        return self._near(x, tol) or self._near(x / 100, tol / 100)

    def unsupported(self, answer: str) -> list[str]:
        """The numbers of ``answer``, as written and once each, no source supports."""
        out: list[str] = []
        for tok in tokenize(answer):
            if not self.supports(tok) and tok.text not in out:
                out.append(tok.text)
        return out


def unsupported_numbers(answer: str, sources: Iterable[Any]) -> list[str]:
    """The numbers of ``answer``, as written, that no source supports.

    ``sources`` are the run's full tool results and the user's messages (the
    brief and the user's turns of the history); see the module docstring for
    the reading rules.

    Examples
    --------
    >>> unsupported_numbers("5,565+ windows remain", [{"n_windows": 16384}])
    ['5,565']
    >>> unsupported_numbers("about 23% wrong", [{"error_rate": 0.2298}])
    []
    >>> unsupported_numbers("about 23% wrong", [{"n_wrong_inside": 23}])
    ['23%']
    >>> unsupported_numbers("0% of them", [{"n_nodata": 0}])
    []
    """
    return NumberPool(sources).unsupported(answer)
