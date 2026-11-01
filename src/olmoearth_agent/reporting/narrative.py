# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Stakeholder narrative: freshness gate and measured accuracy (skill #6).

Assembles a Markdown report from prediction results + the run's
provenance, and **refuses to render stale tiles** past a configurable
window, so a disaster-response brief never shows imagery that has since
gone out of date. Pure functions; no I/O.

Its "How accurate is this map" section states what the request's
verification tools measured, from their recorded results
(``ProvenanceLog.verification``), never from figures the model typed. A
request is one run of the agent: on the web UI, one message, so an estimate
made in an earlier message is not in it. The section states each error rate
with the map it is of and every interval it has, named for what it is (valid
at every sample size; the design's own, exact or by the union bound, or
nominal and approximate where the package's method is a Wilson interval; or,
for windows labelled without a design, the interval they would earn as a
simple random sample, which nothing establishes), the number of labels, the
design and its seed, and the classes whose producer's and user's accuracy
intervals do not overlap, or whose share the map draws too little or too
much (the tool's ``user_vs_producer`` fact); each certified zone with what
alpha bounds and
olmoearth-inferencex's note on what delta counts (``ISSUED_NOTE``), word for
word, and a zone tried after an earlier one on the same design file with
another alpha, delta, rule or labels as a second look, not a certified zone;
each certificate written or checked (``olmoearth_write_certificate``,
``olmoearth_verify_certificate``) with the package's statement,
``ISSUED_NOTE`` and ``ISSUERS_WORD`` (what is taken on the issuer's word),
word for word, since a certificate restates its zone and never replaces that
note, and the tool's summary of what a certificate proves (from the
package's Usage, "certificate and verify"); each which-map verdict with the
test or interval it rests on; labels changed on a design file after a result
was seen (other inline labels, another whole-map figure, or another digest
of the labels); and the notes of the tool and the package that limit each
statement. Every figure is formatted from the
tool's value as the tools format their own sentences
(:func:`~olmoearth_agent.tools.statistical_rules.percent`), or copied from a
sentence the tool wrote. Without such a result the section says plainly
that no error rate was estimated in the request, and names the tools that
estimate one.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from olmoearth_agent.tools.statistical_rules import percent

#: The verification tools whose recorded results the accuracy section states.
PLAN_TOOL = "olmoearth_plan_label_sample"
ESTIMATE_TOOL = "olmoearth_estimate_map_error"
CERTIFY_TOOL = "olmoearth_certify_zone"
#: The certificate a certified zone is packed into, and its check.
WRITE_CERTIFICATE_TOOL = "olmoearth_write_certificate"
VERIFY_CERTIFICATE_TOOL = "olmoearth_verify_certificate"

#: The accuracy section's heading.
ACCURACY_HEADING = "## How accurate is this map"

#: The section's first line when the request measured anything.
MEASURED_LEAD = (
    "Every figure in this section is a result of this request's verification "
    "tools (olmoearth-inferencex), as the tool returned it; none was typed or "
    "worked out for this report."
)
#: Said when no call of the request estimated the map's error rate.
NOT_MEASURED = (
    "No error rate of this map has been estimated in this request: no call of "
    "olmoearth_estimate_map_error or olmoearth_certify_zone graded labels of "
    "it, so this report states no error rate, accuracy or interval for it."
)
#: The tools that estimate it, named after :data:`NOT_MEASURED`.
HOW_TO_MEASURE = (
    "To estimate it, olmoearth_plan_label_sample plans which windows to label, "
    "and olmoearth_estimate_map_error grades the labels: the error rate with "
    "the interval valid at every sample size first, then the one its design "
    "earns. On the web UI each message is a request of "
    "its own: an error rate estimated in an earlier message is stated here only "
    "when olmoearth_estimate_map_error is called again in this request, with "
    "that design_path and its labels."
)
#: Said when the request graded only which of two maps is more accurate.
NEITHER_MAP_MEASURED = (
    "Neither map's own accuracy was measured in this request: which map is more "
    "accurate is not how accurate either map is. A random sample of one map "
    "(olmoearth_plan_label_sample with design='random', then "
    "olmoearth_estimate_map_error) measures its error rate."
)

#: How each interval of an error rate is named: ``valid`` and ``design`` are
#: the estimate's ``intervals``; ``none`` is the one interval of windows
#: labelled without a design.
INTERVAL_NAMES = {
    "valid": "Interval valid at every sample size",
    "design": "Interval this design earns",
    "none": (
        "Interval if these windows are a simple random sample, which no design "
        "establishes"
    ),
}
#: A design interval whose method the package calls nominal (a Wilson
#: interval on the design's effective sample size, under the confidence and
#: proportional designs; its ``VALID_NOTE`` gives the coverage it fell to), or
#: a t interval over tiles: named approximate, never as a plain 95%.
_NOMINAL_METHOD_RE = re.compile(r"\bwilson\b|\bt\s+on\s+\d+\s+df\b", re.I)
#: The level of such an interval.
NOMINAL_LEVEL = " (nominal {}, approximate: not valid at every sample size)"

#: The package's notes on an error rate that limit it, stated word for word.
#: ``scope`` is stated apart (:func:`_scope_lines`): its later sentences give
#: another map's figures (exp88's PASTIS map), which a report of this case
#: does not state as its own.
_ESTIMATE_NOTES = ("warning", "bounds_note", "condition_note", "valid_note")
#: A sentence of a scope note that gives another map's figures or the
#: command line's options: the note is stated up to it.
_FIGURE_OR_OPTION_RE = re.compile(r"\d\s*%|(?<![\w-])--\w")

#: A note of olmoearth-inferencex, as the section introduces it.
_PACKAGE_NOTE = "- olmoearth-inferencex's note: {}"
#: The package's ``warning`` on an error rate that also gives the interval
#: valid at every sample size: each whole-map warning of
#: ``estimate_error_rate`` (thin strata, no labelled window wrong) is of the
#: interval the design earns, never of the valid one.
_PACKAGE_DESIGN_NOTE = (
    "- olmoearth-inferencex's note on the interval this design earns: {}"
)
#: The first sentences of a package scope note, before another map's figures.
_PACKAGE_SCOPE_PART = (
    "- olmoearth-inferencex's note on scope (its first sentences, before "
    "another map's figures and the command line's options): {}"
)
#: A must_state sentence of the tool's, as the section introduces it.
_TOOL_LIMIT = "- Limit stated by the tool: {}"
#: How the section introduces the ``user_vs_producer`` fact: what each class's
#: user's accuracy leaves out (agent trial 12, N3/files: classes called well
#: mapped on their user's accuracy alone).
USER_VS_PRODUCER_LEAD = "What each class's user's accuracy leaves out"

#: Windows labelled without a design: what the package's check of them is not.
NO_DESIGN_LIMIT = (
    "- Labels and design: {n} labels on windows labelled without a design; no "
    "design file and no seed. olmoearth-inferencex checked only that they do "
    "not look like a review set, which does not make them a random sample: "
    "windows clustered by tile, or from the confident end of the map, pass "
    "that check, and the interval above then does not hold."
)
#: The package's check found such windows drawn from the confident end.
CONFIDENT_END = (
    "- olmoearth-inferencex's check found these windows drawn from the "
    "confident end of the map (looks_drawn_from_the_confident_end), where the "
    "map is wrong less often than elsewhere, so this rate can understate the "
    "map's."
)
#: What a certified zone means, in place of the tool's sentence that names its
#: alpha and delta (the verdict above names them).
ZONE_MEANING = (
    "- What it means: the zone's error rate is at most alpha, a statement that "
    "fails on at most delta of random draws; nothing outside the zone is "
    "certified."
)
#: A zone call that changed its alpha, delta or rule after an earlier zone call
#: on the same design file, or its labels after an earlier call graded others
#: there: a second look, which the guarantee does not cover
#: (``post_hoc_alpha``, ``rule_switch_after_failure``).
SECOND_LOOK = (
    "- Not stated as a certified zone: this call ({settings}) came after an "
    "earlier call of this request on the same design file and changed its "
    "{changed} once that result was seen, a second look that falls outside "
    "what the guarantee covers."
)
#: Labels on a design file that differ from an earlier call's on it.
LABELS_CHANGED = (
    "- Labels changed after a result was seen: an earlier call of this request "
    "graded other labels on this design file."
)
#: The map of a design file whose plan is not in the request.
MAP_NOT_IN_REQUEST = (
    "named only by its design file (the plan that drew it is not in this request)"
)
#: A certificate's notes, each word for word from the tool's result: what it
#: proves (the tool's), what delta counts and what the issuer's word covers
#: (olmoearth-inferencex's). The zone's ``ISSUED_NOTE`` is stated with every
#: certificate: the certificate restates the zone, it does not replace it.
_CERTIFICATE_NOTES = (
    ("what_it_proves", "- What a certificate proves (the tool's words): {}"),
    ("issued_note", "- What delta counts (olmoearth-inferencex's note, verbatim): {}"),
    (
        "issuers_word",
        "- Taken on the issuer's word (olmoearth-inferencex's note, verbatim): {}",
    ),
)
#: A certificate checked against the map the certificate itself names.
MAP_NOT_GIVEN = (
    " (the path the certificate names, not a copy given to the check: it shows "
    "only that the certificate agrees with that file)"
)

#: A recorded verification call that computed: its tool, arguments and result.
_Call = tuple[str, dict[str, Any], dict[str, Any]]
#: A zone's figures a statement copies from ``olmoearth_certify_zone``.
_ZONE_KEYS = ("n_zone", "coverage", "alpha", "delta", "rule")
#: A zone's settings, fixed before its labels are seen; the tool's defaults.
_ZONE_SETTINGS: dict[str, Any] = {"alpha": None, "delta": 0.1, "rule": "prefix"}


def _parse_iso(timestamp: str) -> datetime:
    """Parse an ISO-8601 string, tolerating a trailing ``Z``."""
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def freshness_check(
    creation_time: str,
    window_hours: float = 24.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Classify one result as fresh or stale relative to ``window_hours``.

    Returns ``fresh`` (bool or ``None`` if the timestamp can't be parsed),
    ``age_hours``, and ``window_hours``.
    """
    now = now or datetime.now(timezone.utc)
    try:
        created = _parse_iso(creation_time)
    except (ValueError, AttributeError, TypeError):
        return {
            "creation_time": creation_time,
            "fresh": None,
            "age_hours": None,
            "window_hours": window_hours,
            "error": "unparseable timestamp",
        }
    age_hours = (now - created).total_seconds() / 3600.0
    return {
        "creation_time": creation_time,
        "age_hours": round(age_hours, 2),
        "fresh": age_hours <= window_hours,
        "window_hours": window_hours,
    }


def typed_text(title: str, results: Iterable[Mapping[str, Any]]) -> list[str]:
    """The text the report renders from its caller: the title and each result's.

    The title, and each result's ``result_id`` and ``property_names``, as
    :func:`build_narrative` renders them. They are the model's typing, never
    a tool's figures: ``olmoearth_case_narrative`` refuses an accuracy figure
    in them.

    Examples
    --------
    >>> typed_text("Flood brief", [{"result_id": "r1", "property_names": ["p"]}])
    ['Flood brief', 'r1', 'p']
    """
    out = [title]
    for result in results:
        if result.get("result_id") is not None:
            out.append(str(result["result_id"]))
        out.extend(str(p) for p in result.get("property_names") or [])
    return out


# --------------------------------------------------------------------------- the verification calls


def _get(obj: Any, key: str) -> Any:
    """``key`` of a recorded call: a mapping's item or a dataclass's attribute."""
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _computed(verification: Iterable[Any]) -> list[_Call]:
    """``(tool, arguments, result)`` of each verification call that ran and computed.

    A failed call, or one whose tool said the optional package is missing
    (``available`` false), measured nothing.
    """
    out: list[_Call] = []
    for call in verification:
        result = _get(call, "result")
        if not _get(call, "ok") or not isinstance(result, Mapping):
            continue
        if result.get("available") is not True:
            continue
        arguments = _get(call, "arguments")
        args = dict(arguments) if isinstance(arguments, Mapping) else {}
        out.append((str(_get(call, "tool")), args, dict(result)))
    return out


def _fact(result: Mapping[str, Any], fact_id: str) -> Mapping[str, Any] | None:
    """The result's fact ``fact_id``, or ``None``."""
    for item in result.get("facts") or []:
        if isinstance(item, Mapping) and item.get("id") == fact_id:
            return item
    return None


def _count(n: Any) -> str:
    """A count as the tools' sentences write it, with thousands separators."""
    return f"{n:,}" if isinstance(n, int) and not isinstance(n, bool) else str(n)


def _design_path(arguments: Mapping[str, Any], result: Mapping[str, Any]) -> Any:
    """The design file a call read or wrote."""
    return result.get("design_path") or arguments.get("design_path")


def _design_name(result: Mapping[str, Any]) -> Any:
    """The design by the plan's name for it (``random``, ``class``, ...)."""
    return result.get("plan_design") or result.get("design")


def _seed_text(seed: Any) -> str:
    """``seed 7``, or that the call recorded none."""
    return f"seed {seed}" if seed is not None else "seed not recorded"


def _labels_line(
    n: Any, design: Any, seed: Any, path: Any, *, what: str = "labels"
) -> str:
    """The labels (or planned windows), the design, its seed and the design file."""
    return (
        f"- Labels, design and seed: {_count(n)} {what} on the {design} design, "
        f"{_seed_text(seed)}; design file `{path}`."
    )


def _limits(result: Mapping[str, Any]) -> list[str]:
    """The tool's must_state sentences, each as a limit it states."""
    return [
        _TOOL_LIMIT.format(s)
        for s in result.get("must_state") or []
        if isinstance(s, str) and s.strip()
    ]


def _rate(block: Mapping[str, Any]) -> str:
    """A point estimate as a percent, or the range the package gives in its place."""
    if block.get("estimate") is not None:
        return percent(block["estimate"])
    span = block.get("estimate_range")
    if isinstance(span, list) and len(span) == 2:
        return f"between {percent(span[0])} and {percent(span[1])}"
    return "not given"


def _span(block: Mapping[str, Any]) -> str:
    """An interval's ends as percents: ``9.1% to 16.0%``."""
    return f"{percent(block.get('low'))} to {percent(block.get('high'))}"


def _coverage(result: Mapping[str, Any]) -> str | None:
    """``95%`` from the result's nominal coverage, or ``None``."""
    cov = result.get("nominal_coverage")
    if isinstance(cov, (int, float)) and not isinstance(cov, bool):
        return f"{100.0 * float(cov):g}%"
    return None


def _interval_name(which: str, method: Any, result: Mapping[str, Any]) -> str:
    """An interval named for what it is, with its level.

    The design's interval is named nominal and approximate where the
    package's method is a Wilson interval on the design's effective sample
    size (the confidence and proportional designs): the package calls it
    nominal, not valid at every sample size. Every other interval of the
    package covers at least its level (exact, by the union bound, or by the
    betting interval) and is named with it.
    """
    cov = _coverage(result)
    name = INTERVAL_NAMES.get(which, which)
    if which != "valid" and _NOMINAL_METHOD_RE.search(str(method or "")):
        return name + NOMINAL_LEVEL.format(cov or "level")
    return f"{name} ({cov})" if cov else name


def _intervals(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every interval of an error rate, named: the valid one first, the design's next."""
    entries = result.get("intervals")
    if isinstance(entries, list) and entries:
        return [
            {k: e.get(k) for k in ("interval", "low", "high", "method")}
            for e in entries
            if isinstance(e, Mapping)
        ]
    return [
        {
            "interval": "design",
            "low": result.get("low"),
            "high": result.get("high"),
            "method": result.get("method"),
        }
    ]


def _scope_lines(result: Mapping[str, Any]) -> list[str]:
    """The package's ``scope`` note, up to its first sentence with a figure.

    ``SCOPE_CERTIFY`` and the first sentences of ``SCOPE_ESTIMATE`` limit
    the statement (a whole-map figure says nothing of a part predicted from
    other inputs); ``SCOPE_ESTIMATE``'s later sentences give another map's
    rates, and both end with the command line's option, which the agent has
    no tool for. Stated word for word, as far as it goes.
    """
    scope = result.get("scope")
    if not isinstance(scope, str) or not scope.strip():
        return []
    kept: list[str] = []
    for sentence in re.split(r"(?<=\.)\s+", scope.strip()):
        if _FIGURE_OR_OPTION_RE.search(sentence):
            break
        kept.append(sentence)
    if not kept:
        return []
    text = " ".join(kept)
    whole = text == scope.strip()
    return [(_PACKAGE_NOTE if whole else _PACKAGE_SCOPE_PART).format(text)]


def _per_class_notes(
    result: Mapping[str, Any], named: Iterable[Any]
) -> tuple[list[str], list[str]]:
    """The package's notes on a per-class table, and on each class it names.

    The table's ``warning`` and ``valid_note`` (its intervals hold each on
    its own, not jointly), then the ``warning`` of each class the
    ``weakest_classes`` fact names (few labels, few errors, near census, thin
    strata), each word for word.
    """
    table = result.get("per_class")
    if not isinstance(table, Mapping):
        return [], []
    notes = [
        str(table[key])
        for key in ("warning", "valid_note")
        if isinstance(table.get(key), str) and str(table[key]).strip()
    ]
    rows = table.get("per_class")
    rows = rows if isinstance(rows, Mapping) else {}
    by_class: list[str] = []
    for c in named:
        row = rows.get(str(c))
        warning = row.get("warning") if isinstance(row, Mapping) else None
        if isinstance(warning, str) and warning.strip():
            by_class.append(f"- olmoearth-inferencex's note on class {c}: {warning}")
    return notes, by_class


def _map_of_arguments(arguments: Mapping[str, Any]) -> str | None:
    """The map a plan (or an estimate of windows with no design) read, as named."""
    if arguments.get("result_id"):
        return _map_named(
            {
                "result_id": arguments["result_id"],
                "property_name": arguments.get("property_name"),
            }
        )
    if arguments.get("scores_path"):
        return _map_named({"scores_path": arguments["scores_path"]})
    if isinstance(arguments.get("scores"), list):
        return "scores given inline"
    return None


def _map_named(source: Any) -> str:
    """A map as a result or a plan names it."""
    if not isinstance(source, Mapping):
        return "not named"
    if source.get("result_id"):
        prop = source.get("property_name")
        named = f"Studio result `{source['result_id']}`"
        return f"{named} ({prop})" if prop else named
    if source.get("scores_path"):
        return f"scores file `{source['scores_path']}`"
    return "not named"


def _figure(result: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """What a call's labels gave, to tell two label sets of one design apart.

    The ``whole_map_estimate`` (an estimate's, and a zone's from the same
    labels) or the which-map counts: the package computes each from the
    design and the labels alone, so a different figure on one design file is
    a different set of labels.
    """
    whole = _fact(result, "whole_map_estimate")
    if whole is not None:
        return ("rate", whole.get("estimate"), whole.get("low"), whole.get("high"))
    which = _fact(result, "which_map")
    if which is not None:
        return ("which", which.get("n_a_right"), which.get("n_b_right"))
    return None


def _same_labels(first: _Call, later: _Call) -> bool:
    """Whether two calls on one design file graded the same labels, as far as seen.

    Different figures, different digests of the labels (``labels_sha256``,
    which the zone and the certificate return), or different labels given
    inline, are different labels; labels read from a file with no digest are
    compared by their figures only.
    """
    (_t1, a1, r1), (_t2, a2, r2) = first, later
    d1, d2 = r1.get("labels_sha256"), r2.get("labels_sha256")
    if isinstance(d1, str) and isinstance(d2, str) and d1 != d2:
        return False
    f1, f2 = _figure(r1), _figure(r2)
    if f1 is not None and f2 is not None and f1 != f2:
        return False
    for key in ("wrong", "reference"):
        x, y = a1.get(key), a2.get(key)
        if x is not None and y is not None and x != y:
            return False
    return True


def _graded(call: _Call) -> bool:
    """Whether a call graded labels (an estimate, a zone or certificate, a grading)."""
    tool, _a, r = call
    graders = (ESTIMATE_TOOL, CERTIFY_TOOL, WRITE_CERTIFICATE_TOOL)
    return tool in graders and r.get("n_labelled") is not None


def _zone_settings(
    arguments: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    """A zone call's alpha, delta and rule: the result's, else its arguments'."""
    return {
        key: result.get(key, arguments.get(key, default))
        for key, default in _ZONE_SETTINGS.items()
    }


def _settings_text(settings: Mapping[str, Any]) -> str:
    """``alpha=0.1, delta=0.1, rule 'bonferroni'``, as the tool's verdict has them."""

    def num(x: Any) -> str:
        """A number as ``:g`` writes it, anything else as it is."""
        return f"{x:g}" if isinstance(x, (int, float)) else str(x)

    return (
        f"alpha={num(settings['alpha'])}, delta={num(settings['delta'])}, "
        f"rule '{settings['rule']}'"
    )


def _second_looks(calls: list[_Call]) -> dict[int, dict[str, Any]]:
    """Per graded call (by index): labels changed, and what a zone changed.

    In call order, per design file: a call whose labels differ from an
    earlier graded call's has ``labels_changed``; a zone call tested after
    an earlier tested zone call with another alpha, delta or rule (or other
    labels) lists them under ``changed``. The guarantee holds for settings
    and labels fixed before a result was seen.
    """
    looks: dict[int, dict[str, Any]] = {}
    before: dict[str, list[_Call]] = {}
    for i, call in enumerate(calls):
        tool, a, r = call
        path = _design_path(a, r)
        if tool == PLAN_TOOL or not path or not _graded(call):
            continue
        earlier = before.setdefault(str(path), [])
        labels_changed = any(not _same_labels(c, call) for c in earlier)
        changed: list[str] = []
        if tool == CERTIFY_TOOL:
            mine = _zone_settings(a, r)
            zones = [_zone_settings(aa, rr) for t, aa, rr in earlier if t == tool]
            changed = [k for k in _ZONE_SETTINGS if any(z[k] != mine[k] for z in zones)]
            if labels_changed:
                changed.append("labels")
        looks[i] = {"labels_changed": labels_changed, "changed": changed}
        earlier.append(call)
    return looks


def _names(items: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _error_rate(
    arguments: Mapping[str, Any],
    result: Mapping[str, Any],
    seed: Any,
    *,
    map_named: str,
    labels_changed: bool,
) -> tuple[list[str], dict[str, Any]]:
    """An error rate's lines and statement, from ``olmoearth_estimate_map_error``."""
    n, total = result.get("n_labelled"), result.get("n_population")
    valid = result.get("valid_interval")
    point = valid if isinstance(valid, Mapping) else result
    intervals = _intervals(result)
    path = _design_path(arguments, result)
    lines = [
        "### Error rate (`olmoearth_estimate_map_error`)",
        "",
        f"- Map: {map_named}.",
        f"- Error rate: {_rate(point)}, estimated from {_count(n)} labels on a "
        f"population of {_count(total)} windows.",
    ]
    for entry in intervals:
        which = str(entry.get("interval")) if path else "none"
        name = _interval_name(which, entry.get("method"), result)
        lines.append(f"- {name}: {_span(entry)} (method: {entry.get('method')}).")
    design = _design_name(result)
    if path:
        lines.append(_labels_line(n, design, seed, path))
    else:
        lines.append(NO_DESIGN_LIMIT.format(n=_count(n)))
        check = result.get("review_set_check")
        if isinstance(check, Mapping) and check.get(
            "looks_drawn_from_the_confident_end"
        ):
            lines.append(CONFIDENT_END)
    if labels_changed:
        lines.append(LABELS_CHANGED)
    weakest = _fact(result, "weakest_classes")
    if weakest and weakest.get("sentence"):
        lines.append(f"- Per class: {weakest['sentence']}")
    # What a class's user's accuracy leaves out: its producer's accuracy and
    # the share the map draws, as the tool states them (agent trial 12, N3).
    contrast = _fact(result, "user_vs_producer")
    if contrast and contrast.get("sentence"):
        lines.append(f"- {USER_VS_PRODUCER_LEAD}: {contrast['sentence']}")
    table_notes, class_notes = _per_class_notes(
        result, (weakest or {}).get("lowest") or []
    )
    lines += class_notes
    lines += _limits(result)
    if isinstance(result.get("class_map_note"), str):
        lines.append(_TOOL_LIMIT.format(result["class_map_note"]))
    # Beside the interval valid at every sample size, the package's warning is
    # named as of the design's interval (each of its whole-map warnings is).
    two = any(e.get("interval") == "valid" for e in intervals) and bool(path)
    warning = result.get("warning")
    if two and isinstance(warning, str) and warning.strip():
        lines.append(_PACKAGE_DESIGN_NOTE.format(warning))
    keys = [k for k in _ESTIMATE_NOTES if not (two and k == "warning")]
    notes: list[str] = []
    for text in [result.get(key) for key in keys] + table_notes:
        if isinstance(text, str) and text.strip() and text not in notes:
            notes.append(text)
    lines += [_PACKAGE_NOTE.format(text) for text in notes]
    lines += _scope_lines(result)
    statement: dict[str, Any] = {
        "kind": "error_rate",
        "tool": ESTIMATE_TOOL,
        "map": map_named,
        "design": design,
        "seed": seed,
        "design_path": path,
        "n_labelled": n,
        "n_population": total,
        "estimate": point.get("estimate"),
        "intervals": intervals,
        "labels_changed": labels_changed,
    }
    return lines, statement


def _zone(
    arguments: Mapping[str, Any],
    result: Mapping[str, Any],
    seed: Any,
    *,
    map_named: str,
    estimated: bool,
    changed: list[str],
) -> tuple[list[str], dict[str, Any]]:
    """A certified zone's lines and statement, from ``olmoearth_certify_zone``.

    ``estimated`` says an error rate from the same labels is stated already;
    else the zone's own whole-map estimate (the same labels) is. ``changed``
    lists what this call changed after an earlier zone call on the same
    design file (:func:`_second_looks`): the zone is then a second look,
    never stated as certified.
    """
    lines = [
        "### Certified zone (`olmoearth_certify_zone`)",
        "",
        f"- Map: {map_named}.",
    ]
    path = _design_path(arguments, result)
    design = _design_name(result)
    n = result.get("n_labelled")
    certified = bool(result.get("certified"))
    verdict = result.get("verdict")
    stated_as_certified = certified and not changed
    if changed:
        settings = _settings_text(_zone_settings(arguments, result))
        lines.append(SECOND_LOOK.format(settings=settings, changed=_names(changed)))
    if isinstance(verdict, str) and verdict and not (changed and certified):
        lines.append(f"- {verdict[0].upper()}{verdict[1:].rstrip('.')}.")
    elif not verdict and isinstance(result.get("reason"), str):
        lines.append(f"- No certified zone: {result['reason']}")
    if stated_as_certified:
        lines.append(ZONE_MEANING)
    if n is not None:
        lines.append(_labels_line(n, design, seed, path))
    else:
        lines.append(
            f"- Design and seed: the {design} design, {_seed_text(seed)}; design "
            f"file `{path}`."
        )
    if not changed:
        lines += _limits(result)
        if stated_as_certified and isinstance(result.get("issued_note"), str):
            lines.append(
                "- What delta counts (olmoearth-inferencex's note, verbatim): "
                + result["issued_note"]
            )
        if isinstance(result.get("note"), str) and result["note"].strip():
            lines.append(_PACKAGE_NOTE.format(result["note"]))
        lines += _scope_lines(result)
    whole = _fact(result, "whole_map_estimate")
    if not estimated and whole and whole.get("sentence"):
        lines.append(f"- Error rate from the same labels: {whole['sentence']}")
    statement: dict[str, Any] = {
        "kind": "certified_zone",
        "tool": CERTIFY_TOOL,
        "map": map_named,
        "certified": certified,
        "stated_as_certified": stated_as_certified,
        "changed_after_an_earlier_zone": list(changed),
        "design": design,
        "seed": seed,
        "design_path": path,
        "n_labelled": n,
    }
    statement.update({k: result.get(k) for k in _ZONE_KEYS})
    return lines, statement


def _certificate(
    arguments: Mapping[str, Any],
    result: Mapping[str, Any],
    seed: Any,
    *,
    map_named: str,
    labels_changed: bool,
    estimated: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    """A certificate's lines and statement, from ``olmoearth_write_certificate``.

    The package's statement, ``ISSUED_NOTE`` and what is taken on the
    issuer's word (``ISSUERS_WORD``), each word for word, and the tool's
    summary of what a certificate proves; or why none was written (no zone
    certified on the labels). ``estimated`` says an error rate from the same
    labels is stated already; else the call's own whole-map estimate (the
    same labels, as the zone states it) is.
    """
    path = _design_path(arguments, result)
    written = bool(result.get("certificate_written"))
    lines = [
        "### Certificate (`olmoearth_write_certificate`)",
        "",
        f"- Map: {map_named}.",
    ]
    said = result.get("statement")
    if written:
        lines += [
            f"- Written: `{result.get('certificate_path')}`, for the map of "
            f"{result.get('date')} as the issuer states, and verified by "
            "olmoearth-inferencex before it was written.",
            f"- Statement (olmoearth-inferencex's, verbatim): {said}",
        ]
    elif isinstance(result.get("reason"), str):
        lines.append(f"- No certificate: {result['reason']}")
    lines.append(
        _labels_line(result.get("n_labelled"), _design_name(result), seed, path)
    )
    if labels_changed:
        lines.append(LABELS_CHANGED)
    if written:
        lines += [
            template.format(result[key])
            for key, template in _CERTIFICATE_NOTES
            if isinstance(result.get(key), str)
        ]
    lines += _limits(result)
    whole = _fact(result, "whole_map_estimate")
    if not estimated and whole and whole.get("sentence"):
        lines.append(f"- Error rate from the same labels: {whole['sentence']}")
    statement: dict[str, Any] = {
        "kind": "certificate",
        "tool": WRITE_CERTIFICATE_TOOL,
        "map": map_named,
        "written": written,
        "certificate_path": result.get("certificate_path"),
        "date": result.get("date"),
        "design": _design_name(result),
        "seed": seed,
        "design_path": path,
        "n_labelled": result.get("n_labelled"),
        "labels_changed": labels_changed,
    }
    statement.update({k: result.get(k) for k in _ZONE_KEYS})
    return lines, statement


def _certificate_check(result: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """A certificate check's lines and statement, from ``olmoearth_verify_certificate``.

    Verified, the statement as certify run again gives it and ``ISSUED_NOTE``;
    not verified, each failed check in the package's words. The package's
    notes (what is taken on the issuer's word among them) are stated as given.
    """
    verified = bool(result.get("verified"))
    said = result.get("statement")
    read = f"`{result.get('map_read')}`" if result.get("map_read") else "not read"
    lines = [
        "### Certificate checked (`olmoearth_verify_certificate`)",
        "",
        f"- Certificate: `{result.get('certificate_path')}`; map: {read}"
        + ("" if result.get("map_given") else MAP_NOT_GIVEN)
        + ".",
    ]
    if verified:
        lines.append(
            "- Verified: olmoearth-inferencex built it again from the map, its "
            "labels and its settings, and it is the certificate written."
        )
        if isinstance(said, str):
            lines.append(f"- Statement (olmoearth-inferencex's, verbatim): {said}")
    else:
        raw = result.get("checks")
        checks = raw if isinstance(raw, Mapping) else {}
        failed = [
            f"{name}: {(checks.get(name) or {}).get('detail')}"
            for name in result.get("failed_checks") or []
        ]
        lines.append(
            "- Not verified, so its statement is not shown to hold for this map: "
            + ("; ".join(failed) or "no check passed")
            + "."
        )
    window = result.get("window_lookup")
    cell = window.get("window") if isinstance(window, Mapping) else None
    if isinstance(window, Mapping) and isinstance(cell, list) and len(cell) == 2:
        lines.append(f"- Window {cell[0]}, {cell[1]}: {window.get('where')}.")
    lines += [
        template.format(result[key])
        for key, template in _CERTIFICATE_NOTES[:2]
        if verified and isinstance(result.get(key), str)
    ]
    lines += [
        _PACKAGE_NOTE.format(note)
        for note in result.get("notes") or []
        if isinstance(note, str) and note.strip()
    ]
    lines += _limits(result)
    statement: dict[str, Any] = {
        "kind": "certificate_check",
        "tool": VERIFY_CERTIFICATE_TOOL,
        "verified": verified,
        "certificate_path": result.get("certificate_path"),
        "map_read": result.get("map_read"),
        "map_given": bool(result.get("map_given")),
        "failed_checks": list(result.get("failed_checks") or []),
        "seed": result.get("seed"),
        "window": cell,
    }
    return lines, statement


def _which_map(
    arguments: Mapping[str, Any],
    result: Mapping[str, Any],
    seed: Any,
    *,
    labels_changed: bool,
) -> tuple[list[str], dict[str, Any]]:
    """A which-map grading's lines and statement (a ``disagreement`` design)."""
    lines = [
        "### Which of two maps is more accurate (`olmoearth_estimate_map_error`)",
        "",
        f"- Maps: map a is {_map_named(result.get('map_a'))}; map b is "
        f"{_map_named(result.get('map_b'))}.",
    ]
    fact = _fact(result, "which_map")
    if fact and fact.get("sentence"):
        lines.append(f"- Verdict: {fact['sentence']}")
    n = result.get("n_labelled")
    path = _design_path(arguments, result)
    lines.append(_labels_line(n, _design_name(result), seed, path))
    if labels_changed:
        lines.append(LABELS_CHANGED)
    lines += _limits(result)
    for key in ("note", "verdict_note"):
        text = result.get(key)
        if isinstance(text, str) and text.strip():
            lines.append(_PACKAGE_NOTE.format(text))
    difference = result.get("difference")
    if isinstance(difference, Mapping):
        difference = {k: difference.get(k) for k in ("low", "high")}
    statement: dict[str, Any] = {
        "kind": "which_map",
        "tool": ESTIMATE_TOOL,
        "which_map": result.get("which_map"),
        "verdict_rule": result.get("verdict_rule"),
        "n_labelled": n,
        "difference": difference,
        "p_value": (result.get("sign_test") or {}).get("p_value"),
        "design": _design_name(result),
        "seed": seed,
        "design_path": path,
        "labels_changed": labels_changed,
    }
    return lines, statement


def accuracy_section(verification: Iterable[Any]) -> tuple[list[str], dict[str, Any]]:
    """The report's "How accurate is this map" section, from the request's calls.

    Parameters
    ----------
    verification
        The request's verification calls (``ProvenanceLog.verification``, or
        mappings with ``tool``, ``arguments``, ``ok`` and ``result``), in
        call order. Only calls that ran and computed are read.

    Returns
    -------
    tuple
        The section's Markdown lines, and ``accuracy``: ``measured`` (an
        error rate was measured: an estimate, or a zone call that tested
        labels of a random design, certified or not, which states the
        whole-map estimate from them, or a certificate call whose result
        carries that estimate, certificate written or not; a certificate
        result without it states no error rate and measures nothing),
        ``which_map`` (a which-map grading
        was), and ``statements``, each statement's kind,
        map, design, seed, design file, labels and figures, copied from the
        tool result, with whether its labels changed after an earlier result
        on its design file and, for a zone, what it changed after an earlier
        zone call (then not stated as certified).

    Examples
    --------
    >>> lines, accuracy = accuracy_section([])
    >>> accuracy["measured"], lines[2].startswith("No error rate of this map")
    (False, True)
    """
    calls = _computed(verification)
    plans = {
        str(r.get("design_path")): (a, r)
        for tool, a, r in calls
        if tool == PLAN_TOOL and r.get("design_path")
    }

    def seed_of(arguments: Mapping[str, Any], result: Mapping[str, Any]) -> Any:
        """The seed the result returns, else its plan's in this request."""
        if result.get("seed") is not None:
            return result["seed"]
        plan = plans.get(str(_design_path(arguments, result)))
        return plan[1].get("seed") if plan else None

    def map_of(arguments: Mapping[str, Any], result: Mapping[str, Any]) -> str:
        """The map a call graded: its plan's, or its own scores (no design)."""
        path = _design_path(arguments, result)
        if not path:
            return _map_of_arguments(arguments) or "not named"
        plan = plans.get(str(path))
        if plan is None:
            return MAP_NOT_IN_REQUEST
        return _map_of_arguments(plan[0]) or "not named by its plan"

    looks = _second_looks(calls)
    indexed = list(enumerate(calls))
    rates = [
        (i, a, r)
        for i, (tool, a, r) in indexed
        if tool == ESTIMATE_TOOL and "which_map" not in r and "n_labelled" in r
    ]
    graded = [
        (i, a, r)
        for i, (tool, a, r) in indexed
        if tool == ESTIMATE_TOOL and "which_map" in r
    ]
    zones = [(i, a, r) for i, (tool, a, r) in indexed if tool == CERTIFY_TOOL]
    written = [
        (i, a, r) for i, (tool, a, r) in indexed if tool == WRITE_CERTIFICATE_TOOL
    ]
    checked = [r for tool, _a, r in calls if tool == VERIFY_CERTIFICATE_TOOL]
    used = {str(_design_path(a, r)) for tool, a, r in calls if tool != PLAN_TOOL}
    used.discard("None")
    waiting = [
        (a, r)
        for tool, a, r in calls
        if tool == PLAN_TOOL and str(r.get("design_path")) not in used
    ]
    estimated = {str(_design_path(a, r)) for _i, a, r in rates}
    # A certificate call measured an error rate only when its result states
    # the whole-map estimate from its labels, as a zone's does.
    measured = (
        bool(rates)
        or any(r.get("n_labelled") is not None for _i, _a, r in zones)
        or any(_fact(r, "whole_map_estimate") for _i, _a, r in written)
    )

    def look(i: int) -> dict[str, Any]:
        """What call ``i`` changed after an earlier one (:func:`_second_looks`)."""
        return looks.get(i, {"labels_changed": False, "changed": []})

    body: list[str] = []
    statements: list[dict[str, Any]] = []
    for i, a, r in rates:
        lines, statement = _error_rate(
            a,
            r,
            seed_of(a, r),
            map_named=map_of(a, r),
            labels_changed=look(i)["labels_changed"],
        )
        body += [*lines, ""]
        statements.append(statement)
    for i, a, r in zones:
        path = str(_design_path(a, r))
        lines, statement = _zone(
            a,
            r,
            seed_of(a, r),
            map_named=map_of(a, r),
            estimated=path in estimated,
            changed=look(i)["changed"],
        )
        if _fact(r, "whole_map_estimate"):
            estimated.add(path)  # stated once per design file
        body += [*lines, ""]
        statements.append(statement)
    for i, a, r in written:
        path = str(_design_path(a, r))
        changed_labels = look(i)["labels_changed"]
        lines, statement = _certificate(
            a,
            r,
            seed_of(a, r),
            map_named=map_of(a, r),
            labels_changed=changed_labels,
            # other labels give another figure, stated beside LABELS_CHANGED
            estimated=path in estimated and not changed_labels,
        )
        if _fact(r, "whole_map_estimate"):
            estimated.add(path)
        body += [*lines, ""]
        statements.append(statement)
    for r in checked:
        lines, statement = _certificate_check(r)
        body += [*lines, ""]
        statements.append(statement)
    for i, a, r in graded:
        lines, statement = _which_map(
            a, r, seed_of(a, r), labels_changed=look(i)["labels_changed"]
        )
        body += [*lines, ""]
        statements.append(statement)
    if waiting:
        body += ["### Planned, awaiting labels (`olmoearth_plan_label_sample`)", ""]
    for _a, r in waiting:
        body.append(
            _labels_line(
                r.get("budget"),
                r.get("design"),
                r.get("seed"),
                r.get("design_path"),
                what="windows to label",
            )
        )
        statements.append(
            {
                "kind": "plan",
                "tool": PLAN_TOOL,
                "design": r.get("design"),
                "seed": r.get("seed"),
                "design_path": r.get("design_path"),
                "budget": r.get("budget"),
            }
        )
    if waiting:
        body.append("")

    lines = [ACCURACY_HEADING, ""]
    if measured:
        lines += [MEASURED_LEAD, "", *body]
    elif graded:
        lines += [MEASURED_LEAD, "", *body, NEITHER_MAP_MEASURED, ""]
    else:
        lines += [NOT_MEASURED, "", HOW_TO_MEASURE, "", *body]
    accuracy: dict[str, Any] = {
        "measured": measured,
        "which_map": bool(graded),
        "statements": statements,
    }
    return lines, accuracy


def _entry_field(entry: Any, key: str) -> Any:
    """A provenance entry's field: a ``ProvenanceManifest``'s or a dict's."""
    if isinstance(entry, Mapping):
        return entry.get(key)
    return getattr(entry, key, None)


# --------------------------------------------------------------------------- the report


def build_narrative(
    title: str,
    results: list[dict[str, Any]],
    *,
    provenance: Any = None,
    verification: Iterable[Any] | None = None,
    window_hours: float = 24.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a stakeholder Markdown report with a freshness gate and its accuracy.

    Parameters
    ----------
    title
        Report heading.
    results
        Each: ``result_id``, ``tile_urls``, ``property_names``,
        ``creation_time`` (ISO-8601), optional ``result_metadata``.
    provenance
        A ``ProvenanceLog`` (has ``.entries``) or a list of manifest
        dicts. Rendered as an audit section.
    verification
        The request's verification calls, recorded whole (a
        ``ProvenanceLog``'s ``.verification``, which is read when this is
        not given). The accuracy section states their results
        (:func:`accuracy_section`).
    window_hours
        Tiles older than this are gated (struck through, not rendered).

    Returns
    -------
    dict
        ``markdown`` (the report), ``result_count``, ``stale_count``,
        ``gated`` (bool), per-result ``freshness``, ``accuracy`` (what the
        accuracy section states, from :func:`accuracy_section`),
        ``provenance`` (each call and its arguments' hash, as the report
        lists them) and ``not_a_source``: ``["markdown"]``, since the
        report renders the title and results the caller typed, which no
        answer may take as a tool's figures
        (``analysis.output_contract.NOT_A_SOURCE``).

    Examples
    --------
    >>> out = build_narrative("Flood brief", [])
    >>> "## How accurate is this map" in out["markdown"], out["accuracy"]["measured"]
    (True, False)
    """
    freshness = []
    stale_ids: list[str] = []
    for result in results:
        check = freshness_check(result.get("creation_time", ""), window_hours, now)
        entry = {"result_id": result.get("result_id"), **check}
        freshness.append(entry)
        if check.get("fresh") is False:
            stale_ids.append(str(result.get("result_id")))

    lines = [f"# {title}", ""]
    if stale_ids:
        lines.append(
            f"> **Freshness gate:** {len(stale_ids)} of {len(results)} "
            f"result(s) are older than {window_hours}h and were not "
            f"rendered. Re-run the prediction for current tiles."
        )
        lines.append("")

    lines.append("## Results")
    fresh_by_id = {f["result_id"]: f for f in freshness}
    for result in results:
        rid = result.get("result_id")
        check = fresh_by_id[rid]
        if check.get("fresh") is False:
            lines.append(
                f"- ~~{rid}~~ - stale ({check['age_hours']}h old), not rendered"
            )
            continue
        props = ", ".join(result.get("property_names") or []) or "(none)"
        lines.append(f"- **{rid}** - properties: {props}")
        lines.extend(f"  - tile: `{url}`" for url in result.get("tile_urls") or [])
    lines.append("")

    entries = list(getattr(provenance, "entries", provenance) or [])
    if verification is None:
        verification = getattr(provenance, "verification", None) or []
    section, accuracy = accuracy_section(verification)
    lines += section

    calls: list[dict[str, Any]] = []
    for entry in entries:
        call = _entry_field(entry, "api_call")
        req_hash = _entry_field(entry, "request_hash")
        calls.append({"api_call": call, "args_sha": str(req_hash)[:12]})
    if calls:
        lines.append("## Provenance")
        lines.extend(f"- `{c['api_call']}` (args sha {c['args_sha']})" for c in calls)
        lines.append("")

    return {
        "markdown": "\n".join(lines).rstrip() + "\n",
        "result_count": len(results),
        "stale_count": len(stale_ids),
        "gated": bool(stale_ids),
        "freshness": freshness,
        "accuracy": accuracy,
        "provenance": calls,
        "not_a_source": ["markdown"],
    }
