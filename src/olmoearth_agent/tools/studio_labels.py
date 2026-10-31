# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The Studio annotations a model's training selector selects, and what they are for.

``olmoearth_studio_label_inventory`` counts the annotations a Studio model's
training selector selects now, or a project's. exp86 round 9 (B3/studio run
1) answered "no ground-truth labels exist" while the run's model records
named a label field of the user's project; the tools could say only that a
label field existed (``statistical_rules.labels_in_studio_fact``).

The model record (``GET /models/{id}``, read live on 2026-10-31) carries
``data_selectors``, a list of objects shaped as the spec's
``AnnotationSearchRequest``: the kelp model's holds its training project's
``project_id`` as ``{"eq": id, "inc": null}``, every other filter null,
``sort_by`` ``creation_time``, ``sort_direction`` ``asc`` and ``limit``
null. Its ``wizard_answers`` name the label field (``label_field_id``), the
split field (``split_field_id``, the annotation metadata field named
``oe_split``) and the split proportions (train 0.75, val 0.25, test 0, while
557 of the project's 1056 annotations are tagged test). Two live fine-tuned
models (KarstBinary, KarstNumber) name no split field, and the embeddings
model (KarstEmbedding) has no selector. Each selector's filters, of at most
:data:`_MAX_SELECTORS` selectors, are sent to ``POST /annotations/search`` as
they stand, and the annotations are counted by status, geometry type, split,
class, split by class and creation time, beside the model record's own
times; an annotation two selectors select counts once.

Assumed, not verified live: that Studio trained on what the selectors select
(they are read now: an annotation deleted since is not seen, and one updated
since counts as it is now; the counts created or updated after the model's
creation say how many are newer); that several selectors are a union; that
training honoured the split field rather than the proportions (the record
names both, nothing says which); that a filtered search's ``meta.total``
counts the filtered annotations; and the paging (``sort_by``,
``sort_direction``, ``offset`` up to 10000 and a ``creation_time`` floor past
it are the spec's alone). The read is called complete only when each
selector returned as many distinct annotations as its ``meta.total``: offset
pages over equal creation times keep no fixed order (the kelp project's 1056
were created within 2 ms, probe3.out), so a page may repeat some and skip
others.

What the counts are for, the result's ``must_state``: Studio records no
random design for its annotations, so they give no design-based error rate
(``olmoearth_plan_label_sample`` draws one); if Studio trained on what the
selectors select (not confirmed, above), those of split train or val created
and updated before the model's creation fit or tuned it, which is stated
with that condition; those of split test were held out only if training
honoured the split field. A selector not read (one with no filter, or past
:data:`_MAX_SELECTORS`) may select annotations the counts miss, so later
annotations are then not ruled out. No id,
geometry, coordinate, free-text value or person's id is returned, nor
Studio's error text for a search, which may quote the selector: annotations
are counted, never listed, and an annotation's id is a key of the count's
deduplication only.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.studio.client import StudioAccessError
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.registry import Capability, RegisteredTool, ToolContext
from olmoearth_agent.types import ApiEnvelope

#: Annotations asked for per search page.
_PAGE = 1000
#: The most annotations one call counts, over all its selectors; past it the
#: counts cover the annotations read, and the result says so.
_CAP = 20_000
#: The most search pages one call sends, over all its selectors. Pages that
#: hold only annotations counted already (two selectors that overlap, the
#: floor's repeats, a server that ignores the order) do not move the count
#: toward :data:`_CAP`, so this bounds the requests whatever Studio returns.
_MAX_PAGES = 50
#: The most data selectors one call replays; the rest are named as left out.
_MAX_SELECTORS = 10
#: The most keys a selector names as not replayed; the rest are counted.
_MAX_KEYS = 10
#: The splits the split-by-class table keeps apart; the rest are summed.
_MAX_TABLE_SPLITS = 10
#: The classes the split-by-class table keeps apart; the rest are summed.
_MAX_TABLE_CLASSES = 20
#: The largest ``offset`` ``AnnotationSearchRequest`` accepts (the spec of
#: 2026-10-31; ``StudioClient`` refuses a larger one). Past it the paging goes
#: on from the last creation time read (:func:`_read_selector`).
_OFFSET_MAX = 10_000
#: The filters of the spec's ``AnnotationSearchRequest`` (2026-10-31): a
#: selector's filters with a value are replayed as they stand.
_FILTER_KEYS = frozenset(
    {
        "id",
        "task_id",
        "annotator_id",
        "status",
        "project_id",
        "dataset_id",
        "image_id",
        "metadata_filters",
        "has_feedback",
        "has_images",
        "has_labels",
        "creation_time",
        "updated_time",
        "start_time",
        "end_time",
        "intersects_geometry",
        "intersects_area_id",
        "metadata_value",
    }
)
#: A selector's order, never replayed: it does not change which annotations a
#: selector without a ``limit`` or ``offset`` selects.
_ORDER_KEYS = frozenset({"sort_by", "sort_direction"})
#: The order every page is read in: an annotation created while the tool pages
#: comes after the pages read, so no page shifts under the offset.
_ORDER = {"sort_by": "creation_time", "sort_direction": "asc"}
#: The tool's three inputs, of which exactly one is given.
_INPUTS = ("prediction_id", "model_id", "project_id")
#: A Studio id. The spec types them as UUIDs; letters, digits and hyphens keep
#: ``GET /predictions/{id}`` and ``GET /models/{id}`` to the one record, for
#: an id the model typed and one read from a prediction record alike.
_ID = re.compile(r"[0-9A-Za-z-]{1,64}")
#: The statuses of the spec's ``AnnotationStatus``; any other is unrecorded.
_STATUSES = frozenset({"pending", "approved", "rejected"})
#: The split names a text value may hold and still be counted by name.
_SPLIT_NAMES = frozenset({"train", "val", "test"})
#: The proportions of ``wizard_answers.split_proportions`` the result states.
_SPLIT_PROPORTIONS = ("train_prop", "val_prop", "test_prop")
#: The WKT geometry types a count names; any other start is unrecorded.
_WKT_TYPES = frozenset(
    {
        "POINT",
        "LINESTRING",
        "POLYGON",
        "MULTIPOINT",
        "MULTILINESTRING",
        "MULTIPOLYGON",
        "GEOMETRYCOLLECTION",
    }
)
#: The most values one count keeps; the rest are summed under :data:`OTHER`.
_MAX_VALUES = 50
#: The most labelset fields counted for a project with no model record.
_MAX_FIELDS = 10
#: The most characters of a class, split, field or key name the result keeps.
_MAX_NAME = 60
#: The most characters of a model's name the result keeps.
_MAX_MODEL_NAME = 120

#: How a selector's read ended when it returned as many distinct annotations
#: as Studio's ``meta.total`` for it (:meth:`_Selector.end`).
_COMPLETE = "complete"
#: The ways a read falls short (:func:`_read_note`), in the order the summary
#: names the first that holds: the call's cap, its page bound, a floor that
#: would not move, pages that ended short of the total, no total.
_SHORT = ("cap", "pages", "stuck", "gap", "no_total")

#: The key of an annotation whose split, class or field value is not recorded.
UNRECORDED = "unrecorded"
#: The key of a text value: free text is never returned.
FREE_TEXT = "free text (not shown)"
#: The key under which the values past :data:`_MAX_VALUES` are summed.
OTHER = "other values"

#: Every result: nothing records Studio's annotations as a probability sample.
NOT_A_SAMPLE = (
    "Studio records no random design over the map for its annotations, so they "
    "give no design-based error rate for the map."
)
#: A replayed selector whose model record names a split field, every
#: annotation read created and last updated before the model's creation.
#: That training used what the selector selects is assumed (the module
#: docstring), so the sentence says so.
TRAIN_VAL_USED = (
    "If training used the record's selector, its train and val annotations, "
    "none created or updated after the model's creation, fit or tuned it."
)
#: The same, when some were created or updated after it, or that is unknown
#: (a read not complete, a selector not read).
TRAIN_VAL_BEFORE = (
    "Assuming training used the record's selector, train and val annotations "
    "created and updated before the model's creation fit or tuned it; later "
    "ones may not."
)
#: A split field named, and proportions with a test share above 0 or none.
TEST_HELD_OUT_IF = (
    "Test-split annotations were held out only if training honoured the split "
    "field the model record names, which the record cannot confirm."
)
#: A split field named beside proportions whose test share is 0 (the kelp
#: model: 557 annotations tagged test, test_prop 0.0).
TEST_SHARE_ZERO = (
    "Test-split annotations were held out only if training followed the split "
    "field rather than the proportions, whose test share is zero; nothing "
    "records which."
)
#: No split field named, and proportions with a test share above 0 or none.
NO_SPLIT_FIELD = (
    "The model record names no split field, so which of these annotations fit, "
    "tuned or were held out from the model is unknown."
)
#: No split field named, and a test share of 0 (KarstBinary, KarstNumber).
NO_TEST_SHARE = (
    "The model record names no split field and a test share of zero, so no "
    "annotation is recorded as held out from the model."
)
#: No split field named, and some annotations newer than the model, or unknown.
LATER_UNKNOWN = (
    "Annotations the selector selects that were created or updated after the "
    "model's creation may not have trained it, or not as now recorded."
)
#: A project's annotations counted with no model: no training statement.
TRAINING_UNKNOWN = (
    "Which of these annotations any model was trained on is unknown: no model "
    "record was read."
)
#: No selector replayed: no model, no record read, or none replayable.
SELECTOR_UNKNOWN = (
    "The model's training selector is unknown here, so whether and which Studio "
    "annotations trained the model is not known from this run."
)
#: A model record with no selector (the embeddings model has none).
NO_SELECTOR = (
    "The model record names no training selector, so this run cannot say "
    "whether any Studio annotations trained the model."
)
#: The route to an error rate, in every result.
NEXT_STEP = (
    "For the map's error rate: olmoearth_plan_label_sample draws windows of the "
    "map by a design, and olmoearth_estimate_map_error grades the reviewer's "
    "labels on them."
)
#: When no selector was replayed.
PROJECT_STEP = (
    "Given a project, olmoearth_studio_label_inventory with project_id counts "
    "its annotations; whether any of them trained the model stays unknown."
)
#: What the counts and their times can say.
TIMES_NOTE = (
    "The counts are of what the selectors select now: an annotation deleted "
    "since training is not seen, and one updated since counts as it is now. "
    "The model record does not say when training read its annotations or "
    "ended, so one created or updated after the model was created may not "
    "have been trained on."
)
#: What by_class leaves out.
CLASSES_NOTE = (
    "by_class lists the classes the annotations read carry; a class of the "
    "label field that no annotation carries is not listed (the labelset is "
    "not read)."
)
#: A model record that names a split field and split proportions both.
SPLIT_NOTE = (
    "The model record names a split field and split proportions; which of them "
    "training followed is not recorded."
)
#: A project counted with no model record naming its class and split fields.
FIELDS_NOTE = (
    "No model record names the class or split field, so each labelset field is "
    "counted under its own name, over the annotations that carry it."
)
#: The tail of every read note that falls short.
_READ_ONLY = "the counts cover the annotations read only."


def _annotation_rate_claim() -> dict[str, str]:
    """An accuracy or error rate from Studio annotations, stated as the map's.

    No new id: the contract's ids are fixed (``statistical_rules.FIXED_IDS``),
    and ``accuracy_not_measured`` is the one the answer checks read for an
    accuracy figure (``harness/checks.py``, ``_accuracy_not_measured``). Its
    detector is narrower than this ``why``: it reads only a figure
    ``states_accuracy_figure`` finds (an error rate, an accuracy or a share
    wrong, "94% accurate"; not an F1 or an IoU), and only in a run that
    measured none. Once the run holds a ``whole_map_estimate`` fact or a
    successful ``olmoearth_classification_metrics`` call (on labels the
    caller gives, which may be these annotations exported), it is not read,
    and a figure from annotations meets only the number check. In a run that
    measured none, every such figure is flagged, from annotations or not.
    The checks keep one ``why`` per id, the first tool's, so this one also
    holds where the stakeholder report emits the id.
    """
    return rules.forbidden(
        rules.ACCURACY_NOT_MEASURED,
        "no error rate, accuracy or share wrong may be stated for the map unless "
        "olmoearth_estimate_map_error graded labels of a planned sample of it: "
        "Studio records no random design for its annotations, those a model was "
        "fit or tuned on are not independent of it, and a model's training or "
        "validation metrics are no measure of the map",
    )


def _when(value: Any) -> datetime | None:
    """An ISO time as an aware datetime (UTC when it names no zone), or ``None``."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        when = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc)


def _text(value: Any, limit: int = _MAX_NAME) -> str | None:
    """``value`` stripped and cut to ``limit`` characters when it is text."""
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()[:limit]


def _number(value: Any) -> int | float | None:
    """``value`` when it is a number (a bool is not), else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _geometry_type(wkt: Any) -> str:
    """The geometry type a WKT string starts with (``MULTIPOLYGON``), never more."""
    if not isinstance(wkt, str):
        return UNRECORDED
    text = wkt.strip()
    if text[:5].upper() == "SRID=":
        text = text.split(";", 1)[-1]
    words = text.split("(", 1)[0].split()
    head = words[0].upper() if words else ""
    return head if head in _WKT_TYPES else UNRECORDED


def _entry_value(entry: dict[str, Any], *, split: bool = False) -> str:
    """What one metadata entry records, as a count's key.

    A labelset field's ``label_name``; a number's or a bool's value as text;
    a text value only as a split named train, val or test (``split``), since
    free text is never returned (:data:`FREE_TEXT`); :data:`UNRECORDED` when
    the entry holds nothing.
    """
    name = entry.get("label_name")
    if isinstance(name, str) and name.strip():
        return name.strip()[:_MAX_NAME]
    value = entry.get("value")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, str) and value.strip():
        word = value.strip().lower()
        return word if split and word in _SPLIT_NAMES else FREE_TEXT
    if isinstance(value, list) and value:
        return FREE_TEXT
    return UNRECORDED


def _field_value(
    entries: list[dict[str, Any]], field_id: str | None, *, split: bool = False
) -> str:
    """What an annotation records for the field ``field_id`` (:func:`_entry_value`)."""
    if not field_id:
        return UNRECORDED
    for entry in entries:
        if entry.get("metadata_field_id") == field_id:
            return _entry_value(entry, split=split)
    return UNRECORDED


def _top(counts: Counter[str]) -> dict[str, int]:
    """``counts`` largest first: at most :data:`_MAX_VALUES` keys, the rest summed."""
    ranked = counts.most_common()
    out = dict(ranked[:_MAX_VALUES])
    rest = sum(n for _key, n in ranked[_MAX_VALUES:])
    if rest:
        out[OTHER] = out.get(OTHER, 0) + rest
    return out


@dataclass
class _Tally:
    """What the annotations read hold, counted: each annotation once.

    With a split or class field named (``split_field``, ``label_field``, from
    the model record), the annotations are counted by split and class; with
    neither, each labelset field under its own name. ``seen`` holds the ids
    counted, only to count each once; no id leaves this object. ``pages``
    counts the search pages the call sent (:data:`_MAX_PAGES`).
    """

    split_field: str | None = None
    label_field: str | None = None
    model_created: datetime | None = None
    model_updated: datetime | None = None
    read: int = 0
    pages: int = 0
    seen: set[str] = field(default_factory=set)
    status: Counter[str] = field(default_factory=Counter)
    geometry: Counter[str] = field(default_factory=Counter)
    split: Counter[str] = field(default_factory=Counter)
    label: Counter[str] = field(default_factory=Counter)
    split_label: Counter[tuple[str, str]] = field(default_factory=Counter)
    fields: dict[str, Counter[str]] = field(default_factory=dict)
    first: tuple[datetime, str] | None = None
    last: tuple[datetime, str] | None = None
    after_created: int = 0
    after_updated: int = 0
    updated_after_created: int = 0
    untimed: int = 0

    @property
    def by_field(self) -> bool:
        """Whether the model record names the split or the class field."""
        return bool(self.split_field or self.label_field)

    @property
    def newer(self) -> bool:
        """Whether an annotation read may postdate the model's creation.

        One was created or last updated after it, one has no readable
        creation time, or the record gives no creation time to compare with.
        """
        return (
            self.model_created is None
            or self.untimed > 0
            or self.after_created > 0
            or self.updated_after_created > 0
        )

    def is_new(self, record: dict[str, Any]) -> bool:
        """Whether ``record`` is not counted yet (one with no id never is)."""
        key = record.get("id")
        return key is None or str(key) not in self.seen

    def add(self, record: dict[str, Any]) -> None:
        """Count ``record``, unless an annotation of its id was counted already."""
        key = record.get("id")
        if key is not None:
            if str(key) in self.seen:
                return
            self.seen.add(str(key))
        self.read += 1
        status = record.get("status")
        named = status if isinstance(status, str) and status in _STATUSES else None
        self.status[named or UNRECORDED] += 1
        self.geometry[_geometry_type(record.get("geom_wkt"))] += 1
        values = record.get("metadata_values")
        entries = (
            [v for v in values if isinstance(v, dict)]
            if isinstance(values, list)
            else []
        )
        if self.by_field:
            split = _field_value(entries, self.split_field, split=True)
            label = _field_value(entries, self.label_field)
            self.split[split] += 1
            self.label[label] += 1
            self.split_label[split, label] += 1
        else:
            for entry in entries:
                self._count_labelset(entry)
        self._count_time(record.get("creation_time"), record.get("updated_time"))

    def _count_labelset(self, entry: dict[str, Any]) -> None:
        """Count a labelset entry's label under its field's name."""
        name = entry.get("name")
        if entry.get("data_type") != "labelset" or not isinstance(name, str):
            return
        key = name.strip()[:_MAX_NAME]
        if not key or (key not in self.fields and len(self.fields) >= _MAX_FIELDS):
            return
        self.fields.setdefault(key, Counter())[_entry_value(entry)] += 1

    def _count_time(self, created: Any, updated: Any) -> None:
        """Keep the first and last creation times; count those after the model's.

        An annotation last updated after the model's creation is counted too
        (one created after it among them): its split or class may have
        changed since training read it.
        """
        when = _when(created)
        if when is None:
            self.untimed += 1
        else:
            text = str(created).strip()
            if self.first is None or when < self.first[0]:
                self.first = (when, text)
            if self.last is None or when > self.last[0]:
                self.last = (when, text)
            if self.model_created is not None and when > self.model_created:
                self.after_created += 1
            if self.model_updated is not None and when > self.model_updated:
                self.after_updated += 1
        edited, model = _when(updated), self.model_created
        if model is not None and edited is not None and edited > model:
            self.updated_after_created += 1

    def _table(self) -> dict[str, dict[str, int]]:
        """Split by class, the most counted kept apart and the rest summed."""
        splits = {key for key, _n in self.split.most_common(_MAX_TABLE_SPLITS)}
        labels = {key for key, _n in self.label.most_common(_MAX_TABLE_CLASSES)}
        out: dict[str, dict[str, int]] = {}
        for (split, label), n in self.split_label.items():
            cells = out.setdefault(split if split in splits else OTHER, {})
            column = label if label in labels else OTHER
            cells[column] = cells.get(column, 0) + n
        return out

    def counts(self) -> dict[str, Any]:
        """The counts as the result states them: no id, geometry or free text."""
        out: dict[str, Any] = {
            "by_status": _top(self.status),
            "by_geometry_type": _top(self.geometry),
        }
        if self.by_field:
            out["by_split"] = _top(self.split)
            out["by_class"] = _top(self.label)
            out["split_by_class"] = self._table()
        else:
            out["by_labelset_field"] = {k: _top(c) for k, c in self.fields.items()}
        out["creation_time"] = (
            {"first": self.first[1], "last": self.last[1]}
            if self.first is not None and self.last is not None
            else None
        )
        return out


@dataclass
class _Selector:
    """One selector's read: Studio's total, what came back and how it ended.

    ``ids`` holds the ids this selector returned, counted by another
    selector or not, only to compare with ``total``; no id leaves it.
    """

    total: int | None = None
    pages: int = 0
    ids: set[str] = field(default_factory=set)
    unkeyed: int = 0
    ended: str = _COMPLETE

    @property
    def returned(self) -> int:
        """The distinct annotations Studio returned for this selector."""
        return len(self.ids) + self.unkeyed

    def note(self, record: dict[str, Any]) -> None:
        """Note that Studio returned ``record`` for this selector."""
        key = record.get("id")
        if key is None:
            self.unkeyed += 1
        else:
            self.ids.add(str(key))

    def end(self, short: str) -> _Selector:
        """This read, ended: complete if it returned Studio's total, else ``short``."""
        whole = self.total is not None and self.returned >= self.total
        self.ended = _COMPLETE if whole else short
        return self


def _without_nulls(value: Any) -> Any:
    """``value`` less every ``None`` member of its dicts, at any depth."""
    if isinstance(value, dict):
        kept = {key: _without_nulls(item) for key, item in value.items()}
        return {key: item for key, item in kept.items() if item is not None}
    if isinstance(value, list):
        return [_without_nulls(item) for item in value]
    return value


def _replay(selector: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """A data selector as an annotation search's filters, and the keys left out.

    Only the spec's ``AnnotationSearchRequest`` filters (:data:`_FILTER_KEYS`)
    that hold a value are sent, each less its null members (the kelp model's
    ``project_id`` ``{"eq": id, "inc": null}`` is sent as ``{"eq": id}``); a
    filter emptied by that is no filter. The order keys and an ``offset`` of
    0 are left out without a word (:data:`_ORDER_KEYS`); a ``limit``, another
    ``offset`` or any other key that holds a value is named, at most
    :data:`_MAX_KEYS` of them, since it may narrow what training selected,
    and the counts then hold more.
    """
    filters: dict[str, Any] = {}
    left_out: list[str] = []
    for key, value in selector.items():
        kept = _without_nulls(value)
        empty = kept is None or (isinstance(kept, (dict, list)) and not kept)
        start = key == "offset" and kept == 0 and not isinstance(kept, bool)
        if empty or start or key in _ORDER_KEYS:
            continue
        if key in _FILTER_KEYS:
            filters[key] = kept
        else:
            left_out.append(str(key)[:_MAX_NAME])
    named = sorted(left_out)
    if len(named) > _MAX_KEYS:
        named = [*named[:_MAX_KEYS], f"and {len(named) - _MAX_KEYS} more"]
    return filters, named


async def _search(
    ctx: ToolContext, body: dict[str, Any], *, limit: int, offset: int
) -> ApiEnvelope[dict[str, Any]]:
    """One annotation search; a failure keeps its type and HTTP status only.

    The body holds what the model record's selector holds, which may be a
    person's id (``annotator_id``) or coordinates (``intersects_geometry``),
    and Studio's error text may quote it (``StudioClient._unwrap`` raises
    with its ``errors`` as written). A refusal is raised again with Studio's
    code alone, so the registry keeps its hint; anything else as its type and
    status. Either is raised out of the ``except`` block, so its
    ``__context__`` holds no request.
    """
    failure: Exception
    try:
        return await ctx.studio.search_annotations(body, limit=limit, offset=offset)
    except StudioAccessError as exc:
        failure = StudioAccessError(exc.method, exc.path, exc.status_code, exc.code)
    except Exception as exc:  # its text may quote the request: type and status only
        status = getattr(getattr(exc, "response", None), "status_code", None)
        detail = type(exc).__name__
        if isinstance(status, int):
            detail += f", HTTP {status}"
        failure = RuntimeError(f"Studio's annotation search failed ({detail})")
    raise failure


def _bound(tally: _Tally) -> str | None:
    """The call's bound the read has reached ("cap" or "pages"), or ``None``."""
    if tally.read >= _CAP:
        return "cap"
    if tally.pages >= _MAX_PAGES:
        return "pages"
    return None


async def _read_selector(
    ctx: ToolContext, filters: dict[str, Any], tally: _Tally
) -> _Selector:
    """Page one selector's annotations into ``tally``.

    Pages of :data:`_PAGE` in creation-time order, by ``offset`` up to
    :data:`_OFFSET_MAX`; past it from a ``creation_time`` floor (``gte``) at
    the last creation time read, offset 0 again, so the annotations of that
    time come twice and count once. A floor must move later, so the read ends
    whatever order the server keeps. Stops at a short page, at :data:`_CAP`
    annotations counted in the call, at :data:`_MAX_PAGES` pages sent in it,
    or at a floor that would not move; a selector reached after the cap or
    the page bound has its total read from a page of one. The read is
    complete only when the selector returned as many distinct annotations as
    Studio's ``meta.total`` for it (:meth:`_Selector.end`).
    """
    got = _Selector()
    floor: tuple[datetime, str] | None = None
    offset = 0
    while True:
        bound = _bound(tally)
        if bound is not None:
            if got.pages == 0:
                env = await _search(ctx, {**filters, **_ORDER}, limit=1, offset=0)
                got.total = env.total
            return got.end(bound)
        body: dict[str, Any] = {**filters, **_ORDER}
        if floor is not None:
            own = filters.get("creation_time")
            body["creation_time"] = {
                **(own if isinstance(own, dict) else {}),
                "gte": floor[1],
            }
        env = await _search(ctx, body, limit=_PAGE, offset=offset)
        tally.pages += 1
        if got.pages == 0:
            got.total = env.total
        got.pages += 1
        for record in env.records:
            if not isinstance(record, dict):
                continue
            if tally.read >= _CAP and tally.is_new(record):
                got.ended = "cap"
                return got
            got.note(record)
            tally.add(record)
        if len(env.records) < _PAGE:
            return got.end("gap" if got.total is not None else "no_total")
        if offset + _PAGE <= _OFFSET_MAX:
            offset += _PAGE
            continue
        last = env.records[-1]
        at = last.get("creation_time") if isinstance(last, dict) else None
        when = _when(at)
        if when is None or (floor is not None and when <= floor[0]):
            return got.end("stuck")
        floor, offset = (when, str(at).strip()), 0


def _read_note(read: int, total: int | None, ending: str, *, several: bool) -> str:
    """How much of what the selectors select was read and counted, and why."""
    of = ""
    if total is not None:
        of = f" of the {total} that Studio counts (meta.total)"
    if ending == _COMPLETE:
        why = "every annotation selected was read, each counted once."
    elif ending == "cap":
        why = f"the read stopped at this tool's cap of {_CAP} annotations; "
        why += _READ_ONLY
    elif ending == "pages":
        why = f"the read stopped at this tool's {_MAX_PAGES} search pages; "
        why += _READ_ONLY
    elif ending == "stuck":
        why = (
            "more annotations share one creation time than Studio's search "
            "pages past; " + _READ_ONLY
        )
    elif ending == "gap":
        why = (
            "Studio's pages ended before as many distinct annotations as it "
            "counts came back (offset pages over equal creation times may "
            "repeat some and skip others); " + _READ_ONLY
        )
    else:
        why = (
            "Studio gave no total to check the read against, so whether every "
            "annotation selected was read is unknown."
        )
    note = f"Read {read}{of}: {why}"
    if several:
        note += (
            " Studio's totals are per selector, summed; an annotation two "
            "selectors select is counted once here."
        )
    return note


async def _read(
    ctx: ToolContext, selectors: list[dict[str, Any]], tally: _Tally
) -> tuple[dict[str, Any], list[_Selector]]:
    """Read every selector into ``tally``: the summary, and each selector's read."""
    reads = [await _read_selector(ctx, filters, tally) for filters in selectors]
    known = [r.total for r in reads if r.total is not None]
    total = sum(known) if reads and len(known) == len(reads) else None
    short = [r.ended for r in reads if r.ended != _COMPLETE]
    ending = min(short, key=_SHORT.index) if short else _COMPLETE
    several = len(selectors) > 1
    summary: dict[str, Any] = {
        "annotations_read": tally.read,
        "studio_total": total,
        "complete": not short,
        "read_note": _read_note(tally.read, total, ending, several=several),
    }
    return summary, reads


def _contract(
    out: dict[str, Any], must_state: list[str], *, steps: tuple[str, ...] = ()
) -> dict[str, Any]:
    """``out`` with its next steps, its ``must_state`` and the forbidden claim."""
    out["next_steps"] = [NEXT_STEP, *steps]
    return rules.add_contract(
        out, must_state=must_state, forbidden_claims=[_annotation_rate_claim()]
    )


def _unknown(
    out: dict[str, Any],
    reason: str,
    *,
    label: str = "unknown",
    sentence: str = SELECTOR_UNKNOWN,
) -> dict[str, Any]:
    """A result with no selector replayed, and why."""
    out["training_selector"] = label
    out["reason"] = reason
    return _contract(out, [sentence, NOT_A_SAMPLE], steps=(PROJECT_STEP,))


async def _model_record(
    ctx: ToolContext, model_id: str
) -> tuple[dict[str, Any] | None, str]:
    """The model's record, or ``None`` and why it is unknown.

    Any failure leaves the training selector unknown, never empty: a refusal
    of the key (:class:`StudioAccessError`, as ``sampling.model_summary``
    takes it too), a server without a ``/models`` route
    (``StudioClient.models_unavailable``) or anything else, stated by its
    HTTP status or type only.
    """
    try:
        record = await ctx.studio.get_model(model_id)
    except StudioAccessError as exc:
        why = f"Studio refused the model record for this key (HTTP {exc.status_code})"
        return None, why
    except Exception as exc:  # the selector is then unknown, not absent
        if getattr(ctx.studio, "models_unavailable", False):
            return None, (
                "Studio's model records cannot be read here: an earlier /models "
                "request was answered as for a route this server does not serve"
            )
        return None, f"the model record could not be read ({type(exc).__name__})"
    if not isinstance(record, dict) or not record:
        return None, "Studio returned no model record"
    return record, ""


def _model_summary(record: dict[str, Any], wizard: dict[str, Any]) -> dict[str, Any]:
    """The record's fields, bounded; never its requester_id or checkpoint_token."""
    split = wizard.get("split_proportions")
    return {
        "name": _text(record.get("name"), _MAX_MODEL_NAME),
        "model_type": _text(record.get("model_type")),
        "prediction_type": _text(wizard.get("prediction_type")),
        "status": _text(record.get("status")),
        "creation_time": _text(record.get("creation_time")),
        "updated_time": _text(record.get("updated_time")),
        "patch_size": _number(wizard.get("patch_size")),
        "training_project_id": _text(wizard.get("project_id")),
        "label_field_named": bool(wizard.get("label_field_id")),
        "split_field_named": bool(wizard.get("split_field_id")),
        "split_proportions": (
            {k: _number(split.get(k)) for k in _SPLIT_PROPORTIONS if k in split}
            if isinstance(split, dict)
            else None
        ),
    }


def _field_id(value: Any) -> str | None:
    """A field id of the wizard's answers as text, or ``None``."""
    text = str(value).strip() if value else ""
    return text or None


async def _project_inventory(ctx: ToolContext, project_id: str) -> dict[str, Any]:
    """A project's annotations counted, with no training statement but "unknown"."""
    tally = _Tally()
    summary, _reads = await _read(ctx, [{"project_id": {"eq": project_id}}], tally)
    out: dict[str, Any] = {
        "source": "project",
        "project_id": project_id,
        "training_selector": "unknown",
        **summary,
        **tally.counts(),
        "fields_note": FIELDS_NOTE,
    }
    return _contract(out, [NOT_A_SAMPLE, TRAINING_UNKNOWN])


def _model_must_state(
    split_field: str | None, wizard: dict[str, Any], *, newer: bool
) -> list[str]:
    """What a replayed selector's annotations were and were not used for.

    By the record, if Studio trained on what its selector selects (not
    confirmed): train and val fit or tuned the model, unless created or
    updated since its creation, or not all read (``newer``); test was held
    out only if training honoured the split field, and with a test share of
    0 the proportions hold none out. With no split field, a test share of 0 means
    no annotation is recorded as held out.
    """
    split = wizard.get("split_proportions")
    shares = split if isinstance(split, dict) else {}
    zero = _number(shares.get("test_prop")) == 0
    if split_field:
        return [
            NOT_A_SAMPLE,
            TRAIN_VAL_BEFORE if newer else TRAIN_VAL_USED,
            TEST_SHARE_ZERO if zero else TEST_HELD_OUT_IF,
        ]
    must = [NOT_A_SAMPLE, NO_TEST_SHARE if zero else NO_SPLIT_FIELD]
    return [*must, LATER_UNKNOWN] if newer else must


async def _model_inventory(ctx: ToolContext, source: str, ident: str) -> dict[str, Any]:
    """The training selector of a model (or a prediction's), replayed and counted."""
    out: dict[str, Any] = {"source": source.removesuffix("_id")}
    model_id = ident
    if source == "prediction_id":
        prediction = await ctx.studio.get_prediction(ident)
        out["prediction_id"] = ident
        model_id = str(prediction.get("model_id") or "").strip()
        if model_id and not _ID.fullmatch(model_id):
            out["model_id"] = None
            return _unknown(out, "the prediction record's model id is not a Studio id")
    out["model_id"] = model_id or None
    if not model_id:
        return _unknown(out, "the prediction record names no model")
    record, why = await _model_record(ctx, model_id)
    if record is None:
        return _unknown(out, why)
    answers = record.get("wizard_answers")
    wizard: dict[str, Any] = answers if isinstance(answers, dict) else {}
    out["model"] = _model_summary(record, wizard)
    raw = record.get("data_selectors")
    selectors = [s for s in raw if isinstance(s, dict)] if isinstance(raw, list) else []
    if not selectors:
        return _unknown(
            out,
            "the model record names no training selector",
            label="none recorded",
            sentence=NO_SELECTOR,
        )
    replays = [_replay(s) for s in selectors[:_MAX_SELECTORS]]
    filters = [f for f, _keys in replays if f]
    if not filters:
        reason = "no selector holds a filter of Studio's annotation search"
        return _unknown(out, reason, label="not replayed")
    split_field = _field_id(wizard.get("split_field_id"))
    tally = _Tally(
        split_field=split_field,
        label_field=_field_id(wizard.get("label_field_id")),
        model_created=_when(record.get("creation_time")),
        model_updated=_when(record.get("updated_time")),
    )
    summary, reads = await _read(ctx, filters, tally)
    each = iter(reads)  # one read per selector that holds a filter, in order
    out["training_selector"] = "replayed"
    out["selectors"] = [
        {
            "filters": sorted(f),
            "not_replayed": keys,
            "studio_total": next(each).total if f else None,
        }
        for f, keys in replays
    ]
    if len(selectors) > _MAX_SELECTORS:
        out["selectors_left_out"] = len(selectors) - _MAX_SELECTORS
    out.update(summary)
    out.update(tally.counts())
    created = tally.model_created
    out["created_after_model_creation"] = (
        tally.after_created if created is not None else None
    )
    out["updated_after_model_creation"] = (
        tally.updated_after_created if created is not None else None
    )
    out["created_after_model_update"] = (
        tally.after_updated if tally.model_updated is not None else None
    )
    out["times_note"] = TIMES_NOTE
    out["classes_note"] = CLASSES_NOTE
    notes: list[str] = []
    if any(keys for _f, keys in replays):
        notes.append(
            "A selector key was not replayed (named under not_replayed): it is no "
            "filter of Studio's annotation search, or a limit or offset, so the "
            "counts may hold annotations the selector left out."
        )
    if len(filters) < len(replays):
        notes.append(
            "A selector with no filter was not replayed: it would select every "
            "annotation the key can see."
        )
    if len(selectors) > _MAX_SELECTORS:
        notes.append(
            f"Only the first {_MAX_SELECTORS} of the record's {len(selectors)} "
            "selectors were replayed; the counts may miss annotations the others "
            "select."
        )
    if split_field and isinstance(wizard.get("split_proportions"), dict):
        notes.append(SPLIT_NOTE)
    if notes:
        out["selector_notes"] = notes
    # A selector not read (no filter, or past the cap) may select annotations
    # the counts miss: later ones are then not ruled out. A limit or offset
    # not replayed only widens the set counted.
    unread = len(selectors) > _MAX_SELECTORS or len(filters) < len(replays)
    newer = tally.newer or not summary["complete"] or unread
    return _contract(out, _model_must_state(split_field, wizard, newer=newer))


async def _label_inventory(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_studio_label_inventory``.

    Exactly one of ``prediction_id``, ``model_id`` or ``project_id``, a
    Studio id (:data:`_ID`): a prediction gives its model, and a model its
    training selector; a project is counted whole, and which of its
    annotations trained a model stays unknown.

    Raises
    ------
    ValueError
        If none or more than one of the three is given, or it is no Studio id.
    """
    given = {
        key: str(args[key]).strip()
        for key in _INPUTS
        if args.get(key) is not None and str(args[key]).strip()
    }
    if len(given) != 1:
        named = ", ".join(sorted(given)) or "none"
        raise ValueError(
            "give exactly one of prediction_id, model_id or project_id "
            f"(given: {named})"
        )
    source, ident = next(iter(given.items()))
    if not _ID.fullmatch(ident):
        raise ValueError(
            f"{source} is not a Studio id: letters, digits and hyphens, at most 64"
        )
    if source == "project_id":
        return await _project_inventory(ctx, ident)
    return await _model_inventory(ctx, source, ident)


def build_studio_label_tools() -> list[RegisteredTool]:
    """Return the ``olmoearth_studio_label_inventory`` tool (core, read-only)."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_studio_label_inventory",
                description=(
                    "Count the Studio annotations a model's training selector "
                    "selects now, or a project's, by status, geometry, split and "
                    "class. Give exactly one of prediction_id, model_id, "
                    "project_id. No random design recorded: no error rate. "
                    "Read-only."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "prediction_id": {"type": "string"},
                        "model_id": {"type": "string"},
                        "project_id": {"type": "string"},
                    },
                    "required": [],
                },
            ),
            handler=_label_inventory,
            capability=Capability(
                does="count the Studio annotations a model's training selector "
                "selects now, or a project's, by status, split and class"
            ),
        )
    ]
