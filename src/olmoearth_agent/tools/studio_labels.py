# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Which Studio annotations a model was trained on, and what they can be used for.

``olmoearth_studio_label_inventory`` counts the annotations a Studio model's
training selector selects, or a project's. exp86 round 9 (B3/studio run 1)
answered "no ground-truth labels exist" while the run's model records named a
label field of the user's project; the tools could say only that a label
field existed (``statistical_rules.labels_in_studio_fact``).

The model record (``GET /models/{id}``, read live on 2026-10-31) carries
``data_selectors``, a list of objects shaped as the spec's
``AnnotationSearchRequest``: the kelp model's holds its training project's
``project_id`` as ``{"eq": id, "inc": null}``, every other filter null,
``sort_by`` ``creation_time``, ``sort_direction`` ``asc`` and ``limit``
null. Its ``wizard_answers`` name the label field (``label_field_id``), the
split field (``split_field_id``, the annotation metadata field named
``oe_split``) and the split proportions. Each selector's filters are sent to
``POST /annotations/search`` as they stand, and the annotations are counted
by status, geometry type, split, class, split by class and creation time,
beside the model record's own times; an annotation two selectors select
counts once.

Assumed, not verified live: that Studio trained on what the selectors select
now (an annotation added, edited or deleted since counts as it is now; the
counts created after the model's creation and last update say how many are
newer); that several selectors are a union; that training honoured the split
field (the record names it, nothing confirms it); that a filtered search's
``meta.total`` counts the filtered annotations; and the paging (``sort_by``,
``sort_direction``, ``offset`` up to 10000 and a ``creation_time`` floor past
it are the spec's alone).

What the counts are for, the result's ``must_state``: people place
annotations where they choose, so they are no probability sample of the map
and give no design-based error rate (``olmoearth_plan_label_sample`` draws
one); those of split train or val fit or tuned the model; those of split
test were held out only if training honoured the split field. No id,
geometry, coordinate, free-text value or person's id is returned:
annotations are counted, never listed, and an annotation's id is a key of
the count's deduplication only.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.studio.client import StudioAccessError
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.registry import Capability, RegisteredTool, ToolContext

#: Annotations asked for per search page.
_PAGE = 1000
#: The most annotations one call counts, over all its selectors; past it the
#: counts cover the annotations read, and the result says so.
_CAP = 20_000
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
#: The most characters of a class, split or field name the result keeps.
_MAX_NAME = 60

#: The key of an annotation whose split, class or field value is not recorded.
UNRECORDED = "unrecorded"
#: The key of a text value: free text is never returned.
FREE_TEXT = "free text (not shown)"
#: The key under which the values past :data:`_MAX_VALUES` are summed.
OTHER = "other values"

#: Every result: annotations are no probability sample of the map.
NOT_A_SAMPLE = (
    "People place Studio annotations where they choose, not at random over the "
    "map, so they give no design-based error rate for it."
)
#: A replayed selector whose model record names a split field.
TRAIN_VAL_USED = (
    "Annotations of split train or val in the training selector were used to "
    "fit or tune the model."
)
#: The same: the record names the split field and cannot say training used it.
TEST_HELD_OUT_IF = (
    "Test-split annotations were held out only if training honoured the split "
    "field the model record names, which the record cannot confirm."
)
#: A replayed selector whose model record names no split field.
NO_SPLIT_FIELD = (
    "The model record names no split field, so which of these annotations fit, "
    "tuned or were held out from the model is unknown."
)
#: A project's annotations counted with no model: no training statement.
TRAINING_UNKNOWN = (
    "Which of these annotations any model was trained on is unknown: no model "
    "record was read."
)
#: No selector replayed (no model, no record, none recorded or none replayable).
SELECTOR_UNKNOWN = (
    "The model's training selector is unknown here, so which Studio annotations "
    "trained it is not known from this run."
)
#: The route to an error rate, in every result.
NEXT_STEP = (
    "For the map's error rate: olmoearth_plan_label_sample draws windows of the "
    "map by a design, and olmoearth_estimate_map_error grades the reviewer's "
    "labels on them."
)
#: When the training selector is unknown.
PROJECT_STEP = (
    "Given the project whose annotations trained the model, "
    "olmoearth_studio_label_inventory with project_id counts that project's "
    "annotations; which of them trained the model stays unknown."
)
#: What the creation-time counts can say.
TIMES_NOTE = (
    "These are record times: the model record does not say when training read "
    "its annotations or ended, so an annotation created after the model was "
    "created may not have been trained on."
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


def _annotation_rate_claim() -> dict[str, str]:
    """An accuracy or error rate from Studio annotations, stated as the map's.

    The contract's ``accuracy_not_measured`` is the id the answer checks read
    for an accuracy figure in a request that measured none
    (``harness/checks.py``, ``_accuracy_not_measured``): this tool measures
    none, and a figure from these annotations, or the model's training or
    validation metrics, is no measure of the map. The check stops once the
    request estimated the map's error rate from labels of a planned sample.
    """
    return rules.forbidden(
        rules.ACCURACY_NOT_MEASURED,
        "Studio annotations were placed where people chose, not drawn at random, "
        "and those a model was fit or tuned on are not independent of it: no "
        "accuracy, error rate or share wrong computed from them, nor a model's "
        "training or validation metrics, may be stated as the map's; only "
        "olmoearth_estimate_map_error on the labels of a planned sample measures "
        "the map",
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
    counted, only to count each once; no id leaves this object.
    """

    split_field: str | None = None
    label_field: str | None = None
    model_created: datetime | None = None
    model_updated: datetime | None = None
    read: int = 0
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

    @property
    def by_field(self) -> bool:
        """Whether the model record names the split or the class field."""
        return bool(self.split_field or self.label_field)

    def add(self, record: dict[str, Any]) -> None:
        """Count ``record``, unless an annotation of its id was counted already."""
        key = record.get("id")
        if key is not None:
            if str(key) in self.seen:
                return
            self.seen.add(str(key))
        self.read += 1
        status = record.get("status")
        known = isinstance(status, str) and status in _STATUSES
        self.status[status if known else UNRECORDED] += 1
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
        self._count_time(record.get("creation_time"))

    def _count_labelset(self, entry: dict[str, Any]) -> None:
        """Count a labelset entry's label under its field's name."""
        name = entry.get("name")
        if entry.get("data_type") != "labelset" or not isinstance(name, str):
            return
        key = name.strip()[:_MAX_NAME]
        if not key or (key not in self.fields and len(self.fields) >= _MAX_FIELDS):
            return
        self.fields.setdefault(key, Counter())[_entry_value(entry)] += 1

    def _count_time(self, raw: Any) -> None:
        """Keep the first and last creation times; count those after the model's."""
        when = _when(raw)
        if when is None:
            return
        text = str(raw).strip()
        if self.first is None or when < self.first[0]:
            self.first = (when, text)
        if self.last is None or when > self.last[0]:
            self.last = (when, text)
        if self.model_created is not None and when > self.model_created:
            self.after_created += 1
        if self.model_updated is not None and when > self.model_updated:
            self.after_updated += 1

    def _table(self) -> dict[str, dict[str, int]]:
        """Split by class, on the splits and classes :func:`_top` keeps."""
        splits, labels = _top(self.split), _top(self.label)
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
    filter emptied by that is no filter. The order keys are left out without
    a word (:data:`_ORDER_KEYS`); a ``limit``, an ``offset`` or any other key
    that holds a value is named, since it may narrow what training selected,
    and the counts then hold more.
    """
    filters: dict[str, Any] = {}
    left_out: list[str] = []
    for key, value in selector.items():
        kept = _without_nulls(value)
        empty = kept is None or (isinstance(kept, (dict, list)) and not kept)
        if empty or key in _ORDER_KEYS:
            continue
        if key in _FILTER_KEYS:
            filters[key] = kept
        else:
            left_out.append(str(key)[:_MAX_NAME])
    return filters, sorted(left_out)


async def _read_selector(
    ctx: ToolContext, filters: dict[str, Any], tally: _Tally
) -> tuple[int | None, bool]:
    """Page one selector's annotations into ``tally``.

    Pages of :data:`_PAGE` in creation-time order, by ``offset`` up to
    :data:`_OFFSET_MAX`; past it from a ``creation_time`` floor (``gte``) at
    the last creation time read, offset 0 again, so the annotations of that
    time come twice and count once. Stops at a short page, at :data:`_CAP`
    annotations counted in the call, or when a floor would not move (more
    annotations share one creation time than the offset reaches).

    Returns
    -------
    tuple
        Studio's ``meta.total`` for the selector (its first page's), or
        ``None``; and whether every annotation it selects was read.
    """
    total: int | None = None
    floor: str | None = None
    offset, pages = 0, 0
    while tally.read < _CAP:
        body: dict[str, Any] = {**filters, **_ORDER}
        if floor is not None:
            own = filters.get("creation_time")
            body["creation_time"] = {
                **(own if isinstance(own, dict) else {}),
                "gte": floor,
            }
        env = await ctx.studio.search_annotations(body, limit=_PAGE, offset=offset)
        if pages == 0:
            total = env.total
        pages += 1
        for record in env.records:
            if tally.read >= _CAP:
                return total, False
            if isinstance(record, dict):
                tally.add(record)
        if len(env.records) < _PAGE:
            return total, True
        if offset + _PAGE <= _OFFSET_MAX:
            offset += _PAGE
            continue
        last = env.records[-1]
        at = last.get("creation_time") if isinstance(last, dict) else None
        if not isinstance(at, str) or not at or at == floor:
            return total, False
        floor, offset = at, 0
    return total, False


def _read_note(read: int, total: int | None, complete: bool, *, several: bool) -> str:
    """How much of what the selectors select was read and counted."""
    of = ""
    if total is not None:
        of = f" of the {total} that Studio counts (meta.total)"
    if complete:
        note = f"Every annotation selected was read: {read} counted."
    elif read >= _CAP:
        note = (
            f"Read {read}{of}, stopping at this tool's cap of {_CAP}; the counts "
            "cover the annotations read only."
        )
    else:
        note = (
            f"Read {read}{of}: more annotations share one creation time than "
            "Studio's search pages past; the counts cover the annotations read only."
        )
    if several:
        note += (
            " Studio's totals are per selector, summed; an annotation two "
            "selectors select is counted once here."
        )
    return note


async def _read(
    ctx: ToolContext, selectors: list[dict[str, Any]], tally: _Tally
) -> tuple[dict[str, Any], list[int | None]]:
    """Read every selector into ``tally``: the summary, and each selector's total."""
    totals: list[int | None] = []
    complete = True
    for filters in selectors:
        total, whole = await _read_selector(ctx, filters, tally)
        totals.append(total)
        complete = complete and whole
    known = [t for t in totals if t is not None]
    total = sum(known) if totals and len(known) == len(totals) else None
    summary: dict[str, Any] = {
        "annotations_read": tally.read,
        "studio_total": total,
        "complete": complete,
        "read_note": _read_note(
            tally.read, total, complete, several=len(selectors) > 1
        ),
    }
    return summary, totals


def _contract(
    out: dict[str, Any], must_state: list[str], *, steps: tuple[str, ...] = ()
) -> dict[str, Any]:
    """``out`` with its next steps, its ``must_state`` and the forbidden claim."""
    out["next_steps"] = [NEXT_STEP, *steps]
    return rules.add_contract(
        out, must_state=must_state, forbidden_claims=[_annotation_rate_claim()]
    )


def _unknown(
    out: dict[str, Any], reason: str, *, label: str = "unknown"
) -> dict[str, Any]:
    """A result with no selector replayed, and why."""
    out["training_selector"] = label
    out["reason"] = reason
    return _contract(out, [SELECTOR_UNKNOWN, NOT_A_SAMPLE], steps=(PROJECT_STEP,))


async def _model_record(
    ctx: ToolContext, model_id: str
) -> tuple[dict[str, Any] | None, str]:
    """The model's record, or ``None`` and why it is unknown.

    A refusal of the key (:class:`StudioAccessError`) is raised as it is: no
    lookup succeeds with that key. Any other failure, a server without a
    ``/models`` route among them (``StudioClient.models_unavailable``), leaves
    the training selector unknown, never empty.
    """
    try:
        record = await ctx.studio.get_model(model_id)
    except StudioAccessError:
        raise
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
    """The model record's fields stated; never its requester_id or checkpoint_token."""
    split = wizard.get("split_proportions")
    return {
        "name": record.get("name"),
        "status": record.get("status"),
        "creation_time": record.get("creation_time"),
        "updated_time": record.get("updated_time"),
        "patch_size": wizard.get("patch_size"),
        "training_project_id": wizard.get("project_id"),
        "label_field_named": bool(wizard.get("label_field_id")),
        "split_field_named": bool(wizard.get("split_field_id")),
        "split_proportions": (
            {k: v for k, v in split.items() if k in _SPLIT_PROPORTIONS}
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
    summary, _totals = await _read(ctx, [{"project_id": {"eq": project_id}}], tally)
    out: dict[str, Any] = {
        "source": "project",
        "project_id": project_id,
        "training_selector": "unknown",
        **summary,
        **tally.counts(),
        "fields_note": FIELDS_NOTE,
    }
    return _contract(out, [NOT_A_SAMPLE, TRAINING_UNKNOWN])


async def _model_inventory(ctx: ToolContext, source: str, ident: str) -> dict[str, Any]:
    """The training selector of a model (or a prediction's), replayed and counted."""
    out: dict[str, Any] = {"source": source.removesuffix("_id")}
    model_id = ident
    if source == "prediction_id":
        prediction = await ctx.studio.get_prediction(ident)
        out["prediction_id"] = ident
        model_id = str(prediction.get("model_id") or "").strip()
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
        reason = "the model record names no training selector"
        return _unknown(out, reason, label="none recorded")
    replays = [_replay(s) for s in selectors]
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
    summary, totals = await _read(ctx, filters, tally)
    each = iter(totals)  # one total per selector that holds a filter, in order
    out["training_selector"] = "replayed"
    out["selectors"] = [
        {
            "filters": sorted(f),
            "not_replayed": keys,
            "studio_total": next(each) if f else None,
        }
        for f, keys in replays
    ]
    out.update(summary)
    out.update(tally.counts())
    out["created_after_model_creation"] = (
        tally.after_created if tally.model_created is not None else None
    )
    out["created_after_model_update"] = (
        tally.after_updated if tally.model_updated is not None else None
    )
    out["times_note"] = TIMES_NOTE
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
    if split_field and isinstance(wizard.get("split_proportions"), dict):
        notes.append(SPLIT_NOTE)
    if notes:
        out["selector_notes"] = notes
    must = (
        [NOT_A_SAMPLE, TRAIN_VAL_USED, TEST_HELD_OUT_IF]
        if split_field
        else [NOT_A_SAMPLE, NO_SPLIT_FIELD]
    )
    return _contract(out, must)


async def _label_inventory(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_studio_label_inventory``.

    Exactly one of ``prediction_id``, ``model_id`` or ``project_id``: a
    prediction gives its model, and a model its training selector; a project
    is counted whole, and which of its annotations trained a model stays
    unknown.

    Raises
    ------
    ValueError
        If none or more than one of the three is given.
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
                    "Count the Studio annotations a model was trained on, or a "
                    "project's, by status, geometry, split and class. Give "
                    "exactly one of prediction_id, model_id, project_id. "
                    "Annotations are no random sample: no error rate. Read-only."
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
                does="count the Studio annotations a model was trained on, or a "
                "project's, by status, split and class"
            ),
        )
    ]
