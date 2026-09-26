# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-review-set`` tool bundle (skill #18).

Five tools:

- ``olmoearth_review_set`` -- rank windows by the model's own top-1 minus
  top-2 margin and return the ones a reviewer should open first at a budget.
- ``olmoearth_review_set_from_result`` -- the same ranking for a Studio
  prediction result: sample its band on a grid, read a binary score in
  ``[0, 1]`` as ``[1 - s, s]`` (stating that assumption), and rank.
- ``olmoearth_compare_review`` -- compare two inferences of the same windows:
  how much they differ and where, with the side question declined on evidence
  (and, with the ``inferencex`` extra, what a difference means at the maps'
  dates).
- ``olmoearth_grade_review_rule`` -- grade any candidate suspicion signal
  against the margin baseline and a no-model control, with a per-group sign
  test, so a signal that merely sounds principled cannot ship unmeasured.
- ``olmoearth_review_budget_ceiling`` -- the arithmetic that stops a good
  ranker being talked down: no rule can catch more than
  ``min(1, budget / error_rate)`` of the errors at a given budget.

Bring-your-own-scores, like skill #8's embeddings: Studio's pixel-value
returns only ``raw_value`` and ``classification``, never probabilities or
logits. A margin cannot be recovered from a hard class; it can be read from a
Studio regression band that is a binary score in ``[0, 1]`` (decided at 0.5),
which is what ``olmoearth_review_set_from_result`` does, or from any band once
a decision threshold is named. When all you have is hard classes from two or
more Studio results, ``olmoearth_compare_results`` with ``mode='ensemble'``
(skill #9's disagreement signal) is the tool that still works. A model run's
own raster (every pixel's scores, e.g. from a GPU cluster) goes through
``olmoearth_scores_from_file`` (:mod:`olmoearth_agent.tools.scores_file`), whose
scores file these tools read.

No coordinates in, no coordinates out (rule §3.1).
"""

from __future__ import annotations

import asyncio
import calendar
import csv
import hashlib
import io
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from olmoearth_agent.analysis.output_contract import add_must_state
from olmoearth_agent.analysis.raster_compare import (
    declared_range,
    grid_windows,
    result_bbox,
)
from olmoearth_agent.analysis.review_set import (
    COMPARISON_FORBIDDEN,
    DEFAULT_BUDGETS,
    DEFAULT_MAX_LISTED,
    MUST_STATE_MULTICLASS_LOGIT,
    NOT_COVERED,
    attainable_ceiling,
    compare_scores,
    evidence_detail,
    grade_rule,
    regression_scores,
    review_set,
)
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.security.paths import safe_path, workspace_root
from olmoearth_agent.tools import inferencex
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext
from olmoearth_agent.tools.sampling import (
    FAILED,
    NODATA,
    OK,
    SAMPLE_CONCURRENCY,
    default_property,
    read_sample,
    result_nodata_context,
    sample_records,
    select_band,
)

_SCORES_SCHEMA = {
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
    "description": (
        "Per-window model scores, one inner array per window, one number per "
        "class (>= 2 classes). Logits preferred; probabilities accepted."
    ),
}

_SERIES = {"type": "array", "items": {"type": "number"}}

logger = logging.getLogger(__name__)

#: Env var naming the directory scores files are read from and written to.
SCORES_ROOT_ENV = "OLMOEARTH_SCORES_ROOT"

#: The file, under the write root, that holds the full evidence text a result's
#: one-sentence ``evidence_scope`` stands for (named by ``evidence_detail_path``).
EVIDENCE_FILE = "review_set_evidence.json"

#: Differing windows a comparison lists inline; the file at ``differing_path``
#: holds every one (exp86 rounds 6 and 7 read directions and places off an
#: inline listing of 50).
COMPARE_INLINE_LISTED = 10

#: Grid bounds for sampling a Studio result into review windows: every window
#: is one live pixel-value call, so the grid is capped (16 x 16 = 256 calls).
FROM_RESULT_DEFAULT_GRID = 10
FROM_RESULT_MIN_GRID = 2
FROM_RESULT_MAX_GRID = 16

#: The grid a Studio result takes, as stated in the tools' schemas.
FROM_RESULT_GRID_RANGE = f"{FROM_RESULT_MIN_GRID}-{FROM_RESULT_MAX_GRID}"


def result_grid(value: Any) -> tuple[int, int | None]:
    """The side of the square grid a Studio result is sampled on, and the side asked.

    ``value`` is ``N``, ``[N, N]`` or ``None`` (the default side). ``N`` is
    held to :data:`FROM_RESULT_MIN_GRID`-:data:`FROM_RESULT_MAX_GRID`,
    because every window is one live pixel-value call; the caller states the
    hold (:meth:`SampledScores.grid_block`), so it is never silent.

    Parameters
    ----------
    value
        The ``grid`` argument as the model passed it.

    Returns
    -------
    tuple[int, int | None]
        ``(used, requested)``; ``requested`` is ``None`` when no grid was given.

    Raises
    ------
    ValueError
        For a non-square ``[rows, cols]``: a Studio result is sampled square.
    """
    if value is None:
        return FROM_RESULT_DEFAULT_GRID, None
    if isinstance(value, (list, tuple)):
        if len(value) != 2 or int(value[0]) != int(value[1]):
            msg = (
                "a Studio result is sampled on a square N x N grid: pass grid "
                f"as one integer N ({FROM_RESULT_GRID_RANGE}), got {list(value)}"
            )
            raise ValueError(msg)
        value = value[0]
    requested = int(value)
    used = max(FROM_RESULT_MIN_GRID, min(FROM_RESULT_MAX_GRID, requested))
    return used, requested


#: How a sampled value is judged no-data (the same rule as the compare tools).
_NODATA_RULE = (
    "a sample is no-data, and is dropped before ranking, when its band is "
    "missing or NaN, equals the model's nodata_value, lies outside the band's "
    "declared regression range, or carries no class on a classification band"
)


def read_roots() -> list[str]:
    """Directories a scores or design file may be read from."""
    env = os.environ.get(SCORES_ROOT_ENV)
    if env:
        return [os.path.realpath(env)]
    return [os.path.realpath(os.getcwd()), str(workspace_root())]


def write_root() -> Path:
    """Directory the tools write scores and design files to.

    ``OLMOEARTH_SCORES_ROOT`` when set, else the workspace root (rule §3.4's
    path guard), which :func:`load_json_file` also reads from.
    """
    env = os.environ.get(SCORES_ROOT_ENV)
    return Path(env).resolve() if env else workspace_root()


def load_json_file(path: str, what: str = "scores_path") -> Any:
    """Read a ``.json`` file under the scores root; anything else is refused.

    A model-chosen path must not pull arbitrary files into a tool result, so
    only a ``.json`` under ``OLMOEARTH_SCORES_ROOT`` (or, unset, the working
    directory or the workspace root) is readable.
    """
    real = os.path.realpath(path)
    roots = read_roots()
    inside = any(os.path.commonpath([root, real]) == root for root in roots)
    if not real.endswith(".json") or not inside:
        msg = f"{what} must name a .json file under {' or '.join(roots)}"
        raise ValueError(msg)
    with open(real, encoding="utf-8") as fh:
        return json.load(fh)


def write_json_file(name: str, payload: dict[str, Any]) -> str:
    """Write ``payload`` as ``name`` under :func:`write_root`; return the absolute path."""
    root = write_root()
    target = safe_path(name, root=root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")
    return str(target)


def slug(text: str, limit: int = 24) -> str:
    """A filesystem-safe fragment of ``text`` for generated file names."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")[:limit] or "x"


@dataclass
class ScoresFile:
    """A scores file as read: rows, an optional grid, and the windows the rows are.

    ``windows`` is ``None`` when the rows cover the whole grid row-major (the
    original format). A file written from a sampled Studio result skips its
    no-data windows, so it carries ``windows``: the row-major grid index of
    each row.
    """

    scores: list[Any]
    grid: tuple[int, int] | None = None
    windows: list[int] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def load_scores_file(path: str) -> ScoresFile:
    """Read a scores file: a JSON array of rows, or an object with ``scores``.

    The object form may carry ``grid`` ``[rows, cols]`` and ``windows`` (the
    grid index of each row, when no-data windows were left out).
    """
    data = load_json_file(path)
    if isinstance(data, dict):
        grid = data.get("grid")
        windows = data.get("windows")
        return ScoresFile(
            scores=data["scores"],
            grid=(int(grid[0]), int(grid[1])) if grid else None,
            windows=[int(w) for w in windows] if windows is not None else None,
            meta={k: v for k, v in data.items() if k not in ("scores", "windows")},
        )
    return ScoresFile(scores=data)


def _place_rows(
    rows: list[dict[str, Any]], key: str, windows: list[int] | None, cols: int | None
) -> None:
    """Map ``key`` indices back to grid windows, and add ``row``/``col``, in place."""
    for row in rows:
        idx = int(row[key])
        if windows is not None:
            idx = windows[idx]
            row[key] = idx
        if cols:
            row["row"], row["col"] = divmod(idx, cols)


def _file_classes(meta: dict[str, Any], n_rows: int) -> list[int] | None:
    """A scores file's own per-row classes (``map_class``), checked, or ``None``.

    A file from ``olmoearth_scores_from_file`` carries each window's majority
    class beside its row, because a row whose margin is exactly 0 cannot say
    which class won.
    """
    classes = meta.get("map_class")
    if classes is None:
        return None
    if not isinstance(classes, list) or len(classes) != n_rows:
        msg = f"the scores file's map_class does not have one class per row ({n_rows})"
        raise ValueError(msg)
    return [int(c) for c in classes]


def _annotate_from_file(
    out: dict[str, Any],
    meta: dict[str, Any],
    n_rows: int,
    full: list[dict[str, Any]] | None = None,
) -> None:
    """Carry a scores file's signal, provenance, classes and class names into a ranking.

    The classes and names go on the listed rows and on ``full``, the review
    set's every row, when given.
    """
    if isinstance(meta.get("signal"), str):
        out["signal"] = meta["signal"]
    if isinstance(meta.get("provenance"), str):
        out["provenance"] = meta["provenance"]
    names = meta.get("classes") if isinstance(meta.get("classes"), dict) else None
    classes = _file_classes(meta, n_rows)
    for row in [*out.get("review", []), *(full or [])]:
        if classes is not None:
            row["predicted_class"] = classes[int(row["window_index"])]
        if names is not None:
            row["class_name"] = names.get(str(row["predicted_class"]))


def file_model(meta: dict[str, Any]) -> str | None:
    """The model a scores file names (its manifest's repo or id), or ``None``."""
    model = meta.get("model")
    if isinstance(model, dict):
        name = model.get("repo") or model.get("id")
        return str(name) if name else None
    return str(model) if isinstance(model, str) and model else None


def file_warnings(meta: dict[str, Any]) -> list[str]:
    """The warnings a scores file carries (the provider's ``package_warnings``,
    and any ``warnings``), each stated as the provider's, whole."""
    out: list[str] = []
    for key, who in (
        ("package_warnings", "The scores provider (olmoearth-inferencex) warns: "),
        ("warnings", "The scores file warns: "),
    ):
        raw = meta.get(key)
        out.extend(
            who + warning.strip()
            for warning in (raw if isinstance(raw, list) else [])
            if isinstance(warning, str) and warning.strip()
        )
    return out


#: A scores provider's warning, by its opening words, and the limit it states
#: as a ``must_state`` sentence. The provider's own text can end in an
#: instruction no agent tool can carry out (the multi-class warning's "pass
#: form='top1'": no agent tool takes a ``form``), so it is never copied into
#: ``must_state``; a warning not listed here is kept in ``scores_file_warnings``
#: only.
WARNING_LIMITS: tuple[tuple[str, str], ...] = (
    ("multi-class logit margin", MUST_STATE_MULTICLASS_LOGIT),
)


def warning_limits(meta: dict[str, Any]) -> list[str]:
    """The ``must_state`` sentences for the warnings a scores file carries."""
    raw: list[str] = []
    for key in ("package_warnings", "warnings"):
        listed = meta.get(key)
        if isinstance(listed, list):
            raw.extend(w.strip().lower() for w in listed if isinstance(w, str))
    return [
        limit
        for opening, limit in WARNING_LIMITS
        if any(warning.startswith(opening) for warning in raw)
    ]


def file_grid(
    asked: Any, loaded: ScoresFile, *, name: str = "scores file"
) -> tuple[list[int] | None, str | None]:
    """The grid to place a scores file's rows on, and a note when the one asked is set aside.

    A file that names its grid is placed on it, whatever grid the model
    passed: a file that leaves its no-data windows out indexes its rows into
    that grid, and a transposed grid of the same size would pass every
    count check and place every window wrong (exp87 review). A file that
    leaves windows out but names no grid gives no way to check the model's,
    so no grid is used. A whole-grid file without one keeps the model's,
    which the ranking checks against the row count.
    """
    asked_grid = [int(asked[0]), int(asked[1])] if asked else None
    if loaded.grid is not None:
        own = [int(loaded.grid[0]), int(loaded.grid[1])]
        note = (
            f"the grid passed ({asked_grid[0]}x{asked_grid[1]}) was set aside for "
            f"the {name}'s own ({own[0]}x{own[1]})"
            if asked_grid is not None and asked_grid != own
            else None
        )
        return own, note
    if loaded.windows is not None:
        note = (
            f"the grid passed ({asked_grid[0]}x{asked_grid[1]}) was not used: the "
            f"{name} leaves windows out and names no grid to check it against"
            if asked_grid is not None
            else None
        )
        return None, note
    return asked_grid, None


def evidence_detail_path() -> str | None:
    """Save the full evidence text behind ``evidence_scope``; return its path.

    The text is static, so the file is rewritten only when it differs. A
    failure to write never fails the tool: the result then carries no path.
    """
    text = json.dumps(evidence_detail(), indent=1)
    try:
        target = safe_path(EVIDENCE_FILE, root=write_root())
        if not target.is_file() or target.read_text(encoding="utf-8") != text:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
    except (OSError, ValueError):
        logger.warning("could not save the review-set evidence file", exc_info=True)
        return None
    return str(target)


#: A review list file's columns, in order: the first five always (``row`` and
#: ``col`` empty without a grid), the rest when the rows have them. Never a
#: coordinate, and never a caller's own window label (rule §3.1).
REVIEW_LIST_COLUMNS = ("rank", "window_index", "row", "col", "margin")
REVIEW_LIST_OPTIONAL = ("score", "predicted_class", "class_name", "boundary_neighbours")


def _save_review_list(rows: list[dict[str, Any]]) -> str | None:
    """The review set's every row, in review order, to a CSV beside the evidence file.

    Named by a digest of its content, so the same ranking writes the same
    file. A failure to write never fails the ranking: the result then says
    the list could not be saved.
    """
    optional = [c for c in REVIEW_LIST_OPTIONAL if rows and c in rows[0]]
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[*REVIEW_LIST_COLUMNS, *optional],
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    body = buffer.getvalue()
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:10]
    try:
        target = safe_path(f"review_list_{digest}.csv", root=write_root())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    except (OSError, ValueError):
        logger.warning("could not save the review list", exc_info=True)
        return None
    return str(target)


def review_list(
    listed: int, full: list[dict[str, Any]], budget: float
) -> dict[str, Any]:
    """The review list file when the listing is cut short, and the note that says so.

    exp86 round 8 (brief 8 on the cluster, runs 2 and 3) said the rest of the
    819-window review set was saved in review_set_evidence.json, which holds
    evidence text only; nothing held the list. When fewer windows are listed
    than the review set holds, every one is written to a CSV
    (:func:`_save_review_list`) and returned as ``review_list_path`` with
    ``review_list_rows``; ``listing_note`` says what the listing shows, where
    the rest is, and what the evidence file holds.

    Parameters
    ----------
    listed
        Rows listed inline.
    full
        Every row of the review set, in review order.
    budget
        The budget the review set is at.
    """
    total = len(full)
    evidence = (
        f"{EVIDENCE_FILE} (evidence_detail_path) holds evidence text only, no "
        "windows"
    )
    if listed >= total:
        return {
            "listing_note": f"review lists all {total:,} windows of the review set "
            f"at budget {budget:g}, in review order, so no list file is written; "
            f"{evidence}."
        }
    path = _save_review_list(full)
    where = (
        f"the full list of {total:,}, in the same order, is in the CSV at "
        f"review_list_path ({', '.join(REVIEW_LIST_COLUMNS)} and the class or "
        "score; no coordinates)"
        if path
        else f"the full list of {total:,} could not be saved"
    )
    out: dict[str, Any] = {
        "listing_note": f"review lists the first {listed:,} of the {total:,} "
        f"windows in the review set at budget {budget:g}, in review order (not a "
        f"sample of them); {where}; {evidence}."
    }
    if path:
        out["review_list_path"] = path
        out["review_list_rows"] = total
    return out


async def _review_set(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_review_set``."""
    grid = args.get("grid")
    scores = args.get("scores")
    windows: list[int] | None = None
    meta: dict[str, Any] = {}
    grid_note: str | None = None
    if scores is None and args.get("scores_path"):
        # Scores usually live in a file where inference ran; a model cannot relay
        # hundreds of rows inline, and asking it to would test transcription.
        loaded = load_scores_file(str(args["scores_path"]))
        scores, windows, meta = loaded.scores, loaded.windows, loaded.meta
        grid, grid_note = file_grid(grid, loaded)
    if scores is None:
        msg = "pass 'scores' (rows inline) or 'scores_path' (a .json file of rows)"
        raise ValueError(msg)
    order = str(args.get("order", "confidence"))
    if windows is not None and order == "boundary_first":
        msg = (
            "boundary_first needs every window of the grid, and this file leaves "
            "its no-data windows out; use order='confidence'"
        )
        raise ValueError(msg)
    # A file that says what its rows are (logits or probabilities) is believed
    # over detection: a provider's rows are one confidence and zeros.
    file_type = meta.get("score_type")
    budget = float(args.get("budget", 0.05))
    out = review_set(
        scores,
        budget=budget,
        ids=args.get("ids"),
        # A grid whose no-data windows were left out has no full neighbourhood.
        grid=(int(grid[0]), int(grid[1])) if grid and windows is None else None,
        order=order,
        score_type=args.get("score_type")
        or (file_type if file_type in ("logit", "probability") else None),
        error_rate=(
            float(args["error_rate"]) if args.get("error_rate") is not None else None
        ),
        max_listed=int(args.get("max_listed", DEFAULT_MAX_LISTED)),
        score_kind=(
            meta.get("score_kind") if isinstance(meta.get("score_kind"), str) else None
        ),
        model=file_model(meta),
        keep_full=True,
    )
    full = out.pop("review_full")
    if meta:
        _annotate_from_file(out, meta, len(scores), full)
        # A warning the scores file carries travels with every ranking of it
        # (exp86 round 6: no brief-8 answer passed the provider's multi-class
        # warning on): whole in scores_file_warnings, and as the limit it
        # states in must_state, never as the provider's instruction.
        warnings = file_warnings(meta)
        if warnings:
            out["scores_file_warnings"] = warnings
        # A limit that states the suite's figures is stated only where the
        # suite's evidence covers the case, in whole or in part (exp86 round
        # 8: no field carries an experiment's numbers to a case it excludes).
        if out.get("evidence_covers_this_case") not in NOT_COVERED:
            add_must_state(out, warning_limits(meta))
    if grid_note:
        out["grid_note"] = grid_note
    # Row-major index to (row, col), so a caller with a grid need not do the division itself.
    cols = int(grid[1]) if grid else None
    _place_rows(out.get("review", []), "window_index", windows, cols)
    _place_rows(full, "window_index", windows, cols)
    out["evidence_detail_path"] = evidence_detail_path()
    out.update(review_list(len(out.get("review", [])), full, budget))
    return out


#: The date forms ``olmoearth_compare_review`` takes, as the package's
#: ``compare.dates_reading`` reads a string: a day or a start/end interval.
DATE_FORMS = "YYYY-MM-DD, or a YYYY-MM-DD/YYYY-MM-DD period; not a bare year or month"

_BARE_YEAR = re.compile(r"(\d{4})")
_BARE_MONTH = re.compile(r"(\d{4})-(\d{2})")


def _check_date_form(value: Any, name: str) -> None:
    """Refuse a bare year or month, naming the interval that means it.

    ``oe_inferencex.compare.dates_reading`` reads a string as an ISO day or a
    start/end interval and refuses ``'2023'``; the schema says so (exp86 round
    2: every brief-3 cluster run passed ``'2023'`` first). The refusal names the
    whole-period interval, as the package reads a year or month ``datetime64``.
    """
    if not isinstance(value, str):
        return
    text = value.strip()
    if _BARE_YEAR.fullmatch(text):
        period = f"{text}-01-01/{text}-12-31"
        what = "a year"
    elif (month := _BARE_MONTH.fullmatch(text)) and 1 <= int(month[2]) <= 12:
        year, mm = int(month[1]), int(month[2])
        last = calendar.monthrange(year, mm)[1]
        period = f"{text}-01/{text}-{last:02d}"
        what = "a month"
    else:
        return
    msg = (
        f"{name}: {value!r} is {what}, not a date; a map of that whole {what[2:]} "
        f"is the interval {period!r} (dates are {DATE_FORMS})"
    )
    raise ValueError(msg)


def _dates_block(args: dict[str, Any]) -> dict[str, Any]:
    """The package's reading of what a difference means at the maps' dates.

    With the ``inferencex`` extra: ``oe_inferencex.compare.dates_reading`` of
    ``date_a``, ``date_b`` and ``labels_date`` (``status`` "unstated" when none
    was given). Without it: a statement that the dates were not assessed. A
    bare year or month is refused first, with the interval that means it.
    """
    date_a, date_b = args.get("date_a"), args.get("date_b")
    labels_date = args.get("labels_date")
    for name, value in (
        ("date_a", date_a),
        ("date_b", date_b),
        ("labels_date", labels_date),
    ):
        _check_date_form(value, name)
    try:
        compare = inferencex.load("compare")
    except inferencex.InferencexMissingError:
        return {
            "assessed": False,
            "a": date_a,
            "b": date_b,
            "labels": labels_date,
            "reason": "the dates were not assessed: reading what a difference "
            "means across dates needs olmoearth-inferencex (>= 1.3.0)",
            "install": inferencex.INSTALL_HINT,
        }
    reading: dict[str, Any] = dict(compare.dates_reading(date_a, date_b, labels_date))
    reading["assessed"] = True
    return reading


#: Stated with a comparison of maps of different or overlapping times: the
#: dates sentence merged with what labels for one date can grade (exp86 round
#: 8, brief 3 on the cluster and brief 7 on files).
DIFFERENT_TIMES_MUST_STATE = rules.DATED_MAPS_MUST_STATE

#: Stated with a comparison where only one map's date was given.
PARTLY_DATED_MUST_STATE = (
    "Only one map's date was given, so a difference cannot be told from a change "
    "on the ground."
)


def _file_names(metas: list[dict[str, Any]]) -> dict[str, str] | None:
    """The class names both scores files give, when they give the same ones."""
    first, second = (m.get("classes") for m in metas)
    if isinstance(first, dict) and first and first == second:
        return {str(k): str(v) for k, v in first.items()}
    return None


def _save_differing(
    rows: list[dict[str, Any]], grid: Any, args: dict[str, Any]
) -> str | None:
    """Every differing window, in window order, to a file; its path, or ``None``.

    Named by a digest of its rows, so the same comparison writes the same file.
    A failure to write never fails the comparison.
    """
    body = json.dumps(rows, sort_keys=True)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:10]
    payload: dict[str, Any] = {
        "format": "olmoearth-agent/differing@1",
        "n_differing_total": len(rows),
        "listing_order": "every differing window, in window order (row-major, "
        "from row 0); not a sample",
        "grid": [int(grid[0]), int(grid[1])] if grid else None,
        "scores_path_a": args.get("scores_path_a"),
        "scores_path_b": args.get("scores_path_b"),
        "differing": rows,
    }
    try:
        return write_json_file(f"compare_differing_{digest}.json", payload)
    except (OSError, ValueError):
        logger.warning("could not save the differing windows", exc_info=True)
        return None


def _listing_order(n_listed: int, n_total: int, path: str | None) -> str:
    """What the inline listing is, where the rest is, and where to read patterns."""
    if not n_total:
        return "no window differs, so nothing is listed and no file is saved"
    where = (
        f"all {n_total:,} are in the file at differing_path, in the same order"
        if path
        else f"the full listing of {n_total:,} could not be saved"
    )
    return (
        f"the {n_listed} listed windows are the first differing windows in window "
        f"order (from row 0), not a sample of them; {where}. Read which classes "
        "change, in which direction and where from class_changes, class_pairs, "
        "spatial and facts, not from the list"
    )


async def _compare_review(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_compare_review``."""
    asked = args.get("grid")
    grid = [int(asked[0]), int(asked[1])] if asked else None
    sides = []
    side_windows: list[list[int] | None] = []
    metas: list[dict[str, Any]] = []
    # The grids the loaded files name themselves (never the grid the model
    # passed, which file_grid returns for a whole-grid file without one).
    own_grids: list[list[int]] = []
    loaded_any = False
    for key in ("a", "b"):
        scores = args.get(f"scores_{key}")
        windows: list[int] | None = None
        meta: dict[str, Any] = {}
        if scores is None and args.get(f"scores_path_{key}"):
            loaded = load_scores_file(str(args[f"scores_path_{key}"]))
            scores, windows, meta = loaded.scores, loaded.windows, loaded.meta
            loaded_any = True
            if loaded.grid is not None:
                own_grids.append([int(loaded.grid[0]), int(loaded.grid[1])])
        if scores is None:
            msg = f"pass 'scores_{key}' or 'scores_path_{key}' for both inferences"
            raise ValueError(msg)
        sides.append(scores)
        side_windows.append(windows)
        metas.append(meta)
    if side_windows[0] != side_windows[1]:
        msg = (
            "the two inferences cover different windows (their no-data windows "
            "differ); compare them on the windows both predicted"
        )
        raise ValueError(msg)
    windows = side_windows[0]
    # A scores file is placed on its own grid, never on the model's (exp87
    # review: a file that leaves windows out indexes its rows into its grid).
    grid_note: str | None = None
    if loaded_any:
        named = sorted({tuple(g) for g in own_grids})
        if len(named) > 1:
            msg = (
                "the two scores files name different grids ("
                + " and ".join(f"{r}x{c}" for r, c in named)
                + "); compare inferences of the same windows"
            )
            raise ValueError(msg)
        passed = grid
        if named:
            grid = list(named[0])
            if passed is not None and passed != grid:
                grid_note = (
                    f"the grid passed ({passed[0]}x{passed[1]}) was set aside for "
                    f"the scores files' own ({grid[0]}x{grid[1]})"
                )
        elif windows is not None:
            grid = None
            if passed is not None:
                grid_note = (
                    f"the grid passed ({passed[0]}x{passed[1]}) was not used: the "
                    "scores files leave windows out and name no grid to check it "
                    "against"
                )
    # Every differing window comes back; the inline listing is cut below and the
    # whole of it goes to a file.
    out = compare_scores(
        sides[0],
        sides[1],
        grid=(int(grid[0]), int(grid[1])) if grid else None,
        max_listed=None,
        windows=windows,
        class_names=_file_names(metas),
    )
    rows = out.pop("differing")
    # compare_scores' generic winner_without_labels gives way to the statistical
    # rules' reasons below, which name the dates and the labels' date.
    out["forbidden_claims"] = [
        c for c in out.get("forbidden_claims", []) if c not in COMPARISON_FORBIDDEN
    ]
    asked = int(args.get("max_listed", COMPARE_INLINE_LISTED))
    inline = max(0, min(COMPARE_INLINE_LISTED, asked))
    path = _save_differing(rows, grid, args) if rows else None
    out["differing"] = rows[:inline]
    out["n_differing_listed"] = len(out["differing"])
    out["n_differing_total"] = out["n_differing"]
    out["differing_path"] = path
    out["listing_order"] = _listing_order(out["n_differing_listed"], len(rows), path)
    if grid_note:
        out["grid_note"] = grid_note
    dates = _dates_block(args)
    out["dates"] = dates
    if dates.get("assessed"):
        if dates.get("status") in ("different_time", "overlapping_time"):
            # exp86 round 7 (brief 3, cluster): "run a third dated map" to
            # check the change; another unlabelled date cannot settle it. Round
            # 8: labels for "either date", or one reference "plus a date-matched
            # second inference" (brief 7 on files), cannot either.
            one = rules.one_reference_settles_two_dates(
                dates.get("a"), dates.get("b"), dates.get("labels")
            )
            rules.add_contract(out, forbidden_claims=[*rules.across_dates(dates), one])
            out["which_side_is_right"] = (
                "not graded: the maps describe different times, so a window where "
                "they differ may have changed on the ground, and each map needs a "
                "reference of its own date; "
                + (
                    f"labels dated {dates.get('labels')} grade only a map of that "
                    "date"
                    if dates.get("labels")
                    else "no labels_date was given, so no grading is possible even "
                    "with labels"
                )
            )
            add_must_state(out, [DIFFERENT_TIMES_MUST_STATE])
        elif dates.get("status") == "partly_stated":
            out["which_side_is_right"] = (
                "not graded: only one map's date was given, so an error cannot be "
                "told from a change on the ground"
            )
            add_must_state(out, [PARTLY_DATED_MUST_STATE])
            rules.add_contract(
                out,
                forbidden_claims=[rules.one_reference_settles_two_dates(partly=True)],
            )
    # No labels are taken here, whatever the dates; kept once when the dates
    # block has already listed it with the labels' date.
    winner = rules.winner_without_labels(rules.MORE_CONFIDENT_IS_NOT_RIGHT)
    # "Which is right" points to labelling, never of the low-confidence windows
    # only (exp86 round 8, brief 3 on Studio, run 3).
    rules.add_contract(
        out, forbidden_claims=[winner, rules.labelling_low_confidence_only()]
    )
    out["evidence_detail_path"] = evidence_detail_path()
    return out


# --------------------------------------------------------------------------- a Studio result as review windows


@dataclass
class SampledScores:
    """A Studio result sampled on a grid and turned into two-class window scores."""

    result_id: str
    property_name: str | None
    score_kind: str
    assumption: str
    threshold: float
    declared_range: tuple[float, float] | None
    grid: int
    windows: list[int]
    values: list[float]
    scores: list[list[float]]
    centres: list[tuple[float, float]]
    n_sampled: int
    n_nodata: int
    n_failed: int
    model: dict[str, Any] | None = None
    #: The side the caller asked for (``None``: the default was used).
    grid_requested: int | None = None

    def grid_block(self) -> dict[str, Any]:
        """The grid used, the grid asked for, and whether it was held to the range."""
        capped = self.grid_requested is not None and self.grid_requested != self.grid
        block: dict[str, Any] = {
            "grid": f"{self.grid}x{self.grid}",
            "grid_requested": self.grid_requested,
            "grid_capped": capped,
        }
        if capped:
            verb = "capped at" if self.grid == FROM_RESULT_MAX_GRID else "raised to"
            block["grid_note"] = (
                f"grid {self.grid_requested} was {verb} {self.grid}: a Studio "
                f"result is sampled on {FROM_RESULT_GRID_RANGE} points a side "
                "(each point is one pixel-value call), so the grid used is "
                f"{self.grid}x{self.grid} = {self.grid * self.grid} points"
            )
        return block

    def sampling_block(self) -> dict[str, Any]:
        """What was sampled, stated so a grid of points is not read as every pixel."""
        return {
            **self.grid_block(),
            "n_windows_sampled": self.n_sampled,
            "n_valid": len(self.windows),
            "n_nodata_dropped": self.n_nodata,
            "n_failed_dropped": self.n_failed,
            "what_a_window_is": (
                f"a grid of sampled points, not every pixel: the result's extent "
                f"is cut into {self.grid} x {self.grid} cells and each window is "
                "ONE pixel-value sample at its cell centre, so a window stands for "
                "its cell only as far as that pixel does"
            ),
            "nodata_rule": _NODATA_RULE,
        }

    def to_file(self) -> dict[str, Any]:
        """The scores-file payload (coordinates stay in the file, never in chat)."""
        return {
            "format": "olmoearth-agent/scores@1",
            "result_id": self.result_id,
            "property_name": self.property_name,
            "score_kind": self.score_kind,
            "threshold": self.threshold,
            "declared_range": (
                list(self.declared_range) if self.declared_range else None
            ),
            "assumption": self.assumption,
            "grid": [self.grid, self.grid],
            "windows": self.windows,
            "scores": self.scores,
            "values": self.values,
            "centres_lon_lat": [list(c) for c in self.centres],
        }


def _refusal(
    result_id: str, prop: str | None, reason: str, **extra: Any
) -> dict[str, Any]:
    """A ranking the tool declines to make, with the reason and where to go."""
    out: dict[str, Any] = {
        "ranked": False,
        "result_id": result_id,
        "property_name": prop,
        "reason": reason,
    }
    out.update(extra)
    return out


_CLASSIFICATION_REFUSAL = (
    "this result's band is a classification: Studio returns the hard class "
    "only, and no margin can be recovered from a hard class, so there is no "
    "ranking to make from one result. With two or more results of the same "
    "area, use olmoearth_compare_results with mode='ensemble' (disagreement "
    "between them); "
    "with per-class scores from wherever inference ran, use olmoearth_review_set."
)


async def sample_result_scores(
    ctx: ToolContext,
    result_id: str,
    *,
    grid: int,
    property_name: str | None = None,
    threshold: float | None = None,
    grid_requested: int | None = None,
) -> SampledScores | dict[str, Any]:
    """Sample one Studio result on a grid and build per-window two-class scores.

    ``grid`` is the side used (see :func:`result_grid`); ``grid_requested``,
    the side asked for, is carried so the result can say when they differ.
    Returns :class:`SampledScores`, or a refusal dict (``ranked: False``) when
    the band cannot be ranked: a classification band, a regression band that
    is not a ``[0, 1]`` score and no ``threshold``, a missing extent, or fewer
    than two valid windows. The band type is read from the result's declared
    outputs before sampling, so a refusal usually costs no pixel-value call.
    """
    record = await ctx.studio.get_prediction_result(result_id)
    bbox = result_bbox(record)
    if bbox is None:
        return _refusal(
            result_id, property_name, "the result has no extent (missing bounds)"
        )
    nodata = await result_nodata_context(ctx, record)
    name = default_property(record, property_name) or property_name
    decl = nodata.declared.get(str(name)) if name else None
    if decl and decl.get("value_type") == "classification":
        return _refusal(
            result_id,
            name,
            _CLASSIFICATION_REFUSAL,
            use_instead="olmoearth_compare_results",
        )
    rng: tuple[float, float] | None = None
    if decl and decl.get("value_type") == "regression":
        rng = declared_range(None, decl)
        if not (rng == (0.0, 1.0)) and threshold is None:
            return _refusal(
                result_id,
                name,
                f"this regression band declares the range {list(rng) if rng else 'none'}, "
                "not [0, 1]: a margin is a distance from a decision, so ranking it "
                "needs a decision threshold. Pass threshold (the value at which "
                "the map's decision flips) to rank by distance from it.",
                declared_range=list(rng) if rng else None,
            )

    cells = grid_windows(bbox, grid)
    sem = asyncio.Semaphore(SAMPLE_CONCURRENCY)
    records = await sample_records(
        ctx, result_id, [(lo, la) for _r, _c, lo, la in cells], sem
    )
    reads = [read_sample(r, name, nodata) for r in records]

    if decl is None:
        # Nothing declared: decide from the first usable sampled band.
        first = next(
            (rec for rec, read in zip(records, reads) if read[2] == OK and rec), None
        )
        if first is None:
            return _refusal(result_id, name, "no grid point returned a usable value")
        band = select_band(first, name) or {}
        name = band.get("property_name") or name
        if band.get("classification") is not None:
            return _refusal(
                result_id,
                name,
                _CLASSIFICATION_REFUSAL,
                use_instead="olmoearth_compare_results",
            )
        rng = declared_range(band)
        if rng != (0.0, 1.0) and threshold is None:
            return _refusal(
                result_id,
                name,
                f"this regression band declares the range {list(rng) if rng else 'none'}, "
                "not [0, 1]: ranking it needs a decision threshold; pass threshold.",
                declared_range=list(rng) if rng else None,
            )

    windows: list[int] = []
    values: list[float] = []
    centres: list[tuple[float, float]] = []
    n_nodata = n_failed = 0
    for (row, col, lon, lat), (value, categorical, status) in zip(cells, reads):
        if status == OK and not categorical and isinstance(value, (int, float)):
            windows.append(row * grid + col)
            values.append(float(value))
            centres.append((lon, lat))
        elif status == FAILED:
            n_failed += 1
        elif status == NODATA or status == OK:
            n_nodata += 1
    if len(windows) < 2:
        return _refusal(
            result_id,
            name,
            f"only {len(windows)} of {len(cells)} sampled windows hold a value "
            f"({n_nodata} no-data, {n_failed} failed); a ranking needs at least two",
        )
    lo, hi = rng if rng else (None, None)
    rows, kind, assumption = regression_scores(
        values, min_value=lo, max_value=hi, threshold=threshold
    )
    return SampledScores(
        result_id=result_id,
        property_name=name,
        score_kind=kind,
        assumption=assumption,
        threshold=0.5 if kind == "binary_score" else float(threshold or 0.0),
        declared_range=rng,
        grid=grid,
        windows=windows,
        values=values,
        scores=rows,
        centres=centres,
        n_sampled=len(cells),
        n_nodata=n_nodata,
        n_failed=n_failed,
        model=nodata.model,
        grid_requested=grid_requested,
    )


def _budgets(raw: Any) -> list[float]:
    """Validated review budgets, ascending and de-duplicated."""
    budgets = sorted({float(b) for b in (raw or DEFAULT_BUDGETS)})
    bad = [b for b in budgets if not 0.0 < b <= 1.0]
    if bad:
        msg = f"every budget must be in (0, 1], got {bad}"
        raise ValueError(msg)
    return budgets


def _refused_regression(refusal: dict[str, Any]) -> dict[str, Any]:
    """A refused regression band with no threshold, with the claims it forbids.

    A refusal that names a ``declared_range`` is a regression band that is not
    a ``[0, 1]`` score and came with no threshold: it has no margins to review
    by and no error rate (exp86 round 8, brief 3 on Studio: the most ambiguous
    windows of KarstNumber, declared 0.2 to 1.2, offered for review). Any other
    refusal is returned as it is.
    """
    if "declared_range" not in refusal:
        return refusal
    rng = refusal.get("declared_range")
    band = (
        refusal.get("property_name"),
        (float(rng[0]), float(rng[1])) if isinstance(rng, list) else None,
    )
    return rules.add_contract(
        refusal,
        forbidden_claims=[
            rules.review_set_for_unthresholded_regression([band]),
            rules.unthresholded_regression([band]),
        ],
    )


async def _review_set_from_result(
    args: dict[str, Any], ctx: ToolContext
) -> dict[str, Any]:
    """Handler for ``olmoearth_review_set_from_result``."""
    result_id = str(args["result_id"])
    grid, grid_requested = result_grid(args.get("grid"))
    budgets = _budgets(args.get("budgets"))
    threshold = float(args["threshold"]) if args.get("threshold") is not None else None
    error_rate = (
        float(args["error_rate"]) if args.get("error_rate") is not None else None
    )
    max_listed = int(args.get("max_listed", DEFAULT_MAX_LISTED))

    sampled = await sample_result_scores(
        ctx,
        result_id,
        grid=grid,
        property_name=args.get("property_name"),
        threshold=threshold,
        grid_requested=grid_requested,
    )
    if isinstance(sampled, dict):
        return _refused_regression(sampled)

    ranked = review_set(
        sampled.scores,
        budget=budgets[-1],
        error_rate=None,
        max_listed=max_listed,
        score_kind=sampled.score_kind,
        keep_full=True,
    )
    full = ranked.pop("review_full")
    n_valid = len(sampled.windows)
    margins_sorted = sorted(abs(r[1] - r[0]) for r in sampled.scores)
    per_budget = []
    for budget in budgets:
        k = max(1, int(round(budget * n_valid)))
        entry: dict[str, Any] = {
            "budget": budget,
            "n_review": k,
            "realised_budget": round(k / n_valid, 6),
            "margin_cut": round(margins_sorted[min(k, n_valid) - 1], 6),
        }
        if error_rate is not None:
            entry["attainable_ceiling"] = round(
                attainable_ceiling(budget, error_rate), 6
            )
        per_budget.append(entry)
    review = ranked["review"]
    for row in [*review, *full]:
        local = int(row["window_index"])
        row["score"] = round(sampled.values[local], 6)
    _place_rows(review, "window_index", sampled.windows, grid)
    _place_rows(full, "window_index", sampled.windows, grid)

    out: dict[str, Any] = {
        "ranked": True,
        "result_id": result_id,
        "property_name": sampled.property_name,
        "declared_range": (
            list(sampled.declared_range) if sampled.declared_range else None
        ),
        "score_kind": sampled.score_kind,
        "threshold": sampled.threshold,
        "assumption": sampled.assumption,
        "sampling": sampled.sampling_block(),
        "signal": (
            "margin = distance from the decision (lower = more suspect, opened "
            "first); the most confident windows come last, never first"
        ),
        "class_meaning": {
            "0": f"at or below the threshold {sampled.threshold:g}",
            "1": f"above the threshold {sampled.threshold:g}",
        },
        "budgets": per_budget,
        "ceiling": {
            "formula": "min(1, budget / error_rate)",
            "error_rate": error_rate,
            "note": (
                "No ranking can catch more than this share of the errors at a "
                "budget; judge a realised capture against it, not against 1.0."
                + (
                    ""
                    if error_rate is not None
                    else " Pass error_rate (known or estimated) for the number."
                )
            ),
        },
        "review": review,
        "n_review_listed": len(review),
        # The largest budget's review set, whole, when the listing is cut short.
        **review_list(len(review), full, budgets[-1]),
        "margin_summary": ranked["margin_summary"],
        "evidence_scope": ranked["evidence_scope"],
        "evidence_covers_this_case": ranked["evidence_covers_this_case"],
        "evidence_detail_path": evidence_detail_path(),
        "caveats": [
            "Budgets are fractions of the valid sampled windows, not of the map's "
            "pixels.",
            # No upstream figure: no recorded experiment covers a Studio band
            # (exp86 round 8 carried such figures to cases they exclude).
            "This review set is chosen to hold errors; it is not a sample, and "
            "its error rate overstates the map's. To say how wrong the map is, "
            "use olmoearth_plan_label_sample.",
        ],
        "facts": ranked["facts"],
        "must_state": ranked["must_state"],
        "forbidden_claims": ranked["forbidden_claims"],
    }
    if sampled.model:
        out["model"] = {k: v for k, v in sampled.model.items() if k != "nodata_value"}
    if bool(args.get("save_scores", True)):
        name = f"scores_{slug(result_id, 12)}_{slug(str(sampled.property_name))}_g{grid}.json"
        out["scores_path"] = write_json_file(name, sampled.to_file())
        out["next_step"] = (
            "To estimate how wrong this map is, pass scores_path to "
            "olmoearth_plan_label_sample (it chooses which windows to label); "
            "the file also holds each window's location for the reviewer."
        )
    return out


async def _grade_review_rule(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_grade_review_rule``."""
    budgets = args.get("budgets") or list(DEFAULT_BUDGETS)
    return grade_rule(
        args["signal"],
        args["errors"],
        baseline=args.get("baseline"),
        control=args.get("control"),
        groups=args.get("groups"),
        budgets=[float(b) for b in budgets],
        higher_is_suspect=bool(args.get("higher_is_suspect", True)),
    )


async def _budget_ceiling(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_review_budget_ceiling``."""
    budget = float(args["budget"])
    error_rate = float(args["error_rate"])
    ceiling = attainable_ceiling(budget, error_rate)
    out: dict[str, Any] = {
        "budget": budget,
        "error_rate": error_rate,
        "attainable_ceiling": round(ceiling, 6),
        "random_baseline": round(budget, 6),
        "formula": "min(1, budget / error_rate)",
        "note": (
            "A reviewer opening this fraction of windows cannot see more "
            "than this share of the errors, however good the ranking is. "
            "Reading a realised capture against 1.0 instead of against this "
            "ceiling is the most common way a working ranker gets dismissed."
        ),
    }
    observed = args.get("observed_capture")
    if observed is not None:
        obs = float(observed)
        out["observed_capture"] = obs
        out["share_of_ceiling"] = round(obs / ceiling, 6) if ceiling > 0 else None
        out["multiple_of_random"] = round(obs / budget, 6) if budget > 0 else None
        if obs > ceiling + 1e-9:
            out["warning"] = (
                "The observed capture exceeds the attainable ceiling, which is "
                "impossible: the budget, the error rate, or the capture was "
                "measured on a different set of units."
            )
    return out


def build_review_set_tools() -> list[RegisteredTool]:
    """Return the ``olmoearth-review-set`` tool bundle."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_review_set",
                description=(
                    "Which windows a human should check first, without labels, "
                    "from per-class scores (logits or probabilities, inline or "
                    "a scores file) produced wherever inference ran: ranks by "
                    "the model's top-1-minus-top-2 margin, low margin first, at "
                    "your budget, optionally boundary-first. For a STUDIO "
                    "prediction result use olmoearth_review_set_from_result; "
                    "for a model run's raster, olmoearth_scores_from_file "
                    "first; "
                    "with only hard classes from 2+ Studio results, "
                    "olmoearth_compare_results (mode='ensemble'). The review "
                    "set is not a "
                    "sample: never divide its errors by its size (use "
                    "olmoearth_plan_label_sample). The result's "
                    "'evidence_scope' says in one sentence what was measured "
                    "and whether it covers this case. Ranks relative "
                    "suspicion, not a "
                    "calibrated error probability; not OOD detection (that is "
                    "olmoearth_area_of_applicability, skill "
                    "olmoearth-uncertainty). Window indices and ids only, never "
                    "coordinates."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "scores": _SCORES_SCHEMA,
                        "scores_path": {
                            "type": "string",
                            "description": (
                                "Instead of 'scores': path to a .json file holding "
                                "the rows, or an object {'grid': [rows, cols], "
                                "'scores': rows}. Must sit under "
                                "OLMOEARTH_SCORES_ROOT (e.g. from "
                                "olmoearth_scores_from_file). Use this when the "
                                "scores were written by an inference run; do not "
                                "retype them."
                            ),
                        },
                        "budget": {
                            "type": "number",
                            "default": 0.05,
                            "description": (
                                "Fraction of windows a reviewer can open, in "
                                "(0, 1]. 0.05 = the most suspect 5%."
                            ),
                        },
                        "ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional window labels parallel to 'scores'. "
                                "Do not put lat/lon here (rule §3.1)."
                            ),
                        },
                        "grid": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": (
                                "Optional [rows, cols]; windows are read "
                                "row-major. Enables the nine-level boundary "
                                "indicator and boundary-first ordering."
                            ),
                        },
                        "order": {
                            "type": "string",
                            "enum": ["confidence", "boundary_first"],
                            "default": "confidence",
                            "description": (
                                "'confidence' = ascending margin. "
                                "'boundary_first' = windows touching a "
                                "predicted class boundary first (needs grid)."
                            ),
                        },
                        "score_type": {
                            "type": "string",
                            "enum": ["logit", "probability"],
                            "description": (
                                "Auto-detected when omitted. The measured "
                                "evidence is for the logit margin."
                            ),
                        },
                        "error_rate": {
                            "type": "number",
                            "description": (
                                "Optional known/estimated error rate; adds the "
                                "attainable-ceiling arithmetic for this budget."
                            ),
                        },
                        "max_listed": {
                            "type": "integer",
                            "default": DEFAULT_MAX_LISTED,
                            "description": "Cap on inline review rows; a "
                            "review set listed short is saved whole to "
                            "review_list_path.",
                        },
                    },
                    "required": [],
                },
            ),
            handler=_review_set,
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_review_set_from_result",
                description=(
                    "Which windows of a STUDIO prediction result a reviewer "
                    "should check first (or where it is likely wrong); use it "
                    "instead of sampling pixel values by hand. Samples the band "
                    "on a grid over the result's extent (a window = one "
                    "pixel-value at a cell centre; no-data dropped and "
                    "counted), reads a [0, 1] score s as [1 - s, s] and ranks "
                    "by margin: s near 0.5 first, s near 0 or 1 last. Report "
                    "the returned 'assumption': the score is treated as "
                    "P(positive) for ranking only, not as a probability of "
                    "error. Other regression ranges need 'threshold'; a "
                    "classification band is refused. Windows are named by (row, "
                    "col) and index. Not a sample for an error rate: pass the "
                    "returned scores_path to olmoearth_plan_label_sample. Slow: "
                    "grid^2 pixel-value calls."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "result_id": {"type": "string"},
                        "grid": {
                            "type": "integer",
                            "default": FROM_RESULT_DEFAULT_GRID,
                            "minimum": FROM_RESULT_MIN_GRID,
                            "maximum": FROM_RESULT_MAX_GRID,
                            "description": f"N*N windows, {FROM_RESULT_GRID_RANGE}; "
                            "a larger N is capped (the result says so).",
                        },
                        "budgets": dict(
                            _SERIES,
                            description="Fractions of the valid windows.",
                        ),
                        "property_name": {
                            "type": "string",
                            "description": "Defaults to the first band.",
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Required unless the band is a "
                            "[0, 1] score.",
                        },
                        "error_rate": {
                            "type": "number",
                            "description": "Adds the attainable ceiling.",
                        },
                        "max_listed": {
                            "type": "integer",
                            "default": DEFAULT_MAX_LISTED,
                            "description": "Cap on inline review rows; a "
                            "review set listed short is saved whole to "
                            "review_list_path.",
                        },
                        "save_scores": {
                            "type": "boolean",
                            "default": True,
                            "description": "Save scores (and window "
                            "locations) to a file; returns scores_path.",
                        },
                    },
                    "required": ["result_id"],
                },
            ),
            handler=_review_set_from_result,
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_compare_review",
                description=(
                    "Compare TWO inferences of the SAME windows (another "
                    "sensor, date, encoder, before and after fine-tuning): how "
                    "many windows differ, what share, whether on class "
                    "boundaries, and each side's margin there. It does NOT say "
                    "which side is right: without labels that is not resolvable "
                    "(the result's 'evidence_scope' gives the measured reason), "
                    "so decline and say why. Class changes, where they "
                    "concentrate and which side is more confident come as "
                    "counts over every differing window; only the first few "
                    "are listed inline, the rest in a file. Pass "
                    "date_a/date_b (and labels_date) when the maps describe "
                    "dates: across dates a difference can be real change, and "
                    "the 'dates' block says so (needs the inferencex extra; "
                    "otherwise reported as not assessed). Scores inline or as "
                    ".json files under OLMOEARTH_SCORES_ROOT. Window indices "
                    "only."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "scores_a": _SCORES_SCHEMA,
                        "scores_b": _SCORES_SCHEMA,
                        "scores_path_a": {
                            "type": "string",
                            "description": "File for side A.",
                        },
                        "scores_path_b": {
                            "type": "string",
                            "description": "File for side B.",
                        },
                        "grid": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "Optional [rows, cols]; enables the boundary reading.",
                        },
                        "max_listed": {
                            "type": "integer",
                            "default": COMPARE_INLINE_LISTED,
                            "description": "Cap on differing windows listed "
                            f"inline (at most {COMPARE_INLINE_LISTED}; the file "
                            "at differing_path holds all).",
                        },
                        "date_a": {
                            "type": "string",
                            "description": "The date or period map A describes: "
                            f"{DATE_FORMS} (a map of 2023 is "
                            "2023-01-01/2023-12-31).",
                        },
                        "date_b": {
                            "type": "string",
                            "description": "The date or period map B describes: "
                            f"{DATE_FORMS}.",
                        },
                        "labels_date": {
                            "type": "string",
                            "description": "The date or period any labels "
                            f"describe: {DATE_FORMS}; required before grading "
                            "maps of different dates.",
                        },
                    },
                    "required": [],
                },
            ),
            handler=_compare_review,
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_grade_review_rule",
                description=(
                    "Measure whether a CANDIDATE suspicion signal is actually "
                    "better than the model's own confidence at finding errors, "
                    "on units where you DO have labels. Give the candidate "
                    "signal, a 0/1 error indicator, optionally the baseline "
                    "(negated margin) and a no-model control, and optionally "
                    "group ids (scene / event / region). Returns E-AURC "
                    "(lower is better, 0 = oracle), AUROC, error capture at "
                    "each budget with the attainable ceiling, a head-to-head "
                    "verdict, and a one-sided exact sign test across groups -- "
                    "the group is the unit of replication because units inside "
                    "one scene are not independent. Use this before adopting "
                    "ANY new reliability heuristic: upstream this same "
                    "machinery rejected ensemble disagreement, cross-model "
                    "disagreement, two-view disagreement and feature-space "
                    "typicality, all of which sounded principled. A candidate "
                    "that cannot beat the no-model control is measuring the "
                    "scene, not the model. Read-only."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "signal": dict(
                            _SERIES,
                            description="Candidate suspicion score per graded unit.",
                        ),
                        "errors": dict(
                            _SERIES,
                            description=(
                                "0/1 per unit; 1 = the model was wrong there."
                            ),
                        ),
                        "baseline": dict(
                            _SERIES,
                            description=(
                                "The incumbent, normally the NEGATED margin so "
                                "higher means more suspect."
                            ),
                        ),
                        "control": dict(
                            _SERIES,
                            description=(
                                "A no-model control (band index, pixel "
                                "variance, class rarity)."
                            ),
                        ),
                        "groups": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional group id per unit (scene / event / "
                                "region) for the sign test."
                            ),
                        },
                        "budgets": dict(
                            _SERIES,
                            description="Review budgets; default 0.01/0.05/0.10.",
                        ),
                        "higher_is_suspect": {
                            "type": "boolean",
                            "default": True,
                            "description": (
                                "False if a LOW candidate value means suspect."
                            ),
                        },
                    },
                    "required": ["signal", "errors"],
                },
            ),
            handler=_grade_review_rule,
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_review_budget_ceiling",
                description=(
                    "The ceiling arithmetic for a review budget: no ranking "
                    "rule can catch more than min(1, budget/error_rate) of the "
                    "errors when a reviewer only opens 'budget' of the "
                    "windows. Call this whenever anyone quotes an error-capture "
                    "number, including one from another tool or paper: pass "
                    "observed_capture to get the share of the ceiling actually "
                    "reached and the multiple of random review. Judging a "
                    "capture against 1.0 rather than against its ceiling is "
                    "the standard way a working ranker gets dismissed. Pure "
                    "arithmetic; no data leaves the process."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "budget": {
                            "type": "number",
                            "description": "Fraction of windows reviewed, (0, 1].",
                        },
                        "error_rate": {
                            "type": "number",
                            "description": "Fraction of windows that are wrong, [0, 1].",
                        },
                        "observed_capture": {
                            "type": "number",
                            "description": (
                                "Optional realised share of errors captured, to "
                                "score against the ceiling."
                            ),
                        },
                    },
                    "required": ["budget", "error_rate"],
                },
            ),
            handler=_budget_ceiling,
        ),
    ]
