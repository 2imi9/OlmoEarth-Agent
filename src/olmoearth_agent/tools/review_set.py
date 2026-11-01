# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-review-set`` tool bundle (skill #18).

Four tools:

- ``olmoearth_review_set`` -- rank windows by the model's own top-1 minus
  top-2 margin and return the ones a reviewer should open first at a budget.
- ``olmoearth_review_set_from_result`` -- the same ranking for a Studio
  prediction result: sample its band on a grid, read a binary score in
  ``[0, 1]`` as ``[1 - s, s]``, or a class map's top-k scores as its rank-1
  and rank-2 scores ``[p1, p2]`` (stating either assumption), and rank.
- ``olmoearth_compare_review`` -- compare two inferences of the same windows:
  how much they differ and where, with the side question declined on evidence
  (and, with the ``inferencex`` extra, what a difference means at the maps'
  dates).
- ``olmoearth_review_budget_ceiling`` -- the arithmetic that stops a good
  ranker being talked down: no rule can catch more than
  ``min(1, budget / error_rate)`` of the errors at a given budget.

Bring-your-own-scores: Studio's pixel-value
returns only ``raw_value`` and ``classification`` per band, never logits. A
margin cannot be recovered from a hard class; it can be read from a Studio
regression band that is a binary score in ``[0, 1]`` (decided at 0.5), from
a class map with top-k scores (a segmentation prediction submitted with
``probability_output``, whose result carries each rank's class and softmax
score as bands of their own: the margin is the rank-1 score minus the
rank-2), both of which ``olmoearth_review_set_from_result`` reads, or from
any band once a decision threshold is named. When all you have is hard
classes from two or more Studio results, ``olmoearth_compare_results`` with
``mode='ensemble'`` (skill #9's disagreement signal) is the tool that still
works. A model run's own raster (every pixel's scores, e.g. from a GPU
cluster) goes through ``olmoearth_scores_from_file``
(:mod:`olmoearth_agent.tools.scores_file`), whose scores file these tools
read.

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
    STUDIO_WINDOW_CONFIDENCE,
    TOPK_SCORE,
    WINDOW_CONFIDENCE,
    attainable_ceiling,
    compare_scores,
    evidence_detail,
    regression_scores,
    review_set,
    topk_assumption,
)
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.security.paths import safe_path, workspace_root
from olmoearth_agent.tools import inferencex
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.registry import Capability, RegisteredTool, ToolContext
from olmoearth_agent.tools.sampling import (
    FAILED,
    NODATA,
    OK,
    OUTSIDE,
    OUTSIDE_REASON,
    SAMPLE_CONCURRENCY,
    TOPK_NODATA_RULE,
    UNSCORED,
    ResultNodata,
    TopKScores,
    default_property,
    dropped_text,
    read_sample,
    read_topk,
    result_nodata_context,
    sample_records,
    select_band,
    topk_part_reason,
    topk_quantized,
    topk_range_refusal,
    topk_scores,
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


def _file_margin(meta: dict[str, Any], n_rows: int) -> list[float] | None:
    """A scores file's own per-row margins (``margin``), checked, or ``None``.

    A class map's top-k scores file carries each window's margin beside its
    row ``[p1, p2]``: for Studio's uint8 scores it is the levels' difference
    over 254, which the row's gap misses by a level where a rank-2 score is
    raw 0 (:func:`~olmoearth_agent.tools.sampling.read_topk`).
    """
    margin = meta.get("margin")
    if margin is None:
        return None
    if (
        not isinstance(margin, list)
        or len(margin) != n_rows
        or not all(
            isinstance(m, (int, float)) and not isinstance(m, bool) for m in margin
        )
    ):
        msg = f"the scores file's margin does not have one number per row ({n_rows})"
        raise ValueError(msg)
    return [float(m) for m in margin]


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


#: The ``kind`` of a scores file's ``studio`` block pooled from a Studio class
#: map's top-k scores (``tools/studio_scores.TOPK``; that module imports this
#: one, so the value is repeated here).
STUDIO_TOPK = "top_k"


def file_studio_kind(meta: dict[str, Any]) -> str | None:
    """The kind of Studio result a scores file was pooled from, or ``None``.

    ``olmoearth_scores_from_file`` copies the ``studio`` block of a manifest
    that ``olmoearth_scores_from_studio`` wrote (``source.kind``
    ``studio_files``) into its scores file (``scores_file.studio_notes``):
    its ``kind`` is :data:`STUDIO_TOPK` for a class map's top-k softmax
    scores, ``"unit_score"`` for a ``[0, 1]`` score. Any other scores file
    has no such block; a block with no kind gives ``""`` (Studio's files, of
    a kind not stated), and so does a file without the block whose
    ``provenance`` is ``scores_file.STUDIO_PROVENANCE`` (a manifest whose
    ``studio`` value was no block): it is still a Studio model's scores,
    never "not an OlmoEarth model". ``meta`` may be a population's source,
    which keeps the block and the provenance under the same keys.
    """
    studio = meta.get("studio")
    if isinstance(studio, dict):
        kind = studio.get("kind")
        return kind if isinstance(kind, str) else ""
    # scores_file imports this module, so its constant is read at call time.
    from olmoearth_agent.tools import scores_file

    return "" if meta.get("provenance") == scores_file.STUDIO_PROVENANCE else None


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


#: Classes a ``review_set_classes`` sentence names one by one; the rest are
#: counted together.
CLASSES_NAMED = 10

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
    # The note names the file, not the key that holds its path: exp86 round 9
    # (B7/files run 1) told the user the windows were "listed in
    # `differing_path` above", a field the user never sees.
    where = (
        f"the full list of {total:,}, in the same order, is in the CSV "
        f"{os.path.basename(path)} ({', '.join(REVIEW_LIST_COLUMNS)} and the "
        "class or score; no coordinates)"
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


def list_file_fact(
    path: str | None, sentence: str, **fields: Any
) -> dict[str, Any] | None:
    """The ``list_file`` fact: a sentence for the user naming a list's file.

    A comparison or a review set that writes its windows to a file returns
    the path under a key (``differing_path``, ``review_list_path``) the user
    never sees: exp86 round 9 (B7/files run 1) answered "All 1,570 differing
    windows are listed in `differing_path` above", naming the field for the
    file and pointing at a list the answer did not show. The fact's sentence
    names the file by its full path and says what it holds, so an answer
    points to the file. ``None`` when no file was written.
    """
    if not path:
        return None
    return rules.fact("list_file", sentence, path=path, **fields)


def review_list_fact(out: dict[str, Any], budget: float) -> dict[str, Any] | None:
    """The ``list_file`` fact of a review set whose full list went to a CSV."""
    path = out.get("review_list_path")
    if not path:
        return None
    rows = int(out.get("review_list_rows") or 0)
    return list_file_fact(
        path,
        f"The {rows:,} windows of the review set at budget {budget:g} are listed, "
        f"in review order, in {path}.",
        rows=rows,
        order="review order",
    )


def review_set_classes_fact(
    full: list[dict[str, Any]],
    n_listed: int,
    *,
    meaning: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """The ``review_set_classes`` fact: every review-set window by predicted class.

    exp86 round 9 (B8/cluster run 2) said "montane_forest vs woodland_forest
    pairs dominate the list" of an 819-window review set from the 10 windows
    the tool listed (8 of them montane or woodland); over all 819 the
    predicted classes are grassland_barren 256 and shrubland_savanna 246
    before woodland_forest 125 and montane_forest 84. The counts and shares
    here are over every window of the review set, listed or not, so which
    classes it holds most of comes from code. A class is named by the scores
    file's class name, else by ``meaning`` (a Studio score's side of its
    threshold), else by its number; past :data:`CLASSES_NAMED` classes the
    rest are counted together.
    """
    total = len(full)
    if not total:
        return None
    counts: dict[Any, int] = {}
    names: dict[Any, str] = {}
    for row in full:
        k = row.get("predicted_class")
        counts[k] = counts.get(k, 0) + 1
        if row.get("class_name") and k not in names:
            names[k] = str(row["class_name"])
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))

    def label(k: Any) -> str:
        if k in names:
            return names[k]
        if meaning and str(k) in meaning:
            return f"class {k} ({meaning[str(k)]})"
        return f"class {k}"

    shown = ranked[:CLASSES_NAMED]
    parts = [f"{n:,} ({rules.percent(n / total)}) {label(k)}" for k, n in shown]
    parts[0] = parts[0].replace(") ", ") are predicted ", 1)
    rest = ranked[CLASSES_NAMED:]
    if rest:
        n_rest = sum(n for _k, n in rest)
        parts.append(
            f"{n_rest:,} ({rules.percent(n_rest / total)}) in {len(rest)} "
            "other classes"
        )
    listing = (
        "every one of them is listed"
        if n_listed >= total
        else f"these counts cover all {total:,}, not only the {n_listed:,} listed"
    )
    joined = (
        parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f" and {parts[-1]}"
    )
    return rules.fact(
        "review_set_classes",
        f"Of the {total:,} windows in the review set, {joined}; {listing}.",
        n=total,
        n_listed=min(n_listed, total),
        classes=[
            {
                "class": k,
                "name": names.get(k),
                "n": n,
                "share": round(n / total, 6),
            }
            for k, n in ranked
        ],
    )


#: The certified zone a class map with top-k scores can have, named with its
#: review set: a random plan, which makes one possible, not certain.
TOPK_ZONE_STEP = (
    f"For a certified zone of this {rules.TOPK_MAP_KIND}, plan design='random', "
    "fixed before labelling. " + rules.POSSIBLE_NOT_CERTAIN
)


def topk_contract(out: dict[str, Any], top_k: Any) -> None:
    """The contract of a review set of a class map with top-k scores, in place.

    Its scores are Studio's uncalibrated softmax scores, quantized unless
    ``top_k`` (the field read, as the scores file keeps it) says otherwise:
    ``must_state`` says their margin ranks windows and is no probability
    that a class is right. Like a scores map's results, it forbids
    ``certification_guaranteed``: its random plan can certify a zone, which
    is how the harness tells it from a class map without top-k scores
    (``harness.checks._scores_map_in_run``).
    """
    quantized = not (isinstance(top_k, dict) and top_k.get("quantized") is False)
    rules.add_contract(
        out,
        must_state=[rules.topk_scores_must_state(quantized)],
        forbidden_claims=[
            rules.certification_guaranteed(
                "this review set certifies nothing; a random plan of this "
                f"{rules.TOPK_MAP_KIND} makes a certified zone possible, and "
                "olmoearth_certify_zone may certify none"
            )
        ],
    )


async def _review_set(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_review_set``.

    A scores file's rows are not always scores whose type can be detected: a
    class map's top-k rows ``[p1, p2]`` and a file of window confidences
    (``score_kind`` ``window_confidence``, every ``olmoearth_scores_from_file``
    output: each row the window's confidence at its class and 0 elsewhere)
    are an encoding, read as the file's ``score_type`` says, so neither
    reports ``score_type_detected`` (live3, 31 October 2026, showed "logit"
    beside "probability" for rows ``[m, 0]``). A file of window confidences
    pooled from a Studio result's own files (its ``studio`` block or Studio
    provenance, :func:`file_studio_kind`) is scoped as a Studio model's
    scores, which no recorded experiment grades, and one of a class map's
    top-k scores states that its window scores pool Studio's uncalibrated
    softmax scores (``rules.STUDIO_WINDOW_SCORES_MUST_STATE``; a pooled mean
    is not on the 1/254 grid ``olmoearth_scores_from_studio`` states of the
    pixels' scores).
    """
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
    # A class map's top-k scores file keeps each window's margin and rank-1
    # class beside its rows [p1, p2] (olmoearth_review_set_from_result writes
    # them): the rows' arg-max is no class, and their gap misses the margin by
    # a level where a rank-2 score is raw 0.
    topk_file = meta.get("score_kind") == TOPK_SCORE
    score_kind = (
        meta.get("score_kind") if isinstance(meta.get("score_kind"), str) else None
    )
    window_file = score_kind == WINDOW_CONFIDENCE
    studio_kind = file_studio_kind(meta)
    if window_file and studio_kind is not None:
        # A Studio model's scores, pooled from its result's own files: Studio
        # fine-tunes OlmoEarth models, so the scope never says "not an
        # OlmoEarth model" of its model id (live3), and no experiment covers it.
        score_kind = STUDIO_WINDOW_CONFIDENCE
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
        score_kind=score_kind,
        model=file_model(meta),
        keep_full=True,
        map_class=_file_classes(meta, len(scores)) if topk_file else None,
        margin=_file_margin(meta, len(scores)) if topk_file else None,
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
        # A class map's top-k scores, saved by olmoearth_review_set_from_result,
        # keep that ranking's contract (its scores' must_state, and the zone a
        # random plan of it can certify). Their rows are probabilities, as the
        # file says, never detected (p1 + p2 is below 1), and the map's
        # classes are the file's, not the rows' two.
        if topk_file:
            topk_contract(out, meta.get("top_k"))
            out.pop("score_type_detected", None)
            if isinstance(meta.get("classes"), dict):
                out["n_classes"] = len(meta["classes"])
        # Rows [m, 0, ...] encode a window's confidence and class: their type
        # is the file's score_type, and detection would read them as logits.
        if window_file:
            out.pop("score_type_detected", None)
        # A Studio class map's top-k scores, pooled from its own files: what
        # the window scores pool, after the scope's sentence.
        if window_file and studio_kind == STUDIO_TOPK:
            add_must_state(out, [rules.STUDIO_WINDOW_SCORES_MUST_STATE])
    if grid_note:
        out["grid_note"] = grid_note
    # Row-major index to (row, col), so a caller with a grid need not do the division itself.
    cols = int(grid[1]) if grid else None
    _place_rows(out.get("review", []), "window_index", windows, cols)
    _place_rows(full, "window_index", windows, cols)
    out["evidence_detail_path"] = evidence_detail_path()
    out.update(review_list(len(out.get("review", [])), full, budget))
    rules.add_contract(
        out,
        facts=[
            f
            for f in (
                review_set_classes_fact(full, len(out.get("review", []))),
                review_list_fact(out, budget),
            )
            if f
        ],
    )
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
    was given). Without it, or with a release older than
    ``inferencex.MIN_VERSION``: a statement that the dates were not assessed,
    with the line that installs or upgrades it. A bare year or month is
    refused first, with the interval that means it.
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
        _status, install = inferencex.unavailable()
        return {
            "assessed": False,
            "a": date_a,
            "b": date_b,
            "labels": labels_date,
            "reason": "the dates were not assessed: reading what a difference "
            "means across dates needs olmoearth-inferencex "
            f"(>= {inferencex.MIN_VERSION})",
            "install": install,
        }
    reading: dict[str, Any] = dict(compare.dates_reading(date_a, date_b, labels_date))
    reading["assessed"] = True
    return reading


#: Stated with a comparison of maps of different or overlapping times: the
#: dates sentence merged with what labels for one date can grade (exp86 round
#: 8, brief 3 on the cluster and brief 7 on files).
DIFFERENT_TIMES_MUST_STATE = rules.DATED_MAPS_MUST_STATE

#: The next step of a comparison of maps of different or overlapping times,
#: and the only one that applies: each map's own sample, labelled against a
#: reference of that map's own date (agent trial 12 offered a disagreement
#: design and another unlabelled map instead).
EACH_MAP_ITS_OWN_SAMPLE = (
    "Measure each map on its own: olmoearth_plan_label_sample on one map's "
    "scores alone, its windows labelled against a reference of that map's own "
    "date, then olmoearth_estimate_map_error with that plan's design_path; the "
    "same for the other map at its date."
)

#: Stated with a comparison where only one map's date was given.
PARTLY_DATED_MUST_STATE = (
    "Only one map's date was given, so a difference cannot be told from a change "
    "on the ground."
)


def file_row_order(
    meta: dict[str, Any], cols: int | None, windows: list[int] | None, n_rows: int
) -> str | None:
    """How a scores file's grid rows run on the ground, or ``None`` without a georeference.

    A file with each window's centre (``centres_lon_lat``, one per row, as
    ``olmoearth_scores_from_file`` writes for a georeferenced raster and
    ``olmoearth_review_set_from_result`` for a Studio result) is read: row 0
    is north when the first grid row's centres lie north of the last row's.
    A file with a ``transform`` (rasterio's affine order, ``a, b, c, d, e,
    f``) runs north to south when ``e`` is negative. Anything else has no
    georeference: exp86 round 8 (brief 7 on files) called row band 0 of 400
    chips stacked in dataset order "the northmost band".
    """
    centres = meta.get("centres_lon_lat")
    if cols and isinstance(centres, list) and len(centres) == n_rows and n_rows:
        by_row: dict[int, list[float]] = {}
        try:
            for i, centre in enumerate(centres):
                index = windows[i] if windows is not None else i
                by_row.setdefault(int(index) // cols, []).append(float(centre[1]))
        except (IndexError, TypeError, ValueError):
            return None
        first, last = by_row[min(by_row)], by_row[max(by_row)]
        north_first = sum(first) / len(first) - sum(last) / len(last)
        if north_first > 0:
            return "north_to_south"
        if north_first < 0:
            return "south_to_north"
        return None
    transform = meta.get("transform")
    if isinstance(transform, (list, tuple)) and len(transform) >= 6:
        e = transform[4]
        if isinstance(e, (int, float)) and not isinstance(e, bool) and e:
            return "north_to_south" if e < 0 else "south_to_north"
    return None


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
    """What the inline listing is, where the rest is, and where to read patterns.

    The file is named, not the key that holds its path (exp86 round 9,
    B7/files run 1: "listed in `differing_path` above").
    """
    if not n_total:
        return "no window differs, so nothing is listed and no file is saved"
    where = (
        f"all {n_total:,} are in the file {os.path.basename(path)}, in the same "
        "order"
        if path
        else f"the full listing of {n_total:,} could not be saved"
    )
    return (
        f"the {n_listed} listed windows are the first differing windows in window "
        f"order (from row 0), not a sample of them; {where}. Read which classes "
        "change, in which direction and where from class_changes, class_pairs, "
        "spatial and facts, not from the list. "
        # exp86 round 8 (brief 8) named the evidence file as holding a list.
        f"{EVIDENCE_FILE} (evidence_detail_path) holds evidence text only, no "
        "windows"
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
            if meta.get("score_kind") == TOPK_SCORE:
                # Its rows are the rank-1 and rank-2 scores, whose arg-max is
                # no window's class, and this comparison reads classes from rows.
                msg = (
                    f"scores_path_{key} holds a {rules.TOPK_MAP_KIND}'s rank-1 "
                    "and rank-2 scores, not one score per class, so its rows do "
                    "not say each window's class; compare two Studio results "
                    "with olmoearth_compare_results"
                )
                raise ValueError(msg)
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
    # North only for georeferenced scores: inline rows and a file with no
    # window locations or transform have rows that need not run north to south.
    cols = int(grid[1]) if grid else None
    orders = {
        file_row_order(meta, cols, windows, len(rows))
        for meta, rows in zip(metas, sides)
        if meta
    } - {None}
    # Every differing window comes back; the inline listing is cut below and the
    # whole of it goes to a file.
    out = compare_scores(
        sides[0],
        sides[1],
        grid=(int(grid[0]), int(grid[1])) if grid else None,
        max_listed=None,
        windows=windows,
        class_names=_file_names(metas),
        row_order=orders.pop() if len(orders) == 1 else None,
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
    pointer = list_file_fact(
        path,
        f"All {len(rows):,} differing windows are listed, in window order, in "
        f"{path}, each with both maps' class and margin.",
        rows=len(rows),
        order="window order",
    )
    if pointer:
        rules.add_contract(out, facts=[pointer])
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
            # Agent trial 12: a disagreement design offered across dates
            # (B3/cluster run 5, B7/files run 11), and another unlabelled map or
            # "a same-date third inference" to tell change from error (B7/files
            # runs 7 and 12). Only each map's own sample applies.
            out["not_for_these_maps"] = rules.NOT_FOR_MAPS_OF_DIFFERENT_TIMES
            out["next_steps"] = [EACH_MAP_ITS_OWN_SAMPLE]
            rules.add_contract(
                out,
                forbidden_claims=[
                    rules.two_map_design_across_dates(dates.get("a"), dates.get("b"))
                ],
            )
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


def grid_block(grid: int, grid_requested: int | None) -> dict[str, Any]:
    """The grid a Studio result was sampled on, the grid asked for, and any hold.

    Parameters
    ----------
    grid
        The side used (see :func:`result_grid`).
    grid_requested
        The side asked for, or ``None`` when the default was used.

    Returns
    -------
    dict
        ``grid`` (``"NxN"``), ``grid_requested``, ``grid_capped`` and, when
        the side was held to the range, ``grid_note``.
    """
    capped = grid_requested is not None and grid_requested != grid
    block: dict[str, Any] = {
        "grid": f"{grid}x{grid}",
        "grid_requested": grid_requested,
        "grid_capped": capped,
    }
    if capped:
        verb = "capped at" if grid == FROM_RESULT_MAX_GRID else "raised to"
        block["grid_note"] = (
            f"grid {grid_requested} was {verb} {grid}: a Studio "
            f"result is sampled on {FROM_RESULT_GRID_RANGE} points a side "
            "(each point is one pixel-value call), so the grid used is "
            f"{grid}x{grid} = {grid * grid} points"
        )
    return block


def sampling_block(
    grid: int,
    grid_requested: int | None,
    *,
    n_sampled: int,
    n_valid: int,
    n_nodata: int,
    n_failed: int,
    n_outside: int = 0,
    nodata_rule: str = _NODATA_RULE,
) -> dict[str, Any]:
    """What was sampled of a Studio result: a grid of points, never every pixel.

    Shared by the review set of a result (:class:`SampledScores`) and the
    labelled sample of a Studio class map
    (:mod:`olmoearth_agent.tools.estimation`).

    Parameters
    ----------
    grid, grid_requested
        As :func:`grid_block` takes them.
    n_sampled, n_valid, n_nodata, n_failed
        The windows sampled, those holding a usable value, and those dropped
        as no-data or as a failed pixel-value call (any failure but a 404
        beside answered points).
    n_outside
        The windows dropped because Studio answered their pixel-value call
        with HTTP 404 ``not_found_error`` while it answered other points of
        the result (:func:`~olmoearth_agent.tools.sampling.sample_records`),
        read as points where no file is served. Probe 5 (31 October 2026)
        got 60 such 404s on the kelp result's 16 x 16 grid, the same count
        the live run had counted as failed; it did not test which file covers
        which point, so the note states the 404, not that the windows lie
        outside the result's files.
    nodata_rule
        How a sample is judged no-data, as the calling tool applies it.

    Returns
    -------
    dict
        :func:`grid_block`'s keys, the counts (``n_outside_dropped`` among
        them), ``what_a_window_is``, ``nodata_rule`` and, when a window was
        answered with a 404, ``outside_note``.
    """
    block: dict[str, Any] = {
        **grid_block(grid, grid_requested),
        "n_windows_sampled": n_sampled,
        "n_valid": n_valid,
        "n_nodata_dropped": n_nodata,
        "n_failed_dropped": n_failed,
        "n_outside_dropped": n_outside,
        "what_a_window_is": (
            f"a grid of sampled points, not every pixel: the result's extent "
            f"is cut into {grid} x {grid} cells and each window is "
            "ONE pixel-value sample at its cell centre, so a window stands for "
            "its cell only as far as that pixel does"
        ),
        "nodata_rule": nodata_rule,
    }
    if n_outside:
        block["outside_note"] = (
            f"{n_outside} of the {n_sampled} windows were {OUTSIDE_REASON}: they "
            "are dropped and counted apart from failed calls (n_failed_dropped); "
            "which of the result's files covers which point was not tested"
        )
    return block


@dataclass
class SampledScores:
    """A Studio result sampled on a grid and turned into two-class window scores.

    A class map with top-k scores (``score_kind`` :data:`TOPK_SCORE`) has
    rows ``[p1, p2]``, its rank-1 and rank-2 scores, ``values`` its ``p1``,
    no threshold and no declared range of its own; ``margin`` is each
    window's margin as :func:`~olmoearth_agent.tools.sampling.read_topk`
    reads it (for Studio's uint8 scores, the levels' difference over 254,
    which ``p1 - p2`` misses by a level where the rank-2 score is raw 0),
    ``map_class`` codes each window's rank-1 class into ``classes`` (the
    classes as the map writes them) and ``top_k`` names the field read
    (:meth:`~olmoearth_agent.tools.sampling.TopKScores.describe`, with
    ``quantized``); ``threshold_note`` says when a threshold passed was set
    aside. Every other result leaves those six ``None``.
    """

    result_id: str
    property_name: str | None
    score_kind: str
    assumption: str
    threshold: float | None
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
    p1: list[float] | None = None
    map_class: list[int] | None = None
    classes: list[Any] | None = None
    top_k: dict[str, Any] | None = None
    margin: list[float] | None = None
    threshold_note: str | None = None
    #: Windows Studio answered with HTTP 404 where it answered others
    #: (``sampling.OUTSIDE``); ``n_failed`` counts every other failed call.
    n_outside: int = 0

    def grid_block(self) -> dict[str, Any]:
        """The grid used, the grid asked for, and whether it was held to the range."""
        return grid_block(self.grid, self.grid_requested)

    def sampling_block(self) -> dict[str, Any]:
        """What was sampled, stated so a grid of points is not read as every pixel."""
        return sampling_block(
            self.grid,
            self.grid_requested,
            n_sampled=self.n_sampled,
            n_valid=len(self.windows),
            n_nodata=self.n_nodata,
            n_failed=self.n_failed,
            n_outside=self.n_outside,
            nodata_rule=(
                TOPK_NODATA_RULE if self.score_kind == TOPK_SCORE else _NODATA_RULE
            ),
        )

    def to_file(self) -> dict[str, Any]:
        """The scores-file payload (coordinates stay in the file, never in chat).

        A class map's top-k scores also carry what their rows cannot say, as
        ``olmoearth_scores_from_file`` writes it: ``score_type`` (the rows are
        probabilities, whatever their sum), each window's ``p1``, ``margin``
        and class (``map_class``), the classes' names (``classes``), and
        ``top_k``.
        """
        payload: dict[str, Any] = {
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
        if self.map_class is not None:
            payload["score_type"] = "probability"
            payload["p1"] = self.p1
            payload["margin"] = self.margin
            payload["map_class"] = self.map_class
            payload["classes"] = {
                str(c): str(v) for c, v in enumerate(self.classes or [])
            }
            payload["top_k"] = self.top_k
        return payload


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


#: The rankings that remain for a classification band, whose hard class has no
#: margin: disagreement between results, or scores from wherever inference ran.
_CLASSIFICATION_ROUTES = (
    "With two or more results of the same area, olmoearth_compare_results "
    "with mode='ensemble' ranks windows by the disagreement between them; with "
    "per-class scores from wherever inference ran, olmoearth_review_set ranks "
    "them by margin."
)


#: The key of a classification refusal under which :func:`sample_result_scores`
#: hands the grid it already sampled to a caller that asked for it
#: (``keep_class_reads``); the caller pops it, and no result carries it.
CLASS_READS = "_class_reads"


@dataclass
class ClassReads:
    """A classification band's grid samples, kept so they are not sampled twice.

    A result that declares no band type is found to be a classification only
    after its grid is sampled; the labelled sample of a Studio class map
    (:mod:`olmoearth_agent.tools.estimation`) reads its classes from these
    reads instead of sampling the grid again. ``reads[i]`` is window ``i``'s
    ``(value, categorical, status)`` (row-major, as ``cells``), read through
    the result's no-data context; ``model`` is the model behind the result.
    """

    property_name: str | None
    cells: list[tuple[int, int, float, float]]
    reads: list[tuple[Any, bool, str]]
    model: dict[str, Any] | None


def _classification_refusal(result_id: str, name: str | None) -> dict[str, Any]:
    """A classification band refused: the class-map sentence and the routes left.

    Studio returns the band's class only, so no margin can be recovered and
    nothing ranks or certifies its windows; the reason ends with where
    top-k scores come from (``rules.CLASS_MAP_REFUSAL``). ``class_map`` marks
    the refusal for :mod:`olmoearth_agent.tools.estimation`, whose random and
    class designs measure such a map from labels instead. A class map with
    top-k scores is never refused: :func:`sample_result_scores` reads it
    first.
    """
    return _refusal(
        result_id,
        name,
        rules.CLASS_MAP_REFUSAL,
        use_instead="olmoearth_compare_results",
        other_routes=_CLASSIFICATION_ROUTES,
        class_map=True,
    )


async def _sample_topk(
    ctx: ToolContext,
    result_id: str,
    topk: TopKScores,
    nodata: ResultNodata,
    bbox: list[float],
    *,
    grid: int,
    grid_requested: int | None,
    threshold: float | None = None,
) -> SampledScores | dict[str, Any]:
    """A class map with top-k scores on a grid: rows ``[p1, p2]`` and each class.

    One pixel-value call per window returns every band, and
    :func:`~olmoearth_agent.tools.sampling.read_topk` reads its rank-1
    class, its rank-1 and rank-2 scores and their margin, each band by its
    own name. The margin is kept beside the rows (``margin``: for Studio's
    uint8 scores, the levels' difference over 254), ``p1`` is the window's
    top-1 probability (the confidence design reads it) and its class is the
    rank-1 class, coded in the declared classes' order (``map_class``,
    ``classes``). A score range other than ``[0, 254]`` or ``[0, 1]`` is
    refused, before any pixel is sampled when the result declares it, and so
    is a window whose rank-2 score is above its rank-1 score. A window that
    holds a class but no score refuses the sampling: a score never makes a
    window no-data, and leaving it out would drop a valid window of the map
    (a score of 1.0 is stored as the files' no-data value). A ``threshold``
    plays no part, and the result says it was set aside.
    """
    reason = topk_range_refusal(topk, nodata.declared)
    if reason is not None:
        return _refusal(result_id, topk.base, reason, top_k=topk.describe())
    cells = grid_windows(bbox, grid)
    sem = asyncio.Semaphore(SAMPLE_CONCURRENCY)
    records = await sample_records(
        ctx, result_id, [(lo, la) for _r, _c, lo, la in cells], sem
    )
    try:
        reads = [read_topk(r, topk, nodata) for r in records]
    except ValueError as exc:
        return _refusal(result_id, topk.base, str(exc), top_k=topk.describe())
    unscored = sum(1 for _value, status in reads if status == UNSCORED)
    if unscored:
        scores = " or ".join(repr(n) for n in topk.score_properties[:2])
        return _refusal(
            result_id,
            topk.base,
            f"{unscored} of {len(cells)} sampled windows hold a rank-1 class but "
            f"no number in {scores}: a score never makes a window no-data, and "
            "leaving these windows out would drop valid windows of the map, so no "
            "margin is read",
            top_k=topk.describe(),
        )
    windows: list[int] = []
    centres: list[tuple[float, float]] = []
    read_classes: list[Any] = []
    rows: list[list[float]] = []
    margin: list[float] = []
    n_nodata = n_failed = n_outside = 0
    for (row, col, lon, lat), (value, status) in zip(cells, reads):
        if status == OK and value is not None:
            windows.append(row * grid + col)
            centres.append((lon, lat))
            read_classes.append(value[0])
            rows.append([value[1], value[2]])
            margin.append(value[3])
        elif status == FAILED:
            n_failed += 1
        elif status == OUTSIDE:
            n_outside += 1
        else:
            n_nodata += 1
    if len(windows) < 2:
        return _refusal(
            result_id,
            topk.base,
            f"only {len(windows)} of {len(cells)} sampled windows hold a value "
            f"({dropped_text(n_nodata, n_failed, n_outside)}); a ranking needs at "
            "least two",
        )
    codes, classes = topk.code_classes(read_classes)
    quantized = topk_quantized(topk, nodata, records)
    p1 = [r[0] for r in rows]
    return SampledScores(
        result_id=result_id,
        property_name=topk.base,
        score_kind=TOPK_SCORE,
        assumption=topk_assumption(topk.k, quantized=quantized),
        threshold=None,
        declared_range=None,
        grid=grid,
        windows=windows,
        values=p1,
        scores=rows,
        centres=centres,
        n_sampled=len(cells),
        n_nodata=n_nodata,
        n_failed=n_failed,
        n_outside=n_outside,
        model=nodata.model,
        grid_requested=grid_requested,
        p1=p1,
        map_class=codes,
        classes=classes,
        top_k={**topk.describe(), "quantized": quantized},
        margin=margin,
        threshold_note=(
            f"the threshold passed ({threshold:g}) was set aside: a "
            f"{rules.TOPK_MAP_KIND} is ranked by its margin, which needs none"
            if threshold is not None
            else None
        ),
    )


async def sample_result_scores(
    ctx: ToolContext,
    result_id: str,
    *,
    grid: int,
    property_name: str | None = None,
    threshold: float | None = None,
    grid_requested: int | None = None,
    keep_class_reads: bool = False,
) -> SampledScores | dict[str, Any]:
    """Sample one Studio result on a grid and build per-window two-class scores.

    ``grid`` is the side used (see :func:`result_grid`); ``grid_requested``,
    the side asked for, is carried so the result can say when they differ.
    A class map with top-k scores
    (:func:`~olmoearth_agent.tools.sampling.topk_scores`) is read before
    either class refusal, as rows ``[p1, p2]`` with each window's margin and
    rank-1 class beside them (:func:`_sample_topk`; ``threshold`` plays no
    part, and ``threshold_note`` says so), and a ``property_name`` naming
    one of its ranks' scores or classes is refused. Returns
    :class:`SampledScores`, or a refusal dict (``ranked: False``) when the
    band cannot be ranked: a classification band without top-k scores
    (marked ``class_map``:
    :data:`~olmoearth_agent.tools.statistical_rules.CLASS_MAP_REFUSAL` is
    its reason), a regression band that is not a ``[0, 1]`` score and no
    ``threshold``, a top-k score of a range other than ``[0, 254]`` or
    ``[0, 1]``, a top-k window with a class but no score or with a rank-2
    score above its rank-1, a missing extent, or fewer than two valid
    windows. The band type is read from the result's declared outputs
    before sampling, so a refusal usually costs no pixel-value call. A
    result that declares no band type is sampled first; with
    ``keep_class_reads``, a classification band found that way is refused
    with the grid's reads under :data:`CLASS_READS` (a :class:`ClassReads`),
    which the caller pops.
    """
    record = await ctx.studio.get_prediction_result(result_id)
    bbox = result_bbox(record)
    if bbox is None:
        return _refusal(
            result_id, property_name, "the result has no extent (missing bounds)"
        )
    nodata = await result_nodata_context(ctx, record)
    topk = topk_scores(record, property_name)
    if topk is not None:
        if topk.reads_a_part(property_name):
            return _refusal(
                result_id,
                property_name,
                topk_part_reason(str(property_name), topk),
                top_k=topk.describe(),
            )
        return await _sample_topk(
            ctx,
            result_id,
            topk,
            nodata,
            bbox,
            grid=grid,
            grid_requested=grid_requested,
            threshold=threshold,
        )
    name = default_property(record, property_name) or property_name
    decl = nodata.declared.get(str(name)) if name else None
    if decl and decl.get("value_type") == "classification":
        return _classification_refusal(result_id, name)
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
            refusal = _classification_refusal(result_id, name)
            if keep_class_reads:
                # The grid is sampled: a class map's labelled sample reads
                # its classes from here instead of sampling it again.
                refusal[CLASS_READS] = ClassReads(name, cells, reads, nodata.model)
            return refusal
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
    n_nodata = n_failed = n_outside = 0
    for (row, col, lon, lat), (value, categorical, status) in zip(cells, reads):
        if status == OK and not categorical and isinstance(value, (int, float)):
            windows.append(row * grid + col)
            values.append(float(value))
            centres.append((lon, lat))
        elif status == FAILED:
            n_failed += 1
        elif status == OUTSIDE:
            n_outside += 1
        elif status == NODATA or status == OK:
            n_nodata += 1
    if len(windows) < 2:
        return _refusal(
            result_id,
            name,
            f"only {len(windows)} of {len(cells)} sampled windows hold a value "
            f"({dropped_text(n_nodata, n_failed, n_outside)}); a ranking needs at "
            "least two",
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
        n_outside=n_outside,
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
    windows of KarstNumber, declared 0.2 to 1.2, offered for review). A
    classification band's refusal (``class_map``) forbids a certified zone of
    it (:func:`~olmoearth_agent.tools.statistical_rules.zone_for_a_class_map`).
    Any other refusal is returned as it is.
    """
    if refusal.get("class_map"):
        return rules.add_contract(
            refusal, forbidden_claims=[rules.zone_for_a_class_map()]
        )
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

    # A class map's top-k rows are probabilities whatever their sum (below 1
    # with more than two classes, and by the quantization), their arg-max is
    # never the window's class, and their margin is read from the bands'
    # levels: all three are passed, never detected.
    topk = sampled.map_class is not None
    ranked = review_set(
        sampled.scores,
        budget=budgets[-1],
        error_rate=None,
        max_listed=max_listed,
        score_type="probability" if topk else None,
        score_kind=sampled.score_kind,
        keep_full=True,
        map_class=sampled.map_class,
        margin=sampled.margin,
    )
    full = ranked.pop("review_full")
    n_valid = len(sampled.windows)
    margins_sorted = sorted(
        sampled.margin
        if sampled.margin is not None
        else [abs(r[1] - r[0]) for r in sampled.scores]
    )
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
    names = sampled.classes or []
    for row in [*review, *full]:
        local = int(row["window_index"])
        row["score"] = round(sampled.values[local], 6)
        if topk and 0 <= int(row["predicted_class"]) < len(names):
            row["class_name"] = str(names[int(row["predicted_class"])])
    _place_rows(review, "window_index", sampled.windows, grid)
    _place_rows(full, "window_index", sampled.windows, grid)
    # A class map's classes are its own (``classes`` below), not two sides of
    # a threshold.
    meaning: dict[str, Any] = {}
    if not topk:
        cut = float(sampled.threshold or 0.0)
        meaning["class_meaning"] = {
            "0": f"at or below the threshold {cut:g}",
            "1": f"above the threshold {cut:g}",
        }

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
            "margin = the rank-1 score minus the rank-2 score (lower = more "
            "suspect, opened first); score is the rank-1 score; the most "
            "confident windows come last, never first"
            if topk
            else "margin = distance from the decision (lower = more suspect, "
            "opened first); the most confident windows come last, never first"
        ),
        **meaning,
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
    if topk:
        # The map's classes as it writes them, and the field read.
        out["classes"] = {str(c): v for c, v in enumerate(names)}
        out["top_k"] = sampled.top_k
        topk_contract(out, sampled.top_k)
    if sampled.threshold_note:
        out["threshold_note"] = sampled.threshold_note
    rules.add_contract(
        out,
        facts=[
            f
            for f in (
                review_set_classes_fact(
                    full, len(review), meaning=out.get("class_meaning")
                ),
                review_list_fact(out, budgets[-1]),
                # exp86 round 9 (B3/studio run 1): "no ground-truth labels
                # exist", of models fine-tuned on the project's labels
                rules.labels_in_studio_fact([sampled.model] if sampled.model else []),
            )
            if f
        ],
    )
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
        if topk:
            out["next_step"] += f" {TOPK_ZONE_STEP}"
    return out


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
                    "calibrated error probability; not OOD detection. Window "
                    "indices and ids only, never coordinates."
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
            capability=Capability(
                does="rank windows for review by the model's top-1 minus top-2 "
                "margin, from per-class scores inline or in a scores file; a "
                "review set is not a sample",
                needs=(
                    "2 or more class scores per window (a hard class has no margin)",
                ),
            ),
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
                    "error. A class map with top-k scores ranks by rank-1 minus "
                    "rank-2 score. Other regression ranges need 'threshold'; a "
                    "class map without top-k scores is refused. Windows are "
                    "named by (row, col) and index. Not a sample for an error "
                    "rate: pass the returned scores_path to "
                    "olmoearth_plan_label_sample. Slow: grid^2 pixel-value calls."
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
                            "description": "Required for a regression band "
                            "not in [0, 1].",
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
            capability=Capability(
                does="rank a Studio result's sampled windows for review by "
                "distance from its decision or its top-k margin",
                needs=("threshold, for a regression band not in [0, 1]",),
                cannot=("rank a class map with no top-k scores (no margin)",),
            ),
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
            capability=Capability(
                does="how two inferences of the same windows differ (counts, "
                "class changes, boundaries, each side's margin)",
                needs=("both inferences over the same windows",),
                cannot=(
                    "say which map is right, even with labels_date (it takes no "
                    "labels)",
                ),
            ),
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
            capability=Capability(
                does="the most errors any ranking can catch at a review budget, "
                "min(1, budget / error_rate)"
            ),
        ),
    ]
