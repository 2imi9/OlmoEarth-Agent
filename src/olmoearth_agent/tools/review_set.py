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
more Studio results, skill #9's ``olmoearth_ensemble_uncertainty`` is the tool
that still works.

No coordinates in, no coordinates out (rule §3.1).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from olmoearth_agent.analysis.raster_compare import (
    declared_range,
    grid_windows,
    result_bbox,
)
from olmoearth_agent.analysis.review_set import (
    DEFAULT_BUDGETS,
    DEFAULT_MAX_LISTED,
    attainable_ceiling,
    compare_scores,
    grade_rule,
    regression_scores,
    review_set,
)
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.security.paths import safe_path, workspace_root
from olmoearth_agent.tools import inferencex
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

#: Env var naming the directory scores files are read from and written to.
SCORES_ROOT_ENV = "OLMOEARTH_SCORES_ROOT"

#: Grid bounds for sampling a Studio result into review windows: every window
#: is one live pixel-value call, so the grid is capped (16 x 16 = 256 calls).
FROM_RESULT_DEFAULT_GRID = 10
FROM_RESULT_MAX_GRID = 16

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


async def _review_set(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_review_set``."""
    grid = args.get("grid")
    scores = args.get("scores")
    windows: list[int] | None = None
    if scores is None and args.get("scores_path"):
        # Scores usually live in a file where inference ran; a model cannot relay
        # hundreds of rows inline, and asking it to would test transcription.
        loaded = load_scores_file(str(args["scores_path"]))
        scores, windows = loaded.scores, loaded.windows
        if grid is None and loaded.grid is not None:
            grid = list(loaded.grid)
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
    out = review_set(
        scores,
        budget=float(args.get("budget", 0.05)),
        ids=args.get("ids"),
        # A grid whose no-data windows were left out has no full neighbourhood.
        grid=(int(grid[0]), int(grid[1])) if grid and windows is None else None,
        order=order,
        score_type=args.get("score_type"),
        error_rate=(
            float(args["error_rate"]) if args.get("error_rate") is not None else None
        ),
        max_listed=int(args.get("max_listed", DEFAULT_MAX_LISTED)),
    )
    # Row-major index to (row, col), so a caller with a grid need not do the division itself.
    _place_rows(
        out.get("review", []), "window_index", windows, int(grid[1]) if grid else None
    )
    return out


def _dates_block(args: dict[str, Any]) -> dict[str, Any]:
    """The package's reading of what a difference means at the maps' dates.

    With the ``inferencex`` extra: ``oe_inferencex.compare.dates_reading`` of
    ``date_a``, ``date_b`` and ``labels_date`` (``status`` "unstated" when none
    was given). Without it: a statement that the dates were not assessed.
    """
    date_a, date_b = args.get("date_a"), args.get("date_b")
    labels_date = args.get("labels_date")
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


async def _compare_review(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_compare_review``."""
    grid = args.get("grid")
    sides = []
    side_windows: list[list[int] | None] = []
    for key in ("a", "b"):
        scores = args.get(f"scores_{key}")
        windows: list[int] | None = None
        if scores is None and args.get(f"scores_path_{key}"):
            loaded = load_scores_file(str(args[f"scores_path_{key}"]))
            scores, windows = loaded.scores, loaded.windows
            if grid is None and loaded.grid is not None:
                grid = list(loaded.grid)
        if scores is None:
            msg = f"pass 'scores_{key}' or 'scores_path_{key}' for both inferences"
            raise ValueError(msg)
        sides.append(scores)
        side_windows.append(windows)
    if side_windows[0] != side_windows[1]:
        msg = (
            "the two inferences cover different windows (their no-data windows "
            "differ); compare them on the windows both predicted"
        )
        raise ValueError(msg)
    windows = side_windows[0]
    out = compare_scores(
        sides[0],
        sides[1],
        grid=(int(grid[0]), int(grid[1])) if grid and windows is None else None,
        max_listed=int(args.get("max_listed", DEFAULT_MAX_LISTED)),
    )
    if windows is not None:
        for row in out.get("differing", []):
            row.pop("row", None)
            row.pop("col", None)
        _place_rows(
            out.get("differing", []),
            "window_index",
            windows,
            int(grid[1]) if grid else None,
        )
    dates = _dates_block(args)
    out["dates"] = dates
    if dates.get("assessed"):
        if dates.get("status") in ("different_time", "overlapping_time"):
            out["which_side_is_right"] = (
                "not graded: the maps describe different times, so a window where "
                "they differ may have changed on the ground; "
                + (
                    "labels would be graded against their own date "
                    f"({dates.get('labels')}) only"
                    if dates.get("labels")
                    else "no labels_date was given, so no grading is possible even "
                    "with labels"
                )
            )
        elif dates.get("status") == "partly_stated":
            out["which_side_is_right"] = (
                "not graded: only one map's date was given, so an error cannot be "
                "told from a change on the ground"
            )
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

    def sampling_block(self) -> dict[str, Any]:
        """What was sampled, stated so a grid of points is not read as every pixel."""
        return {
            "grid": f"{self.grid}x{self.grid}",
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
    "area, use olmoearth_ensemble_uncertainty (disagreement between them); "
    "with per-class scores from wherever inference ran, use olmoearth_review_set."
)


async def sample_result_scores(
    ctx: ToolContext,
    result_id: str,
    *,
    grid: int,
    property_name: str | None = None,
    threshold: float | None = None,
) -> SampledScores | dict[str, Any]:
    """Sample one Studio result on a grid and build per-window two-class scores.

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
            use_instead="olmoearth_ensemble_uncertainty",
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
                use_instead="olmoearth_ensemble_uncertainty",
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
    )


def _budgets(raw: Any) -> list[float]:
    """Validated review budgets, ascending and de-duplicated."""
    budgets = sorted({float(b) for b in (raw or DEFAULT_BUDGETS)})
    bad = [b for b in budgets if not 0.0 < b <= 1.0]
    if bad:
        msg = f"every budget must be in (0, 1], got {bad}"
        raise ValueError(msg)
    return budgets


async def _review_set_from_result(
    args: dict[str, Any], ctx: ToolContext
) -> dict[str, Any]:
    """Handler for ``olmoearth_review_set_from_result``."""
    result_id = str(args["result_id"])
    grid = max(
        2, min(FROM_RESULT_MAX_GRID, int(args.get("grid", FROM_RESULT_DEFAULT_GRID)))
    )
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
    )
    if isinstance(sampled, dict):
        return sampled

    ranked = review_set(
        sampled.scores, budget=budgets[-1], error_rate=None, max_listed=max_listed
    )
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
    for row in review:
        local = int(row["window_index"])
        row["score"] = round(sampled.values[local], 6)
    _place_rows(review, "window_index", sampled.windows, grid)

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
        "margin_summary": ranked["margin_summary"],
        "evidence": ranked["evidence"],
        "caveats": list(ranked["caveats"])
        + [
            "Budgets are fractions of the valid sampled windows, not of the map's "
            "pixels.",
            "This review set is chosen to hold errors; it is not a sample. Its "
            "error rate overstates the map's (1.8 to 5.8 times upstream, exp78). "
            "To say how wrong the map is, use olmoearth_plan_label_sample.",
        ],
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
                    "with only hard classes from 2+ Studio results, "
                    "olmoearth_ensemble_uncertainty. The review set is not a "
                    "sample: never divide its errors by its size (use "
                    "olmoearth_plan_label_sample). Measured: the margin beat "
                    "the best no-model control on all 24 tasks of Ai2's "
                    "embedding suite; the result's 'evidence' and 'caveats' "
                    "blocks give the rest. Ranks relative suspicion, not a "
                    "calibrated error probability; not OOD detection (pair with "
                    "olmoearth_area_of_applicability). Window indices and ids "
                    "only, never coordinates."
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
                                "OLMOEARTH_SCORES_ROOT. Use this when the scores "
                                "were written by an inference run; do not retype "
                                "them."
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
                            "description": "Cap on inline review rows.",
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
                            "description": f"N*N windows, 2-{FROM_RESULT_MAX_GRID}.",
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
                            "description": "Cap on inline review rows.",
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
                    "(the more confident side is right on only 51-70% of "
                    "differing windows), so decline and say why. Pass "
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
                            "default": DEFAULT_MAX_LISTED,
                            "description": "Cap on listed differing windows.",
                        },
                        "date_a": {
                            "type": "string",
                            "description": "ISO date or period (start/end) map A "
                            "describes.",
                        },
                        "date_b": {
                            "type": "string",
                            "description": "ISO date or period map B describes.",
                        },
                        "labels_date": {
                            "type": "string",
                            "description": "ISO date or period any labels "
                            "describe; required before grading maps of "
                            "different dates.",
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
                    "capture of 0.45 against 1.0 rather than against a ceiling "
                    "of 0.50 is the standard way a working ranker gets "
                    "dismissed. Pure arithmetic; no data leaves the process."
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
