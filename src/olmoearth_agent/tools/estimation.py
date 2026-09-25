# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""How wrong is the map? Design-based estimation (skill #18, second half).

The review set answers *which windows to open first*; it cannot say how wrong
the map is, because it is chosen to hold errors: labelling it and dividing
overstates the error rate (1.8 to 5.8 times the true rate on the companion
package's exp78). An error rate needs a labelled **sample drawn by a design**,
and its interval is the one that design earns -- a stratified or targeted draw
does not get the simple-random-sample formula. Three tools wrap
``oe_inferencex.estimate`` (``olmoearth-inferencex`` >= 1.3.0, the optional
``inferencex`` extra, imported lazily):

- ``olmoearth_plan_label_sample`` -- choose which windows to label
  (``sample_for_estimation``; the confidence design by default, ``random``
  when a certified zone is wanted) and save the design to a file;
- ``olmoearth_estimate_map_error`` -- the error rate with the design's
  interval and ``method`` (``estimate_error_rate``), per-class accuracy with
  reference classes (``estimate_per_class``), or, for windows labelled with no
  design, ``estimate_from_indices``, which refuses a review set;
- ``olmoearth_certify_zone`` -- the largest most-confident share of the map
  whose error rate is at most ``alpha`` (``certify_zone``), random designs
  only.

Windows are addressed by index and grid ``(row, col)`` (rule §3.1); the
locations a reviewer needs stay in the files.

Every output also carries ``next_steps``, written by code from the design and
the outcome, and the output contract's ``facts``, ``must_state`` and
``forbidden_claims`` (:mod:`olmoearth_agent.tools.statistical_rules`): exp86's
answers proposed a looser alpha or another rule after nothing certified,
offered to certify from a stratified design, and worked out 69/300 by hand.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

from olmoearth_agent.analysis.review_set import (
    DEFAULT_MAX_LISTED,
    detect_score_type,
    margins,
    predicted_classes,
)
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools import inferencex
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext
from olmoearth_agent.tools.review_set import (
    FROM_RESULT_DEFAULT_GRID,
    FROM_RESULT_GRID_RANGE,
    FROM_RESULT_MAX_GRID,
    SampledScores,
    load_json_file,
    load_scores_file,
    read_roots,
    result_grid,
    sample_result_scores,
    slug,
    write_json_file,
    write_root,
)

#: Designs offered (the package's tile design needs tile ids the agent lacks).
DESIGNS = ("confidence", "proportional", "random")

#: Rules offered for the certified zone (the package's "plugin" has no guarantee).
ZONE_RULES = ("prefix", "bonferroni")

#: Format tag of a design file written by olmoearth_plan_label_sample.
DESIGN_FORMAT = "olmoearth-agent/label-design@1"

#: Stated with every certify_zone output: exp86 round 2's model recommended a
#: confidence design for certification, which the tool refuses.
DESIGN_REQUIREMENT = (
    "Certification needs design='random' (a simple random sample of the map); "
    "olmoearth_certify_zone refuses a 'confidence' or 'proportional' design, so "
    "recommend only a random design for a certified zone."
)

#: Stated with every plan and estimate, because the trial's model broke both.
INTERVAL_RULES = (
    "A targeted review set (e.g. the windows olmoearth_review_set lists) is "
    "not a sample: its error rate overstates the map's, 1.8 to 5.8 times on "
    "the package's exp78.",
    "The simple-random-sample interval p +/- 1.96 sqrt(p(1 - p)/n) does not "
    "apply to a stratified or targeted design, nor to a pool of both; report "
    "the interval and method olmoearth_estimate_map_error returns instead.",
)

_SCORES_SCHEMA = {
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
    "description": "Per-window scores, one row per window, one number per class.",
}


def _jsonable(value: Any) -> Any:
    """Numpy arrays/scalars to lists/numbers, NaN to None, recursively."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):  # numpy array or scalar, without importing numpy
        return _jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _grid_pair(value: Any) -> tuple[int, int] | None:
    """A grid given as N (N x N) or [rows, cols], or ``None``."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _softmax_top(row: list[float]) -> float:
    """The largest softmax probability of a logit row."""
    top = max(row)
    exps = [math.exp(v - top) for v in row]
    return 1.0 / sum(exps)


@dataclass
class Population:
    """The windows a design is drawn from, in one flattened index space.

    ``margin`` and ``p1`` are NaN, and ``map_class`` is -1, where a window has
    no value (no-data); the package leaves such windows out of the population.
    """

    margin: list[float]
    p1: list[float]
    map_class: list[int]
    grid: tuple[int, int] | None
    score_kind: str
    source: dict[str, Any] = field(default_factory=dict)
    centres: dict[int, tuple[float, float]] = field(default_factory=dict)

    @property
    def n_valid(self) -> int:
        """Windows with a finite margin (the population the package samples)."""
        return sum(1 for m in self.margin if math.isfinite(m))


def population_from_rows(
    rows: list[Any],
    *,
    grid: tuple[int, int] | None = None,
    windows: list[int] | None = None,
    score_kind: str = "class_scores",
    source: dict[str, Any] | None = None,
    centres: list[Any] | None = None,
    p1: list[Any] | None = None,
    map_class: list[Any] | None = None,
) -> Population:
    """Build a :class:`Population` from per-window score rows.

    ``windows`` (with ``grid``) says which grid window each row is, when
    no-data windows were left out; otherwise row ``i`` is window ``i``.
    ``margin`` is top-1 minus top-2, ``p1`` the top-1 probability (softmax
    of logits), ``map_class`` the arg-max, as ``olmoearth_review_set`` reads
    the same rows. A scores file that carries its own ``p1`` and
    ``map_class`` per row (``olmoearth_scores_from_file``: the pooled top-1
    probability of the model's pixels and the window's majority class) is
    read as it says, since its rows hold one confidence and zeros.
    """
    matrix = [[float(v) for v in row] for row in rows]
    if not matrix:
        raise ValueError("no windows: pass at least one score row")
    width = len(matrix[0])
    if width < 2 or any(len(r) != width for r in matrix):
        raise ValueError("every row needs the same number (>= 2) of class scores")
    if any(not math.isfinite(v) for r in matrix for v in r):
        raise ValueError("a score is not finite; leave no-data windows out")
    kind = detect_score_type(matrix)
    marg = margins(matrix)
    cls = predicted_classes(matrix)
    top1 = [max(r) if kind == "probability" else _softmax_top(r) for r in matrix]
    if p1 is not None:
        top1 = [float(v) for v in p1]
        if len(top1) != len(matrix) or not all(0.0 <= v <= 1.0 for v in top1):
            raise ValueError("the scores file's p1 needs one probability per row")
    if map_class is not None:
        cls = [int(c) for c in map_class]
        if len(cls) != len(matrix):
            raise ValueError("the scores file's map_class needs one class per row")
    if windows is not None:
        if grid is None:
            raise ValueError("a scores file with 'windows' needs its 'grid'")
        if len(windows) != len(matrix):
            raise ValueError(f"{len(windows)} windows for {len(matrix)} score rows")
        n_cells = grid[0] * grid[1]
        index = list(windows)
    else:
        n_cells = len(matrix)
        index = list(range(n_cells))
        if grid is not None and grid[0] * grid[1] != n_cells:
            raise ValueError(f"grid {list(grid)} does not match {n_cells} windows")
    if len(set(index)) != len(index) or any(not 0 <= i < n_cells for i in index):
        raise ValueError("window indices must be distinct and inside the grid")
    pop = Population(
        margin=[math.nan] * n_cells,
        p1=[math.nan] * n_cells,
        map_class=[-1] * n_cells,
        grid=grid,
        score_kind=score_kind,
        source=dict(source or {}),
    )
    for local, w in enumerate(index):
        pop.margin[w] = marg[local]
        pop.p1[w] = top1[local]
        pop.map_class[w] = cls[local]
    if centres is not None and len(centres) == len(index):
        pop.centres = {w: (float(c[0]), float(c[1])) for w, c in zip(index, centres)}
    return pop


def _from_sampled(sampled: SampledScores) -> Population:
    """A :class:`Population` from a Studio result sampled on a grid."""
    return population_from_rows(
        sampled.scores,
        grid=(sampled.grid, sampled.grid),
        windows=sampled.windows,
        score_kind=sampled.score_kind,
        source={
            "result_id": sampled.result_id,
            "property_name": sampled.property_name,
            "threshold": sampled.threshold,
            "assumption": sampled.assumption,
            "sampling": sampled.sampling_block(),
        },
        centres=sampled.centres,
    )


async def _population(
    args: dict[str, Any], ctx: ToolContext
) -> Population | dict[str, Any]:
    """The population from ``scores``, ``scores_path`` or a Studio ``result_id``."""
    if args.get("scores") is not None:
        return population_from_rows(
            args["scores"],
            grid=_grid_pair(args.get("grid")),
            source={"input": "inline scores"},
        )
    if args.get("scores_path"):
        path = str(args["scores_path"])
        loaded = load_scores_file(path)
        meta = loaded.meta
        source: dict[str, Any] = {"scores_path": path}
        for key in ("result_id", "property_name", "threshold", "assumption"):
            if meta.get(key) is not None:
                source[key] = meta[key]
        if isinstance(meta.get("provenance"), str):
            source["provenance"] = meta["provenance"]
            source["model"] = meta.get("model")
        return population_from_rows(
            loaded.scores,
            grid=loaded.grid,
            windows=loaded.windows,
            score_kind=str(meta.get("score_kind") or "class_scores"),
            source=source,
            centres=meta.get("centres_lon_lat"),
            p1=meta.get("p1"),
            map_class=meta.get("map_class"),
        )
    if args.get("result_id"):
        # N or [N, N], held to 2-16 and stated in the result, never silently.
        grid, grid_requested = result_grid(args.get("grid"))
        sampled = await sample_result_scores(
            ctx,
            str(args["result_id"]),
            grid=grid,
            property_name=args.get("property_name"),
            threshold=(
                float(args["threshold"]) if args.get("threshold") is not None else None
            ),
            grid_requested=grid_requested,
        )
        if isinstance(sampled, dict):
            return sampled
        return _from_sampled(sampled)
    raise ValueError(
        "pass 'scores', 'scores_path' (e.g. from olmoearth_review_set_from_result) or 'result_id'"
    )


def _window_row(pop: Population, index: int) -> dict[str, Any]:
    """A window by index and, with a grid, its (row, col); never coordinates."""
    row: dict[str, Any] = {"window_index": index}
    if pop.grid:
        row["row"], row["col"] = divmod(index, pop.grid[1])
    row["map_class"] = pop.map_class[index]
    return row


def _write_csv(name: str, pop: Population, to_label: list[dict[str, Any]]) -> str:
    """The reviewer's sheet beside the design: one row per window, blank labels."""
    target = write_root() / name
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = ["order", "window_index", "row", "col", "map_class"]
    if pop.centres:
        fields += ["lon", "lat"]
    fields += ["stratum", "wrong", "reference_class"]
    with open(target, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for entry in to_label:
            line = dict(entry)
            centre = pop.centres.get(int(entry["window_index"]))
            if centre:
                line["lon"], line["lat"] = centre
            writer.writerow(line)
    return str(target)


def _budget_refusal(budget: int, pop: Population) -> str:
    """Why ``budget`` cannot be planned from ``pop``, with only the options that work.

    Reached for a budget below 1, for inline or file scores (the whole map)
    given a budget above their valid windows, and for a Studio result sampled
    below the largest grid: at grid 16 the plan takes every valid window and
    states the labels left over instead (:func:`_at_studio_ceiling`). Below
    it a finer grid is offered only while one can still reach the budget, or
    else grid 16 with the same budget, and the full raster is a direct model
    run read through ``olmoearth_scores_from_file``. exp86 round 1's model was
    told "a finer grid" at the cap and retried grids until the turn cap.
    """
    n_valid = pop.n_valid
    if budget < 1:
        return f"budget {budget} must be at least 1"
    if not (pop.source.get("result_id") and pop.grid):
        return (
            f"budget {budget} is more than the {n_valid} valid windows of these "
            f"scores, so at most {n_valid} labels can be planned from them "
            f"({budget - n_valid} of the {budget} would have no window); plan a "
            f"budget of at most {n_valid}"
        )
    side = pop.grid[0]
    n_points = pop.grid[0] * pop.grid[1]
    most = FROM_RESULT_MAX_GRID * FROM_RESULT_MAX_GRID
    sampling = pop.source.get("sampling") or {}
    head = ""
    if sampling.get("grid_capped"):
        # Below the largest grid a grid is capped only by raising one below 2;
        # a grid capped at 16 is planned at its ceiling and never refused.
        head = (
            f"grid {sampling.get('grid_requested')} was raised to {side} (a "
            f"Studio result takes {FROM_RESULT_GRID_RANGE}); "
        )
    works = [f"a budget of at most {n_valid}"]
    if budget <= most:
        works.append(
            f"a finer grid, up to {FROM_RESULT_MAX_GRID} ({FROM_RESULT_MAX_GRID}x"
            f"{FROM_RESULT_MAX_GRID} = {most} points at most, fewer once no-data "
            f"is dropped), with the same budget: at grid {FROM_RESULT_MAX_GRID} "
            "a budget above the valid windows is planned at all of them, and "
            "the labels left over are stated"
        )
        limit = ""
    else:
        # No grid reaches the budget, but grid 16 plans at its ceiling and
        # states the labels left over (the unused_labels fact), so the model
        # keeps the user's budget and never works the difference out.
        works.append(
            f"grid={FROM_RESULT_MAX_GRID} with the same budget of {budget}: the "
            f"plan then takes every valid window of the {FROM_RESULT_MAX_GRID}x"
            f"{FROM_RESULT_MAX_GRID} grid and states how many of the {budget} "
            "labels go unused"
        )
        limit = (
            f" No grid reaches {budget} labels: {FROM_RESULT_MAX_GRID}x"
            f"{FROM_RESULT_MAX_GRID} = {most} points is the most a Studio result "
            "allows."
        )
    works.append(
        "for the whole map, a direct model run's scores raster read through "
        "olmoearth_scores_from_file (every window of the raster is then in the "
        "population)"
    )
    rest = " (the rest were no-data or failed)" if n_valid < n_points else ""
    return (
        f"{head}budget {budget} is more than the {n_valid} valid windows: a "
        f"Studio result sampled at {side}x{side} = {n_points} points has "
        f"{n_valid} valid windows here{rest}, so at most {n_valid} labels can be "
        f"planned from it.{limit} What works: "
        + "; ".join(works[:-1])
        + f"; or, {works[-1]}."
    )


def _budget_error(budget: int, pop: Population) -> ValueError:
    """The budget refusal as an error, with the labels left over where they are known.

    Inline or file scores are the whole map, so a budget above their valid
    windows leaves a known number of labels with no window: the error carries
    the ``unused_labels`` fact (the registry puts it on the failed envelope).
    Below the largest Studio grid a finer grid holds more windows, so the
    count is not known there, and a budget below 1 has none.
    """
    message = _budget_refusal(budget, pop)
    studio = bool(pop.source.get("result_id") and pop.grid)
    if budget < 1 or studio:
        return ValueError(message)
    return rules.refusal(
        message, facts=[_unused_labels(budget, pop.n_valid, pop, refused=True)]
    )


def _at_studio_ceiling(pop: Population) -> bool:
    """True for a Studio result sampled on the largest grid: no call holds more windows.

    There the ceiling is the tool's sampling cap, not the map, so a larger
    budget is planned at every valid window and the rest stated as unused.
    Below it a finer grid holds more, and inline or file scores are the whole
    map; both keep the budget refusal (:func:`_budget_refusal`).
    """
    return bool(
        pop.source.get("result_id") and pop.grid and pop.grid[0] >= FROM_RESULT_MAX_GRID
    )


def _studio_scope(pop_source: dict[str, Any], n_windows: Any) -> list[str]:
    """The scope limit of a rate over a Studio result's sampled grid points."""
    if not pop_source.get("result_id"):
        return []
    return [
        f"The rate describes the map at the {n_windows} sampled grid points "
        "of a Studio result (one pixel each), not every pixel of the map."
    ]


def _unused_labels(
    requested: int, planned: int, pop: Population, *, refused: bool = False
) -> dict[str, Any]:
    """The ``unused_labels`` fact: the labels a budget leaves with no window.

    ``planned`` is the plan's size: every valid window of a Studio result at
    the grid's ceiling, or, with ``refused``, the most a refused budget over
    inline or file scores could have planned (their valid windows).
    """
    n = requested - planned
    if refused:
        sentence = (
            f"Of the {requested} labels requested, at most {planned} can be "
            f"planned: these scores hold {planned} valid windows, so {n} labels "
            "would have no window to go to; the call was refused and no plan "
            "was written."
        )
        return rules.fact(
            "unused_labels", sentence, n=n, requested=requested, planned=planned
        )
    side = pop.grid[0] if pop.grid else FROM_RESULT_MAX_GRID
    sentence = (
        f"Of the {requested} labels requested, {planned} are planned: a Studio "
        f"result sampled at {side}x{side} (the most it allows) has {planned} "
        f"valid windows here, and the plan labels every one of them, so {n} "
        "labels have no window to go to."
    )
    return rules.fact(
        "unused_labels", sentence, n=n, requested=requested, planned=planned
    )


def _plan_next_steps(
    design: str, csv_path: str, planned: int, pop: Population, unused: int
) -> list[str]:
    """What follows a plan, from its design (and a budget above the ceiling)."""
    steps = [
        f"Label every one of the {planned} windows in {csv_path} (wrong, and "
        "reference_class for per-class accuracy); olmoearth_estimate_map_error "
        "refuses a design with any window unlabelled.",
        "Then olmoearth_estimate_map_error with design_path and the filled sheet "
        "(labels_path) gives the error rate and the interval this design earns.",
    ]
    if design == "random":
        steps.append(
            "For a certified zone, olmoearth_certify_zone with the same "
            "design_path and labels, at an alpha (and delta and rule) fixed now, "
            "before any label is seen: the guarantee covers only that alpha."
        )
    else:
        steps.append(
            f"This {design} design gives an estimate, not a certification: "
            "olmoearth_certify_zone refuses it. A certified zone needs a separate "
            "plan with design='random', fixed before labelling."
        )
    if unused:
        steps.append(
            "For more windows than the grid holds, a direct model run's scores "
            "raster read through olmoearth_scores_from_file puts every window "
            "of the map in the population."
        )
    return steps


def _whole_map_estimate(
    est: dict[str, Any], *, certify: bool = False
) -> dict[str, Any]:
    """The ``whole_map_estimate`` fact from the package's error-rate estimate.

    Fields ``estimate``, ``low``, ``high`` and ``design`` (the output
    contract's); the sentence also names the labelled count, the population,
    the method and, when the package gives one, its warning (a stratified
    design with starved strata, or with no or every labelled window wrong).
    """
    cov = est.get("nominal_coverage")
    level = f"{100.0 * float(cov):g}% " if isinstance(cov, (int, float)) else ""
    sentence = (
        f"The map's error rate is estimated at {rules.percent(est.get('estimate'))} "
        f"({level}interval {rules.percent(est.get('low'))} to "
        f"{rules.percent(est.get('high'))}) from {est.get('n_labelled')} "
        f"labelled windows of a population of {est.get('n_population')}; "
        f"method: {est.get('method')}."
    )
    if est.get("warning"):
        sentence += f" The package warns: {str(est['warning']).rstrip('.')}."
    if certify:
        sentence += (
            " It is the estimate for the whole population from the same labels, "
            "not a certification."
        )
    fields = ("estimate", "low", "high", "design")
    return rules.fact("whole_map_estimate", sentence, **{k: est.get(k) for k in fields})


def _refused_band(refusal: dict[str, Any]) -> dict[str, Any]:
    """A Studio band the plan cannot use, with the rules for a regression one.

    A regression band that is not a ``[0, 1]`` score and came with no
    threshold (the refusal names its ``declared_range``) has no error rate:
    exp86 round 7's answers offered one for such a band.
    """
    if "declared_range" not in refusal:
        return refusal
    rng = refusal.get("declared_range")
    band = (
        refusal.get("property_name"),
        (float(rng[0]), float(rng[1])) if isinstance(rng, list) else None,
    )
    refusal["next_steps"] = [
        "Pass threshold (the value at which the map's decision flips) to plan a "
        "labelled sample of this band; without one it has no error rate to "
        "estimate."
    ]
    return rules.add_contract(
        refusal, forbidden_claims=[rules.unthresholded_regression([band])]
    )


async def _plan_label_sample(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_plan_label_sample``."""
    try:
        estimate = inferencex.load("estimate")
    except inferencex.InferencexMissingError:
        return inferencex.missing_extra("Planning a labelled sample")
    design = str(args.get("design", "confidence"))
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {list(DESIGNS)}, got {design!r}")
    budget = int(args["budget"])
    seed = int(args.get("seed", 0))
    max_listed = int(args.get("max_listed", DEFAULT_MAX_LISTED))
    pop = await _population(args, ctx)
    if isinstance(pop, dict):
        return _refused_band(pop)
    n_valid = pop.n_valid
    # At the largest Studio grid a budget above the valid windows plans every
    # one of them and states the labels left over (exp86: "your extra 127
    # labels have nowhere to go", 300 - 173 worked out by the model); any
    # other budget above them is refused, with the options that work.
    if budget < 1 or (budget > n_valid and not _at_studio_ceiling(pop)):
        raise _budget_error(budget, pop)
    requested, budget = budget, min(budget, n_valid)
    sample = estimate.sample_for_estimation(
        pop.margin,
        budget,
        design=design,
        p1=pop.p1 if design == "confidence" else None,
        seed=seed,
    )
    indices = [int(i) for i in sample["indices"].tolist()]
    stratum_of: dict[int, int] = {}
    if "strata" in sample:
        stratum_of = {
            int(w): int(h)
            for w, h in zip(
                sample["strata_of_population"].tolist(), sample["strata"].tolist()
            )
        }
    to_label = []
    for order, idx in enumerate(indices):
        entry = {"order": order, **_window_row(pop, idx)}
        if stratum_of:
            entry["stratum"] = stratum_of.get(idx)
        to_label.append(entry)

    # A digest of the population keeps two plans of different maps apart.
    digest = hashlib.sha256(json.dumps(_jsonable(pop.margin)).encode()).hexdigest()
    origin = slug(str(pop.source.get("result_id") or "scores"), 12)
    stem = f"design_{origin}_{digest[:8]}_{design}_{budget}_s{seed}"
    payload = {
        "format": DESIGN_FORMAT,
        "package": {"name": inferencex.DISTRIBUTION, "version": inferencex.version()},
        "design": design,
        "budget": budget,
        "budget_requested": requested,
        "seed": seed,
        "sample": _jsonable(sample),
        "population": {
            "n_windows": len(pop.margin),
            "n_valid": n_valid,
            "grid": list(pop.grid) if pop.grid else None,
            "score_kind": pop.score_kind,
            "margin": _jsonable(pop.margin),
            "map_class": pop.map_class,
            "source": _jsonable(pop.source),
        },
        "to_label": to_label,
    }
    design_path = write_json_file(f"{stem}.json", payload)
    csv_path = _write_csv(f"{stem}_to_label.csv", pop, to_label)
    out: dict[str, Any] = {
        "available": True,
        "design": design,
        "budget": budget,
        "budget_requested": requested,
        "n_population": int(sample["n_population"]),
        "seed": seed,
        "design_path": design_path,
        "labels_csv_path": csv_path,
        "windows": to_label[:max_listed],
        "n_windows_listed": min(len(to_label), max_listed),
        "how_to_label": (
            "For each window, in this order, set wrong = 1 when the map's class "
            "(map_class) is not what is on the ground there, else 0 (and "
            "reference_class to the true class id for per-class accuracy). Fill "
            "the CSV (it holds each window's location) or pass the list; then "
            "call olmoearth_estimate_map_error with design_path."
        ),
        "interval_rules": list(INTERVAL_RULES),
        "population": {
            "score_kind": pop.score_kind,
            "what_the_rate_will_be_of": (
                "the valid windows of this population"
                + (
                    ": for a Studio result, the grid's sampled points, so the "
                    "rate describes the map at those points"
                    if pop.source.get("result_id")
                    else (
                        ": every valid window of the model run's map, each "
                        "graded on its majority class (map_class)"
                        if pop.score_kind == "window_confidence"
                        else ""
                    )
                )
            ),
        },
    }
    if "sizes" in sample:
        out["strata_sizes"] = _jsonable(sample["sizes"])
        out["allocation"] = _jsonable(sample["allocation"])
    if design == "confidence":
        out["design_note"] = (
            "Strata by confidence margin, labels allocated from the model's own "
            "top-1 score; the score only allocates labels, so the estimate stays "
            "design-unbiased whatever its calibration. A certified zone needs "
            "design='random'."
        )
    if pop.source.get("assumption"):
        out["assumption"] = pop.source["assumption"]
    if pop.source.get("sampling"):
        # The grid a Studio result was sampled on, and whether it was capped.
        out["sampling"] = pop.source["sampling"]
    if pop.score_kind in ("binary_score", "threshold_distance"):
        out["class_meaning"] = {
            "0": "at or below the decision threshold",
            "1": "above the decision threshold",
        }
    if sample.get("note"):
        out["note"] = sample["note"]
    unused = requested - budget
    out["next_steps"] = _plan_next_steps(design, csv_path, budget, pop, unused)
    facts = [_unused_labels(requested, budget, pop)] if unused else []
    claims = [rules.error_rate_without_labels()]
    # The sheet keeps the package's draw order: a random draw's order is
    # random, so its first rows are a smaller random sample; a stratified
    # draw lists the strata in turn, least confident first, so its first rows
    # are one stratum (a single non-empty stratum is a random draw in effect).
    if design == "random" or sum(1 for n in sample.get("sizes", []) if n) < 2:
        facts.append(rules.prefix_is_random_sample(budget, _scores_route(pop)))
    else:
        claims.append(rules.subset_labelling_sufficient(design))
    return rules.add_contract(
        out,
        facts=facts,
        must_state=(
            [f"The plan holds {budget} windows, not the {requested} requested."]
            if unused
            else []
        )
        + _studio_scope(pop.source, n_valid),
        forbidden_claims=claims,
    )


def _scores_route(pop: Population) -> str | None:
    """The argument olmoearth_estimate_map_error takes this population's scores by.

    ``window_indices`` need the scores they index: the inline rows again, or
    the same scores file. A Studio result sampled by the plan itself has none.
    """
    if pop.source.get("scores_path"):
        return "scores_path"
    if pop.source.get("input") == "inline scores":
        return "scores"
    return None


def _load_design(path: str) -> dict[str, Any]:
    """Read and check a design file written by olmoearth_plan_label_sample."""
    design = load_json_file(path, what="design_path")
    if not isinstance(design, dict) or design.get("format") != DESIGN_FORMAT:
        raise ValueError(
            "design_path is not a design file from olmoearth_plan_label_sample"
        )
    return design


def _margin(values: list[Any]) -> list[float]:
    """A stored margin list back to floats, None to NaN."""
    return [math.nan if v is None else float(v) for v in values]


def _read_labels_csv(path: str) -> dict[int, tuple[str, str]]:
    """``window_index -> (wrong, reference_class)`` from a filled-in CSV."""
    real = os.path.realpath(path)
    roots = read_roots()
    if not real.endswith(".csv") or not any(
        os.path.commonpath([r, real]) == r for r in roots
    ):
        raise ValueError(
            f"labels_path must name a .csv file under {' or '.join(roots)}"
        )
    out: dict[int, tuple[str, str]] = {}
    with open(real, encoding="utf-8", newline="") as fh:
        for line in csv.DictReader(fh):
            out[int(line["window_index"])] = (
                (line.get("wrong") or "").strip(),
                (line.get("reference_class") or "").strip(),
            )
    return out


def _labels(
    args: dict[str, Any], indices: list[int]
) -> tuple[list[int], list[int] | None]:
    """The reviewer's ``wrong`` (and optional reference classes), in design order."""
    wrong = args.get("wrong")
    reference = args.get("reference")
    if wrong is None and args.get("labels_path"):
        sheet = _read_labels_csv(str(args["labels_path"]))
        missing = [i for i in indices if i not in sheet or sheet[i][0] == ""]
        if missing:
            raise ValueError(
                f"{len(missing)} of {len(indices)} design windows have no 'wrong' "
                f"label yet (first: window {missing[0]}); every drawn window must "
                "be labelled, and no others"
            )
        wrong = [sheet[i][0] for i in indices]
        refs = [sheet[i][1] for i in indices]
        if reference is None and all(refs):
            reference = refs
    if wrong is None:
        raise ValueError(
            "pass 'wrong' (0/1 per window, in the design's order) or 'labels_path'"
        )
    values = [int(float(w)) for w in wrong]
    if len(values) != len(indices):
        raise ValueError(
            f"{len(values)} labels for the {len(indices)} windows of the design"
        )
    if any(v not in (0, 1) for v in values):
        raise ValueError("wrong must be 0 or 1 per window")
    ref = [int(float(r)) for r in reference] if reference is not None else None
    return values, ref


async def _estimate_map_error(
    args: dict[str, Any], _ctx: ToolContext
) -> dict[str, Any]:
    """Handler for ``olmoearth_estimate_map_error``."""
    try:
        estimate = inferencex.load("estimate")
    except inferencex.InferencexMissingError:
        return inferencex.missing_extra("Estimating a map's error rate")
    if args.get("design_path"):
        design = _load_design(str(args["design_path"]))
        sample = design["sample"]
        indices = [int(i) for i in sample["indices"]]
        wrong, reference = _labels(args, indices)
        out: dict[str, Any] = {
            "available": True,
            "design_path": str(args["design_path"]),
        }
        out.update(_jsonable(estimate.estimate_error_rate(sample, wrong)))
        if reference is not None:
            per_class = estimate.estimate_per_class(
                sample,
                reference,
                design["population"]["map_class"],
                n_classes=int(args["n_classes"]) if args.get("n_classes") else None,
            )
            out["per_class"] = _jsonable(per_class)
        population = design["population"]
    elif args.get("window_indices") is not None:
        # Windows labelled with no design: the package treats them as a random
        # sample only after checking they could be one (a review set is refused).
        pop = population_from_rows(**_rows_args(args))
        indices = [int(i) for i in args["window_indices"]]
        wrong, _ref = _labels(args, indices)
        out = {"available": True, "design": "none (checked as a random sample)"}
        out.update(
            _jsonable(estimate.estimate_from_indices(indices, wrong, pop.margin))
        )
        population = {"source": pop.source, "score_kind": pop.score_kind}
    else:
        raise ValueError(
            "pass design_path (from olmoearth_plan_label_sample), or "
            "window_indices with scores/scores_path for windows labelled without a design"
        )
    out["interval_rules"] = list(INTERVAL_RULES)
    out["report"] = (
        "Report the estimate with [low, high] and the method string as returned; "
        "the rate is of the design's population windows"
        + (
            " (a Studio result's sampled grid points)"
            if (population.get("source") or {}).get("result_id")
            else ""
        )
        + "."
    )
    stratified = out.get("design") in ("confidence", "proportional")
    out["next_steps"] = _estimate_next_steps(
        str(out.get("design")), from_design=bool(args.get("design_path"))
    )
    return rules.add_contract(
        out,
        facts=[_whole_map_estimate(out)],
        must_state=_studio_scope(
            population.get("source") or {}, out.get("n_population")
        ),
        forbidden_claims=(
            [
                rules.simple_random_interval(out.get("method")),
                rules.certify_from_nonrandom_design(out.get("design")),
            ]
            if stratified
            else []
        ),
    )


def _estimate_next_steps(design: str, *, from_design: bool) -> list[str]:
    """What follows an estimate, from its design."""
    steps = [
        "Report the estimate with its interval and method as returned (the fact "
        "whole_map_estimate states them); the rate is of the design's "
        "population."
    ]
    if design in ("confidence", "proportional"):
        steps.append(
            f"No zone can be certified from this {design} design: "
            "olmoearth_certify_zone needs design='random'. A certified zone "
            "needs a new plan with design='random', every drawn window labelled, "
            "and alpha, delta and rule fixed before labelling."
        )
    elif from_design:
        steps.append(
            "For a certified zone from these labels, olmoearth_certify_zone with "
            "the same design_path and labels, at the alpha the map's use "
            "requires, not one chosen from these results."
        )
    else:
        steps.append(
            "These windows were checked as a random sample and estimated as one; "
            "a certified zone needs a design file: plan one with "
            "olmoearth_plan_label_sample(design='random') before labelling."
        )
    steps.append(
        "A narrower interval needs more labels under a new plan with a larger "
        "budget, fixed before any of its labels is seen."
    )
    return steps


def _rows_args(args: dict[str, Any]) -> dict[str, Any]:
    """Keyword arguments for :func:`population_from_rows` from inline or file scores."""
    if args.get("scores") is not None:
        return {
            "rows": args["scores"],
            "grid": _grid_pair(args.get("grid")),
            "source": {"input": "inline scores"},
        }
    if args.get("scores_path"):
        loaded = load_scores_file(str(args["scores_path"]))
        return {
            "rows": loaded.scores,
            "grid": loaded.grid,
            "windows": loaded.windows,
            "score_kind": str(loaded.meta.get("score_kind") or "class_scores"),
            "source": {
                "scores_path": str(args["scores_path"]),
                "result_id": loaded.meta.get("result_id"),
            },
            "p1": loaded.meta.get("p1"),
            "map_class": loaded.meta.get("map_class"),
        }
    raise ValueError(
        "window_indices need the scores they index: pass scores or scores_path"
    )


async def _certify_zone(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_certify_zone``."""
    try:
        estimate = inferencex.load("estimate")
    except inferencex.InferencexMissingError:
        return inferencex.missing_extra("Certifying a trusted zone")
    design = _load_design(str(args["design_path"]))
    if design.get("design") != "random":
        refused: dict[str, Any] = {
            "available": True,
            "certified": False,
            "design": design.get("design"),
            "reason": (
                "a certified zone needs a simple random sample of the map: its "
                "guarantee rests on the labels inside each candidate zone being a "
                "random sample of that zone, which a stratified design is not. "
                "Plan one with olmoearth_plan_label_sample(design='random')."
            ),
            "design_requirement": DESIGN_REQUIREMENT,
            "next_steps": _certify_next_steps(None),
        }
        alpha_asked = args.get("alpha")
        return rules.add_contract(
            refused,
            forbidden_claims=[
                rules.certify_from_nonrandom_design(design.get("design")),
                rules.post_hoc_alpha(
                    float(alpha_asked)
                    if isinstance(alpha_asked, (int, float))
                    else None
                ),
                rules.rule_switch_after_failure(
                    certified=None, levels=None, delta=None, rule=None
                ),
            ],
        )
    rule = str(args.get("rule", "prefix"))
    if rule not in ZONE_RULES:
        raise ValueError(f"rule must be one of {list(ZONE_RULES)}, got {rule!r}")
    alpha = float(args["alpha"])
    delta = float(args.get("delta", 0.1))
    indices = [int(i) for i in design["sample"]["indices"]]
    wrong, _ref = _labels(args, indices)
    margin = _margin(design["population"]["margin"])
    zone = estimate.certify_zone(margin, indices, wrong, alpha, delta=delta, rule=rule)
    zone_indices = zone.pop("zone_indices_in_order", None)
    out: dict[str, Any] = {
        "available": True,
        "certified": zone.get("coverage") is not None,
    }
    out.update(_jsonable(zone))
    if zone_indices is not None:
        grid = design["population"].get("grid")
        windows = [int(i) for i in zone_indices.tolist()]
        stem = os.path.splitext(os.path.basename(str(args["design_path"])))[0]
        out["zone_path"] = write_json_file(
            f"{stem}_zone_a{alpha:g}.json",
            {
                "alpha": alpha,
                "delta": delta,
                "rule": rule,
                "window_indices": windows,
                "rows_cols": (
                    [list(divmod(w, int(grid[1]))) for w in windows] if grid else None
                ),
            },
        )
    if out["certified"]:
        out["verdict"] = (
            f"certified: the {zone['n_zone']} most confident windows (coverage "
            f"{zone['coverage']:g} of the population) under rule '{rule}' at "
            f"alpha={alpha:g}, delta={delta:g}"
        )
        out["meaning"] = (
            "The zone is the n_zone most confident windows of the design's "
            f"population; its error rate is at most alpha={alpha:g}, a statement "
            f"that fails on at most delta={delta:g} of random draws. Nothing "
            "outside the zone is certified."
        )
    else:
        out["verdict"] = (
            f"No zone is certified at alpha={alpha:g}, delta={delta:g} under "
            f"rule '{rule}': no share of the map is shown to be wrong at most "
            f"alpha={alpha:g} of the time."
        )
    out["levels_tested"] = _levels_tested(out, alpha, delta, rule)
    # exp86 round 1 (brief 6): with nothing certified at 0.05, every answer
    # read a level's upper_bound as a certification at a looser alpha.
    out["reading_levels"] = (
        "Each level's upper_bound is that zone's own bound at delta, before the "
        "rule is applied. A level whose upper_bound is below alpha, this alpha "
        "or any other, is not thereby certified: certification is the "
        "package's exact test under the chosen rule (levels_tested says how "
        "this rule tested which levels), and which levels are tested depends "
        "on alpha. Only a level marked accepted at this call's alpha is "
        "certified. Another alpha tests other levels, so this output says "
        "nothing about it; and the guarantee holds for an alpha fixed before "
        "the labels were seen, so trying looser alphas until one certifies is "
        "not covered by it."
    )
    out["design_requirement"] = DESIGN_REQUIREMENT
    # exp86 round 6 (B6/files): the answers derived "~23%" (69/300) from the
    # 100% level; the package's whole-map estimate on the same labels is given.
    whole = _whole_map_estimate(
        _jsonable(estimate.estimate_error_rate(design["sample"], wrong)),
        certify=True,
    )
    out["next_steps"] = _certify_next_steps(out)
    return rules.add_contract(
        out,
        facts=[whole],
        must_state=(
            [
                f"Only the {zone['n_zone']} most confident windows (coverage "
                f"{zone['coverage']:g}) are certified; nothing outside the zone "
                "is."
            ]
            if out["certified"]
            else []
        )
        + _studio_scope(design["population"].get("source") or {}, out["n_population"]),
        forbidden_claims=[
            rules.post_hoc_alpha(alpha),
            rules.rule_switch_after_failure(
                certified=bool(out["certified"]),
                levels=out.get("levels") or [],
                delta=delta,
                rule=rule,
            ),
        ],
    )


def _certify_next_steps(out: dict[str, Any] | None) -> list[str]:
    """What follows a certification, from the design and the outcome.

    ``out`` is ``None`` for a refused (stratified) design. Nothing certified
    under a random design: label more windows under a NEW random design fixed
    in advance, or report the whole-map estimate; never a looser alpha and
    never another rule (exp86 rounds 1 to 7, brief 6).
    """
    if out is None:
        return [
            "Estimate the map's error rate from these labels: "
            "olmoearth_estimate_map_error with this design_path and the same "
            "labels gives the estimate and the interval this design earns.",
            "A certified zone needs a new plan: olmoearth_plan_label_sample with "
            "design='random', every drawn window labelled, and alpha, delta and "
            "rule fixed before labelling.",
        ]
    estimate_step = (
        "report the whole-map estimate and interval (the fact whole_map_estimate "
        "states them)"
    )
    if out.get("certified"):
        return [
            f"Report the zone as returned ({out['verdict']}); its windows are "
            "in zone_path.",
            "Nothing outside the zone is certified; for the whole map, "
            + estimate_step
            + ".",
        ]
    return [
        "Label more windows under a NEW random design fixed in advance: "
        "olmoearth_plan_label_sample with design='random' and a larger budget, "
        "with alpha, delta and rule set before any of its labels is seen.",
        "Or " + estimate_step + ".",
        "Not a looser alpha and not another rule on these labels: either is "
        "chosen after seeing this result, which the guarantee does not cover "
        "(forbidden_claims).",
    ]


def _levels_tested(
    out: dict[str, Any], alpha: float, delta: float, rule: str
) -> dict[str, Any]:
    """Which levels the rule tested, at what per-level delta, and whether any passed.

    exp86 round 2 (brief 6, run 3): the model wrote "bonferroni tests every
    level at delta/18", counting the listed levels itself. The count and the
    per-level delta are the package's (``apply_zone_rule`` divides delta by
    the number of levels tested), stated here so no number is the model's.
    """
    levels = out.get("levels") or []
    n_levels = len(levels)
    n_accepted = sum(1 for lv in levels if lv.get("accepted"))
    by_rule: dict[str, float | None] = {"prefix": None, "bonferroni": None}
    if n_levels == 0:
        how = (
            "no level was tested: even a zone with no error among its labels "
            "would need more labels than the design holds (the package's note "
            "says how many)"
        )
    else:
        # The levels tested depend on alpha, delta and the labels, not on the
        # rule, so both rules' per-level delta over these levels is exact.
        by_rule = {"prefix": delta, "bonferroni": delta / n_levels}
        bonferroni = (
            f"each of the {n_levels} levels at delta/{n_levels} = "
            f"{delta:g}/{n_levels} = {delta / n_levels:.6g}"
        )
        prefix = (
            f"the {n_levels} levels at delta = {delta:g} each, from the smallest "
            "zone up, stopping at the first whose p_value is above it"
        )
        if rule == "bonferroni":
            how = (
                f"bonferroni tests {bonferroni}: a level is accepted when its "
                "p_value is at most that, whatever the other levels do (prefix "
                f"would test {prefix})"
            )
        else:
            how = (
                f"prefix tests {prefix}; the levels after that one are not "
                f"accepted (bonferroni would test {bonferroni})"
            )
    return {
        "n_levels": n_levels,
        "coverages": [lv.get("coverage") for lv in levels],
        "per_level_delta": by_rule.get(rule),
        "per_level_delta_by_rule": by_rule,
        "how_the_rule_tests_them": how,
        "n_accepted": n_accepted,
        "any_level_certifies": n_accepted > 0,
        "why_these_levels": (
            "a zone must hold at least "
            f"{out.get('min_labels_to_certify')} labels to be certified at "
            f"alpha={alpha:g}, delta={delta:g} (min_labels_to_certify), so with "
            f"{out.get('n_labelled')} labels no zone below coverage c_min = "
            f"{float(out.get('c_min') or 0.0):.6g} is tested; another alpha "
            "tests other levels"
        ),
    }


_DESIGN_PATH = {"type": "string", "description": "From olmoearth_plan_label_sample."}
_WRONG = {
    "type": "array",
    "items": {"type": "integer"},
    "description": "0/1 per window in the design's order; 1 = map is wrong there.",
}
_LABELS_PATH = {
    "type": "string",
    "description": "Or the design's _to_label.csv with 'wrong' filled in.",
}


def build_estimation_tools() -> list[RegisteredTool]:
    """Return the design-based estimation tools (optional ``inferencex`` extra)."""
    source_props: dict[str, Any] = {
        "scores": _SCORES_SCHEMA,
        "scores_path": {
            "type": "string",
            "description": "Scores .json under OLMOEARTH_SCORES_ROOT (from "
            "olmoearth_scores_from_file or olmoearth_review_set_from_result).",
        },
        "result_id": {
            "type": "string",
            "description": "Or a Studio result, sampled on a grid.",
        },
        "grid": {
            "type": ["integer", "array"],
            "items": {"type": "integer"},
            "description": "Inline scores: N (N x N) or [rows, cols] of the rows. "
            f"A Studio result_id: N, {FROM_RESULT_GRID_RANGE} (N x N sampled "
            f"points, default {FROM_RESULT_DEFAULT_GRID}; [N, N] also taken); a "
            "larger N is capped, and the result's 'sampling' says so.",
        },
        "property_name": {"type": "string"},
        "threshold": {"type": "number"},
    }
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_plan_label_sample",
                description=(
                    "How wrong is my map / how to spend N labels: chooses which "
                    "windows to label for an honest error rate "
                    "(olmoearth-inferencex design: 'confidence' by default, "
                    "'random' when a certified zone is wanted) from scores "
                    "inline, a scores file (from olmoearth_scores_from_file "
                    "or olmoearth_review_set_from_result) or a Studio "
                    "result_id; "
                    "saves the design and lists windows by index and (row, "
                    "col). Do not design your own mix or promise an interval: a "
                    "targeted review set is not a sample (its error rate is "
                    "inflated), and the simple-random-sample interval does not "
                    "apply to a stratified or targeted design. The interval "
                    "comes from olmoearth_estimate_map_error. Needs the "
                    "inferencex extra."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        **source_props,
                        "budget": {
                            "type": "integer",
                            "description": "Windows the reviewer can label.",
                        },
                        "design": {
                            "type": "string",
                            "enum": list(DESIGNS),
                            "default": "confidence",
                            "description": "'random' if a certified zone is wanted.",
                        },
                        "seed": {"type": "integer", "default": 0},
                        "max_listed": {
                            "type": "integer",
                            "default": DEFAULT_MAX_LISTED,
                            "description": "Cap on windows listed inline.",
                        },
                    },
                    "required": ["budget"],
                },
            ),
            handler=_plan_label_sample,
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_estimate_map_error",
                description=(
                    "The map's error rate from the reviewer's 0/1 labels on a "
                    "design from olmoearth_plan_label_sample (design_path + "
                    "wrong, or the filled labels CSV), with the interval that "
                    "design earns and the package's method string; per-class "
                    "accuracy with reference classes. Windows labelled without "
                    "a design: window_indices + scores/scores_path, checked as "
                    "a random sample (a review set is refused: its rate is "
                    "inflated). Report estimate, low, high and method as "
                    "returned; never substitute p +/- 1.96 sqrt(p(1-p)/n). "
                    "Needs the inferencex extra."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "design_path": _DESIGN_PATH,
                        "wrong": _WRONG,
                        "labels_path": _LABELS_PATH,
                        "reference": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Optional true class per window "
                            "(design order), for per-class accuracy.",
                        },
                        "n_classes": {"type": "integer"},
                        "window_indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "No design: the labelled windows.",
                        },
                        "scores": _SCORES_SCHEMA,
                        "scores_path": {"type": "string"},
                        "grid": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                    "required": [],
                },
            ),
            handler=_estimate_map_error,
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_certify_zone",
                description=(
                    "The largest most-confident share of the map whose error "
                    "rate is at most alpha, certified by exact tests (the "
                    "statement fails on at most delta of draws). Needs a "
                    "design='random' plan and its labels; a stratified design "
                    "is refused. rule 'prefix' (default) or 'bonferroni' (no "
                    "assumption). Needs the inferencex extra."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "design_path": _DESIGN_PATH,
                        "wrong": _WRONG,
                        "labels_path": _LABELS_PATH,
                        "alpha": {
                            "type": "number",
                            "description": "Maximum error rate, e.g. 0.05.",
                        },
                        "delta": {
                            "type": "number",
                            "default": 0.1,
                            "description": "Failure probability.",
                        },
                        "rule": {
                            "type": "string",
                            "enum": list(ZONE_RULES),
                            "default": "prefix",
                        },
                    },
                    "required": ["design_path", "alpha"],
                },
            ),
            handler=_certify_zone,
        ),
    ]
