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
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext
from olmoearth_agent.tools.review_set import (
    FROM_RESULT_DEFAULT_GRID,
    FROM_RESULT_MAX_GRID,
    SampledScores,
    load_json_file,
    load_scores_file,
    read_roots,
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
) -> Population:
    """Build a :class:`Population` from per-window score rows.

    ``windows`` (with ``grid``) says which grid window each row is, when
    no-data windows were left out; otherwise row ``i`` is window ``i``.
    ``margin`` is top-1 minus top-2, ``p1`` the top-1 probability (softmax
    of logits), ``map_class`` the arg-max, as ``olmoearth_review_set`` reads
    the same rows.
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
    p1 = [max(r) if kind == "probability" else _softmax_top(r) for r in matrix]
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
        pop.p1[w] = p1[local]
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
        source = {"scores_path": path}
        for key in ("result_id", "property_name", "threshold", "assumption"):
            if meta.get(key) is not None:
                source[key] = meta[key]
        return population_from_rows(
            loaded.scores,
            grid=loaded.grid,
            windows=loaded.windows,
            score_kind=str(meta.get("score_kind") or "class_scores"),
            source=source,
            centres=meta.get("centres_lon_lat"),
        )
    if args.get("result_id"):
        grid = max(
            2,
            min(FROM_RESULT_MAX_GRID, int(args.get("grid", FROM_RESULT_DEFAULT_GRID))),
        )
        sampled = await sample_result_scores(
            ctx,
            str(args["result_id"]),
            grid=grid,
            property_name=args.get("property_name"),
            threshold=(
                float(args["threshold"]) if args.get("threshold") is not None else None
            ),
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
        return pop
    n_valid = pop.n_valid
    if not 0 < budget <= n_valid:
        raise ValueError(
            f"budget {budget} must be between 1 and the {n_valid} valid windows; "
            "label fewer windows, or build a larger population (a finer grid, "
            "or scores for more windows)"
        )
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
                    else ""
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
    if pop.score_kind in ("binary_score", "threshold_distance"):
        out["class_meaning"] = {
            "0": "at or below the decision threshold",
            "1": "above the decision threshold",
        }
    if sample.get("note"):
        out["note"] = sample["note"]
    return out


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
    return out


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
        return {
            "available": True,
            "certified": False,
            "design": design.get("design"),
            "reason": (
                "a certified zone needs a simple random sample of the map: its "
                "guarantee rests on the labels inside each candidate zone being a "
                "random sample of that zone, which a stratified design is not. "
                "Plan one with olmoearth_plan_label_sample(design='random')."
            ),
        }
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
    out["meaning"] = (
        "The zone is the n_zone most confident windows of the design's "
        f"population; its error rate is at most alpha={alpha:g}, a statement "
        f"that fails on at most delta={delta:g} of random draws. Nothing "
        "outside the zone is certified."
    )
    return out


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
            "description": "Scores .json under OLMOEARTH_SCORES_ROOT (e.g. from "
            "olmoearth_review_set_from_result).",
        },
        "result_id": {
            "type": "string",
            "description": "Or a Studio result, sampled on a grid.",
        },
        "grid": {
            "type": ["integer", "array"],
            "items": {"type": "integer"},
            "description": "N (N x N) or [rows, cols].",
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
                    "inline, a scores file (e.g. from "
                    "olmoearth_review_set_from_result) or a Studio result_id; "
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
                            "description": "'random' if a certified zone " "is wanted.",
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
