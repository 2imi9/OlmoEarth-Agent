# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""How wrong is the map? Design-based estimation (skill #5, second half).

The review set answers *which windows to open first*; it cannot say how wrong
the map is, because it is chosen to hold errors: labelling it and dividing
overstates the error rate (1.8 to 5.8 times the true rate on the companion
package's exp78). An error rate needs a labelled **sample drawn by a design**,
and its interval is the one that design earns -- a stratified or targeted draw
does not get the simple-random-sample formula. Three tools wrap
``oe_inferencex.estimate`` (``olmoearth-inferencex`` >= 1.20.3, the optional
``inferencex`` extra, imported lazily):

- ``olmoearth_plan_label_sample`` -- choose which windows to label
  (``sample_for_estimation``; the confidence design by default, ``random``
  when a certified zone is wanted) and save the design to a file;
- ``olmoearth_estimate_map_error`` -- the error rate with the interval
  valid at every sample size first, then the design's interval and
  ``method`` (``estimate_error_rate(valid=True)``, whatever the design),
  per-class accuracy with reference classes (``estimate_per_class(valid=True)``;
  the fact ``weakest_classes`` ranks them), or, for windows labelled with no
  design, ``estimate_from_indices``, which refuses a review set;
- ``olmoearth_certify_zone`` -- the largest most-confident share of the map
  whose error rate is at most ``alpha`` (``certify_zone``), random designs
  only; a certified zone's ``meaning`` ends with the package's statement of
  what delta counts (``oe_inferencex.certify.zone.ISSUED_NOTE``), as
  ``oe-inferencex certify`` prints it.

Which of two maps is more accurate (the which-map route). Where two maps of
the same windows at the same time give the same class, both are right or
both are wrong, so their accuracies differ only through the windows where
they differ. ``olmoearth_plan_label_sample`` with ``design='disagreement'``
reads each window's class in two Studio results of the same area (sampled by
``olmoearth_compare_results``' own pair sampler) or in two scores files of
one window grid, refuses maps that ``oe_inferencex.compare.dates_reading``
does not read as one time, and draws the windows to label where they differ
(``estimate.sample_disagreement``), with ``plan.plan_which_map``'s budget
advice; ``olmoearth_estimate_map_error`` on that design grades the
reviewer's classes with ``estimate.compare_from_disagreement``: which map is
more accurate, or neither, by the exact sign test while the labels are at
most a fifth of the differing windows and by the interval past that, and
the 95% interval on the difference in accuracy. No class map needs a score.

A Studio class map with top-k scores (a segmentation prediction submitted
with ``probability_output``: each window's class, and each rank's softmax
score as a band of its own) is a map with confidence. The review set's
sampler reads it (``score_kind`` ``topk_score``): its margin is the rank-1
score minus the rank-2, its ``p1`` the rank-1 score and its class the
rank-1 class, so it takes every design, ranking ones included, and a random
design's labels can certify a zone; its reviewer writes the class seen, as
for a class map without scores (:func:`_labels_are_classes`).

A Studio class map without top-k scores (a classification result: each
window's class, no probability). Nothing ranks its windows, so it has no
review set and no certified zone (:func:`_has_no_confidence`), but a sample
drawn without its confidence still measures it.
``olmoearth_plan_label_sample`` reads it with ``olmoearth_compare_results``'
own sampler (each window's class, as the two-map design reads it) and draws
``design='random'`` or ``design='class'``, the class design of Olofsson et
al. (2014): the package's condition design with each window's map class as
its condition and ``class_floor`` labels per class at least, or all the
windows of a smaller class (``sample_for_estimation``, as ``oe-inferencex
sample --design class`` draws it), with ``plan.plan_error_rate``'s label
budget for a random sample's interval width and, for a stated class share,
``plan.plan_class``'s. The ranking designs are refused for it with the
reason. The rate is of the
result's sampled grid points (at most 16 x 16, one pixel each), not every
pixel. ``olmoearth_estimate_map_error`` on its labels reports
``estimate_error_rate(valid=True)``'s interval valid at every sample size
first, then the design's own, each named, with the package's notes word for
word, as for every design of one map: under the random design the two are
the same exact interval, under the class design the second is the union
bound over the classes, and both cover at least 95% at every sample size
(under a scores map's confidence and proportional designs the second is
Wilson's on the effective sample size, which the package calls nominal).
With reference classes, ``estimate_per_class(valid=True)`` the same way per
class; the class design's results are worded as the package words them
(``common.wording.class_wording``). ``olmoearth_certify_zone`` and the
review-set tools refuse a class map without top-k scores in one sentence
(:data:`~olmoearth_agent.tools.statistical_rules.CLASS_MAP_NO_CONFIDENCE`),
followed by where top-k scores come from
(:data:`~olmoearth_agent.tools.statistical_rules.CLASS_MAP_REFUSAL`).

A sample a certificate can rest on. ``olmoearth_plan_label_sample`` with
``design='random'`` and ``seed_from_map`` on a direct run's scores file
(``olmoearth_scores_from_file``, every channel kept) runs ``oe-inferencex
sample RASTER --design random --seed-from-map --alpha A`` on the run's raster
in a working folder under the scores root (:data:`CERTIFICATE_DIR`): its seed
is taken from the map's digest and the draw's settings, alpha, delta and the
rule fixed now, and the agent's sample is the same draw, checked window by
window against the package's sheet. ``olmoearth_certify_zone`` then keeps to
the plan's alpha, delta and rule, as ``oe-inferencex certify`` does; the
labels first graded on the plan (by the estimate, the zone or the
certificate) are written into the package's sheet and other labels are
refused afterwards (:func:`hold_plan_labels`); and
``olmoearth_write_certificate`` (:mod:`olmoearth_agent.tools.certificate`)
packs the zone into a certificate.

Windows are addressed by index and grid ``(row, col)`` (rule 1); the
locations a reviewer needs stay in the files.

These package calls run in a worker thread (``asyncio.to_thread``), as
the command line does (:func:`inferencex.run_cli_async`): the estimate of a
design file of one map (its valid per-class table takes seconds per class
under a stratified design), the zone (``certify_zone`` orders every window
of the map) and the map-seeded draw. Run inline, each would hold the event
loop that serves the web bridge's other sessions; an exception raised in the
thread reaches the handler as it would inline. A plain plan's draw, the
which-map grading and windows labelled without a design still run on the
event loop. The estimate fixes a ``seed_from_map`` plan's labels from its
worker thread, so the package's sheet is read, checked and written under one
lock (:func:`fill_sheet`): two sessions grading one plan cannot both write
it, and no reader sees it half written.

Every output also carries ``next_steps``, written by code from the design and
the outcome, and the output contract's ``facts``, ``must_state`` and
``forbidden_claims`` (:mod:`olmoearth_agent.tools.statistical_rules`): exp86's
answers proposed a looser alpha or another rule after nothing certified,
offered to certify from a stratified design, and worked out 69/300 by hand.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeGuard

from olmoearth_agent.analysis.output_contract import (
    MUST_STATE_MAX,
    MUST_STATE_MAX_WORDS,
    word_count,
)
from olmoearth_agent.analysis.raster_compare import declared_fields, declared_range
from olmoearth_agent.analysis.review_set import (
    DEFAULT_MAX_LISTED,
    TOPK_SCORE,
    detect_score_type,
    margins,
    predicted_classes,
)
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.security.paths import safe_path
from olmoearth_agent.tools import inferencex
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.compare import (
    PAIR_MAX_GRID,
    _mixed_types,
    read_pair,
    sample_pair,
)
from olmoearth_agent.tools.registry import Capability, RegisteredTool, ToolContext
from olmoearth_agent.tools.review_set import (
    CLASS_READS,
    DATE_FORMS,
    FROM_RESULT_DEFAULT_GRID,
    FROM_RESULT_GRID_RANGE,
    FROM_RESULT_MAX_GRID,
    FROM_RESULT_MIN_GRID,
    PARTLY_DATED_MUST_STATE,
    STUDIO_TOPK,
    ClassReads,
    SampledScores,
    _check_date_form,
    file_studio_kind,
    load_json_file,
    load_scores_file,
    read_roots,
    result_grid,
    sample_result_scores,
    sampling_block,
    slug,
    write_json_file,
    write_root,
)
from olmoearth_agent.tools.sampling import (
    FAILED,
    NODATA_RULE,
    OK,
    OUTSIDE,
    default_property,
    dropped_text,
    prediction_period,
    topk_part_reason,
    topk_scores,
)

#: The design of two maps: a random sample of the windows where their classes
#: differ (``oe_inferencex.estimate.sample_disagreement``), for which of the
#: two is more accurate.
TWO_MAP_DESIGN = "disagreement"

#: The class design (Olofsson et al. 2014): the map's classes are the strata;
#: each class gets at least ``min_per_class`` labels, or all its windows, and
#: the rest go in proportion to class size. It is the package's condition
#: design with each window's map class as its condition and ``class_floor``
#: set (``sample_for_estimation``; ``oe-inferencex sample --design class``).
CLASS_DESIGN = "class"

#: The designs of one map (the package's tile design needs tile ids the agent
#: lacks).
ONE_MAP_DESIGNS = ("confidence", "proportional", "random", CLASS_DESIGN)

#: The designs a class map takes: neither reads a confidence.
CLASS_MAP_DESIGNS = ("random", CLASS_DESIGN)

#: The designs whose interval is not the simple random sample's, and which
#: certify nothing.
STRATIFIED_DESIGNS = ("confidence", "proportional", CLASS_DESIGN)

#: The designs whose own interval the package calls nominal: Wilson's on the
#: design's effective sample size, for the whole map and for each class's
#: user's accuracy (its ``VALID_NOTE``); approximate, never exact.
NOMINAL_DESIGNS = ("confidence", "proportional")

#: The design of windows labelled without one (``window_indices``): the
#: package estimates them as a simple random sample once its check finds they
#: are not a review set (``estimated_as``), which does not make them one.
NO_DESIGN = "none (checked as a random sample)"

#: ``score_kind`` of a class map's population: each window's class, no score.
CLASS_MAP = "class_map"

#: The ``score_kind`` values whose labels are reference classes written as the
#: map writes its classes (the class labelling path, per-class accuracy by
#: class name): a class map without top-k scores, which has no confidence, and
#: one with them (``topk_score``), which has.
CLASS_LABELLED = (CLASS_MAP, TOPK_SCORE)

#: The interval width (high minus low) the label-budget advice is for unless
#: the caller states one (``oe_inferencex.plan.plan_error_rate``).
DEFAULT_WIDTH = 0.10

#: The class design's labels per class at least, as the schema states it: the
#: package's ``estimate.CLASS_FLOOR`` (Olofsson et al. suggest 50 to 100),
#: which the handler reads (the package is optional, the schema is not).
DEFAULT_MIN_PER_CLASS = 50

#: How the reviewer labels a class map's sheet: the class seen, as the map
#: writes its classes; ``wrong`` follows from it. ``{blind}`` is how to label
#: blind under the plan's design (:func:`_how_to_label_a_class_map`).
HOW_TO_LABEL_A_CLASS_MAP = (
    "For each window, in this order, write in reference_class the class you "
    "see there, as map_class writes classes; wrong (1 where it is not "
    "map_class, else 0) is read from it, and may be filled instead for the "
    "error rate alone. {blind} Fill the CSV (it holds each window's location) "
    "or pass reference in the same order; then call "
    "olmoearth_estimate_map_error with design_path."
)

#: How to label a class map blind: a random sheet's rows are in random order,
#: so hiding the map's class is enough; the class design's rows run class by
#: class, so the order names the class too (the package's ``CLASS_BLIND_NOTE``,
#: whose ``condition`` column the agent's sheet does not have).
BLIND_RANDOM = "To label blind, hide map_class."
BLIND_BY_CLASS = (
    "To label blind, hide map_class and stratum and show the rows in a "
    "shuffled order (they run class by class), writing each label on its own "
    "row; the sheet has no condition column."
)


def _how_to_label_a_class_map(design: str) -> str:
    """How the reviewer labels a class map's sheet under ``design``."""
    blind = BLIND_BY_CLASS if design == CLASS_DESIGN else BLIND_RANDOM
    return HOW_TO_LABEL_A_CLASS_MAP.format(blind=blind)


#: The keys of ``plan.plan_error_rate``'s and ``plan.plan_class``'s results a
#: plan returns (the budget-by-budget table is left out; ``labels`` and its
#: probability are the advice).
_ADVICE_KEYS = (
    "question",
    "n_population",
    "width",
    "error_rate",
    "error_rate_stated",
    "share",
    "drawn_ratio",
    "n_classes",
    "power",
    "labels",
    "probability_at_labels",
    "checked_up_to",
    "census_checked",
    "every_budget_checked",
    "runs_reaching_power",
    "refusal",
    "rate_note",
    "note",
    "interval_note",
)

#: Designs offered.
DESIGNS = (*ONE_MAP_DESIGNS, TWO_MAP_DESIGN)

#: Rules offered for the certified zone (the package's "plugin" has no guarantee).
ZONE_RULES = ("prefix", "bonferroni")

#: Format tag of a design file written by olmoearth_plan_label_sample.
DESIGN_FORMAT = "olmoearth-agent/label-design@1"

#: Under the scores root, the working folder of each sample drawn for a
#: certificate (``seed_from_map``): the package's sheet and sidecar, the zone
#: ``certify`` writes and the bundle (``olmoearth_write_certificate``).
CERTIFICATE_DIR = "certificates"

#: The package's sheet of a sample drawn for a certificate, in its working
#: folder: ``oe-inferencex sample`` writes its sidecar beside it
#: (``sample.json``), and ``oe-inferencex certify`` its zone
#: (``sample_zone.json``, with the zone mask ``sample_zone.npy``).
CERT_SAMPLE = "sample.csv"

#: The certificate bundle's folder inside a plan's working folder
#: (``olmoearth_write_certificate``).
CERT_BUNDLE = "certificate"

#: delta when a plan drawn with ``seed_from_map`` names none: the default of
#: ``olmoearth_certify_zone``, and the package's ``ZONE_DELTA``.
DEFAULT_DELTA = 0.1

#: Said when a sample drawn for a certificate already holds other labels.
LABELS_FIXED = (
    "a certificate rests on the labels first given for its sample: labels "
    "changed after a result was seen are a second test, which neither "
    "certify's guarantee nor a certificate covers"
)

#: Said with a plan drawn with ``seed_from_map``: what its seed shows.
SEED_FROM_MAP_NOTE = (
    "The seed is taken from the map's digest and the draw's settings, alpha, "
    "delta and the rule included (oe-inferencex sample --seed-from-map): "
    "anyone holding the map draws these windows again, and a certificate "
    "shows these settings were fixed before any label. It cannot show the "
    "sample was drawn only once."
)
#: The same, for a zone read from a Studio result's own files: the map the
#: seed is drawn from is the raster the agent decoded from Studio's native
#: file (``olmoearth_scores_from_studio``), which a holder of Studio's file
#: does not hold.
SEED_FROM_STUDIO_RASTER_NOTE = (
    "The seed is taken from the digest of the raster the agent decoded from "
    "Studio's file and the draw's settings, alpha, delta and the rule included "
    "(oe-inferencex sample --seed-from-map): anyone holding that decoded "
    "raster, not Studio's own file, draws these windows again, and a "
    "certificate shows these settings were fixed before any label. It cannot "
    "show the sample was drawn only once."
)


def _certificate_holder(decoded: bool) -> str:
    """Who can check a certificate: the map's holder, or the decoded raster's.

    ``decoded`` for a zone read from a Studio result's own files, whose map is
    the raster the agent decoded, not Studio's native file.
    """
    if decoded:
        return (
            "anyone holding the decoded raster this plan drew from (not Studio's "
            "own file)"
        )
    return "anyone holding the map"


#: Stated with every certify_zone output: exp86 round 2's model recommended a
#: confidence design for certification, which the tool refuses, and round
#: 8's answers promised "a guaranteed-certifiable region" from a random one.
DESIGN_REQUIREMENT = (
    "Certification needs design='random' (a simple random sample of the map); "
    "olmoearth_certify_zone refuses a 'confidence' or 'proportional' design, so "
    "recommend only a random design for a certified zone. " + rules.POSSIBLE_NOT_CERTAIN
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

#: A class's user's accuracy is ranked only on at least this many windows the
#: map puts in it (the ``weakest_classes`` fact); below it one window more or
#: fewer moves the estimate by a fifth or more.
MIN_TO_RANK = 5

#: The lowest classes the ``weakest_classes`` sentence names: every class with
#: none of its map-labelled windows correct, then the next lowest up to this.
N_WEAKEST = 3

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
    A class map (``score_kind`` :data:`CLASS_MAP`) has no confidence: its
    ``margin`` is 0 wherever a window holds a class, which only marks the
    population, its ``p1`` NaN, and ``classes`` the class each code stands
    for, as the map writes it. A class map with top-k scores (``score_kind``
    ``topk_score``) has its rank-1 minus rank-2 score as ``margin``, its
    rank-1 score as ``p1``, and ``classes`` the same way. ``class_names``
    names a scores map's class codes where its scores file does
    (``olmoearth_scores_from_file`` writes them); it only labels a plan's
    strata (:meth:`class_name`).
    """

    margin: list[float]
    p1: list[float]
    map_class: list[int]
    grid: tuple[int, int] | None
    score_kind: str
    source: dict[str, Any] = field(default_factory=dict)
    centres: dict[int, tuple[float, float]] = field(default_factory=dict)
    classes: list[Any] | None = None
    class_names: dict[int, str] = field(default_factory=dict)

    @property
    def n_valid(self) -> int:
        """Windows with a finite margin (the population the package samples)."""
        return sum(1 for m in self.margin if math.isfinite(m))

    def class_name(self, code: Any) -> str | None:
        """Class ``code``'s name, as the map or its scores file writes it, if known."""
        if not isinstance(code, int) or code < 0:
            return None
        if self.classes is not None and code < len(self.classes):
            name = str(self.classes[code]).strip()
            return name or None
        return self.class_names.get(code)


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
    margin: list[Any] | None = None,
) -> Population:
    """Build a :class:`Population` from per-window score rows.

    ``windows`` (with ``grid``) says which grid window each row is, when
    no-data windows were left out; otherwise row ``i`` is window ``i``.
    ``margin`` is top-1 minus top-2, ``p1`` the top-1 probability (softmax
    of logits), ``map_class`` the arg-max, as ``olmoearth_review_set`` reads
    the same rows. A scores file that carries its own ``p1`` and
    ``map_class`` per row (``olmoearth_scores_from_file``: the pooled top-1
    probability of the model's pixels and the window's majority class) is
    read as it says, since its rows hold one confidence and zeros; so is the
    ``margin`` a class map's top-k scores carry (for Studio's uint8 scores,
    the levels' difference over 254, which the gap of ``[p1, p2]`` misses by
    a level where a rank-2 score is raw 0).
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
    if margin is not None:
        marg = [float(v) for v in margin]
        if len(marg) != len(matrix) or not all(math.isfinite(v) for v in marg):
            raise ValueError("the scores file's margin needs one number per row")
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
    """A :class:`Population` from a Studio result sampled on a grid.

    A class map with top-k scores brings its ``p1``, margins and rank-1
    classes (its rows' arg-max is no class), the classes as the map writes
    them, the field read (``source["top_k"]``) and, when a threshold was
    passed, that it was set aside (``source["threshold_note"]``).
    """
    source: dict[str, Any] = {
        "result_id": sampled.result_id,
        "property_name": sampled.property_name,
        "threshold": sampled.threshold,
        "assumption": sampled.assumption,
        "sampling": sampled.sampling_block(),
    }
    if sampled.top_k is not None:
        source["top_k"] = sampled.top_k
    if sampled.threshold_note:
        source["threshold_note"] = sampled.threshold_note
    pop = population_from_rows(
        sampled.scores,
        grid=(sampled.grid, sampled.grid),
        windows=sampled.windows,
        score_kind=sampled.score_kind,
        source=source,
        centres=sampled.centres,
        p1=sampled.p1,
        map_class=sampled.map_class,
        margin=sampled.margin,
    )
    if sampled.classes is not None:
        pop.classes = list(sampled.classes)
    return pop


def _topk_file_classes(raw: Any) -> list[Any]:
    """A top-k scores file's classes (code to name), as a list by code.

    ``olmoearth_review_set_from_result`` names every code from 0 up; a file
    that does not is refused, since its classes are matched against the
    reviewer's.
    """
    names = _file_class_names(raw)
    if sorted(names) != list(range(len(names))):
        msg = (
            "this scores file of a class map's top-k scores does not name its "
            "classes 0 to K-1: read the result again with "
            "olmoearth_review_set_from_result"
        )
        raise ValueError(msg)
    return [names[c] for c in range(len(names))]


def _topk_file_margin(meta: dict[str, Any]) -> list[Any] | None:
    """A top-k scores file's own margins, or ``None`` for any other file.

    ``olmoearth_review_set_from_result`` writes each window's margin beside
    its row ``[p1, p2]`` (for Studio's uint8 scores, the levels' difference
    over 254); another file's ``margin`` key, if any, is not read.
    """
    margin = meta.get("margin")
    if meta.get("score_kind") != TOPK_SCORE or margin is None:
        return None
    if not isinstance(margin, list):
        raise ValueError("the scores file's margin needs one number per row")
    return margin


def _file_class_names(raw: Any) -> dict[int, str]:
    """A scores file's ``classes`` (code to name, as a mapping or a list) by code.

    The names only label strata and classes, so an entry that is no class
    code and name is left out rather than refused: a key ``int`` cannot read
    ("-1", or "²", which ``str.isdigit`` passes but ``str.isdecimal``, the
    digits ``int`` reads, does not), or a name that is not a non-empty
    string.
    """
    pairs: list[tuple[Any, Any]] = []
    if isinstance(raw, dict):
        pairs = list(raw.items())
    elif isinstance(raw, list):
        pairs = list(enumerate(raw))
    return {
        int(str(key).strip()): name.strip()
        for key, name in pairs
        if str(key).strip().isdecimal() and isinstance(name, str) and name.strip()
    }


async def _population(
    args: dict[str, Any], ctx: ToolContext, *, class_map: bool = False
) -> Population | dict[str, Any]:
    """The population from ``scores``, ``scores_path`` or a Studio ``result_id``.

    A Studio classification band is refused as the review set refuses it
    (the refusal is marked ``class_map``); with ``class_map``, for a design
    that reads no confidence, it is read as a class map instead
    (:func:`_class_map_population`), from the grid already sampled when the
    result declares no band type.
    """
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
        for key in ("result_id", "property_name", "threshold", "assumption", "top_k"):
            if meta.get(key) is not None:
                source[key] = meta[key]
        if isinstance(meta.get("provenance"), str):
            source["provenance"] = meta["provenance"]
            source["model"] = meta.get("model")
        if isinstance(meta.get("studio"), dict):
            # Pooled from a Studio result's own files: the design file keeps
            # how, so its estimate and certification state what the plan does.
            source["studio"] = meta["studio"]
        pop = population_from_rows(
            loaded.scores,
            grid=loaded.grid,
            windows=loaded.windows,
            score_kind=str(meta.get("score_kind") or "class_scores"),
            source=source,
            centres=meta.get("centres_lon_lat"),
            p1=meta.get("p1"),
            map_class=meta.get("map_class"),
            margin=_topk_file_margin(meta),
        )
        pop.class_names = _file_class_names(meta.get("classes"))
        if pop.score_kind == TOPK_SCORE:
            # A class map's top-k scores: its reviewer writes classes as the
            # map writes them, which these name.
            pop.classes = _topk_file_classes(meta.get("classes"))
        return pop
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
            keep_class_reads=class_map,
        )
        if isinstance(sampled, dict):
            kept = sampled.pop(CLASS_READS, None)
            if class_map and sampled.get("class_map"):
                return await _class_map_population(
                    args, ctx, grid, grid_requested, kept
                )
            return sampled
        return _from_sampled(sampled)
    raise ValueError(
        "pass 'scores', 'scores_path' (e.g. from olmoearth_review_set_from_result) or 'result_id'"
    )


async def _class_map_population(
    args: dict[str, Any],
    ctx: ToolContext,
    grid: int,
    grid_requested: int | None,
    kept: ClassReads | None = None,
) -> Population:
    """A Studio class map's windows: each one's class, as the two-map design reads it.

    The result is read and sampled by ``olmoearth_compare_results``' own
    sampler (:func:`~olmoearth_agent.tools.compare.read_pair` and
    :func:`~olmoearth_agent.tools.compare.sample_pair` with this one result),
    so a window's class and the no-data rule are the two-map design's, and
    its classes are coded as that design codes them (:func:`_code_classes`).
    The population is every window that holds a class: its margin is 0
    there and NaN elsewhere, since the package samples the windows with a
    finite margin; nothing ranks them. A result that declares no band type
    was sampled by the review set's sampler before it refused the band:
    ``kept`` holds those reads (same grid, same no-data context), and the
    grid is not sampled again.
    """
    rid = str(args["result_id"]).strip()
    if kept is None:
        pair = await read_pair(ctx, [rid])
        if isinstance(pair, dict):
            raise ValueError(str(pair.get("reason")))
        sampled = await sample_pair(
            ctx, pair, prop=args.get("property_name"), grid=grid
        )
        if isinstance(sampled, dict):
            raise rules.refusal(
                str(sampled.get("reason")),
                forbidden_claims=sampled.get("forbidden_claims") or [],
            )
        reads = sampled.reads[0]
        cells = sampled.cells
        prop = sampled.names[0]
        model = sampled.models[0] if sampled.models else None
    else:
        reads, cells = kept.reads, kept.cells
        prop, model = kept.property_name, kept.model
        if _mixed_types([reads]):
            # As the pair sampler refuses a band read as numbers and classes.
            msg = (
                "sampled values mix numeric and categorical bands; pass "
                "property_name to pin one field"
            )
            raise ValueError(msg)
    values = [
        _class_value(value) if status == OK and categorical else None
        for value, categorical, status in reads
    ]
    codes, _none, classes = _code_classes(values, [])
    n_points = grid * grid
    n_valid = sum(1 for c in codes if c >= 0)
    n_failed = sum(1 for _v, _c, status in reads if status == FAILED)
    # Points Studio answered with HTTP 404 (where it answered others) are neither
    # failed calls nor no-data pixels.
    n_outside = sum(1 for _v, _c, status in reads if status == OUTSIDE)
    n_nodata = n_points - n_valid - n_failed - n_outside
    if n_valid < 2:
        msg = (
            f"only {n_valid} of {n_points} sampled windows hold a class "
            f"({dropped_text(n_nodata, n_failed, n_outside)}); a labelled sample "
            "needs at least two"
        )
        raise ValueError(msg)
    return Population(
        margin=[0.0 if c >= 0 else math.nan for c in codes],
        p1=[math.nan] * n_points,
        map_class=codes,
        grid=(grid, grid),
        score_kind=CLASS_MAP,
        source={
            "result_id": rid,
            "property_name": prop,
            "model": (model or {}).get("name"),
            "sampling": sampling_block(
                grid,
                grid_requested,
                n_sampled=n_points,
                n_valid=n_valid,
                n_nodata=n_nodata,
                n_failed=n_failed,
                n_outside=n_outside,
                nodata_rule=NODATA_RULE,
            ),
        },
        centres={r * grid + c: (lon, lat) for r, c, lon, lat in cells},
        classes=classes,
    )


def _window_row(pop: Population, index: int) -> dict[str, Any]:
    """A window by index and, with a grid, its (row, col); never coordinates.

    A class map's window shows its class as the map writes it.
    """
    row: dict[str, Any] = {"window_index": index}
    if pop.grid:
        row["row"], row["col"] = divmod(index, pop.grid[1])
    code = pop.map_class[index]
    row["map_class"] = pop.classes[code] if pop.classes and code >= 0 else code
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
            "and points Studio answers with HTTP 404 are dropped), with the same "
            f"budget: at grid {FROM_RESULT_MAX_GRID} "
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
    if rest and sampling.get("n_outside_dropped"):
        rest = " (the rest were no-data, failed, or answered HTTP 404)"
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


def _label_every_window(planned: int, csv_path: str) -> str:
    """The first step of a plan of scores: label every drawn window."""
    return (
        f"Label every one of the {planned} windows in {csv_path} (wrong, and "
        "reference_class for per-class accuracy); olmoearth_estimate_map_error "
        "refuses a design with any window unlabelled."
    )


def _plan_next_steps(
    design: str, csv_path: str, planned: int, pop: Population, unused: int
) -> list[str]:
    """What follows a plan, from its design (and a budget above the ceiling).

    Every plan's steps name a certified zone, as this plan's follow-up or as
    what a separate random plan is for, so each says a random design makes
    one possible, not certain (exp86 round 8, B5/files/2). A class map's
    steps say instead that it has no certified zone
    (:data:`~olmoearth_agent.tools.statistical_rules.CLASS_MAP_NO_CONFIDENCE`);
    a class map with top-k scores is labelled as a class map is (the class
    seen) and certified as a scores map is.
    """
    estimate_step = (
        "Then olmoearth_estimate_map_error with design_path and the filled sheet "
        "(labels_path) gives the error rate with the interval valid at every "
        "sample size first, then the interval this design earns."
    )
    classes = pop.score_kind in CLASS_LABELLED
    if classes or design == CLASS_DESIGN:
        estimate_step = (
            "Then olmoearth_estimate_map_error with design_path and the filled "
            "sheet (labels_path) gives the error rate with the interval valid at "
            "every sample size first, then the interval this design earns, and "
            "each class's accuracy."
        )
    label_step = (
        f"Write in reference_class, for every one of the {planned} windows in "
        f"{csv_path}, the class seen there as map_class writes classes (wrong "
        "is then read from it); olmoearth_estimate_map_error refuses a design "
        "with any window unlabelled."
        if classes
        else _label_every_window(planned, csv_path)
    )
    if pop.score_kind == CLASS_MAP:
        steps = [label_step, estimate_step, rules.CLASS_MAP_NO_CONFIDENCE]
    elif design == "random":
        steps = [
            label_step,
            estimate_step,
            "For a certified zone, olmoearth_certify_zone with the same "
            "design_path and labels, at an alpha (and delta and rule) fixed now, "
            "before any label is seen: the guarantee covers only that alpha. "
            + rules.POSSIBLE_NOT_CERTAIN,
        ]
    else:
        steps = [
            label_step,
            estimate_step,
            f"This {design} design gives an estimate, not a certification: "
            "olmoearth_certify_zone refuses it. A certified zone needs a separate "
            "plan with design='random', fixed before labelling. "
            + rules.POSSIBLE_NOT_CERTAIN,
        ]
    if unused:
        steps.append(
            "For more windows than the grid holds, a direct model run's scores "
            "raster read through olmoearth_scores_from_file puts every window "
            "of the map in the population."
        )
    return steps


def _whole_map_estimate(
    est: dict[str, Any],
    *,
    certify: bool = False,
    uncertified: list[dict[str, Any]] | None = None,
    why_field: str | None = "levels_tested",
) -> dict[str, Any]:
    """The ``whole_map_estimate`` fact from the package's error-rate estimate.

    Fields ``estimate``, ``low``, ``high`` and ``design`` (the output
    contract's); the sentence also names the labelled count, the population,
    the method and, when the package gives one, its warning (a stratified
    design with starved strata, or with no or every labelled window wrong).

    ``uncertified`` is the levels of a certify call that certified nothing
    (``[]`` when it tested none): the sentence adds that the rate describes
    the whole map, not its most confident part, and gives the evidence on
    that part as the result does, the smallest zone tested with its coverage,
    labels and wrong count (field ``smallest_zone_tested``). Agent trial 12
    (B6/files runs 8, 9 and 13) read the whole-map rate against alpha as why
    no zone certified (``whole_map_rate_for_a_zone``). When none was tested,
    ``why_field`` is the result's field that says why (``levels_tested`` in
    olmoearth_certify_zone's, ``note`` in olmoearth_write_certificate's, which
    has no ``levels_tested``: the step 11 review), or ``None`` for none.
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
    found: dict[str, Any] = {k: est.get(k) for k in fields}
    if uncertified is not None:
        sentence += " It describes the whole map, not its most confident part: "
        if uncertified:
            first = uncertified[0]
            sentence += (
                "the smallest zone tested, at coverage "
                f"{float(first.get('coverage') or 0.0):g} of the map, held "
                f"{first.get('n_labelled_inside')} labels with "
                f"{first.get('n_wrong_inside')} wrong."
            )
            found["smallest_zone_tested"] = {
                k: first.get(k)
                for k in ("coverage", "n_labelled_inside", "n_wrong_inside")
            }
        else:
            sentence += "no zone was tested" + rules.says_why(why_field) + "."
            found["smallest_zone_tested"] = None
    return rules.fact("whole_map_estimate", sentence, **found)


def _names(items: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


#: The package's flags for a class's share as the map draws it, against the
#: interval of its share on the ground (``estimate_per_class``'s ``drawn``).
DRAWN_FLAGS = ("too little", "too much", "within")

#: How the ``user_vs_producer`` fact opens. The package's valid per-class
#: intervals hold each on its own at 95%, not jointly (its
#: ``PER_CLASS_VALID_NOTE``), so over many classes one can be named by chance.
USER_VS_PRODUCER_OPENING = (
    "Beside each class's user's accuracy, from olmoearth-inferencex's "
    "per-class rows (intervals valid at every sample size, each holding on its "
    "own at 95% and not jointly, unless said): "
)
#: How a drawn flag says which interval it is read on: the exact one holds
#: for all classes together (random and class designs), the valid one alone.
DRAWN_EXACT = "exact intervals, holding for all classes together at 95%"
DRAWN_ON_ITS_OWN_INTERVAL = "flagged on that interval alone"
#: How a class whose valid intervals the package did not compute names the
#: design's it is read on instead: exact under the random and class designs,
#: Wilson's on the effective sample size under the confidence and
#: proportional ones, which the package calls nominal.
DESIGN_INTERVALS = " (the design's intervals)"
DESIGN_INTERVALS_NOMINAL = (
    " (the design's nominal intervals, not valid at every sample size)"
)

#: Most classes whose producer's accuracy lies below their user's accuracy
#: get a ``must_state`` sentence: after the valid interval's sentence the
#: contract's :data:`~olmoearth_agent.analysis.output_contract.MUST_STATE_MAX`
#: leaves room for two, one beside a Studio result's scope.
MAX_MISSED_STATED = MUST_STATE_MAX - 1


def _bounded(q: Any) -> TypeGuard[dict[str, Any]]:
    """Whether ``q`` is a per-class quantity of the package with both ends given."""
    if not isinstance(q, dict):
        return False
    low, high = q.get("low"), q.get("high")
    return isinstance(low, (int, float)) and isinstance(high, (int, float))


def _quantity(q: dict[str, Any], interval: str) -> dict[str, Any]:
    """A per-class quantity's point and ends, as the package gives them, named."""
    out: dict[str, Any] = {k: q.get(k) for k in ("estimate", "low", "high")}
    if q.get("estimate") is None and q.get("estimate_range") is not None:
        out["estimate_range"] = q.get("estimate_range")
    out["interval"] = interval
    return out


def _producer_accuracy(row: dict[str, Any]) -> dict[str, Any] | None:
    """A class's producer's accuracy: its valid interval when given, else the design's.

    The design's is the exact interval where the package gives one
    (``producer_accuracy_exact``: the random and class designs), else its
    ``producer_accuracy``. ``interval`` says which.
    """
    valid = row.get("valid")
    valid = valid if isinstance(valid, dict) else {}
    if _bounded(valid.get("producer_accuracy")):
        return _quantity(valid["producer_accuracy"], "valid")
    for key in ("producer_accuracy_exact", "producer_accuracy"):
        if _bounded(row.get(key)):
            return _quantity(row[key], "design")
    return None


def _drawn(row: dict[str, Any]) -> dict[str, Any] | None:
    """Whether the map draws a class too little, too much or within its share.

    The exact flag where the package gives one (``drawn`` with
    ``reference_share_exact``: the random and class designs), whose intervals
    hold for all classes together at 95%; else the flag of the valid
    interval of the class's share on the ground
    (``valid.reference_share.drawn``), which holds on its own and not
    jointly, so the package flags a share outside it "on that interval
    alone" (its ``PER_CLASS_VALID_NOTE``). With the map's share
    (``map_share``) and that interval, each the package's; the interval's
    ``interval`` says which (the step 11 review: a valid flag over eight or
    more classes wrongly names one with a chance well above 5%).
    """
    valid = row.get("valid")
    valid = valid if isinstance(valid, dict) else {}
    share = valid.get("reference_share")
    exact = row.get("reference_share_exact")
    if _bounded(exact) and row.get("drawn") in DRAWN_FLAGS:
        flag, ref = row["drawn"], _quantity(exact, "exact")
    elif _bounded(share) and share.get("drawn") in DRAWN_FLAGS:
        flag, ref = share["drawn"], _quantity(share, "valid")
    else:
        return None
    return {"flag": flag, "map_share": row.get("map_share"), "reference_share": ref}


def _class_pair(
    row: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """A class's user's and producer's accuracy: both valid, else both the design's."""
    valid = row.get("valid")
    valid = valid if isinstance(valid, dict) else {}
    user, producer = valid.get("user_accuracy"), valid.get("producer_accuracy")
    if _bounded(user) and _bounded(producer):
        return "valid", user, producer
    user = row.get("user_accuracy")
    producer = row.get("producer_accuracy_exact")
    if not _bounded(producer):
        producer = row.get("producer_accuracy")
    if _bounded(user) and _bounded(producer):
        return "design", user, producer
    return None


def _share_text(q: dict[str, Any]) -> str:
    """``93.3% (86.7% to 97.2%)``, or ``86.7% to 97.2%`` when no point is given."""
    ends = f"{rules.percent(q.get('low'))} to {rules.percent(q.get('high'))}"
    point = q.get("estimate")
    if isinstance(point, (int, float)):
        return f"{rules.percent(point)} ({ends})"
    return ends


def _missed_must_state(k: Any, producer: dict[str, Any]) -> str | None:
    """The ``must_state`` of a class the map finds less of than it calls.

    From the producer's side only: every answer that reports the estimate
    gives the user's accuracy already, and with both sides' numbers in it
    such an answer held half of the sentence's key terms without a word on
    what the map misses (the step 11 review, on trial 12's N3 answers).

    The class is named once: a class map's class as it writes it can be a
    Studio label of several words ("Permanent water bodies"), and the
    contract holds at most :data:`MUST_STATE_MAX_WORDS` words a sentence.
    ``None`` for a name so long the sentence would still hold more: that
    class is then left to the ``user_vs_producer`` fact, which names it.
    """
    sentence = (
        f"Class {k}: the map finds only {rules.percent(producer['low'])} to "
        f"{rules.percent(producer['high'])} of the class's ground "
        "(producer's accuracy); it misses the rest."
    )
    return sentence if word_count(sentence) <= MUST_STATE_MAX_WORDS else None


def _user_vs_producer(
    per_class: dict[str, Any],
    names: dict[int, str] | None = None,
    *,
    written: list[Any] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """The ``user_vs_producer`` fact and the ``must_state`` sentences it asks for.

    Agent trial 12 (N3/files, three runs) called classes 4 and 7 well mapped
    on their user's accuracy (93.3%, 97 of 104) while the same result gave
    class 4 a valid producer's accuracy of 69.0% (60.2% to 76.8%) and its
    share drawn too little (32.8% of the map against 45.7%, 40.0% to
    51.4%, on the ground). From the package's per-class rows only, the fact
    names every class whose user's and producer's accuracy intervals do not
    overlap (both valid where the package gives both, else both the
    design's, named nominal under the confidence and proportional designs
    as :func:`_weakest_classes` names them), and every class the map draws
    too little or too much (:func:`_drawn`: the exact flag that holds for
    all classes together where the package gives one), each in one sentence
    with its intervals. The valid intervals hold each on its own, not
    jointly, and the sentence says so. A class whose producer's interval
    lies wholly below its user's gets a ``must_state`` sentence
    (:func:`_missed_must_state`; a written class too long for the contract's
    words gets none and stays in the fact), the lowest producer's accuracy
    first, at most :data:`MAX_MISSED_STATED` (the caller keeps fewer beside a
    Studio result's scope), so an answer that reports the estimate also
    states what the map misses of that class.

    ``names`` gives a scores map's class names by class id, shown beside the
    id. ``written`` is a class map's classes by code, as the map writes them
    (as :func:`_weakest_classes` takes them): each class is then named as
    written, in the sentences and the ``must_state``, with its internal code
    as ``code`` (the step 11 review: a map writing ids [1, 2, 3, 5] codes
    them 0 to 3, and "class 3" named its class 5). ``(None, [])`` when no
    class qualifies.
    """
    table = per_class.get("per_class")
    if not isinstance(table, dict):
        return None, []
    nominal = per_class.get("design") in NOMINAL_DESIGNS
    sentences: list[str] = []
    classes: list[dict[str, Any]] = []
    missed: list[tuple[float, int, str]] = []
    for key in sorted(table, key=int):
        row, c = table[key], int(key)
        if not isinstance(row, dict):
            continue
        # A class as the map writes it: its id, or a class map's written class.
        k = written[c] if written is not None and c < len(written) else c
        said: list[str] = []
        entry: dict[str, Any] = {"class": k}
        if written is not None:
            entry["code"] = c
        pair = _class_pair(row)
        if pair is not None:
            kind, user, producer = pair
            ua, pa = _share_text(user), _share_text(producer)
            if producer["high"] < user["low"]:
                said.append(
                    f"what the map calls class {k} is right {ua} of the time, but "
                    f"it finds only {pa} of the ground that is class {k}"
                )
                entry["producer_below_user"] = True
                point = producer.get("estimate")
                lowest = point if isinstance(point, (int, float)) else producer["high"]
                limit = _missed_must_state(k, producer)
                if limit is not None:
                    missed.append((float(lowest), c, limit))
            elif user["high"] < producer["low"]:
                said.append(
                    f"the map finds {pa} of the ground that is class {k}, but what "
                    f"it calls class {k} is right only {ua} of the time"
                )
                entry["user_below_producer"] = True
            if said:
                entry["intervals"] = kind
                entry["user_accuracy"] = _quantity(user, kind)
                entry["producer_accuracy"] = _quantity(producer, kind)
        drawn = _drawn(row)
        if drawn is not None and drawn["flag"] != "within":
            ref = drawn["reference_share"]
            # The exact flag holds for all classes together; the valid one
            # on its own interval only (the package's PER_CLASS_VALID_NOTE).
            exact = ref["interval"] == "exact"
            which = DRAWN_EXACT if exact else DRAWN_ON_ITS_OWN_INTERVAL
            said.append(
                f"the map draws class {k} {drawn['flag']}: on "
                f"{rules.percent(drawn['map_share'])} of the windows, against "
                f"{_share_text(ref)} of the ground ({which})"
            )
            entry["drawn"] = drawn
        if not said:
            continue
        name = (names or {}).get(c) if written is None else None
        label = f"Class {k}" + (f" ({name})" if name else "")
        if entry.get("intervals") == "design":
            # The package's valid intervals were not computed (too large);
            # under the confidence and proportional designs the design's are
            # nominal, never a plain 95% interval (the step 11 review).
            label += DESIGN_INTERVALS_NOMINAL if nominal else DESIGN_INTERVALS
        sentences.append(f"{label}: " + "; ".join(said) + ".")
        classes.append(entry)
    if not sentences:
        return None, []
    sentence = USER_VS_PRODUCER_OPENING + " ".join(sentences)
    missed.sort()
    stated = [text for _high, _c, text in missed[:MAX_MISSED_STATED]]
    return rules.fact("user_vs_producer", sentence, classes=classes), stated


def _weakest_classes(
    per_class: dict[str, Any], names: list[Any] | None = None
) -> dict[str, Any] | None:
    """The ``weakest_classes`` fact: the classes ranked by user's accuracy, lowest first.

    exp86 round 8 (B5/files/2) named class 1 (user's accuracy 22.5%) and
    class 6 (45.1%) the weakest, while class 5 had none of its 10
    map-labelled windows correct (interval 0 to 0.278, below both): the model
    ranked the nine rows of the package's table by eye and left the worst out.
    Code ranks them here. A class is ranked on at least :data:`MIN_TO_RANK`
    windows the map puts in it; ``correct`` is the confusion matrix's
    diagonal (rows the map's class, columns the reference's), ``labelled``
    its row. The sentence names every ranked class with none correct first
    (they tie at 0, and leaving one out was the defect), then the next
    lowest, up to :data:`N_WEAKEST` named and never every ranked class: with
    two ranked, one is the lowest; with one, nothing is ranked against it and
    ``lowest`` is empty (the fix-r8 review: on a map with a rare positive
    class, the one ranked class, 97.7% correct, was named "the lowest" while
    the positive class, too few to rank, went unnamed). It names the classes
    the map or the labels hold on fewer windows as too few to rank, and a
    class with enough windows but no estimate as such, never dropping one.
    It says which accuracy this is: a class's producer's accuracy, what the
    map misses, is another quantity in the table. ``names`` (a class map's
    classes, by code) puts each ranked class's name beside its count.
    ``None`` without a per-class table.

    A table of ``estimate_per_class(valid=True)`` gives each class's user's
    accuracy an interval valid at every sample size (its row's ``valid``):
    the sentence states it first, then the design's, as the estimate's
    ``report`` asks, and each ranked entry carries it as ``valid_low`` and
    ``valid_high``. The ranking stays on the package's estimate. Under the
    confidence and proportional designs the design's interval is Wilson's
    on the effective sample size, which the package calls nominal, and is
    named so, never as a plain 95% interval.

    Each ranked entry also carries, from the package's row, the class's
    producer's accuracy (``producer_accuracy``: the valid interval when
    given, else the design's) and whether the map draws the class too
    little, too much or within its share on the ground (``drawn``, with the
    map's share and that share's interval): agent trial 12 (N3/files) called
    a class well mapped on its user's accuracy alone. The sentence is
    unchanged; :func:`_user_vs_producer` states them.
    """
    table = per_class.get("per_class")
    if not isinstance(table, dict) or not table:
        return None
    confusion = per_class.get("confusion_counts") or []

    def diagonal(c: int) -> int | None:
        row = confusion[c] if c < len(confusion) else None
        return int(row[c]) if isinstance(row, list) and c < len(row) else None

    ranked: list[dict[str, Any]] = []
    too_few: list[dict[str, Any]] = []
    no_estimate: list[dict[str, Any]] = []
    for key in sorted(table, key=int):
        row, c = table[key], int(key)
        labelled = int(row.get("n_labelled_map_class") or 0)
        ua = row.get("user_accuracy")
        if labelled >= MIN_TO_RANK:
            if not isinstance(ua, dict) or ua.get("estimate") is None:
                # enough windows but no estimate: listed, never dropped
                no_estimate.append(
                    {"class": c, "labelled": labelled, "correct": diagonal(c)}
                )
                continue
            entry: dict[str, Any] = {
                "class": c,
                "correct": diagonal(c),
                "labelled": labelled,
                **{k: ua.get(k) for k in ("estimate", "low", "high")},
            }
            valid = row.get("valid")
            valid_ua = valid.get("user_accuracy") if isinstance(valid, dict) else None
            if isinstance(valid_ua, dict) and None not in (
                valid_ua.get("low"),
                valid_ua.get("high"),
            ):
                entry["valid_low"] = valid_ua["low"]
                entry["valid_high"] = valid_ua["high"]
            # What the map misses beside what it calls the class: the
            # producer's accuracy and whether the map draws the class too
            # little or too much, from the package's row (agent trial 12, N3).
            producer = _producer_accuracy(row)
            if producer is not None:
                entry["producer_accuracy"] = producer
            drawn = _drawn(row)
            if drawn is not None:
                entry["drawn"] = drawn
            ranked.append(entry)
        elif labelled or row.get("map_share") or row.get("n_labelled_reference_class"):
            # a class neither the map nor the labels hold is not in this map
            too_few.append({"class": c, "labelled": labelled, "correct": diagonal(c)})

    def none_right(r: dict[str, Any]) -> bool:
        if r["correct"] is not None:
            return bool(r["correct"] == 0)
        return bool(r["estimate"] == 0)

    def rank(r: dict[str, Any]) -> tuple[bool, float, float, int]:
        # a bound the package left non-finite comes back None (_jsonable):
        # it sorts as the widest, never as a comparison of None with a float
        high = r["high"] if isinstance(r["high"], (int, float)) else 1.0
        return (not none_right(r), float(r["estimate"]), float(high), r["class"])

    ranked.sort(key=rank)
    zero = [r for r in ranked if none_right(r)]
    rest = [r for r in ranked if not none_right(r)]
    # Every class with none correct, then the next lowest, but never every
    # ranked class: the highest of them is not "the lowest" of anything.
    cap = min(N_WEAKEST, len(ranked) - 1)
    named = [] if len(ranked) < 2 else zero + rest[: max(0, cap - len(zero))]

    cov = per_class.get("nominal_coverage")
    level = f"{100.0 * float(cov):g}% " if isinstance(cov, (int, float)) else ""
    # The design's per-class interval under the confidence and proportional
    # designs is Wilson's on the effective sample size (nominal); under the
    # random and class designs the user's accuracy is exact.
    nominal = "nominal " if per_class.get("design") in NOMINAL_DESIGNS else ""
    what = (
        "user's accuracy (of the windows the map puts in a class, the share the "
        "labels agree with)"
    )

    def described(r: dict[str, Any]) -> str:
        count = (
            f"{r['correct']} of {r['labelled']} correct"
            if r["correct"] is not None
            else f"{r['labelled']} map-labelled windows"
        )
        c = r["class"]
        name = f", {names[c]}" if names and c < len(names) else ""
        design_ends = f"{rules.percent(r['low'])} to {rules.percent(r['high'])}"
        if "valid_low" in r:
            intervals = (
                f"{level}interval valid at every sample size "
                f"{rules.percent(r['valid_low'])} to "
                f"{rules.percent(r['valid_high'])}, {nominal}{level}interval this "
                f"design earns {design_ends}"
            )
        else:
            intervals = f"{nominal}{level}interval {design_ends}"
        return f"class {c} ({count}{name}; {rules.percent(r['estimate'])}, {intervals})"

    estimated = " with an estimate" if no_estimate else ""
    if not ranked:
        sentence = (
            f"No class has at least {MIN_TO_RANK} map-labelled windows{estimated}, "
            f"so none is ranked by {what}."
        )
    elif len(ranked) == 1:
        sentence = (
            f"Only one class has at least {MIN_TO_RANK} map-labelled "
            f"windows{estimated}, {described(ranked[0])}, so no class is ranked "
            f"against it by {what}."
        )
    elif len(named) == len(ranked):
        sentence = (
            f"Ranked by {what} over the {len(ranked)} classes with at least "
            f"{MIN_TO_RANK} map-labelled windows, none has any correct: "
            + _names([described(r) for r in named])
            + "."
        )
    else:
        sentence = (
            f"Ranked by {what} over the {len(ranked)} classes with at least "
            f"{MIN_TO_RANK} map-labelled windows, the lowest "
            f"{'are' if len(named) > 1 else 'is'} "
            + _names([described(r) for r in named])
            + "."
        )
    if too_few:
        many = len(too_few) > 1
        sentence += (
            f" Class{'es' if many else ''} "
            + _names([str(r["class"]) for r in too_few])
            + f" {'have' if many else 'has'} fewer than {MIN_TO_RANK} map-labelled "
            f"windows ({_names([str(r['labelled']) for r in too_few])}), too few "
            "to rank."
        )
    if no_estimate:
        many = len(no_estimate) > 1
        sentence += (
            f" Class{'es' if many else ''} "
            + _names([str(r["class"]) for r in no_estimate])
            + f" {'have' if many else 'has'} at least {MIN_TO_RANK} map-labelled "
            f"windows ({_names([str(r['labelled']) for r in no_estimate])}) but no "
            "user's accuracy estimate, so not ranked."
        )
    # Under the class design each class is a stratum labelled at random within
    # it, so its user's accuracy is correct over labelled, as under a random one.
    if ranked and per_class.get("design") not in (None, "random", CLASS_DESIGN):
        sentence += (
            " Under this design each estimate is weighted, so it need not equal "
            "correct over labelled."
        )
    return rules.fact(
        "weakest_classes",
        sentence,
        accuracy="user's",
        min_labelled=MIN_TO_RANK,
        lowest=[r["class"] for r in named],
        ranked=ranked,
        too_few_to_rank=too_few,
        no_estimate=no_estimate,
    )


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


def _refused_class_map(refusal: dict[str, Any], design: str) -> dict[str, Any]:
    """A ranking design refused for a Studio class map, with the designs it takes.

    The confidence and proportional designs stratify the map by its
    confidence margin, which a class map without top-k scores does not have
    (the reason ends with where top-k scores come from); the refusal comes
    before any pixel is sampled when the result declares its band type, and
    after the grid is sampled otherwise (the band's type is known only then).
    """
    out: dict[str, Any] = {
        "ranked": False,
        "result_id": refusal.get("result_id"),
        "property_name": refusal.get("property_name"),
        "class_map": True,
        "design": design,
        "reason": (
            f"design='{design}' stratifies the map by its confidence margin, which "
            "a class map without top-k scores does not have. " + rules.CLASS_MAP_REFUSAL
        ),
        "designs_for_a_class_map": list(CLASS_MAP_DESIGNS),
        "next_steps": [
            "Plan a sample of this class map with design='random' (a simple "
            "random sample) or design='class' (strata by map class, at least "
            "min_per_class labels each, or all the windows of a smaller "
            "class), then olmoearth_estimate_map_error on the labels."
        ],
    }
    return rules.add_contract(
        out,
        forbidden_claims=[
            rules.error_rate_without_labels(),
            rules.zone_for_a_class_map(),
        ],
    )


def _class_names(pop: Population, condition: list[int]) -> dict[int, str]:
    """Each map class's name as the class design's stratum: as the map writes it.

    A scores map's class is its code, named ``class N`` as
    ``oe-inferencex sample --design class`` names it; so is a class map's
    class whose name is empty, reserved or taken, since the package refuses
    those.
    """
    names: dict[int, str] = {}
    for code in sorted({c for c in condition if c >= 0}):
        name = ""
        if pop.classes is not None and code < len(pop.classes):
            name = str(pop.classes[code]).strip()
        # "unrecorded" is the package's name for windows with no condition
        if not name or name == "unrecorded" or name in names.values():
            name = f"class {code}"
        names[code] = name
    return names


#: What a confidence or proportional plan's ``strata_sizes`` and
#: ``allocation`` are, in its result (``designs.confidence_strata``).
STRATA_MEANING = (
    "The strata are olmoearth-inferencex's quantile strata of the confidence "
    "margin, stratum 0 the least confident; strata_sizes and allocation list "
    "their windows and labels in that order."
)


def _per_stratum(sample: dict[str, Any], pop: Population) -> list[dict[str, Any]]:
    """Each stratum of a class design with its class, windows and labels.

    The package's condition design (the class design is one) takes as strata
    the condition values present in the population, in ascending order
    (``common.sampling._condition_index``), so a map that holds no window of
    a class has no stratum for it and every later stratum's index runs ahead
    of its class: on agent trial 12's map, ``condition.values`` were [0, 1,
    2, 4, 5, 6, 7, 8] and the model read stratum 4 (class 5, 471 windows) as
    class 4 (5,180). Stratum ``h`` is ``condition["values"][h]``, its windows
    ``sizes[h]`` and its labels ``allocation[h]``, each the package's; the
    name is the scores file's where known. A Studio class map's condition
    values are the agent's internal codes (:func:`_code_classes`: 0 to K-1
    over the classes present, sorted), so its ``class`` is the class as the
    map writes it (``pop.classes[code]``) and the code is ``code``: a map
    writing ids [1, 2, 3, 5] has codes 0 to 3, and code 3 is its class 5
    (the step 11 review). A class map with top-k scores is read the same
    way, its codes in its declared classes' order.
    """
    values = list(sample["condition"]["values"])
    written = pop.classes if pop.score_kind in CLASS_LABELLED else None
    rows: list[dict[str, Any]] = []
    for h, (size, labels) in enumerate(zip(sample["sizes"], sample["allocation"])):
        value = values[h] if h < len(values) else None
        entry: dict[str, Any] = {"stratum": h, "class": value}
        if written is not None and isinstance(value, int) and 0 <= value < len(written):
            entry["class"] = _jsonable(written[value])
            entry["code"] = value
        else:
            name = pop.class_name(value)
            if name is not None:
                entry["class_name"] = name
        entry["windows"] = int(size)
        entry["labels"] = int(labels)
        rows.append(entry)
    return rows


def _draw(
    estimate: Any, pop: Population, design: str, budget: int, seed: int, floor: int
) -> dict[str, Any]:
    """The package's draw of ``design`` over ``pop``.

    The class design is the package's condition design with each window's
    map class as its condition and ``floor`` labels per class at least
    (``class_floor``), its sample marked as ``oe-inferencex sample --design
    class`` marks its sidecar (``condition.by`` "map class",
    ``min_per_class``), so that ``estimate_error_rate`` and
    ``estimate_per_class`` read its strata as the map's classes.
    """
    if design != CLASS_DESIGN:
        return dict(
            estimate.sample_for_estimation(
                pop.margin,
                budget,
                design=design,
                p1=pop.p1 if design == "confidence" else None,
                seed=seed,
            )
        )
    condition = [
        c if math.isfinite(m) else -1 for c, m in zip(pop.map_class, pop.margin)
    ]
    try:
        sample = dict(
            estimate.sample_for_estimation(
                pop.margin,
                budget,
                design="condition",
                seed=seed,
                condition=condition,
                condition_names=_class_names(pop, condition),
                class_floor=floor,
            )
        )
    except ValueError as exc:
        msg = f"{exc} (here: min_per_class, default {estimate.CLASS_FLOOR}, or budget)"
        raise ValueError(msg) from None
    sample["condition"] = {
        **sample["condition"],
        "by": "map class",
        "min_per_class": floor,
    }
    return sample


def _scores_raster(pop: Population) -> tuple[str, dict[str, Any]]:
    """The raster a direct run's scores file was read from, and the file's fields.

    A sample a certificate rests on is drawn from the map itself, so it needs
    the scores file ``olmoearth_scores_from_file`` writes, which records its
    raster's path, read over every channel: a certificate is of the map as
    the package reads it.
    """
    path = pop.source.get("scores_path")
    meta = load_scores_file(str(path)).meta if path else {}
    raster = meta.get("raster") if isinstance(meta.get("raster"), dict) else {}
    if not path or meta.get("score_kind") != "window_confidence" or not raster:
        msg = (
            "seed_from_map draws from the map itself: it needs a scores file "
            "that olmoearth_scores_from_file wrote from a model run's output "
            "GeoTIFF (score_kind window_confidence), passed as scores_path, and "
            "is refused for any other scores file, inline scores or a Studio "
            "result (no raster to digest)"
        )
        raise ValueError(msg)
    if not raster.get("path"):
        msg = (
            "this scores file records no raster path (written before seed_from_map "
            "existed): read the run again with olmoearth_scores_from_file"
        )
        raise ValueError(msg)
    fill = meta.get("fill_class")
    if isinstance(fill, dict) and fill.get("excluded"):
        msg = (
            "a certificate is of the map as the package reads it, every channel; "
            "this scores file leaves the label-fill channel out: read the run "
            "again with include_fill_class=true and plan on that file"
        )
        raise ValueError(msg)
    real = os.path.realpath(str(raster["path"]))
    roots = read_roots()
    if not any(os.path.commonpath([r, real]) == r for r in roots):
        raise ValueError(f"the scores file's raster is not under {' or '.join(roots)}")
    if not os.path.isfile(real):
        raise ValueError(f"the scores file's raster {raster.get('file')!r} is gone")
    return real, meta


def labels_sha256(wrong: list[int]) -> str:
    """The first 12 hex digits of the sha256 of the labels, in the design's order.

    The digest the provenance summary gives inline labels
    (``provenance.log.labels_digest`` of the same list), so a reader matches
    a result's labels with the call that gave them. Two results on one design
    file with different digests graded different labels, which the report
    states (``reporting.narrative.LABELS_CHANGED``).

    Examples
    --------
    >>> labels_sha256([0, 1, 0]) == labels_sha256([0, 1, 0]) != labels_sha256([1, 0, 0])
    True
    """
    blob = json.dumps([int(w) for w in wrong])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def plan_work_dir(block: dict[str, Any]) -> Path:
    """The working folder of a plan drawn with ``seed_from_map``, checked.

    The design file names it (``certificate.work_dir``), and the certificate
    route writes there: the labels into the package's sheet, ``certify``'s
    zone and the bundle. So it must be a folder the plan tool makes, under
    ``<scores root>/certificates/``; a design file naming any other folder
    (even one under the scores root, such as another sheet of the user's) is
    refused.

    Parameters
    ----------
    block
        The design file's ``certificate`` block.

    Returns
    -------
    Path
        The folder, resolved.

    Examples
    --------
    >>> plan_work_dir({"work_dir": "/elsewhere"})  # doctest: +SKIP
    Traceback (most recent call last):
    ValueError: the plan's working folder is not one ...
    """
    base = os.path.realpath(write_root() / CERTIFICATE_DIR)
    raw = str(block.get("work_dir") or "").strip()
    real = os.path.realpath(raw) if raw else ""
    if not real or real == base or os.path.commonpath([base, real]) != base:
        msg = (
            "the plan's working folder is not one olmoearth_plan_label_sample "
            f"wrote (under {base}); plan again with seed_from_map"
        )
        raise ValueError(msg)
    return Path(real)


#: Held while a plan's sheet is read, checked and written (:func:`fill_sheet`)
#: or read for labels (:func:`_sheet_holds_labels`). The estimate fixes the
#: labels from a worker thread while the zone and the certificate fix them on
#: the event loop; the web bridge serves several sessions, so two of them
#: grading one plan would otherwise both read an empty ``wrong`` column and
#: both write, each told its own labels were the ones fixed.
_SHEET_LOCK = threading.Lock()


def _sheet_holds_labels(sheet: Path) -> bool:
    """Whether the package's sheet has a ``wrong`` cell filled in."""
    with _SHEET_LOCK:
        if not sheet.is_file():
            return False
        with open(sheet, encoding="utf-8", newline="") as fh:
            return any((row.get("wrong") or "").strip() for row in csv.DictReader(fh))


def _replace_sheet(sheet: Path, header: list[str], body: list[list[str]]) -> None:
    """Write the sheet whole: a file beside it first, then moved over it.

    ``os.replace`` within one folder is atomic, so a reader (the plan's
    check, the package's ``certify`` in its own process) finds the sheet as
    it was or as written, never cut short; a write that fails leaves the
    sheet as it was and no file beside it.
    """
    staged = sheet.with_name(sheet.name + ".tmp")
    try:
        with open(staged, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(body)
        os.replace(staged, sheet)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def fill_sheet(sheet: Path, indices: list[int], wrong: list[int]) -> None:
    """Write the reviewer's labels into the ``wrong`` column of the package's sheet.

    The sheet is the one ``oe-inferencex sample`` wrote for the plan; its rows
    must be the plan's windows in the plan's order (its ``index`` column).
    Every other cell is written back as read, so the sample a certificate's
    verify draws again is the same. Labels are fixed the first time they are
    written: a sheet that already holds other labels is refused
    (:data:`LABELS_FIXED`), the same labels again change nothing. The read,
    the check and the write are one step under :data:`_SHEET_LOCK`, whichever
    thread calls it, and the sheet is replaced whole
    (:func:`_replace_sheet`): of two sessions grading one plan with
    different labels, the second is refused.

    Parameters
    ----------
    sheet
        The package's ``sample.csv`` in the plan's working folder.
    indices
        The plan's windows, in order.
    wrong
        1 where the map is wrong, else 0, in the same order.

    Examples
    --------
    >>> fill_sheet(Path("certificates/d/sample.csv"), [4, 9], [0, 1])  # doctest: +SKIP
    """
    with _SHEET_LOCK:
        with open(sheet, encoding="utf-8", newline="") as fh:
            rows = [row for row in csv.reader(fh) if row]
        header = rows[0] if rows else []
        if "index" not in header or "wrong" not in header:
            raise ValueError(f"{sheet.name} is not a sheet oe-inferencex sample wrote")
        at_index, at_wrong = header.index("index"), header.index("wrong")
        body = [row + [""] * (len(header) - len(row)) for row in rows[1:]]
        if [int(row[at_index]) for row in body] != list(indices):
            msg = (
                f"{sheet.name} does not list the plan's windows in its order; plan "
                "again with seed_from_map"
            )
            raise ValueError(msg)
        new = [str(int(value)) for value in wrong]
        held = [row[at_wrong].strip() for row in body]
        other = [i for i, old, value in zip(indices, held, new) if old and old != value]
        if other:
            msg = (
                f"this sample's sheet ({sheet.name}) already holds other labels on "
                f"{len(other)} of {len(indices)} windows (first: window {other[0]}): "
                f"{LABELS_FIXED}. Other labels need a new sample"
            )
            raise ValueError(msg)
        if held == new:
            return
        for row, value in zip(body, new):
            row[at_wrong] = value
        _replace_sheet(sheet, header, body)


def hold_plan_labels(design: dict[str, Any], wrong: list[int]) -> Path | None:
    """Fix a ``seed_from_map`` plan's labels in its sheet the first time they are graded.

    ``olmoearth_estimate_map_error``, ``olmoearth_certify_zone`` and
    ``olmoearth_write_certificate`` write the labels they grade on such a plan
    into the package's sheet (:func:`fill_sheet`), which refuses other labels
    than the ones it holds: a certificate rests on labels fixed before a
    result was seen, and relabelling until a zone certifies is what it cannot
    show. Any other plan, or a plan whose sheet is gone (no certificate can
    then be written), is left as it is.

    Parameters
    ----------
    design
        The design file, as read.
    wrong
        The labels, in the design's order.

    Returns
    -------
    Path or None
        The sheet written, or ``None``.

    Examples
    --------
    >>> hold_plan_labels({"design": "random", "sample": {"indices": [1]}}, [0])
    """
    block = design.get("certificate")
    if not isinstance(block, dict) or block.get("seed_rule") != "from_map":
        return None
    sheet = plan_work_dir(block) / CERT_SAMPLE
    if not sheet.is_file():
        return None
    fill_sheet(sheet, [int(i) for i in design["sample"]["indices"]], wrong)
    return sheet


async def _map_seeded_draw(
    estimate: Any,
    pop: Population,
    args: dict[str, Any],
    budget: int,
    stem: str,
    folder: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The random draw a certificate rests on, and what fixes it.

    ``oe-inferencex sample RASTER --design random --seed-from-map --alpha A``
    (with ``--delta``, ``--rule``, the scores file's ``--patch`` and
    ``--logits``) is run by :func:`inferencex.run_cli_async` in this plan's
    working folder under the scores root (:data:`CERTIFICATE_DIR`): the seed
    is 128 bits of sha256 over the map's digest and those settings
    (``certify.certificate.draw_seed``), and the sheet and sidecar it writes
    are what ``certify`` and ``certificate`` read. The agent's sample is then
    the package's draw in Python with that seed (``sample_for_estimation``,
    ``stable=True``, the fixed random order a certificate's verify reads),
    checked to be the same windows, with the same confidence and class, as
    the package's sheet; otherwise the scores file is not the package's
    reading of the raster, and the plan is refused. The draw is a function
    of the map and the settings, so the same plan lands in the same folder:
    once its sheet holds labels, or a certificate was written there, it is
    not drawn again (that would clear the labels), and the plan is refused
    with the design file to grade instead. ``stem`` names the plan's design
    file and ``folder`` its working folder, which the plan names by the
    margins' digest alone: the same map under another scores file, whose
    design file has a name of its own, is drawn in the same folder, and its
    labels are fixed there once.

    Returns
    -------
    tuple
        The sample, as :func:`_draw` returns one, and the design file's
        ``certificate`` block (seed, map digest, alpha, delta, rule, raster,
        working folder, sheet, the run's date window).
    """
    raster, meta = _scores_raster(pop)
    if args.get("alpha") is None:
        msg = (
            "seed_from_map fixes certify's alpha with the draw, before any label: "
            "pass alpha (and delta or rule if not the defaults)"
        )
        raise ValueError(msg)
    alpha = float(args["alpha"])
    delta = float(args.get("delta", DEFAULT_DELTA))
    rule = str(args.get("rule", "prefix"))
    if rule not in ZONE_RULES:
        raise ValueError(f"rule must be one of {list(ZONE_RULES)}, got {rule!r}")
    work = safe_path(f"{CERTIFICATE_DIR}/{folder}", root=write_root())
    sheet = work / CERT_SAMPLE
    bundle = work / CERT_BUNDLE
    if bundle.exists() or _sheet_holds_labels(sheet):
        held = "a certificate" if bundle.exists() else "labels"
        # This plan's design file, or the one an earlier version named after
        # the folder; another scores file of the same map has one of its own.
        named = [safe_path(f"{s}.json", root=write_root()) for s in (stem, folder)]
        design_file = next((p for p in named if p.is_file()), None)
        grade = (
            f"Grade it from that plan's design file, {design_file}"
            if design_file is not None
            else "Grade it from the design file of the plan that drew it"
        )
        msg = (
            "this sample (the same map, budget, alpha, delta and rule) was drawn "
            f"before, and its working folder {work} holds {held}: drawing it "
            f"again would clear its labels. {grade}; {LABELS_FIXED}"
        )
        raise ValueError(msg)
    work.mkdir(parents=True, exist_ok=True)
    argv = [
        "sample",
        raster,
        "--design",
        "random",
        "--budget",
        str(budget),
        "--seed-from-map",
        f"--alpha={alpha!r}",
        f"--delta={delta!r}",
        f"--rule={rule}",
        "--patch",
        str(int(meta.get("patch_px") or 4)),
        "--out",
        str(sheet),
    ]
    if meta.get("values") == "logits":
        argv.append("--logits")
    await inferencex.run_cli_async(argv, cwd=work)
    side = json.loads((work / "sample.json").read_text(encoding="utf-8"))
    seed = side.get("seed")
    if side.get("seed_rule") != "from_map" or not isinstance(seed, int):
        raise ValueError("oe-inferencex sample did not take its seed from the map")
    # The draw orders every window of the map: in a worker thread, as the
    # command line before it, so the event loop keeps serving other sessions.
    drawn_here = await asyncio.to_thread(
        estimate.sample_for_estimation,
        pop.margin,
        budget,
        design="random",
        seed=seed,
        stable=True,
    )
    sample = dict(drawn_here)
    drawn = [int(i) for i in side.get("indices") or []]
    ours = [int(i) for i in sample["indices"].tolist()]
    if ours != drawn or side.get("n_population") != pop.n_valid:
        msg = (
            f"the package drew {len(drawn)} windows of {side.get('n_population')} "
            f"from the raster, and these scores hold {pop.n_valid} valid windows "
            "that the same seed draws differently: the scores file is not the "
            "package's reading of this raster; read the run again"
        )
        raise ValueError(msg)
    with open(sheet, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            i = int(row["index"])
            conf, cls = float(row["confidence"]), int(row["map_class"])
            off = abs(conf - pop.margin[i]) > 1e-6 * max(1.0, abs(conf))
            if off or cls != pop.map_class[i]:
                msg = (
                    f"window {i}: the package's sheet shows confidence {conf:g} "
                    f"and class {cls}, the scores file {pop.margin[i]:g} and "
                    f"{pop.map_class[i]}; read the run again"
                )
                raise ValueError(msg)
    plan = side.get("certify") if isinstance(side.get("certify"), dict) else {}
    block = {
        "seed_rule": "from_map",
        "seed": seed,
        "map_sha256": side.get("map_sha256"),
        "alpha": alpha,
        "delta": delta,
        "rule": rule,
        "level_cut": plan.get("level_cut"),
        "raster": raster,
        "work_dir": str(work),
        "sample_csv": str(sheet),
        "date_window": meta.get("date_window"),
    }
    return sample, block


def _label_budget_advice(
    args: dict[str, Any], pop: Population
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The package's label-budget advice for a class map or the class design.

    ``plan.plan_error_rate`` for the whole-map error rate's 95% interval at
    the stated ``width`` (:data:`DEFAULT_WIDTH` unless given), for a map wrong
    at ``expected_error_rate`` (the package's 50% unless given), and, with a
    stated ``class_share``, ``plan.plan_class`` for one class of that share
    drawn by the map at half of it, its intervals held together over the
    map's classes. Both plan random labels; their numbers are the package's,
    kept as returned less the budget-by-budget table. A width or rate the
    package refuses is refused (no file is written yet); a class share it
    refuses is stated as its refusal.
    """
    plan_pkg = inferencex.load("plan")
    width = float(args.get("width", DEFAULT_WIDTH))
    rate = args.get("expected_error_rate")
    n_valid = pop.n_valid
    planned = _jsonable(
        plan_pkg.plan_error_rate(
            n_valid, width, float(rate) if rate is not None else None
        )
    )
    error_rate = {k: planned[k] for k in _ADVICE_KEYS if k in planned}
    error_rate["planned_by"] = "oe_inferencex.plan.plan_error_rate"
    share = args.get("class_share")
    if share is None:
        return error_rate, None
    n_classes = len({c for c, m in zip(pop.map_class, pop.margin) if math.isfinite(m)})
    try:
        by_class = _jsonable(plan_pkg.plan_class(n_valid, float(share), n_classes))
    except ValueError as exc:
        return error_rate, {
            "question": "class",
            "share": float(share),
            "n_classes": n_classes,
            "refusal": str(exc),
            "planned_by": "oe_inferencex.plan.plan_class",
        }
    one_class = {k: by_class[k] for k in _ADVICE_KEYS if k in by_class}
    one_class["planned_by"] = "oe_inferencex.plan.plan_class"
    return error_rate, one_class


#: Appended to the class design's ``label_budget`` fact: the budget plans a
#: random sample's interval, not either interval of this design, and the
#: union bound's width against a random sample's is the package's
#: conditional statement (``estimate.error_rate._estimate_core``'s docstring
#: and ``CLASS_WHOLE_MAP``), not an unconditional one.
CLASS_DESIGN_BUDGET_NOTE = (
    " This budget is for a random sample's interval, not this class design's: "
    "olmoearth_estimate_map_error reports first the interval valid at every "
    "sample size, then the design's union bound over the classes, which the "
    "package says is wider than a random sample's interval when the classes' "
    "error rates are close and can be narrower when they are far apart; if "
    "only the whole-map rate is needed, design='random' can give a narrower "
    "interval."
)


def _label_budget_fact(
    advice: dict[str, Any], design: str, planned: int
) -> dict[str, Any]:
    """The ``label_budget`` fact: ``plan.plan_error_rate``'s advice, in a sentence.

    The rate it plans at is the caller's, or the package's 50% when none is
    stated (``error_rate_stated``), which the sentence then names as an
    assumption (the advice's ``rate_note`` says more); under the class design
    it adds :data:`CLASS_DESIGN_BUDGET_NOTE`.
    """
    n = advice.get("n_population")
    width = 100.0 * float(advice.get("width", DEFAULT_WIDTH))
    labels = advice.get("labels")
    prob = advice.get("probability_at_labels")
    if labels is None or not isinstance(prob, (int, float)):
        why = str(advice.get("refusal") or "it advises none").rstrip(".")
        sentence = (
            "No label budget is advised for a 95% interval no wider than "
            f"{width:g} points over the {n} valid windows: the package says {why}."
        )
    else:
        rate = rules.percent(advice.get("error_rate"))
        wrong = (
            f"wrong {rate} of the time"
            if advice.get("error_rate_stated")
            else f"assumed wrong {rate} of the time (the package's planning rate "
            "when none is stated: see rate_note)"
        )
        sentence = (
            f"Planned before any label by the package, for a random sample of "
            f"the {n} valid windows of a map {wrong}: {labels} labels give a 95% "
            f"interval no wider than {width:g} points with probability "
            f"{float(prob):.3g}; this plan draws {planned}."
        )
    if design == CLASS_DESIGN:
        sentence += CLASS_DESIGN_BUDGET_NOTE
    return rules.fact(
        "label_budget",
        sentence,
        width=advice.get("width"),
        error_rate=advice.get("error_rate"),
        labels=labels,
        probability=prob,
        n_population=n,
        planned=planned,
    )


def _class_budget_fact(advice: dict[str, Any], design: str) -> dict[str, Any]:
    """The ``class_budget`` fact: ``plan.plan_class``'s advice, in a sentence."""
    share = rules.percent(advice.get("share"))
    labels = advice.get("labels")
    prob = advice.get("probability_at_labels")
    if labels is None or not isinstance(prob, (int, float)):
        why = str(advice.get("refusal") or "it advises none").rstrip(".")
        sentence = (
            f"No label budget is advised for a class that is {share} of the map: "
            f"the package says {why}."
        )
    else:
        sentence = (
            f"Planned before any label by the package, for random labels: a class "
            f"that is {share} of the {advice.get('n_population')} valid windows, "
            f"drawn by the map at {float(advice.get('drawn_ratio', 0.5)):g} of its "
            f"share, is flagged drawn too little with probability {float(prob):.3g} "
            f"from {labels} labels, the intervals of all {advice.get('n_classes')} "
            "classes holding together at 95%."
        )
    if design == CLASS_DESIGN:
        sentence += (
            " The class design gives every class at least min_per_class labels, "
            "or all the windows of a smaller class, which this does not plan."
        )
    return rules.fact(
        "class_budget",
        sentence,
        share=advice.get("share"),
        n_classes=advice.get("n_classes"),
        labels=labels,
        probability=prob,
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
    if design == TWO_MAP_DESIGN:
        return await _plan_two_maps(args, ctx, estimate)
    if args.get("result_id_b") or args.get("scores_path_b"):
        # A map b with a one-map design would be dropped without a word.
        raise ValueError(
            "result_id_b and scores_path_b are map b of design='disagreement' "
            "(which of two maps is more accurate): pass design='disagreement', "
            "or leave them out to plan a sample of one map"
        )
    budget = int(args["budget"])
    seed = int(args.get("seed", 0))
    from_map = bool(args.get("seed_from_map", False))
    if from_map and design != "random":
        msg = (
            "seed_from_map draws the sample a certificate rests on, a simple "
            "random sample: pass design='random'"
        )
        raise ValueError(msg)
    if from_map and seed != 0:
        raise ValueError("give seed or seed_from_map, not both")
    if not from_map:
        # As `oe-inferencex sample` refuses --alpha, --delta and --rule without
        # --seed-from-map: given here, they would be taken and then forgotten.
        stray = [
            key
            for key, default in (("alpha", None), ("delta", DEFAULT_DELTA))
            if args.get(key) is not None and float(args[key]) != default
        ]
        if args.get("rule") is not None and str(args["rule"]) != "prefix":
            stray.append("rule")
        if stray:
            msg = (
                f"{' and '.join(stray)} given without seed_from_map: a plan fixes "
                "certify's alpha, delta and rule with the draw only with "
                "seed_from_map (a random plan on a direct run's scores file, for a "
                "certificate); otherwise give them to olmoearth_certify_zone, "
                "chosen before the labels are read"
            )
            raise ValueError(msg)
    max_listed = int(args.get("max_listed", DEFAULT_MAX_LISTED))
    floor = int(args.get("min_per_class", estimate.CLASS_FLOOR))
    # A Studio classification band is read as a class map only for a design
    # that reads no confidence; a ranking design is refused for it.
    pop = await _population(args, ctx, class_map=design in CLASS_MAP_DESIGNS)
    if isinstance(pop, dict):
        if pop.get("class_map"):
            return _refused_class_map(pop, design)
        return _refused_band(pop)
    class_map = pop.score_kind == CLASS_MAP
    # A class map, with top-k scores or without, is labelled with the class
    # seen; only one without them has no confidence (class_map).
    labels_classes = pop.score_kind in CLASS_LABELLED
    n_valid = pop.n_valid
    # At the largest Studio grid a budget above the valid windows plans every
    # one of them and states the labels left over (exp86: "your extra 127
    # labels have nowhere to go", 300 - 173 worked out by the model); any
    # other budget above them is refused, with the options that work.
    if budget < 1 or (budget > n_valid and not _at_studio_ceiling(pop)):
        raise _budget_error(budget, pop)
    requested, budget = budget, min(budget, n_valid)
    # The package's label budgets, before the draw: a width it refuses writes
    # no file.
    advice: tuple[dict[str, Any], dict[str, Any] | None] | None = None
    if class_map or design == CLASS_DESIGN:
        advice = _label_budget_advice(args, pop)
    # The population as the design file holds it, which the later tools read:
    # certify_zone its grid, the estimate its score kind, classes, class names
    # and source, the certificate the source's Studio block.
    population: dict[str, Any] = {
        "n_windows": len(pop.margin),
        "n_valid": n_valid,
        "grid": list(pop.grid) if pop.grid else None,
        "score_kind": pop.score_kind,
        "margin": _jsonable(pop.margin),
        "map_class": pop.map_class,
        "source": _jsonable(pop.source),
    }
    if labels_classes:
        population["classes"] = _jsonable(pop.classes)
    elif pop.class_names:
        # A scores file's class names, which the estimate's per-class facts
        # name each class by.
        population["class_names"] = {str(k): v for k, v in pop.class_names.items()}
    # A digest of that population and of what the draw reads beside the
    # margins (p1 for the confidence design, min_per_class for the class
    # design) keeps two plans apart that the later tools would read
    # differently: the same margins on two window grids, under two scores
    # files, with other map classes, another p1 or another floor, at one
    # design, budget and seed, would share a file name, and the later plan
    # would overwrite the earlier one's design.
    drawn_from: dict[str, Any] = {}
    if design == "confidence":
        drawn_from["p1"] = pop.p1
    elif design == CLASS_DESIGN:
        drawn_from["min_per_class"] = floor
    digest = hashlib.sha256(
        json.dumps(_jsonable([population, drawn_from]), sort_keys=True).encode()
    ).hexdigest()
    origin = slug(str(pop.source.get("result_id") or "scores"), 12)
    certificate: dict[str, Any] | None = None
    if from_map:
        # alpha, delta and the rule enter the seed: each gets its own plan.
        fixed = (args.get("alpha"), args.get("delta", DEFAULT_DELTA))
        tag = slug(f"a{fixed[0]}_d{fixed[1]}_{args.get('rule', 'prefix')}", 40)
        stem = f"design_{origin}_{digest[:8]}_{design}_{budget}_smap_{tag}"
        # The working folder keeps the name it had, from the margins' digest:
        # the draw is the map's and the settings', and its sheet's labels are
        # fixed once (LABELS_FIXED), so the same map under another scores
        # file, or a plan an earlier version drew, finds that folder again.
        keyed: Any = [pop.margin, pop.map_class] if class_map else pop.margin
        margin_key = hashlib.sha256(json.dumps(_jsonable(keyed)).encode()).hexdigest()
        folder = f"design_{origin}_{margin_key[:8]}_{design}_{budget}_smap_{tag}"
        sample, certificate = await _map_seeded_draw(
            estimate, pop, args, budget, stem, folder
        )
        seed = int(certificate["seed"])
    else:
        stem = f"design_{origin}_{digest[:8]}_{design}_{budget}_s{seed}"
        sample = _draw(estimate, pop, design, budget, seed, floor)
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

    payload = {
        "format": DESIGN_FORMAT,
        "package": {"name": inferencex.DISTRIBUTION, "version": inferencex.version()},
        "design": design,
        "budget": budget,
        "budget_requested": requested,
        "seed": seed,
        "sample": _jsonable(sample),
        "population": population,
        "to_label": to_label,
    }
    if certificate is not None:
        payload["certificate"] = certificate
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
            _how_to_label_a_class_map(design)
            if labels_classes
            else "For each window, in this order, set wrong = 1 when the map's "
            "class (map_class) is not what is on the ground there, else 0 (and "
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
    if "sizes" in sample and isinstance(sample.get("condition"), dict):
        # The class design's strata are the classes the map holds, in
        # ascending order: a map missing a class shifts every later stratum,
        # so stratum h need not be class h (agent trial 12, N1/files: class 4
        # given stratum 4's 471 windows instead of its own 5,180).
        out["per_stratum"] = _per_stratum(sample, pop)
    elif "sizes" in sample:
        out["strata_sizes"] = _jsonable(sample["sizes"])
        out["allocation"] = _jsonable(sample["allocation"])
        out["strata_meaning"] = STRATA_MEANING
    if design == "confidence":
        out["design_note"] = (
            "Strata by confidence margin, labels allocated from the model's own "
            "top-1 score; the score only allocates labels, so the estimate stays "
            "design-unbiased whatever its calibration. A certified zone needs "
            "design='random', which makes one possible, not certain."
        )
    if design == CLASS_DESIGN:
        out["min_per_class"] = floor
        out["design_note"] = (
            "Strata are the map's classes (Olofsson et al. 2014): each class gets "
            f"at least {floor} labels, or all its windows if it has fewer, and "
            "the rest of the budget goes to the classes in proportion to their "
            "size. Each class's labels are a simple random sample of it, so "
            "olmoearth_estimate_map_error gives each class's user's accuracy an "
            "exact interval; for the whole map it reports first the interval "
            "valid at every sample size, then the design's union bound over the "
            "classes. per_stratum gives each stratum's class, as the map writes "
            "it, windows and labels: the strata are the classes the map holds, "
            "in order, so stratum h need not be class h."
        )
        # The package's own note on labelling this design's sheet blind.
        out["class_blind_note"] = str(estimate.CLASS_BLIND_NOTE)
    if labels_classes:
        out["map_classes"] = {str(c): v for c, v in enumerate(pop.classes or [])}
    if class_map:
        out["class_map_note"] = rules.CLASS_MAP_NO_CONFIDENCE
    if pop.source.get("assumption"):
        out["assumption"] = pop.source["assumption"]
    if pop.source.get("threshold_note"):
        # A threshold passed for a class map with top-k scores, set aside.
        out["threshold_note"] = pop.source["threshold_note"]
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
    if certificate is not None:
        # A zone read from Studio's own files: the map the seed is drawn from
        # is the raster the agent decoded, not Studio's file.
        decoded = file_studio_kind(pop.source) is not None
        out["seed_from_map"] = {
            "seed": seed,
            "map_sha256": certificate["map_sha256"],
            "alpha": certificate["alpha"],
            "delta": certificate["delta"],
            "rule": certificate["rule"],
            "note": SEED_FROM_STUDIO_RASTER_NOTE if decoded else SEED_FROM_MAP_NOTE,
            "package_sheet": certificate["sample_csv"],
        }
        out["next_steps"].append(
            f"After labelling: olmoearth_certify_zone with this design_path at "
            f"alpha={certificate['alpha']:g} (alpha, delta and rule are fixed by "
            "this plan), then olmoearth_write_certificate with the same labels "
            "and the map's date packs a certified zone into a certificate "
            f"{_certificate_holder(decoded)} can check."
        )
    facts = [_unused_labels(requested, budget, pop)] if unused else []
    if advice is not None:
        # The package's numbers, before any label (plan.plan_error_rate and,
        # for a stated class share, plan.plan_class).
        out["label_budget"] = advice[0]
        facts.append(_label_budget_fact(advice[0], design, budget))
        if advice[1] is not None:
            out["class_budget"] = advice[1]
            facts.append(_class_budget_fact(advice[1], design))
    claims = [rules.error_rate_without_labels()]
    # The sheet keeps the package's draw order: a random draw's order is
    # random, so its first rows are a smaller random sample, a fallback only;
    # a stratified draw lists the strata in turn, least confident first (the
    # class design: class by class), so its first rows are one stratum (a
    # single non-empty stratum is a random draw in effect). Every plan writes
    # a design file, whose routes need every row labelled (agent trial 12,
    # N2/cluster run 23: "A partial labelling is fine" of a certificate's plan).
    random_order = (
        design == "random" or sum(1 for n in sample.get("sizes", []) if n) < 2
    )
    route = _scores_route(pop)
    for_certificate = certificate is not None
    if random_order:
        facts.append(
            rules.prefix_is_random_sample(
                budget,
                route,
                design=design,
                certificate=for_certificate,
                class_map=class_map,
            )
        )
    claims.append(
        rules.subset_labelling_sufficient(
            design,
            by_class=design == CLASS_DESIGN,
            random_order=random_order,
            route=route,
            certificate=for_certificate,
            class_map=class_map,
        )
    )
    if class_map:
        # No design certifies a class map: it has no confidence to rank by.
        claims.append(rules.zone_for_a_class_map())
    else:
        # Every plan's next steps name a certified zone (exp86 round 8,
        # B5/files/2).
        claims.append(
            rules.certification_guaranteed(
                f"this plan's {budget} windows certify nothing until labelled and "
                "tested at an alpha fixed before labelling, and may certify no "
                "zone then"
                if design == "random"
                else f"this {design} design cannot certify at all, and a separate "
                "random plan may certify nothing"
            )
        )
    return rules.add_contract(
        out,
        facts=facts,
        must_state=(
            [f"The plan holds {budget} windows, not the {requested} requested."]
            if unused
            else []
        )
        + _studio_scope(pop.source, n_valid)
        + _topk_must_state(pop.score_kind, pop.source),
        forbidden_claims=claims,
    )


def _topk_must_state(score_kind: Any, source: dict[str, Any]) -> list[str]:
    """What a result on a class map's top-k scores states about them, or nothing.

    A plan, whose confidence design allocates labels by the rank-1 score,
    and a certified zone, the windows of the largest margins: the scores
    are Studio's uncalibrated softmax scores, quantized unless the field
    read (``source["top_k"]``) says otherwise
    (:func:`~olmoearth_agent.tools.statistical_rules.topk_scores_must_state`).
    A scores file pooled from a Studio class map's own files
    (``olmoearth_scores_from_studio``; ``source["studio"]``, kind
    :data:`~olmoearth_agent.tools.review_set.STUDIO_TOPK`) states what its
    window scores pool
    (:data:`~olmoearth_agent.tools.statistical_rules.STUDIO_WINDOW_SCORES_MUST_STATE`):
    its windows are ranked by their pooled top-1 score, not a margin, and a
    pooled mean is not on the pixels' 1/254 grid. Each caller places it as
    the pixel-value route's sentence, within the cap of three; the estimate
    of windows labelled with no design states it too, for such a file only.
    """
    if score_kind == TOPK_SCORE:
        top_k = source.get("top_k")
        quantized = not (isinstance(top_k, dict) and top_k.get("quantized") is False)
        return [rules.topk_scores_must_state(quantized)]
    if file_studio_kind(source) == STUDIO_TOPK:
        return [rules.STUDIO_WINDOW_SCORES_MUST_STATE]
    return []


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


def _has_no_confidence(design: dict[str, Any]) -> bool:
    """Whether a design file samples a class map without top-k scores.

    Such a map has no confidence to rank its windows by: no certified zone
    (``olmoearth_certify_zone`` refuses it, and every result forbids
    ``zone_for_a_class_map``). A class map with top-k scores has one, its
    margin.
    """
    return bool((design.get("population") or {}).get("score_kind") == CLASS_MAP)


def _labels_are_classes(design: dict[str, Any]) -> bool:
    """Whether a design file's labels are the classes seen, as the map writes them.

    A class map, with top-k scores or without (:data:`CLASS_LABELLED`): its
    reviewer writes the class seen in each window (the sheet's
    ``reference_class``), ``wrong`` follows from it, and per-class accuracy
    names each class as the map writes it.
    """
    population = design.get("population") or {}
    return population.get("score_kind") in CLASS_LABELLED


def _class_map_sheet(
    path: str, indices: list[int]
) -> tuple[list[str] | None, list[str] | None]:
    """A class map's filled sheet as ``(reference, wrong)``, in the design's order.

    ``reference_class`` is what the reviewer is asked to write. Filled on
    every drawn window, it is the label, and ``wrong`` is returned beside it
    only when filled on every window too (the caller then holds the two to
    each other); ``wrong`` filled on some windows only is refused, since it
    is read from ``reference_class``. Filled on some windows only,
    ``reference_class`` is refused, naming the first window without one,
    whatever ``wrong`` holds. Empty on every window, ``wrong`` is the label,
    and the first window with neither is named.
    """
    sheet = _read_labels_csv(path)
    rows = [sheet.get(i, ("", "")) for i in indices]
    n = len(indices)
    no_class = [i for i, (_w, seen) in zip(indices, rows) if not seen]
    no_wrong = [i for i, (w, _seen) in zip(indices, rows) if not w]
    if not no_class:
        if no_wrong and len(no_wrong) < n:
            msg = (
                f"wrong is filled on {n - len(no_wrong)} of {n} design windows "
                f"and blank on the rest (first: window {no_wrong[0]}): it is read "
                "from reference_class, so leave it blank on every window or fill "
                "it on every window"
            )
            raise ValueError(msg)
        refs = [seen for _w, seen in rows]
        return refs, (None if no_wrong else [w for w, _seen in rows])
    if len(no_class) < n:
        msg = (
            f"{len(no_class)} of {n} design windows have no reference_class "
            f"(first: window {no_class[0]}): write the class seen in every drawn "
            "window, or leave reference_class blank on every window to grade "
            "the error rate from wrong alone"
        )
        raise ValueError(msg)
    if no_wrong:
        msg = (
            f"{len(no_wrong)} of {n} design windows have no label yet (first: "
            f"window {no_wrong[0]}): write the class seen in reference_class (or "
            "wrong, 0 or 1, for the error rate alone) for every drawn window, "
            "and no others"
        )
        raise ValueError(msg)
    return None, [w for w, _seen in rows]


def _class_map_labels(
    args: dict[str, Any], indices: list[int], population: dict[str, Any]
) -> tuple[list[int], list[int] | None, list[Any], list[str]]:
    """A class map's labels: wrong, reference codes, the class table and the outsiders.

    The reviewer writes the class seen in each window (``reference``, or the
    sheet's ``reference_class``) as ``map_class`` writes classes, matched as
    the two-map design matches them (:func:`_class_key`); ``wrong`` is 1
    where it is not the window's map class. A class the map never writes
    gets a code after the map's (the package calls it never predicted) and
    is listed; references that all lie outside the map's classes are
    refused. ``wrong`` alone gives the error rate with no per-class table;
    given with the references, it must agree with them window by window.
    The sheet (``labels_path``) is read the same way: a ``reference_class``
    filled on some windows only is refused, naming the first without one; a
    full ``reference_class`` is the label, and a ``wrong`` column filled
    beside it (on every window, or on none) is held to it as the arguments
    are; with no ``reference_class`` at all, ``wrong`` is the label.
    """
    classes = list(population.get("classes") or [])
    map_class = [int(c) for c in population["map_class"]]
    wrong_in = args.get("wrong")
    reference = args.get("reference")
    if wrong_in is None and reference is None and args.get("labels_path"):
        reference, wrong_in = _class_map_sheet(str(args["labels_path"]), indices)
    if wrong_in is None and reference is None:
        msg = (
            "pass reference (the class seen in each window, as map_class writes "
            "classes, in the design's order), wrong (0/1 per window) or "
            "labels_path (the design's sheet)"
        )
        raise ValueError(msg)
    table = list(classes)
    codes: list[int] | None = None
    outside: list[str] = []
    derived: list[int] | None = None
    if reference is not None:
        refs = ["" if r is None else str(r).strip() for r in reference]
        if len(refs) != len(indices):
            msg = f"{len(refs)} references for the {len(indices)} windows of the design"
            raise ValueError(msg)
        blank = [i for i, r in zip(indices, refs) if r in ("", "?")]
        if blank:
            msg = (
                f"{len(blank)} of {len(indices)} design windows have no class "
                f"(first: window {blank[0]}); one map's estimate needs the class "
                "seen in every drawn window ('?' is taken by the two-map design "
                "only)"
            )
            raise ValueError(msg)
        code_of = {_class_key(v): c for c, v in enumerate(table)}
        codes = []
        for r in refs:
            key = _class_key(r)
            if key not in code_of:
                code_of[key] = len(table)
                table.append(_class_value(r))
                outside.append(r)
            codes.append(code_of[key])
        if all(c >= len(classes) for c in codes):
            msg = (
                f"none of the {len(refs)} references ({', '.join(outside)}) is a "
                f"class the map writes ({', '.join(str(c) for c in classes)}): "
                "graded as given, every window would count as wrong. Write each "
                "window's class as map_class writes classes"
            )
            raise ValueError(msg)
        derived = [int(c != map_class[i]) for c, i in zip(codes, indices)]
    if wrong_in is None:
        return derived or [], codes, table, outside
    values = [int(float(w)) for w in wrong_in]
    if len(values) != len(indices):
        msg = f"{len(values)} labels for the {len(indices)} windows of the design"
        raise ValueError(msg)
    if any(v not in (0, 1) for v in values):
        raise ValueError("wrong must be 0 or 1 per window")
    if derived is not None:
        differ = [i for i, a, b in zip(indices, values, derived) if a != b]
        if differ:
            msg = (
                f"wrong disagrees with the class seen on {len(differ)} of "
                f"{len(indices)} windows (first: window {differ[0]}): wrong is 1 "
                "exactly where the class seen is not map_class; pass one of them, "
                "or correct it"
            )
            raise ValueError(msg)
    return values, codes, table, outside


#: What each interval of a valid-first estimate is, as the result names it.
VALID_WHAT = (
    "valid at every sample size, for every map and number of labels "
    "(estimate_error_rate(valid=True), as oe-inferencex estimate --valid gives "
    "it); report it first"
)
DESIGN_WHAT = (
    "the interval this design earns, as oe-inferencex estimate gives it without "
    "--valid"
)
#: How a valid-first estimate's sentence introduces the package's
#: ``warning``: every whole-map warning of ``estimate_error_rate`` is of the
#: design's interval or its variance (thin strata, no labelled window wrong),
#: never of the one valid at every sample size.
DESIGN_WARNING_LEAD = "The package warns, of the interval this design earns"

#: How the confidence and proportional designs' estimates are reported. The
#: interval each design earns is Wilson's on the design's effective sample
#: size, which the package's ``VALID_NOTE`` calls nominal (it covered 91.3% at
#: 100 labels on a 200-window map, exp105): approximate, never exact. The
#: valid interval was 1.6 to 1.7 times its width on the package's demo tile
#: (``VALID_NOTE``), and its ``VALID_HELP`` expects it to be wider.
NOMINAL_DESIGN_HOW = (
    "Report valid_interval first, the interval valid at every sample size, "
    "then the interval this design earns, Wilson's on the design's effective "
    "sample size, which the package calls nominal: approximate, not valid at "
    "every sample size (valid_note gives the coverage it fell to), so never "
    "call it exact. The valid interval is expected to be the wider of the two."
)

#: How a valid-first estimate is reported, by its design. Under a random
#: sample the design's interval is the same exact interval as the valid one
#: (the package's ``VALID_NOTE``); under the class design it is the union
#: bound over the classes, which covers at least 95% by construction
#: (``CLASS_WHOLE_MAP``), and the valid interval was 0.6 times its width on
#: the package's demo tile (``VALID_NOTE``); under the confidence and
#: proportional designs it is nominal (:data:`NOMINAL_DESIGN_HOW`).
VALID_FIRST_HOW = {
    "random": (
        "Report valid_interval, the interval valid at every sample size; under "
        "a random sample the interval this design earns is the same exact "
        "interval (valid_note), so the two entries of intervals have the same "
        "ends and one statement covers both."
    ),
    CLASS_DESIGN: (
        "Report valid_interval first, the interval valid at every sample size, "
        "then the interval this design earns, the union bound over the "
        "classes; both cover at least 95% at every sample size, and valid_note "
        "gives the package's measure of their widths."
    ),
    "confidence": NOMINAL_DESIGN_HOW,
    "proportional": NOMINAL_DESIGN_HOW,
}


def _interval_entry(which: str, what: str, block: dict[str, Any]) -> dict[str, Any]:
    """One interval of a valid-first estimate, named, with the package's numbers."""
    entry: dict[str, Any] = {"interval": which, "what": what}
    for key in ("estimate", "estimate_range", "low", "high", "half_width", "form"):
        if key in block:
            entry[key] = block[key]
    entry["method"] = block.get("method")
    return entry


def _valid_first_table(
    per_class: dict[str, Any], names: list[Any] | None
) -> dict[str, Any]:
    """``estimate_per_class(valid=True)``'s table with each valid interval first.

    Every key and number is the package's: the overall accuracy's valid
    interval comes before the rest, each class's ``valid`` intervals before
    its design intervals, and a class map's row names its class.
    """
    rows: dict[str, Any] = {}
    for key, row in (per_class.get("per_class") or {}).items():
        c = int(key)
        lead: dict[str, Any] = {"class": names[c]} if names and c < len(names) else {}
        rest = {k: v for k, v in row.items() if k != "valid"}
        rows[key] = {**lead, "valid": row.get("valid"), **rest}
    head = {"overall_accuracy_valid": per_class.get("overall_accuracy_valid")}
    others = {
        k: v
        for k, v in per_class.items()
        if k not in ("overall_accuracy_valid", "per_class")
    }
    return {**head, **others, "per_class": rows}


def _valid_first_estimate(res: dict[str, Any], valid: dict[str, Any]) -> dict[str, Any]:
    """The ``whole_map_estimate`` fact of a valid-first estimate: both intervals, named.

    Fields ``estimate``, ``low`` and ``high`` are the valid interval's (its
    own point; a range when the package gives one), ``design_low`` and
    ``design_high`` the design's; the sentence names each interval and its
    method, the labelled count, the population and any warning of the
    package, named as of the interval this design earns: the package's
    whole-map warnings (strata too thin for a variance, no labelled window
    wrong) are of that interval or its variance, and the valid one takes no
    variance.
    """
    cov = res.get("nominal_coverage")
    level = f"{100.0 * float(cov):g}% " if isinstance(cov, (int, float)) else ""
    point = valid.get("estimate")
    span = valid.get("estimate_range") or [None, None]
    head = (
        f"The map's error rate is estimated at {rules.percent(point)}"
        if point is not None
        else f"The map's error rate lies between {rules.percent(span[0])} and "
        f"{rules.percent(span[1])}"
    )
    sentence = (
        f"{head} from {res.get('n_labelled')} labelled windows of a population of "
        f"{res.get('n_population')}: the {level}interval valid at every sample "
        f"size is {rules.percent(valid.get('low'))} to "
        f"{rules.percent(valid.get('high'))} ({valid.get('method')}); the "
        f"interval this design earns is {rules.percent(res.get('low'))} to "
        f"{rules.percent(res.get('high'))} (method: {res.get('method')})."
    )
    if res.get("warning"):
        warning = str(res["warning"]).rstrip(".")
        sentence += f" {DESIGN_WARNING_LEAD}: {warning}."
    return rules.fact(
        "whole_map_estimate",
        sentence,
        estimate=point,
        low=valid.get("low"),
        high=valid.get("high"),
        design=res.get("design"),
        interval="valid",
        design_low=res.get("low"),
        design_high=res.get("high"),
    )


def _estimate_valid_first(
    args: dict[str, Any], design: dict[str, Any], estimate: Any
) -> dict[str, Any]:
    """``olmoearth_estimate_map_error`` on a design file of one map, any design.

    A scores map's confidence, proportional or random design, a class map's
    random or class design, or the class design of a scores map. The error
    rate is ``estimate_error_rate(valid=True)``: its ``valid_interval``,
    valid at every sample size, is reported first, the interval the design
    earns second, each named (``intervals``), with the package's note on the
    two (``valid_note``, ``VALID_NOTE`` word for word). How the second is
    reported depends on the design (:data:`VALID_FIRST_HOW`): under the
    random design the two are the same exact interval; under the class
    design the second is the union bound over the classes, which covers at
    least 95% too; under the confidence and proportional designs it is
    Wilson's on the design's effective sample size, which the package calls
    nominal (:data:`NOMINAL_DESIGN_HOW`). With reference classes,
    ``estimate_per_class(valid=True)``'s table the same way: each class's
    valid intervals first, its own ``valid_note``. The class design's
    results are worded as the package words them
    (``common.wording.class_wording``: its strata are the map's classes).
    A ``seed_from_map`` plan's labels are fixed in its sheet the first time
    they are graded (:func:`hold_plan_labels`). Every number is the
    package's. The contract: the valid interval and the design in
    ``must_state``; for a stratified design, no simple-random-sample
    interval and no certified zone from it; for a class map without top-k
    scores, no certified zone (``zone_for_a_class_map``), while a class map
    with them is labelled as a class map and certified as a scores map. It
    runs in a worker thread (``_estimate_map_error``), so it reads and writes
    files as a plain function does, and fixes a plan's labels in its sheet
    under the sheet's lock (:func:`fill_sheet`), as the zone and the
    certificate do on the event loop.
    """
    wording = inferencex.load("common.wording")
    sample = design["sample"]
    population = design.get("population") or {}
    indices = [int(i) for i in sample["indices"]]
    plan_design = str(design.get("design"))
    # Two questions: whether the map has no confidence (no certified zone),
    # and whether its labels are the classes seen; a class map with top-k
    # scores answers no and yes.
    class_map = _has_no_confidence(design)
    labels_classes = _labels_are_classes(design)
    names: list[Any] | None = None
    outside: list[str] = []
    n_classes: int | None
    if labels_classes:
        wrong, reference, names, outside = _class_map_labels(args, indices, population)
        n_classes = len(names)
    else:
        wrong, reference = _labels(args, indices)
        n_classes = int(args["n_classes"]) if args.get("n_classes") else None
    hold_plan_labels(design, wrong)  # a seed_from_map plan's, fixed once
    # The class design's strata are the map's classes, which the package's
    # condition-design texts call conditions: worded as the package words
    # them (the identity for a random design).
    words = wording.class_wording(sample)
    res = words(_jsonable(estimate.estimate_error_rate(sample, wrong, valid=True)))
    valid = res["valid_interval"]
    out: dict[str, Any] = {
        "available": True,
        "design_path": str(args["design_path"]),
        "plan_design": plan_design,
        # The design file's seed: the same plan with it draws these windows.
        "seed": design.get("seed"),
        "valid_interval": valid,
    }
    out.update(res)
    out["intervals"] = [
        _interval_entry("valid", VALID_WHAT, valid),
        _interval_entry("design", DESIGN_WHAT, res),
    ]
    out["valid_note"] = str(estimate.VALID_NOTE)
    if labels_classes:
        # The class each code of the per-class table stands for: the map's
        # classes, then any class seen that the map never writes.
        out["classes"] = {str(c): v for c, v in enumerate(names or [])}
    if class_map:
        out["class_map_note"] = rules.CLASS_MAP_NO_CONFIDENCE
    if outside:
        out["references_outside_the_maps_classes"] = sorted(set(outside))
    per_class: dict[str, Any] | None = None
    if reference is not None:
        table = words(
            _jsonable(
                estimate.estimate_per_class(
                    sample,
                    reference,
                    population["map_class"],
                    n_classes=n_classes,
                    valid=True,
                )
            )
        )
        out["per_class"] = _valid_first_table(table, names)
        per_class = table
    out["interval_rules"] = list(INTERVAL_RULES)
    studio = bool((population.get("source") or {}).get("result_id"))
    out["report"] = (
        VALID_FIRST_HOW.get(plan_design, VALID_FIRST_HOW[CLASS_DESIGN])
        + " Give each interval's estimate, low, high and method as returned, "
        "named for what it is; with reference classes, each class's valid "
        "intervals before the design's. valid_note and the per-class "
        "valid_note are the package's notes for estimate --valid: its first "
        "interval is valid_interval, the line below it the design's. The rate "
        "is of the design's population windows"
        + (
            " (a Studio result's sampled grid points, one pixel each, not every "
            "pixel)"
            if studio
            else ""
        )
        + "."
    )
    weakest = _weakest_classes(per_class, names) if per_class else None
    out["next_steps"] = _estimate_next_steps(
        plan_design,
        from_design=True,
        per_class=bool(weakest and weakest.get("lowest")),
        valid_first=True,
        class_map=class_map,
    )
    claims: list[dict[str, str]] = []
    if plan_design in STRATIFIED_DESIGNS:
        claims.append(rules.simple_random_interval(res.get("method")))
    if class_map:
        claims.append(rules.zone_for_a_class_map())
    else:
        if plan_design in STRATIFIED_DESIGNS:
            claims.append(rules.certify_from_nonrandom_design(plan_design))
        # Every estimate's next steps name a certified zone: exp86 round 8
        # (B5/files/2) read them as "a guaranteed-certifiable region".
        claims.append(_estimate_certification_claim(plan_design, from_design=True))
    stated = rules.valid_interval_must_state(
        valid.get("estimate"),
        valid["low"],
        valid["high"],
        plan_design,
        int(res["n_labelled"]),
        class_map=class_map,
    )
    scope = _studio_scope(population.get("source") or {}, res.get("n_population"))
    facts = [_valid_first_estimate(res, valid)] + ([weakest] if weakest else [])
    missed: list[str] = []
    if per_class:
        # What each class's user's accuracy leaves out (agent trial 12, N3):
        # the classes the map finds less of than it calls, or draws too
        # little or too much, from the package's rows; the must_state ones
        # come after the valid interval and the scope, within the contract's
        # cap. A class map's classes as it writes them, never its internal
        # codes (the step 11 review); a scores map's ids with its file's names.
        if labels_classes:
            contrast, missed = _user_vs_producer(per_class, written=names or [])
        else:
            labels_of = _file_class_names(population.get("class_names"))
            contrast, missed = _user_vs_producer(per_class, labels_of)
        if contrast is not None:
            facts.append(contrast)
    room = max(0, MUST_STATE_MAX - 1 - len(scope))
    # A class map's top-k scores, stated in the slot the classes it finds
    # less of leave free: those are this estimate's own finding, while the
    # scores only allocated a confidence design's labels (the interval does
    # not rest on them) and the plan whose design this grades states them.
    topk = _topk_must_state(
        population.get("score_kind"), population.get("source") or {}
    )
    stated_missed = missed[:room]
    free = room - len(stated_missed)
    return rules.add_contract(
        out,
        facts=facts,
        must_state=[stated, *scope, *stated_missed, *topk[:free]],
        forbidden_claims=claims,
    )


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
        if design.get("design") == TWO_MAP_DESIGN:
            return _grade_two_maps(args, design, estimate)
        # Every one-map design file (a scores map's confidence, proportional or
        # random design, a class map's, the class design's): the interval valid
        # at every sample size first, then the design's own. In a worker
        # thread: the valid per-class table takes seconds per class under a
        # stratified design (the package's VALID_HELP), and the event loop
        # serves the web bridge's other sessions.
        return await asyncio.to_thread(_estimate_valid_first, args, design, estimate)
    if args.get("window_indices") is None:
        raise ValueError(
            "pass design_path (from olmoearth_plan_label_sample), or "
            "window_indices with scores/scores_path for windows labelled without a design"
        )
    # Windows labelled with no design: the package treats them as a random
    # sample only after checking they could be one (a review set is refused).
    pop = population_from_rows(**_rows_args(args))
    indices = [int(i) for i in args["window_indices"]]
    wrong, _ref = _labels(args, indices)
    out: dict[str, Any] = {"available": True}
    out.update(_jsonable(estimate.estimate_from_indices(indices, wrong, pop.margin)))
    # The package estimates them as a simple random sample (its design
    # "random"), which its check of them does not establish: the design is
    # none, set after the package's result so that "random" cannot stand.
    out["estimated_as"] = out.pop("design", None)
    out["design"] = NO_DESIGN
    out["interval_rules"] = list(INTERVAL_RULES)
    studio = bool(pop.source.get("result_id"))
    out["report"] = (
        "Report the estimate with [low, high] and the method string as returned; "
        "the rate is of the design's population windows"
        + (" (a Studio result's sampled grid points)" if studio else "")
        + "."
    )
    out["next_steps"] = _estimate_next_steps(NO_DESIGN, from_design=False)
    # A Studio class map's top-k scores pooled from its own files say what
    # they are here too, as on the design route; the pixel-value top-k
    # route's sentence stays on its design route, as before.
    pooled = (
        [rules.STUDIO_WINDOW_SCORES_MUST_STATE]
        if file_studio_kind(pop.source) == STUDIO_TOPK
        else []
    )
    return rules.add_contract(
        out,
        facts=[_whole_map_estimate(out)],
        must_state=_studio_scope(pop.source, out.get("n_population")) + pooled,
        # Every estimate's next steps name a certified zone: exp86 round 8
        # (B5/files/2) read them as "a guaranteed-certifiable region".
        forbidden_claims=[_estimate_certification_claim(NO_DESIGN, from_design=False)],
    )


def _estimate_certification_claim(design: str, *, from_design: bool) -> dict[str, str]:
    """``certification_guaranteed`` for an estimate, with what its labels allow.

    As in :func:`_estimate_next_steps`: windows labelled with no design are
    estimated as a random sample but have no design file to certify from.
    """
    if design in STRATIFIED_DESIGNS:
        detail = (
            f"these labels of a {design} design cannot certify at all, and a new "
            "random plan may certify nothing"
        )
    elif from_design:
        detail = (
            "these labels certify a zone only if olmoearth_certify_zone, at an "
            "alpha the map's use requires, finds one, and it may find none"
        )
    else:
        detail = (
            "these windows have no design file to certify from, and a random "
            "plan may certify nothing"
        )
    return rules.certification_guaranteed(detail)


def _estimate_next_steps(
    design: str,
    *,
    from_design: bool,
    per_class: bool = False,
    valid_first: bool = False,
    class_map: bool = False,
) -> list[str]:
    """What follows an estimate, from its design (and a per-class table).

    A valid-first estimate (any design file of one map) reports the interval
    valid at every sample size first; a class map's steps say it has no
    certified zone, in place of the certification step.
    """
    steps = [
        (
            "Report the interval valid at every sample size first (valid_interval; "
            "the fact whole_map_estimate states both), then the interval this "
            "design earns, each named for what it is; the rate is of the design's "
            "population."
            if valid_first
            else "Report the estimate with its interval and method as returned "
            "(the fact whole_map_estimate states them); the rate is of the "
            "design's population."
        )
    ]
    if per_class:
        # exp86 round 8 (B5/files/2) ranked the per-class table by eye and
        # left out the class with none of its windows correct.
        steps.append(
            "Name the weakest classes as the fact weakest_classes ranks them (by "
            f"user's accuracy, over classes with at least {MIN_TO_RANK} "
            "map-labelled windows), not from the table by eye."
        )
    # Each certification step says a random design makes a certified zone
    # possible, not certain: exp86 round 8 (B5/files/2) read "a certified zone
    # needs a new plan with design='random'" as "a guaranteed-certifiable
    # region". A class map has no certified zone, whatever the design.
    if class_map:
        steps.append(rules.CLASS_MAP_NO_CONFIDENCE)
    elif design in STRATIFIED_DESIGNS:
        steps.append(
            f"No zone can be certified from this {design} design: "
            "olmoearth_certify_zone needs design='random'. A certified zone "
            "needs a new plan with design='random', every drawn window labelled, "
            "and alpha, delta and rule fixed before labelling. "
            + rules.POSSIBLE_NOT_CERTAIN
        )
    elif from_design:
        steps.append(
            "For a certified zone from these labels, olmoearth_certify_zone with "
            "the same design_path and labels, at the alpha the map's use "
            "requires, not one chosen from these results. " + rules.POSSIBLE_NOT_CERTAIN
        )
    else:
        steps.append(
            "These windows were checked as a random sample and estimated as one; "
            "a certified zone needs a design file: plan one with "
            "olmoearth_plan_label_sample(design='random') before labelling. "
            + rules.POSSIBLE_NOT_CERTAIN
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
        source: dict[str, Any] = {
            "scores_path": str(args["scores_path"]),
            "result_id": loaded.meta.get("result_id"),
        }
        if isinstance(loaded.meta.get("studio"), dict):
            # Pooled from a Studio result's own files, as the plan keeps it.
            source["studio"] = loaded.meta["studio"]
        return {
            "rows": loaded.scores,
            "grid": loaded.grid,
            "windows": loaded.windows,
            "score_kind": str(loaded.meta.get("score_kind") or "class_scores"),
            "source": source,
            "p1": loaded.meta.get("p1"),
            "map_class": loaded.meta.get("map_class"),
            "margin": _topk_file_margin(loaded.meta),
        }
    raise ValueError(
        "window_indices need the scores they index: pass scores or scores_path"
    )


def _certify_refused_for_a_class_map(design: dict[str, Any]) -> dict[str, Any]:
    """``olmoearth_certify_zone`` on a class map's design: refused, in one sentence.

    A certified zone is the most confident share of a map, and a class map
    without top-k scores has no confidence to rank by, so its random design
    is refused as its class design is (the reason adds where top-k scores
    come from); the labels still give its error rate and per-class
    accuracy, and labels where it differs from another map of one time say
    which of the two is more accurate.
    """
    out: dict[str, Any] = {
        "available": True,
        "certified": False,
        "design": design.get("design"),
        "seed": design.get("seed"),
        "class_map": True,
        "reason": rules.CLASS_MAP_REFUSAL,
        "next_steps": [
            "Estimate the map's error rate and each class's accuracy from these "
            "labels: olmoearth_estimate_map_error with this design_path and the "
            "same labels gives the interval valid at every sample size first, "
            "then the interval this design earns.",
            "Which of this map and another of the same area and time is more "
            "accurate: olmoearth_plan_label_sample with design='disagreement' on "
            "the two results, then olmoearth_estimate_map_error on the class seen "
            "in each planned window.",
        ],
    }
    return rules.add_contract(out, forbidden_claims=[rules.zone_for_a_class_map()])


async def _certify_zone(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_certify_zone``."""
    try:
        estimate = inferencex.load("estimate")
        zone_module = inferencex.load("certify.zone")
    except inferencex.InferencexMissingError:
        return inferencex.missing_extra("Certifying a trusted zone")
    design = _load_design(str(args["design_path"]))
    if _has_no_confidence(design):
        return _certify_refused_for_a_class_map(design)
    if design.get("design") != "random":
        two_maps = design.get("design") == TWO_MAP_DESIGN
        refused: dict[str, Any] = {
            "available": True,
            "certified": False,
            "design": design.get("design"),
            "seed": design.get("seed"),
            "reason": (
                "a certified zone needs a simple random sample of the map: its "
                "guarantee rests on the labels inside each candidate zone being a "
                "random sample of that zone, which "
                + (
                    "a sample of the windows where two maps differ is not (it says "
                    "which map is more accurate, not where either is right). "
                    if two_maps
                    else "a stratified design is not. "
                )
                + "Plan one with olmoearth_plan_label_sample(design='random')."
            ),
            "design_requirement": DESIGN_REQUIREMENT,
            "next_steps": _certify_next_steps(None, two_maps=two_maps),
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
                rules.certification_guaranteed(
                    f"these labels of a {design.get('design')} design cannot "
                    "certify at all, and a new random plan may certify nothing"
                ),
            ],
        )
    # A plan drawn with seed_from_map fixed alpha, delta and the rule with its
    # draw (they enter its seed), as `oe-inferencex certify` keeps to them.
    plan_fixed = design.get("certificate")
    fixed: dict[str, Any] = plan_fixed if isinstance(plan_fixed, dict) else {}
    rule = str(args.get("rule", fixed.get("rule", "prefix")))
    if rule not in ZONE_RULES:
        raise ValueError(f"rule must be one of {list(ZONE_RULES)}, got {rule!r}")
    alpha = float(args["alpha"])
    delta = float(args.get("delta", fixed.get("delta", DEFAULT_DELTA)))
    for key, value in (("alpha", alpha), ("delta", delta), ("rule", rule)):
        if fixed and value != fixed.get(key):
            msg = (
                f"this plan was drawn with seed_from_map, which fixed {key} "
                f"{fixed.get(key)} before any label; {key} {value} on its labels "
                "would be a second test"
            )
            raise rules.refusal(msg, forbidden_claims=[rules.post_hoc_alpha(alpha)])
    indices = [int(i) for i in design["sample"]["indices"]]
    if _labels_are_classes(design):
        # A class map with top-k scores: its sheet holds the class seen, and
        # wrong follows from it, as its estimate reads it.
        wrong = _class_map_labels(args, indices, design["population"])[0]
    else:
        wrong, _ref = _labels(args, indices)
    # A seed_from_map plan's labels are fixed in its sheet the first time they
    # are graded: a zone on other labels afterwards is refused.
    hold_plan_labels(design, wrong)
    # certify_zone orders every window of the map: in a worker thread, so the
    # event loop keeps serving the web bridge's other sessions. A class map's
    # top-k margin orders it as any score's margin does: higher, more trusted.
    margin = _margin(design["population"]["margin"])
    zone = await asyncio.to_thread(
        estimate.certify_zone, margin, indices, wrong, alpha, delta=delta, rule=rule
    )
    zone_indices = zone.pop("zone_indices_in_order", None)
    out: dict[str, Any] = {
        "available": True,
        "certified": zone.get("coverage") is not None,
        # The design file's design and seed: the same plan draws these labels.
        "design": design.get("design"),
        "seed": design.get("seed"),
        # Which labels: the report tells two label sets of one design apart.
        "labels_sha256": labels_sha256(wrong),
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
        # What delta counts, verbatim in the package's words: certify_zone's
        # result does not carry it, and `oe-inferencex certify` prints it with
        # every zone it certifies. A certified zone is not a 1 - delta chance
        # of being right.
        out["meaning"] = (
            "The zone is the n_zone most confident windows of the design's "
            f"population; its error rate is at most alpha={alpha:g}, a statement "
            f"that fails on at most delta={delta:g} of random draws. Nothing "
            "outside the zone is certified. " + str(zone_module.ISSUED_NOTE)
        )
        # The package's note on its own, for a report that states it verbatim
        # (olmoearth_case_narrative).
        out["issued_note"] = str(zone_module.ISSUED_NOTE)
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
    sampled = await asyncio.to_thread(
        estimate.estimate_error_rate, design["sample"], wrong
    )
    # Nothing certified: the whole-map rate is not the most confident part's,
    # and the evidence on that part is the smallest zone tested (agent trial
    # 12, B6/files runs 8, 9 and 13: "nowhere near the 5% bar anywhere").
    levels = list(out.get("levels") or [])
    uncertified = None if out["certified"] else levels
    whole = _whole_map_estimate(
        _jsonable(sampled), certify=True, uncertified=uncertified
    )
    out["next_steps"] = _certify_next_steps(out)
    if fixed and out["certified"]:
        decoded = file_studio_kind(design["population"].get("source") or {})
        holder = _certificate_holder(decoded is not None)
        out["next_steps"].append(
            f"Pack this zone into a certificate {holder} can check: "
            "olmoearth_write_certificate with this design_path, the same labels "
            "and the map's date."
        )
    zone_claims = (
        []
        if out["certified"]
        else [rules.whole_map_rate_for_a_zone(alpha, levels[0] if levels else None)]
    )
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
        + _studio_scope(design["population"].get("source") or {}, out["n_population"])
        + _topk_must_state(
            design["population"].get("score_kind"),
            design["population"].get("source") or {},
        ),
        forbidden_claims=[
            rules.post_hoc_alpha(alpha),
            rules.rule_switch_after_failure(
                certified=bool(out["certified"]),
                levels=out.get("levels") or [],
                delta=delta,
                rule=rule,
            ),
            rules.certification_guaranteed(
                f"this call certified the {zone['n_zone']} most confident windows "
                f"at alpha={alpha:g}, and a new random design with a larger budget "
                "may certify a smaller zone, or none"
                if out["certified"]
                else f"this random design of {out.get('n_labelled')} labels "
                f"certified nothing at alpha={alpha:g}, and a new one with a "
                "larger budget may certify nothing either"
            ),
            *zone_claims,
        ],
    )


def _certify_next_steps(
    out: dict[str, Any] | None, *, two_maps: bool = False
) -> list[str]:
    """What follows a certification, from the design and the outcome.

    ``out`` is ``None`` for a refused (stratified, or two-map) design; with
    ``two_maps`` its labels say which of the two maps is more accurate.
    Nothing certified under a random design: label more windows under a NEW
    random design fixed in advance, or report the whole-map estimate; never a
    looser alpha and never another rule (exp86 rounds 1 to 7, brief 6). Every
    step that names a new random design says it makes a certified zone
    possible, not certain (exp86 round 8, B5/files/2: "a
    guaranteed-certifiable region").
    """
    if out is None:
        return [
            (
                "Grade these labels instead: olmoearth_estimate_map_error with "
                "this design_path and the class the reviewer saw in each window "
                "says which of the two maps is more accurate, or neither."
                if two_maps
                else "Estimate the map's error rate from these labels: "
                "olmoearth_estimate_map_error with this design_path and the same "
                "labels gives the estimate with the interval valid at every "
                "sample size first, then the interval this design earns."
            ),
            "A certified zone needs a new plan: olmoearth_plan_label_sample with "
            "design='random', every drawn window labelled, and alpha, delta and "
            "rule fixed before labelling. " + rules.POSSIBLE_NOT_CERTAIN,
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
            "A larger zone needs a NEW random design with a larger budget, and "
            "alpha, delta and rule fixed before labelling; it makes a larger "
            "zone possible, not certain: it may certify a smaller zone, or none.",
        ]
    return [
        "Label more windows under a NEW random design fixed in advance: "
        "olmoearth_plan_label_sample with design='random' and a larger budget, "
        "with alpha, delta and rule set before any of its labels is seen. "
        + rules.POSSIBLE_NOT_CERTAIN,
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


# --------------------------------------------------------------------------- which of two maps is more accurate

#: The difference in accuracy a two-map plan's budget advice is for, unless
#: the caller states one (``oe_inferencex.plan.plan_which_map``).
DEFAULT_DIFFERENCE = 0.05

#: The dates readings under which the two-map design is refused: the maps are
#: not known to describe one time (``oe_inferencex.compare.dates_reading``).
NOT_ONE_TIME = ("different_time", "overlapping_time", "partly_stated")

#: Stated with a two-map plan or grade whose maps' dates were not given.
UNSTATED_DATES_MUST_STATE = (
    "The maps' dates were not given: a window where they differ is an error in "
    "one map only if both show one time."
)

#: The inputs of a two-map plan, stated when they are wrong.
TWO_MAP_INPUTS = (
    "design='disagreement' compares two maps of one time: pass two Studio "
    "results of the same area (result_id for map a, result_id_b for map b) or "
    "two scores files of one window grid (scores_path and scores_path_b)"
)

#: How the reviewer labels a two-map sheet (as ``oe-inferencex sample
#: --other`` writes it: the class seen, or ? where it cannot be judged).
HOW_TO_LABEL_TWO_MAPS = (
    "For each window, in this order, write in reference_class the class you see "
    "there at the time both maps describe, written as class_a and class_b write "
    "classes, or ? where it cannot be judged (keep the row); hide class_a and "
    "class_b to label blind. Fill the CSV (it holds each window's location "
    "when the maps give one) or pass reference in the same order; then call "
    "olmoearth_estimate_map_error with design_path."
)


@dataclass
class TwoMaps:
    """Two maps on one window grid: each window's class in each map.

    ``class_a`` and ``class_b`` are codes into ``classes`` (the classes as the
    maps write them), ``-1`` where a map holds no class; ``source`` names each
    map (``map_a``, ``map_b``); ``dates`` is the package's reading of their
    dates. ``centres`` (window -> lon, lat) are for the labelling sheet only.
    """

    class_a: list[int]
    class_b: list[int]
    classes: list[Any]
    grid: tuple[int, int] | None
    source: dict[str, dict[str, Any]]
    dates: dict[str, Any]
    centres: dict[int, tuple[float, float]] = field(default_factory=dict)
    sampling: dict[str, Any] | None = None
    assumption: str | None = None
    notes: list[str] = field(default_factory=list)
    studio: bool = False


def _class_value(value: Any) -> Any:
    """A class as a map writes it: a whole number as an int, anything else as given."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _class_key(value: Any) -> str:
    """How a class is matched: its text, case and outer spaces aside; ``3.0`` is ``3``."""
    text = str(_class_value(value)).strip().lower()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def _key_order(key: str) -> tuple[int, float, str]:
    """Numbers first, by value, then names, so codes follow a map's own class ids."""
    try:
        return (0, float(key), key)
    except ValueError:
        return (1, 0.0, key)


def _code_classes(
    values_a: list[Any], values_b: list[Any]
) -> tuple[list[int], list[int], list[Any]]:
    """Both maps' window classes as codes into one table (``-1`` where a map holds none).

    A class is one entry however each map writes it (:func:`_class_key`), so
    two maps that write ``water`` and ``Water`` agree there.
    """
    first: dict[str, Any] = {}
    for value in [*values_a, *values_b]:
        if value is not None:
            first.setdefault(_class_key(value), _class_value(value))
    keys = sorted(first, key=_key_order)
    code = {key: i for i, key in enumerate(keys)}

    def codes(values: list[Any]) -> list[int]:
        """One map's classes as codes, ``-1`` where it holds none."""
        return [code[_class_key(v)] if v is not None else -1 for v in values]

    return codes(values_a), codes(values_b), [first[key] for key in keys]


def _dates_refusal(reading: dict[str, Any]) -> ValueError:
    """Why two maps not known to describe one time get no two-map design.

    Across dates a window where the maps differ can be a change on the ground,
    right in one map and wrong in the other whatever either model did; the
    reviewer writes one class per window, seen at one date, so the labels
    would grade both maps against that date (the package's Usage, "Across
    dates the identity fails").
    """
    status = reading.get("status")
    said = str(reading.get("reading") or "").rstrip(".")
    if status == "partly_stated":
        return rules.refusal(
            f"the two-map design needs both maps' dates, and {said} "
            f"(dates_reading: {status}); pass the other map's date as date_a or "
            f"date_b ({DATE_FORMS})",
            must_state=[PARTLY_DATED_MUST_STATE],
            forbidden_claims=[rules.one_reference_settles_two_dates(partly=True)],
        )
    return rules.refusal(
        f"the two maps are not of one time: {said} (dates_reading: {status}). "
        "Labels on the windows where they differ would grade both maps against "
        "the one date the labels show, counting a change on the ground as an "
        "error of the other map, so they cannot say which map is more accurate; "
        "each map needs a sample of its own, labelled at its own date",
        must_state=[rules.DATED_MAPS_MUST_STATE],
        forbidden_claims=[
            *rules.across_dates(reading),
            rules.one_reference_settles_two_dates(reading.get("a"), reading.get("b")),
            rules.two_map_design_across_dates(reading.get("a"), reading.get("b")),
        ],
    )


def _two_map_dates(periods: list[str | None], args: dict[str, Any]) -> dict[str, Any]:
    """The package's reading of the two maps' dates; refused unless one time is possible.

    A Studio result's date is its prediction's period (start and end times);
    ``date_a`` / ``date_b`` give a map's date where none is read (a scores
    file, or a prediction with no times). Different, overlapping or half-given
    dates are refused (:func:`_dates_refusal`); none given at all is read as
    unstated, which the plan and the grade then state.
    """
    dates_pkg = inferencex.load("compare")
    dates: list[str | None] = []
    sources: list[str | None] = []
    for side, period in zip(("a", "b"), periods):
        given = args.get(f"date_{side}")
        _check_date_form(given, f"date_{side}")
        if period:
            dates.append(period)
            sources.append("the Studio prediction's start and end times")
        elif given:
            dates.append(str(given).strip())
            sources.append(f"date_{side}")
        else:
            dates.append(None)
            sources.append(None)
    try:
        reading = dict(dates_pkg.dates_reading(dates[0], dates[1]))
    except (TypeError, ValueError) as exc:
        msg = f"the maps' dates could not be read: {exc}; dates are {DATE_FORMS}"
        raise ValueError(msg) from None
    reading["source_a"], reading["source_b"] = sources
    if reading.get("status") in NOT_ONE_TIME:
        raise _dates_refusal(reading)
    return reading


def _unthresholded_refusal(
    bands: list[tuple[str | None, tuple[float, float] | None]],
) -> ValueError:
    """A regression band with no decision has no class to compare."""
    named = "; ".join(
        f"{name!r} (declared range {list(rng) if rng else 'none'})"
        for name, rng in bands
    )
    return rules.refusal(
        f"the two-map design compares each window's class, and {named} is a "
        "regression band with no decision: pass threshold (the value at which "
        "the map's decision flips; class 1 above it), or compare class maps",
        forbidden_claims=[rules.unthresholded_regression(bands)],
    )


def _window_class(value: Any, categorical: bool, cut: float) -> Any:
    """A sampled value as a class: its class, or 1 above ``cut`` and 0 at or below."""
    if categorical:
        return _class_value(value)
    return 1 if float(value) > cut else 0


async def _studio_two_maps(args: dict[str, Any], ctx: ToolContext) -> TwoMaps:
    """Two Studio results' classes on one grid, by compare_results' pair sampler.

    The dates are read from the results' predictions before any pixel is
    sampled, and so is a regression band's declared range; a band with no
    decision (not ``[0, 1]``, no ``threshold``) is refused. A ``[0, 1]`` score
    is cut at 0.5, as the estimation tools decide it; ``threshold`` cuts
    either band there. The grid is held to the pair mode's cap.
    """
    ids = [str(args["result_id"]).strip(), str(args["result_id_b"]).strip()]
    if ids[0] == ids[1]:
        msg = (
            f"map a and map b are the same result ({ids[0]}): one map always "
            "agrees with itself; pass two results"
        )
        raise ValueError(msg)
    pair = await read_pair(ctx, ids)
    if isinstance(pair, dict):
        raise ValueError(str(pair.get("reason")))
    dates = _two_map_dates([prediction_period(p) for p in pair.predictions], args)
    prop = args.get("property_name")
    # A class map with top-k scores is compared on its class; a rank's score
    # or class named instead is refused, never read as a map of its own.
    tops = [topk_scores(record, prop) for record in pair.records]
    for top in tops:
        if top is not None and top.reads_a_part(prop):
            raise ValueError(topk_part_reason(str(prop), top))
    threshold = float(args["threshold"]) if args.get("threshold") is not None else None
    if threshold is None:
        undecided: list[tuple[str | None, tuple[float, float] | None]] = []
        for record, top in zip(pair.records, tops):
            name = top.base if top is not None else default_property(record, prop)
            decl = declared_fields(record).get(str(name)) if name else None
            if decl and decl.get("value_type") == "regression":
                rng = declared_range(None, decl)
                if rng != (0.0, 1.0) and (name, rng) not in undecided:
                    undecided.append((name, rng))
        if undecided:
            raise _unthresholded_refusal(undecided)
    side, requested = result_grid(args.get("grid"))
    grid = min(side, PAIR_MAX_GRID)
    sampled = await sample_pair(ctx, pair, prop=prop, grid=grid)
    if isinstance(sampled, dict):
        reason = str(sampled.get("reason"))
        if "property_a" in sampled:
            # Pair mode's refusal offers allow_different_properties, which
            # this design has no use for: one property's classes are compared.
            names = [
                (sampled.get(k) or {}).get("property_name")
                for k in ("property_a", "property_b")
            ]
            reason = (
                f"the two results measure different properties ({names[0]!r} and "
                f"{names[1]!r}): the two-map design compares one property's "
                "classes in both maps; pass results of one property, or pin "
                "property_name"
            )
        raise rules.refusal(
            reason,
            forbidden_claims=sampled.get("forbidden_claims") or [],
        )
    if not sampled.categorical and threshold is None and sampled.unthresholded:
        raise _unthresholded_refusal(sampled.unthresholded)
    cut = 0.5 if threshold is None else threshold
    values: list[list[Any]] = [[], []]
    for read_a, read_b in zip(sampled.reads[0], sampled.reads[1]):
        both = read_a[2] == OK and read_b[2] == OK
        for k, (value, categorical, _status) in enumerate((read_a, read_b)):
            values[k].append(_window_class(value, categorical, cut) if both else None)
    class_a, class_b, classes = _code_classes(values[0], values[1])
    n_points = grid * grid
    n_compared = sum(1 for a, b in zip(class_a, class_b) if a >= 0 and b >= 0)
    capped = requested is not None and requested != grid
    sampling: dict[str, Any] = {
        "grid": f"{grid}x{grid}",
        "grid_requested": requested,
        "grid_capped": capped,
        "n_windows_sampled": n_points,
        "n_windows_compared": n_compared,
        "n_windows_dropped": n_points - n_compared,
        "what_a_window_is": (
            "a grid of sampled points, not every pixel: the results' shared "
            f"extent is cut into {grid} x {grid} cells and each window is ONE "
            "pixel-value sample of each result at its cell centre; a window is "
            "compared where both results hold a class there"
        ),
        "nodata_rule": NODATA_RULE,
    }
    if capped:
        sampling["grid_note"] = (
            f"grid {requested} was held to {grid}: two Studio results are "
            f"sampled on {FROM_RESULT_MIN_GRID}-{PAIR_MAX_GRID} points a side "
            "(each point is one pixel-value call per result), so the grid used "
            f"is {grid}x{grid} = {n_points} points"
        )
    assumption = None
    if not sampled.categorical:
        assumption = (
            f"the regression bands are read as two classes: 1 above {cut:g}, 0 at "
            "or below it"
            + (
                " (a [0, 1] score decided at 0.5, as the estimation tools decide "
                "it; pass threshold for another cut)"
                if threshold is None
                else ""
            )
        )
    source = {
        f"map_{side_name}": {
            "result_id": rid,
            "property_name": name,
            "model": (model or {}).get("name"),
        }
        for side_name, rid, name, model in zip(
            ("a", "b"), sampled.ids, sampled.names, sampled.models
        )
    }
    return TwoMaps(
        class_a=class_a,
        class_b=class_b,
        classes=classes,
        grid=(grid, grid),
        source=source,
        dates=dates,
        centres={r * grid + c: (lon, lat) for r, c, lon, lat in sampled.cells},
        sampling=sampling,
        assumption=assumption,
        studio=True,
    )


#: How far apart (in degrees) one window's centre may lie in two scores files
#: of one grid: about 0.1 m, rounding only.
CENTRE_TOLERANCE_DEG = 1e-6


def _moved_window(
    a: dict[int, tuple[float, float]], b: dict[int, tuple[float, float]]
) -> int | None:
    """The first window both files locate whose centres differ, or ``None``.

    Only windows both files carry a centre for are compared: a window that is
    no-data in one map has none there, and a file without a georeference has
    none at all, so it cannot be checked this way.
    """
    for index in sorted(a.keys() & b.keys()):
        (lon_a, lat_a), (lon_b, lat_b) = a[index], b[index]
        if (
            abs(lon_a - lon_b) > CENTRE_TOLERANCE_DEG
            or abs(lat_a - lat_b) > CENTRE_TOLERANCE_DEG
        ):
            return index
    return None


async def _file_two_maps(args: dict[str, Any], ctx: ToolContext) -> TwoMaps:
    """Two scores files' window classes (``map_class``, or the arg-max), on one grid.

    The files must have one grid: the same shape and, where both give a
    window's centre (``centres_lon_lat``), the same centre for it.
    """
    paths = [str(args["scores_path"]), str(args["scores_path_b"])]
    if os.path.realpath(paths[0]) == os.path.realpath(paths[1]):
        msg = (
            "map a and map b are the same scores file: one map always agrees "
            "with itself; pass two files"
        )
        raise ValueError(msg)
    pops: list[Population] = []
    for path in paths:
        pop = await _population({"scores_path": path}, ctx)
        if isinstance(pop, dict):  # only a Studio result is refused this way
            raise ValueError(str(pop.get("reason")))
        pops.append(pop)
    # A class map's top-k scores file codes its classes; compared, and shown
    # to the reviewer, as the map writes them (a scores map's codes as they are).
    named = [
        pop.classes if pop.score_kind == TOPK_SCORE and pop.classes else None
        for pop in pops
    ]
    if len(pops[0].margin) != len(pops[1].margin) or pops[0].grid != pops[1].grid:
        shapes = [
            f"{p.grid[0]}x{p.grid[1]}" if p.grid else f"{len(p.margin)} windows"
            for p in pops
        ]
        msg = (
            f"the two scores files are not of one window grid ({shapes[0]} and "
            f"{shapes[1]}): the two-map design compares the same windows in both "
            "maps"
        )
        raise ValueError(msg)
    moved = _moved_window(pops[0].centres, pops[1].centres)
    if moved is not None:
        msg = (
            "the two scores files are not of one window grid: their grids have "
            f"the same shape, but window {moved} has another centre in each file, "
            "so they cover different ground; the two-map design compares the same "
            "windows in both maps"
        )
        raise ValueError(msg)
    dates = _two_map_dates([None, None], args)
    values = [
        [
            (
                (names[pop.map_class[i]] if names else pop.map_class[i])
                if math.isfinite(pop.margin[i])
                else None
            )
            for i in range(len(pop.margin))
        ]
        for pop, names in zip(pops, named)
    ]
    class_a, class_b, classes = _code_classes(values[0], values[1])
    keep = ("scores_path", "result_id", "property_name", "model")
    return TwoMaps(
        class_a=class_a,
        class_b=class_b,
        classes=classes,
        grid=pops[0].grid,
        source={
            f"map_{side}": {k: v for k, v in pop.source.items() if k in keep}
            for side, pop in zip(("a", "b"), pops)
        },
        dates=dates,
        centres=dict(pops[0].centres),
    )


async def _two_maps(args: dict[str, Any], ctx: ToolContext) -> TwoMaps:
    """The two maps of a two-map plan: two Studio results, or two scores files."""
    studio = [args.get("result_id"), args.get("result_id_b")]
    files = [args.get("scores_path"), args.get("scores_path_b")]
    if args.get("scores") is not None or (any(studio) and any(files)):
        raise ValueError(TWO_MAP_INPUTS)
    if all(studio):
        maps = await _studio_two_maps(args, ctx)
    elif all(files):
        maps = await _file_two_maps(args, ctx)
    else:
        raise ValueError(TWO_MAP_INPUTS)
    pairs = list(zip(maps.class_a, maps.class_b))
    compared = [(a, b) for a, b in pairs if a >= 0 and b >= 0]
    if compared and not {a for a, _b in compared} & {b for _a, b in compared}:
        maps.notes.append(
            "map a's classes and map b's share none on the windows compared, so "
            "every window differs: check that the two maps write the same classes "
            "before labelling"
        )
    return maps


def _write_pairs_csv(
    name: str, to_label: list[dict[str, Any]], centres: dict[int, tuple[float, float]]
) -> str:
    """The reviewer's two-map sheet: each window's two classes, reference_class blank."""
    target = write_root() / name
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = ["order", "window_index", "row", "col", "class_a", "class_b"]
    if centres:
        fields += ["lon", "lat"]
    fields.append("reference_class")
    with open(target, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for entry in to_label:
            line = dict(entry)
            centre = centres.get(int(entry["window_index"]))
            if centre:
                line["lon"], line["lat"] = centre
            writer.writerow(line)
    return str(target)


def _budget_advice_fact(advice: dict[str, Any], planned: int) -> dict[str, Any]:
    """The ``which_map_budget`` fact: ``plan_which_map``'s advice, in a sentence.

    The package plans, before any label, how many labels among the differing
    windows name the more accurate map with the probability asked for when
    the accuracies differ by the difference stated; the sentence repeats its
    numbers and, when it advises no budget, its own reason.
    """
    diff = float(advice.get("difference", DEFAULT_DIFFERENCE))
    n_pop, n_dis = advice.get("n_population"), advice.get("n_disagree")
    labels = advice.get("labels")
    prob = advice.get("probability_at_labels")
    if labels is None or not isinstance(prob, (int, float)):
        why = str(advice.get("refusal") or "it advises none").rstrip(".")
        sentence = (
            f"No label budget is advised for a difference in accuracy of {diff:g} "
            f"over the {n_pop} windows compared: the package says {why}."
        )
    else:
        sentence = (
            f"Planned before any label, for maps whose accuracies over the {n_pop} "
            f"windows compared differ by {diff:g}: {labels} labels drawn among the "
            f"{n_dis} windows where they differ name the more accurate map with "
            f"probability {float(prob):.3g} at {advice.get('planned_for')}; this "
            f"plan draws {planned}."
        )
    return rules.fact(
        "which_map_budget",
        sentence,
        difference=diff,
        labels=labels,
        probability=prob,
        power=advice.get("power"),
        n_population=n_pop,
        n_disagree=n_dis,
        planned=planned,
    )


async def _plan_two_maps(
    args: dict[str, Any], ctx: ToolContext, estimate: Any
) -> dict[str, Any]:
    """Handler for ``olmoearth_plan_label_sample`` with ``design='disagreement'``."""
    try:
        plan_pkg = inferencex.load("plan")
    except inferencex.InferencexMissingError:
        return inferencex.missing_extra("Planning a labelled sample of two maps")
    budget = int(args["budget"])
    if budget < 1:
        raise ValueError(f"budget {budget} must be at least 1")
    seed = int(args.get("seed", 0))
    max_listed = int(args.get("max_listed", DEFAULT_MAX_LISTED))
    difference = float(args.get("difference", DEFAULT_DIFFERENCE))
    if not 0.0 < difference < 1.0:
        msg = (
            "difference is a difference in accuracy, between 0 and 1 (0.05 is "
            f"5 points), got {difference:g}"
        )
        raise ValueError(msg)
    maps = await _two_maps(args, ctx)
    valid = [a >= 0 and b >= 0 for a, b in zip(maps.class_a, maps.class_b)]
    n_compared = sum(valid)
    if not n_compared:
        msg = (
            f"no window holds a class in both maps (each of the {len(valid)} "
            "windows is no-data, failed or answered HTTP 404 where Studio answered "
            "other points, in one of them), so none can be compared"
        )
        raise ValueError(msg)
    try:
        sample = estimate.sample_disagreement(
            maps.class_a, maps.class_b, budget, valid=valid, seed=seed
        )
    except ValueError as exc:
        # The package's words: the maps never differ, so nothing is to label.
        raise ValueError(f"{exc} ({n_compared} windows compared)") from None
    n_pop, n_dis = int(sample["n_population"]), int(sample["n_disagree"])
    indices = [int(i) for i in sample["indices"].tolist()]
    planned = len(indices)
    to_label: list[dict[str, Any]] = []
    for order, idx in enumerate(indices):
        entry: dict[str, Any] = {"order": order, "window_index": idx}
        if maps.grid:
            entry["row"], entry["col"] = divmod(idx, maps.grid[1])
        entry["class_a"] = maps.classes[maps.class_a[idx]]
        entry["class_b"] = maps.classes[maps.class_b[idx]]
        to_label.append(entry)
    advice = _jsonable(plan_pkg.plan_which_map(n_pop, n_dis, difference))

    # A digest of what the grade reads from the design file (both maps'
    # classes, their sources and their dates) keeps two plans apart that it
    # would grade differently: a dated and an undated plan of the same two
    # maps, or one map under two names, at one budget and seed would share a
    # file name, and the later plan would overwrite the earlier one's design.
    digested = [maps.class_a, maps.class_b, maps.classes, maps.source, maps.dates]
    digest = hashlib.sha256(
        json.dumps(_jsonable(digested), sort_keys=True).encode()
    ).hexdigest()
    origin = slug(
        "_".join(str(s.get("result_id") or "scores") for s in maps.source.values()),
        24,
    )
    stem = f"design_{origin}_{digest[:8]}_{TWO_MAP_DESIGN}_{planned}_s{seed}"
    payload = {
        "format": DESIGN_FORMAT,
        "package": {"name": inferencex.DISTRIBUTION, "version": inferencex.version()},
        "design": TWO_MAP_DESIGN,
        "budget": planned,
        "budget_requested": budget,
        "seed": seed,
        "sample": _jsonable(sample),
        "class_a": [maps.class_a[i] for i in indices],
        "class_b": [maps.class_b[i] for i in indices],
        "classes": _jsonable(maps.classes),
        "population": {
            "n_windows": len(maps.class_a),
            "n_valid": n_pop,
            "grid": list(maps.grid) if maps.grid else None,
            "source": maps.source,
            "studio": maps.studio,
            "dates": maps.dates,
        },
        "budget_advice": advice,
        "to_label": to_label,
    }
    design_path = write_json_file(f"{stem}.json", payload)
    csv_path = _write_pairs_csv(f"{stem}_to_label.csv", to_label, maps.centres)
    out: dict[str, Any] = {
        "available": True,
        "design": TWO_MAP_DESIGN,
        "budget": planned,
        "budget_requested": budget,
        "n_population": n_pop,
        "n_disagree": n_dis,
        "seed": seed,
        "map_a": maps.source["map_a"],
        "map_b": maps.source["map_b"],
        "dates": maps.dates,
        "design_path": design_path,
        "labels_csv_path": csv_path,
        "windows": to_label[:max_listed],
        "n_windows_listed": min(planned, max_listed),
        "how_to_label": HOW_TO_LABEL_TWO_MAPS,
        "design_note": (
            f"Windows drawn at random among the {n_dis} of the {n_pop} windows "
            f"compared where the two maps' classes differ (seed {seed}): where the "
            "maps agree, both are right or both are wrong, so only the windows "
            "where they differ can say which map is more accurate."
        ),
        "budget_advice": advice,
    }
    if maps.sampling:
        out["sampling"] = maps.sampling
    if maps.assumption:
        out["assumption"] = maps.assumption
    if maps.notes:
        out["notes"] = list(maps.notes)
    out["next_steps"] = [
        f"Write in reference_class, for every one of the {planned} windows in "
        f"{csv_path}, the class seen there at the time both maps describe, or ? "
        "where it cannot be judged; hide class_a and class_b to label blind.",
        "Then olmoearth_estimate_map_error with design_path and the filled sheet "
        "(labels_path) says which map is more accurate, by the exact sign test "
        "while the labels are at most a fifth of the differing windows and by the "
        "interval past that, with the difference in accuracy and its 95% interval.",
        "The number of windows to label is fixed now, before any label is read: "
        "labelling more after reading these is a second test.",
    ]
    facts = [_budget_advice_fact(advice, planned)]
    must_state = [
        rules.NOT_EITHER_ACCURACY,
        *_two_map_scope(maps.studio, maps.dates, n_pop),
    ]
    if budget > planned:
        unused = budget - planned
        facts.append(
            rules.fact(
                "unused_labels",
                f"Of the {budget} labels requested, {planned} are planned: the two "
                f"maps differ on {n_dis} of the {n_pop} windows compared, and the "
                f"plan labels every one of them, so {unused} labels have no window "
                "to go to.",
                n=unused,
                requested=budget,
                planned=planned,
            )
        )
        must_state.append(
            f"The plan holds {planned} windows, not the {budget} requested."
        )
    return rules.add_contract(
        out,
        facts=facts,
        must_state=must_state,
        forbidden_claims=[
            rules.more_accurate_without_labels(n_dis, n_pop),
            rules.error_rate_without_labels(),
            # Its grading reads the whole sheet, as every design file's does.
            rules.subset_labelling_sufficient(TWO_MAP_DESIGN),
        ],
    )


def _two_map_scope(
    studio: bool, dates: dict[str, Any], n_compared: Any, *, graded: bool = False
) -> list[str]:
    """What a two-map plan or grade covers: unstated dates, and a Studio grid's points.

    A grade also says it grades these two maps, not the models that made
    them (``rules.MAPS_NOT_MODELS``), last: what the figures cover comes
    first, so beside the verdict and ``rules.NOT_EITHER_ACCURACY`` the
    contract's three sentences keep it only for dated scores files.
    """
    said: list[str] = []
    if dates.get("status") == "unstated":
        said.append(UNSTATED_DATES_MUST_STATE)
    if studio:
        said.append(
            f"The comparison covers the {n_compared} sampled grid points where both "
            "Studio results hold a class (one pixel each), not every pixel."
        )
    if graded:
        said.append(rules.MAPS_NOT_MODELS)
    return said


def _two_map_references(args: dict[str, Any], indices: list[int]) -> list[str]:
    """The class the reviewer saw in each design window, in order (``?``: not judged)."""
    if args.get("wrong") is not None:
        msg = (
            "a design='disagreement' plan is graded from the class the reviewer "
            "saw in each window (reference, or the sheet's reference_class), not "
            "from wrong"
        )
        raise ValueError(msg)
    reference = args.get("reference")
    if reference is None and args.get("labels_path"):
        sheet = _read_labels_csv(str(args["labels_path"]))
        reference = [sheet[i][1] if i in sheet else "" for i in indices]
    if reference is None:
        msg = (
            "pass reference (the class seen in each window, in the design's "
            "order; ? where it cannot be judged) or labels_path (the design's "
            "sheet with reference_class filled in)"
        )
        raise ValueError(msg)
    refs = ["" if r is None else str(r).strip() for r in reference]
    if len(refs) != len(indices):
        msg = f"{len(refs)} references for the {len(indices)} windows of the design"
        raise ValueError(msg)
    missing = [i for i, r in zip(indices, refs) if not r]
    if missing:
        msg = (
            f"{len(missing)} of {len(indices)} design windows have no "
            f"reference_class yet (first: window {missing[0]}); write the class "
            "seen in every drawn window, or ? where it cannot be judged"
        )
        raise ValueError(msg)
    return refs


def _which_map_fact(res: dict[str, Any]) -> dict[str, Any]:
    """The ``which_map`` fact: the package's verdict, what it rests on, and its counts.

    Written from ``compare_from_disagreement``'s result: the map its
    ``verdict`` names (or neither), by the exact sign test (with its p-value
    when a map leads) or the interval, the labelled windows among the
    differing ones and how many are right in each map, and the 95% interval
    on the difference in accuracy over the windows compared. The answer
    checks read ``more_accurate`` and the counts (``harness/checks.py``).
    """
    verdict = res.get("verdict")
    n, n_dis, n_pop = (
        int(res["n_labelled"]),
        int(res["n_disagree"]),
        int(res["n_population"]),
    )
    a, b, c, u = (
        int(res[k]) for k in ("n_a_right", "n_b_right", "n_neither", "n_unjudged")
    )
    diff = res["difference"]
    lo, hi = 100.0 * float(diff["low"]), 100.0 * float(diff["high"])
    level = f"{100.0 * float(res.get('conf', 0.95)):g}%"
    p_value = (res.get("sign_test") or {}).get("p_value")
    counts = f"a right on {a}, b on {b}, neither on {c}" + (
        f", {u} not judged and counted for each map both ways" if u else ""
    )
    if res.get("verdict_rule") == "sign":
        lead = a > b + u or b > a + u  # else no test was run: p is a placeholder
        p_text = (
            f", p = {float(p_value):.3g}"
            if isinstance(p_value, (int, float)) and (lead or not u)
            else ""
        )
        basis = (
            f"the exact sign test on the {n} labelled windows of the {n_dis} where "
            f"the maps differ ({counts}{p_text})"
        )
    else:
        basis = (
            f"the {level} interval on the difference, from {n} labelled windows of "
            f"the {n_dis} where the maps differ ({counts})"
        )
    if verdict in ("a", "b"):
        other = "b" if verdict == "a" else "a"
        head = f"Map {verdict} is more accurate than map {other} by {basis}"
    else:
        head = f"Neither map is shown more accurate by {basis}"
    sentence = (
        f"{head}; the difference in accuracy, a minus b, over the {n_pop} windows "
        f"compared lies between {lo:+.1f} and {hi:+.1f} points ({level} interval)."
    )
    return rules.fact(
        "which_map",
        sentence,
        more_accurate=verdict if verdict in ("a", "b") else None,
        rule=res.get("verdict_rule"),
        n_labelled=n,
        n_disagree=n_dis,
        n_population=n_pop,
        n_a_right=a,
        n_b_right=b,
        n_neither=c,
        n_unjudged=u,
        low=diff.get("low"),
        high=diff.get("high"),
        p_value=p_value,
    )


def _no_reference_in_the_maps_classes(
    n_judged: int, outside: list[str], classes: list[Any]
) -> ValueError:
    """Why references that all lie outside the maps' classes are not graded.

    Graded as given, each would be right in neither map, so the labels would
    name neither map whatever the maps did. Almost always the classes are
    written otherwise (codes for names, say) rather than absent from both maps.
    """
    msg = (
        f"none of the {n_judged} judged references ({', '.join(outside)}) is a "
        f"class either map writes ({', '.join(str(c) for c in classes)}): graded "
        "as given, every window would count as right in neither map and the "
        "labels would name neither map more accurate whatever the maps did. "
        "Write each window's class as class_a and class_b write classes, or ? "
        "where it cannot be judged"
    )
    return ValueError(msg)


def _grade_two_maps(
    args: dict[str, Any], design: dict[str, Any], estimate: Any
) -> dict[str, Any]:
    """``olmoearth_estimate_map_error`` on a ``design='disagreement'`` plan.

    Every number is ``compare_from_disagreement``'s, on the design's sample
    and the two maps' classes it recorded: the reviewer's classes are matched
    to the maps' (:func:`_class_key`), a ``?`` goes in as unjudged, and a
    class neither map writes is right in neither (said in the
    ``references_outside_the_maps_classes`` fact; references that all lie
    outside the maps' classes are refused). The answer's contract: the
    ``which_map`` fact, the verdict sentence with its test or interval and
    its labels (``rules.which_map_must_state``), that it grades these two
    maps and ranks no model (``rules.MAPS_NOT_MODELS``, for dated scores
    files), and no map named more accurate beyond what the package
    names (``more_accurate_without_its_test``).
    """
    sample = design["sample"]
    indices = [int(i) for i in sample["indices"]]
    references = _two_map_references(args, indices)
    classes = list(design.get("classes") or [])
    code_of = {_class_key(value): code for code, value in enumerate(classes)}
    unjudged = [r == "?" for r in references]
    codes = [
        -1 if skip else code_of.get(_class_key(r), -1)
        for r, skip in zip(references, unjudged)
    ]
    judged = [r for r, skip in zip(references, unjudged) if not skip]
    n_outside = sum(1 for r in judged if _class_key(r) not in code_of)
    outside = sorted({r for r in judged if _class_key(r) not in code_of})
    if judged and n_outside == len(judged):
        raise _no_reference_in_the_maps_classes(len(judged), outside, classes)
    res = _jsonable(
        estimate.compare_from_disagreement(
            sample, design["class_a"], design["class_b"], codes, unjudged=unjudged
        )
    )
    population = design.get("population") or {}
    source = population.get("source") or {}
    verdict = res.get("verdict")
    n = int(res["n_labelled"])
    out: dict[str, Any] = {
        "available": True,
        "design_path": str(args["design_path"]),
        # The design file's seed: the same plan with it draws these windows.
        "seed": design.get("seed"),
    }
    out.update(res)
    out["which_map"] = verdict or "undetermined"
    out["map_a"], out["map_b"] = source.get("map_a"), source.get("map_b")
    out["more_accurate_map"] = source.get(f"map_{verdict}") if verdict else None
    # The package that holds estimate (>= 1.20.3) holds decide too.
    decide = inferencex.load("decide")
    head = decide.sign_test_head(res)
    if head:
        # The package's own sentence for a verdict of the exact sign test.
        out["sign_test_statement"] = head
    if outside:
        out["references_outside_the_maps_classes"] = outside
    out["report"] = (
        "Report which_map with the test or interval it rests on (verdict_rule: "
        "the exact sign test while the labels are at most a fifth of the "
        "differing windows, the interval past that) and the number of labels "
        "(n_labelled), and the difference in accuracy, a minus b, with its 95% "
        "interval over the windows compared, as returned; note and verdict_note "
        "are the package's own words."
    )
    if verdict in ("a", "b"):
        rule_words = (
            "the exact sign test"
            if res.get("verdict_rule") == "sign"
            else "the interval"
        )
        other = "b" if verdict == "a" else "a"
        out["next_steps"] = [
            f"Report that map {verdict} is more accurate with what names it, as "
            f"the fact which_map states it: the test or interval and the {n} "
            "labels.",
            "Report the difference in accuracy, a minus b, with its 95% interval "
            f"as returned: it is over the {res.get('n_population')} windows "
            "compared and says nothing of either map's own accuracy.",
        ]
        detail = (
            f"these {n} labels name map {verdict} by {rule_words}; map {other} is "
            f"not more accurate by them, and map {verdict} is named with that test "
            f"or interval and the {n} labels"
        )
    else:
        out["next_steps"] = [
            f"Report that these {n} labels name neither map more accurate, with "
            "the difference in accuracy, a minus b, and its 95% interval as "
            "returned.",
            decide.comparison_second_look(),
            "More labels need a new plan (olmoearth_plan_label_sample with "
            "design='disagreement' and a larger budget), its size fixed before "
            "any of its labels is read.",
        ]
        detail = (
            f"these {n} labels of the {res.get('n_disagree')} windows where the "
            "maps differ name neither map, so neither may be called more "
            "accurate, better or right from them"
        )
    # Within the contract's cap of three, rules.MAPS_NOT_MODELS is the one
    # left out when the dates are unstated or the maps are Studio results.
    must_state = [
        rules.which_map_must_state(res),
        rules.NOT_EITHER_ACCURACY,
        *_two_map_scope(
            bool(population.get("studio")),
            population.get("dates") or {},
            res.get("n_population"),
            graded=True,
        ),
    ][:MUST_STATE_MAX]
    facts = [_which_map_fact(res)]
    if outside:
        facts.append(
            rules.fact(
                "references_outside_the_maps_classes",
                f"Judged windows given a class neither map writes: {n_outside} of "
                f"{len(judged)} ({', '.join(outside)}); each counts as right in "
                "neither map, so check its spelling against class_a and class_b.",
                n=n_outside,
                n_judged=len(judged),
                classes=outside,
            )
        )
    return rules.add_contract(
        out,
        facts=facts,
        must_state=must_state,
        forbidden_claims=[rules.more_accurate_without_its_test(detail)],
    )


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
            "larger N is capped, and the result's 'sampling' says so. Two "
            f"results: at most {PAIR_MAX_GRID}.",
        },
        "property_name": {"type": "string"},
        "threshold": {"type": "number"},
    }
    # Map b and the budget advice of design='disagreement' (which of two maps
    # is more accurate).
    two_map_props: dict[str, Any] = {
        "result_id_b": {
            "type": "string",
            "description": "Map b of design='disagreement'.",
        },
        "scores_path_b": {
            "type": "string",
            "description": "Map b's scores file, same window grid.",
        },
        "date_a": {
            "type": "string",
            "description": "Map a's date where Studio gives none: YYYY-MM-DD or "
            "start/end.",
        },
        "date_b": {"type": "string"},
        "difference": {
            "type": "number",
            "default": DEFAULT_DIFFERENCE,
            "description": "Accuracy difference the budget advice is for.",
        },
    }
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_plan_label_sample",
                description=(
                    "How wrong is my map / how to spend N labels: chooses which "
                    "windows to label for an honest error rate ('confidence' by "
                    "default, 'random' when a certified zone is wanted) from "
                    "scores inline, a scores file (olmoearth_scores_from_file, "
                    "olmoearth_review_set_from_result) or a Studio "
                    "result_id; "
                    "saves the design and lists windows by index and (row, "
                    "col). Do not design your own mix or promise an interval: a "
                    "targeted review set is not a sample (its error rate is "
                    "inflated), and the simple-random-sample interval does not "
                    "apply to a stratified or targeted design. Which of two maps "
                    "is more accurate: design='disagreement' on two Studio "
                    "results (result_id, result_id_b) or scores files "
                    "(scores_path, scores_path_b) of one time. A Studio class "
                    "map without top-k scores takes 'random' or 'class' only. "
                    "seed_from_map ('random', for a certificate) needs a scores "
                    "file olmoearth_scores_from_file wrote from a model run's "
                    "GeoTIFF; any other scores file, inline scores or Studio "
                    "result is refused. Needs the inferencex extra."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        **source_props,
                        **two_map_props,
                        "budget": {
                            "type": "integer",
                            "description": "Windows the reviewer can label.",
                        },
                        "design": {
                            "type": "string",
                            "enum": list(DESIGNS),
                            "default": "confidence",
                            "description": "'random' if a certified zone is "
                            "wanted; 'class': strata by map class; "
                            "'disagreement': two maps.",
                        },
                        "min_per_class": {
                            "type": "integer",
                            "default": DEFAULT_MIN_PER_CLASS,
                            "description": "'class': labels per class at least "
                            "(all of a smaller class).",
                        },
                        "width": {
                            "type": "number",
                            "default": DEFAULT_WIDTH,
                            "description": "Class map or 'class': interval "
                            "width the label budget is for.",
                        },
                        "expected_error_rate": {"type": "number"},
                        "class_share": {
                            "type": "number",
                            "description": "A class's share, for its budget.",
                        },
                        "seed": {"type": "integer", "default": 0},
                        "seed_from_map": {
                            "type": "boolean",
                            "default": False,
                            "description": "Only a scores file "
                            "olmoearth_scores_from_file wrote from a model run's "
                            "GeoTIFF; seed from the map's digest, for a "
                            "certificate; fixes alpha, delta, rule.",
                        },
                        "alpha": {"type": "number"},
                        "delta": {"type": "number", "default": DEFAULT_DELTA},
                        "rule": {
                            "type": "string",
                            "enum": list(ZONE_RULES),
                            "default": "prefix",
                        },
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
            capability=Capability(
                does="choose windows to label for an error rate ('confidence', "
                "'class'; 'random' for a certified zone) or where two maps "
                "differ ('disagreement')",
                needs=(
                    "threshold, for a Studio regression band not in [0, 1]",
                    "a budget of at most the valid windows (Studio grid 16 plans "
                    "all)",
                    "'disagreement': two maps of one time on one window grid",
                    "'random' or 'class' for a class map with no top-k scores",
                    "seed_from_map: a scores file from a model run's GeoTIFF",
                ),
            ),
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_estimate_map_error",
                description=(
                    "The map's error rate from the reviewer's 0/1 labels on a "
                    "design from olmoearth_plan_label_sample (design_path + "
                    "wrong, or the filled labels CSV), with the interval valid "
                    "at every sample size first, then the one that design "
                    "earns, and the package's method string; per-class "
                    "accuracy with reference classes. Windows labelled without "
                    "a design: window_indices + scores/scores_path, checked as "
                    "a random sample (a review set is refused: its rate is "
                    "inflated). Report each interval and its method as "
                    "returned; never substitute p +/- 1.96 sqrt(p(1-p)/n). "
                    "A design='disagreement' plan takes reference (the class "
                    "seen in each window, '?' if it cannot be judged) and says "
                    "which map is more accurate, or neither, by the package's "
                    "sign test or interval, with the accuracy difference's "
                    "interval. Needs the inferencex extra."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "design_path": _DESIGN_PATH,
                        "wrong": _WRONG,
                        "labels_path": _LABELS_PATH,
                        "reference": {
                            "type": "array",
                            "items": {"type": ["integer", "string"]},
                            "description": "Optional true class per window "
                            "(design order), for per-class accuracy; for "
                            "'disagreement', the class seen ('?': unjudged).",
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
            capability=Capability(
                does="the map's error rate, the valid interval first, then its "
                "design's, from the reviewer's 0/1 labels; per-class accuracy; "
                "which of two maps is more accurate",
                needs=(
                    "a label on every window of a planned design (a class for "
                    "'disagreement'), or on random windows",
                ),
                cannot=(
                    "use a review set, or windows chosen by margin, as the sample",
                ),
            ),
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_certify_zone",
                description=(
                    "The largest most-confident share of the map whose error "
                    "rate is at most alpha, certified by exact tests (the "
                    "statement fails on at most delta of draws); it may "
                    "certify none. Needs a "
                    "design='random' plan and its labels; a stratified design, "
                    "or a class map without top-k scores, is refused. rule "
                    "'prefix' (default) or 'bonferroni' (no assumption). Needs "
                    "the inferencex extra."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "design_path": _DESIGN_PATH,
                        "wrong": _WRONG,
                        "labels_path": _LABELS_PATH,
                        "alpha": {
                            "type": "number",
                            "description": "Maximum error rate, e.g. 0.05; a "
                            "seed_from_map plan's own.",
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
            capability=Capability(
                does="the largest most-confident share of the map whose error "
                "rate is at most alpha, by exact tests; it may certify none",
                needs=("a design='random' plan and its labels",),
                cannot=("certify a class map with no top-k scores",),
            ),
        ),
    ]
