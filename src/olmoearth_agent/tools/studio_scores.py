# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""A Studio result's own model output files as scores files (skill #18).

Studio's files endpoint (``GET /prediction-results/files?download_token=``)
returns a zip of a result's model output at native resolution. Measured live on
31 October 2026 (``StudioClient.prediction_result_files``): the zip holds
``<prediction name>/result_epsg<code>_<i>_native.tif``, one GeoTIFF per UTM
zone, and, for a class map with top-k scores, a ``README.md``. Two kinds of
result are read, decided from the result's ``result_metadata`` before anything
is downloaded:

- a class map with top-k scores (a segmentation prediction submitted with
  ``probability_output``): ``2k`` uint8 bands, interleaved rank-1 class,
  rank-1 score, rank-2 class, rank-2 score and so on, no-data 255. The README
  gives ``probability = (value - 1) / 254``, calls value 0 "reserved and
  unused", says a pixel is invalid only where band 1 holds the no-data value
  (a score band may hold 255 on a valid pixel) and that the scores are
  uncalibrated softmax scores. A README that no longer gives the formula is a
  changed format and refused. Scores are decoded with it; a raw 0 reads 0.0
  and is counted per rank. The kelp result measured that night contradicts
  the README: its raw rank-1 plus rank-2 scores are 254 on every valid pixel,
  so its rank-2 score is raw 0 on many of them and ``raw / 254`` fits better.
  For a two-class top-2 file the result reports, as facts, how many valid
  pixels hold that sum and the decoded sum's range. Each declared class gets a
  channel, in ``allowed_values`` order; a class outside a pixel's top-k is
  written 0, and stated as unknown, not 0, when ``k`` is below the number of
  classes.
- one regression field declared ``[0, 1]`` and no classification field (a
  KarstBinary-like score): one float32 band, no-data -1.0. Its values were
  measured from -0.0147 to 1.0070, which no probability is: values outside
  ``[0, 1]`` are refused, or clipped and counted with ``clip_out_of_range``.
  Reading the score as the probability of the positive class is an
  assumption, and the result says so.

A class map without top-k scores is refused, with where top-k scores come
from; so is any other result. Each zone is written as a run directory under
``<scores root>/studio_results/``: ``scores.tif``, a ``(C, H, W)`` float32
probability raster with NaN no-data on the zone's own grid, and a
``manifest.json`` whose ``source.kind`` is ``studio_files``. The zone is then
read by ``olmoearth_scores_from_file``'s own handler, so it is pooled exactly
as a direct run is (``tools/scores_file.py``), and its scores file is what the
review set, the label designs, the estimate and the certificate route read. A
zone written before from the same native file (by its sha256) is reused.

Assumed and unverified: the README's layout for ``k`` above 2 and a README at
the zip's root (both read as the kelp result's); the result's model is its
prediction's ``model_id`` (best effort), and the manifest's ``revision`` is the
result id, since Studio exposes no model revision; reading a ``[0, 1]`` score
as a probability. The download token is a credential: it goes to the client
alone, never into a result, a manifest, an error or a log, and neither does a
person's id. No coordinates in the result (rule §3.1): zones by file name,
windows by grid row and column.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import io
import json
import math
import os
import re
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.security.paths import safe_path
from olmoearth_agent.studio.client import DEFAULT_MAX_FILES_BYTES
from olmoearth_agent.tools import inferencex, scores_file
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.registry import Capability, RegisteredTool, ToolContext
from olmoearth_agent.tools.review_set import slug, write_root

#: The folder under the scores root that holds the zones' run directories.
STUDIO_DIR = "studio_results"

#: A zone's native GeoTIFF in the zip, the only entries read as rasters: one
#: folder, a plain name (:func:`_plain`), and the file name Studio gives it.
NATIVE_TIF = re.compile(
    r"^(?P<folder>[^/]+)/(?P<file>result_epsg(?P<epsg>[0-9]+)_(?P<i>[0-9]+)"
    r"_native\.tif)$"
)

#: A top-k result's README, in the prediction's folder or at the zip's root.
README = re.compile(r"^(?:(?P<folder>[^/]+)/)?README\.md$")

#: What a top-k result's README must still say: its score formula.
README_FORMULA = "(value - 1) / 254"

#: Most bytes, uncompressed, of the entries read from one zip, checked against
#: the sizes the zip declares before any entry is read (a zip bomb); zipfile
#: never decompresses an entry past its declared size.
MAX_UNCOMPRESSED_BYTES = 2 * 1024**3

#: Most bytes of one README read.
MAX_README_BYTES = 1 << 20

#: ``max_mb``'s default, and its cap: the client's own, 2 GiB.
DEFAULT_MAX_MB = 1024
MAX_MB = DEFAULT_MAX_FILES_BYTES // 2**20

#: The two kinds of result read, by what ``result_metadata`` declares.
TOPK, UNIT = "top_k", "unit_score"

#: The raster each zone's run directory holds.
RASTER = "scores.tif"

#: How a top-k score is decoded, stated in every result of one.
TOPK_DECODE = (
    "probability = (value - 1) / 254, the zip README's formula; raw 0 reads "
    "0.0 and raw 255 reads 1.0"
)

#: must_state of a class map with top-k scores.
TOPK_SCORES_SENTENCE = (
    "These scores are Studio's uncalibrated softmax scores, quantized to "
    "1/254: they rank windows and are not probabilities that a class is right."
)
#: must_state when ``k`` is below the number of classes.
UNRETAINED_SENTENCE = (
    "Classes outside a pixel's top-k scores are unknown, not zero, though the "
    "probability raster writes them as 0."
)
#: must_state of a ``[0, 1]`` score.
UNIT_ASSUMPTION_SENTENCE = (
    "Reading Studio's [0, 1] score as the probability of the positive class is "
    "an assumption: Studio declares only its range."
)
#: must_state when a ``[0, 1]`` score's values were clipped.
CLIPPED_SENTENCE = (
    "Values outside [0, 1] were clipped into it on request, so those pixels' "
    "scores are not the model's own."
)
#: must_state of both kinds. Studio's pixel-value read a pixel on the edge of
#: two zones from the neighbouring zone's web-optimized file, 2 levels off the
#: native raster (31 October 2026).
NATIVE_SENTENCE = (
    "These are Studio's native-resolution files; olmoearth_pixel_value reads a "
    "web-optimized copy, which can differ at a zone edge."
)

#: The keys of ``olmoearth_scores_from_file``'s result a zone does not repeat:
#: the zone states them itself, or the result once for every zone.
_POOLED_LEFT_OUT = frozenset(
    {"available", "provenance", "facts", "next_step", "scores_path"}
)


@dataclass(frozen=True)
class FilesPlan:
    """What a result's files hold, decided from its ``result_metadata``.

    ``kind`` is :data:`TOPK` or :data:`UNIT`; ``property_name`` the class
    property ``P`` or the score field. For top-k, ``values`` and ``labels``
    are the declared classes in ``allowed_values`` order (channel ``c`` is
    class ``values[c]``) and ``k`` the ranks the result declares.
    """

    kind: str
    property_name: str
    values: tuple[int, ...] = ()
    labels: tuple[str, ...] = ()
    k: int = 0

    @property
    def channels(self) -> int:
        """Channels of the probability raster written for each zone."""
        return len(self.values) if self.kind == TOPK else 1

    @property
    def class_names(self) -> dict[str, str]:
        """The manifest's classes by channel; a one-band score's two classes."""
        if self.kind == TOPK:
            return {str(c): label for c, label in enumerate(self.labels)}
        return {"0": f"not {self.property_name}", "1": self.property_name}


@dataclass(frozen=True)
class NativeEntry:
    """A zone's native GeoTIFF in the zip, with its EPSG code and index."""

    info: zipfile.ZipInfo
    file: str
    epsg: str
    index: str


@dataclass
class ZipEntries:
    """The entries of a files zip that are read, and how many are not."""

    tifs: list[NativeEntry]
    readmes: list[zipfile.ZipInfo]
    ignored: int


@dataclass
class Zone:
    """One zone read from the zip: its file, run directory and counts."""

    file: str
    run_dir: Path
    sha256: str
    reused: bool
    counts: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResultInfo:
    """What a zone's manifest records of the result; never its download token."""

    result_id: str
    prediction_id: str | None
    name: str
    model_id: str | None
    date_window: dict[str, str] | None


# --------------------------------------------------------------------------- the plan


def _fields(raw: Any) -> list[dict[str, Any]]:
    """The declared fields that name their property."""
    return [
        f
        for f in raw or []
        if isinstance(f, dict) and isinstance(f.get("property_name"), str)
    ]


def _number(value: Any) -> float | None:
    """``value`` as a float when it is a number (never a bool), else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _declares(entry: dict[str, Any], low: float, high: float) -> bool:
    """Whether a regression field declares exactly ``[low, high]``."""
    low_ok = _number(entry.get("min_value")) == low
    return low_ok and _number(entry.get("max_value")) == high


def _scored(entry: dict[str, Any], regressions: dict[str, dict[str, Any]]) -> bool:
    """Whether a class field has a confidence band and a declared rank-1 score."""
    rank1 = f"{entry['property_name']}_rank1_prob"
    return entry.get("confidence_band_index") is not None and rank1 in regressions


def _layout_refusal(name: str, declared: Any, expected: int) -> ValueError:
    """A field whose band is not where the files README's layout puts it."""
    return ValueError(
        f"result_metadata puts {name!r} in band {declared}; the files README's "
        "layout (rank-r class in band 2r-1, its score in band 2r) puts it in "
        f"band {expected}: the format changed, so nothing was downloaded"
    )


def _class_values(base: dict[str, Any]) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """The class property's declared classes: uint8 values and their labels."""
    name = str(base["property_name"])
    values: list[int] = []
    labels: list[str] = []
    for entry in base.get("allowed_values") or []:
        if not isinstance(entry, dict):
            continue
        number = _number(entry.get("value"))
        if number is None or not number.is_integer() or not 0 <= number <= 254:
            raise ValueError(
                f"{name!r} declares the class value {entry.get('value')!r}; a top-k "
                "file's class bands hold uint8 class values 0 to 254 (255 is no-data)"
            )
        values.append(int(number))
        labels.append(str(entry.get("label") or int(number)))
    if len(set(values)) != len(values):
        raise ValueError(f"{name!r} declares a class value twice")
    if len(values) < 2:
        raise ValueError(
            f"{name!r} declares {len(values)} class(es); a probability raster "
            "needs two or more"
        )
    return tuple(values), tuple(labels)


def _topk_plan(
    base: dict[str, Any],
    classes: list[dict[str, Any]],
    regressions: dict[str, dict[str, Any]],
) -> FilesPlan:
    """The plan of a class map with top-k scores, its layout checked.

    The README's layout, as the kelp result's metadata declares it: the class
    property ``P`` in band 1 with its confidence (``P_rank1_prob``) in band 2,
    ``P_rank<r>_class`` in band ``2r - 1`` and ``P_rank<r>_prob`` in band
    ``2r``, each score declared ``[0, 254]``. A margin needs two ranks.
    """
    name = str(base["property_name"])
    k = 0
    while f"{name}_rank{k + 1}_prob" in regressions:
        k += 1
    if k < 2:
        raise ValueError(
            f"{name!r} declares its rank-1 score only: a margin, and these files' "
            f"second rank, need two. {rules.TOPK_FROM_PROBABILITY_OUTPUT}"
        )
    by_name = {str(f["property_name"]): f for f in classes}
    for r in range(1, k + 1):
        score_name = f"{name}_rank{r}_prob"
        score = regressions[score_name]
        if not _declares(score, 0.0, 254.0):
            raise ValueError(
                f"{score_name!r} declares [{score.get('min_value')}, "
                f"{score.get('max_value')}]; this tool reads Studio's uint8 top-k "
                "scores, declared [0, 254]"
            )
        if score.get("band_index") != 2 * r:
            raise _layout_refusal(score_name, score.get("band_index"), 2 * r)
        klass = base if r == 1 else by_name.get(f"{name}_rank{r}_class")
        if klass is None:
            continue  # the README's layout alone places it
        where = klass.get("band_index")
        if where != 2 * r - 1:
            raise _layout_refusal(str(klass["property_name"]), where, 2 * r - 1)
        confidence = klass.get("confidence_band_index")
        if confidence not in (None, 2 * r):
            raise _layout_refusal(
                f"{klass['property_name']} confidence", confidence, 2 * r
            )
    values, labels = _class_values(base)
    return FilesPlan(TOPK, name, values=values, labels=labels, k=k)


def files_plan(record: dict[str, Any]) -> FilesPlan:
    """What a result's files hold, from its ``result_metadata``, or a refusal.

    A class map with top-k scores is a classification field with a
    ``confidence_band_index`` whose ``<P>_rank<r>_prob`` regression fields
    are declared ``[0, 254]``; a ``[0, 1]`` score is one regression field
    declared ``[0, 1]`` and no classification field. Anything else is refused
    before the download: a class map without top-k scores with where they
    come from (``probability_output``), carrying the contract's
    ``zone_for_a_class_map``.

    Parameters
    ----------
    record
        The result's record, as ``GET /prediction-results/{id}`` returns it.

    Returns
    -------
    FilesPlan
        The kind of result and its declared classes.

    Raises
    ------
    ValueError
        For any result this tool does not read, saying what it reads.

    Examples
    --------
    >>> files_plan({"result_metadata": {"regression_fields": [
    ...     {"property_name": "s", "band_index": 1, "min_value": 0, "max_value": 1}
    ... ]}}).kind
    'unit_score'
    """
    meta = record.get("result_metadata")
    if not isinstance(meta, dict):
        raise ValueError(
            "Studio's record of this result declares no outputs "
            "(result_metadata), so what its files hold is unknown; nothing was "
            "downloaded"
        )
    classes = _fields(meta.get("classification_fields"))
    declared = _fields(meta.get("regression_fields"))
    regressions = {str(f["property_name"]): f for f in declared}
    bases = [f for f in classes if _scored(f, regressions)]
    if len(bases) > 1:
        names = ", ".join(repr(f["property_name"]) for f in bases)
        raise ValueError(
            f"the result declares {len(bases)} class maps with top-k scores "
            f"({names}); this tool reads a result with one"
        )
    if bases:
        return _topk_plan(bases[0], classes, regressions)
    if classes:
        names = ", ".join(repr(f["property_name"]) for f in classes)
        raise rules.refusal(
            f"the result's {names} is a class map without top-k scores: its files "
            f"hold classes only, with no score to read. {rules.CLASS_MAP_REFUSAL}",
            forbidden_claims=[rules.zone_for_a_class_map()],
        )
    if len(regressions) == 1:
        (only,) = regressions.values()
        if _declares(only, 0.0, 1.0):
            if only.get("band_index") not in (None, 1):
                raise ValueError(
                    f"result_metadata puts {only['property_name']!r} in band "
                    f"{only.get('band_index')}; a [0, 1] score's file has one band"
                )
            return FilesPlan(UNIT, str(only["property_name"]))
    shown = ", ".join(
        f"{name!r} [{f.get('min_value')}, {f.get('max_value')}]"
        for name, f in regressions.items()
    )
    raise ValueError(
        "this tool reads a class map with top-k scores or one regression field "
        f"declared [0, 1]; the result declares {shown or 'no output field'}. A "
        "regression band in other units has no probability to pool: "
        "olmoearth_review_set_from_result ranks it by its distance from a "
        "threshold you give"
    )


# --------------------------------------------------------------------------- the zip


def _plain(folder: str | None) -> bool:
    """Whether a zip entry's folder is a plain name (no ``.``, ``..`` or ``\\``)."""
    return folder is None or (folder not in (".", "..") and "\\" not in folder)


def zip_entries(zf: zipfile.ZipFile, plan: FilesPlan) -> ZipEntries:
    """The native GeoTIFFs (and a top-k result's README) a files zip holds.

    Only an entry named ``<folder>/result_epsg<code>_<i>_native.tif``, its
    folder a plain name, is read as a zone, and a top-k result's
    ``README.md`` for its formula; every other entry is counted and ignored.
    No entry is ever extracted by its name. The sizes the zip declares for
    the entries read are checked against :data:`MAX_UNCOMPRESSED_BYTES`
    before any is read.

    Raises
    ------
    ValueError
        For a zip with no native GeoTIFF, two of one name, a README over
        :data:`MAX_README_BYTES` or entries over the cap.
    """
    tifs: list[NativeEntry] = []
    readmes: list[zipfile.ZipInfo] = []
    ignored = 0
    for info in zf.infolist():
        if info.is_dir():
            continue
        tif = NATIVE_TIF.fullmatch(info.filename)
        readme = README.fullmatch(info.filename)
        if tif is not None and _plain(tif["folder"]):
            if any(t.file == tif["file"] for t in tifs):
                raise ValueError(
                    f"the files zip holds {tif['file']} twice; which is the zone's "
                    "is unknown, so none was read"
                )
            tifs.append(NativeEntry(info, tif["file"], tif["epsg"], tif["i"]))
        elif plan.kind == TOPK and readme is not None and _plain(readme["folder"]):
            if info.file_size > MAX_README_BYTES:
                raise ValueError(
                    f"the files zip's README.md is {info.file_size} bytes, more "
                    f"than the {MAX_README_BYTES} this tool reads"
                )
            readmes.append(info)
        else:
            ignored += 1
    if not tifs:
        raise ValueError(
            "the files zip holds no <prediction>/result_epsg<code>_<i>_native.tif, "
            "the layout Studio's files had when measured on 31 October 2026"
        )
    total = sum(entry.info.file_size for entry in tifs)
    total += sum(info.file_size for info in readmes)
    if total > MAX_UNCOMPRESSED_BYTES:
        raise ValueError(
            f"the files zip declares {total} bytes uncompressed in the entries "
            f"read, more than the {MAX_UNCOMPRESSED_BYTES} this tool reads; "
            "nothing was read"
        )
    tifs.sort(key=lambda entry: entry.file)
    return ZipEntries(tifs, readmes, ignored)


def _read_entry(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    """One entry's bytes; a corrupt entry is a refusal naming it."""
    try:
        return zf.read(info)
    except (zipfile.BadZipFile, EOFError) as exc:
        name = info.filename
        raise ValueError(f"the files zip's {name!r} is corrupt ({exc})") from None


def check_readme(zf: zipfile.ZipFile, readmes: list[zipfile.ZipInfo]) -> bool:
    """Check that a top-k result's README still gives the score formula.

    Returns whether it calls value 0 "reserved and unused", as the README of
    31 October 2026 did.

    Raises
    ------
    ValueError
        With no README, or one without :data:`README_FORMULA` (a changed format).
    """
    if not readmes:
        raise ValueError(
            "the files zip of a class map with top-k scores holds no README.md, "
            "which says how its scores are stored; the format changed, so nothing "
            "was read"
        )
    unused = True
    for info in readmes:
        text = _read_entry(zf, info).decode("utf-8", errors="replace")
        if README_FORMULA not in text:
            raise ValueError(
                f"the files zip's {info.filename!r} does not give the score formula "
                f"{README_FORMULA} that the files of 31 October 2026 gave; the "
                "format changed, so no score was decoded"
            )
        unused = unused and "reserved and unused" in text.lower()
    return unused


# --------------------------------------------------------------------------- a zone


def _check_values(name: str, channels: int, height: int, width: int) -> None:
    """Refuse a zone whose probability raster is over ``scores_file.MAX_VALUES``."""
    if channels * height * width > scores_file.MAX_VALUES:
        raise ValueError(
            f"{name}: its probability raster holds {channels} x {height} x "
            f"{width} values, more than {scores_file.MAX_VALUES:,} this tool "
            "reads; the zone was not written"
        )


def decode_topk(
    np: Any, src: Any, plan: FilesPlan, name: str
) -> tuple[Any, dict[str, Any]]:
    """A zone's top-k bands as a ``(C, H, W)`` float32 probability raster.

    A pixel is valid where band 1 (the rank-1 class) is not the file's no-data
    value, and nothing else decides it: a score band may hold 255 on a valid
    pixel, which reads 1.0. Scores are decoded as ``(value - 1) / 254`` (the
    README), a raw 0 as 0.0. Channel ``c`` holds rank ``r``'s score where
    rank ``r``'s class is ``values[c]``, 0 for a class not among the pixel's
    top-k, and NaN on every channel where the pixel is invalid.

    Returns
    -------
    tuple
        The raster, and the zone's counts: valid pixels, rank-1 class counts,
        raw 0 and raw 255 scores per rank and, for a two-class top-2 file, the
        pixels whose raw scores sum to 254 and the decoded sum's range.

    Raises
    ------
    ValueError
        For bands that are not ``2k`` of uint8, no no-data value, no valid
        pixel, a class not declared, one class at two ranks of a pixel, or a
        rank's score above the rank's before it.
    """
    k, values = plan.k, plan.values
    dtypes = sorted({str(d) for d in src.dtypes})
    if src.count != 2 * k or dtypes != ["uint8"]:
        raise ValueError(
            f"{name} holds {src.count} band(s) of {', '.join(dtypes)}; a top-{k} "
            f"file holds {2 * k} uint8 bands (the README's layout)"
        )
    nodata = src.nodata
    if nodata is None:
        raise ValueError(
            f"{name} declares no no-data value, which says where band 1 has no "
            "pixel (the README's validity rule)"
        )
    height, width = int(src.height), int(src.width)
    _check_values(name, plan.channels, height, width)
    lut = np.full(256, -1, dtype=np.int16)
    for channel, value in enumerate(values):
        lut[value] = channel
    valid = src.read(1) != nodata
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError(f"{name} has no valid pixel: band 1 is no-data everywhere")
    out = np.full((plan.channels, height, width), np.nan, dtype=np.float32)
    out[:, valid] = 0.0
    codes: list[Any] = []
    raws: list[Any] = []
    probs: list[Any] = []
    raw_zero: list[int] = []
    raw_255: list[int] = []
    for r in range(1, k + 1):
        klass, raw = src.read(2 * r - 1), src.read(2 * r)
        code = lut[klass]
        stray = valid & (code < 0)
        if stray.any():
            shown = [int(v) for v in np.unique(klass[stray])[:6]]
            raise ValueError(
                f"{name}: rank {r}'s class band holds {shown} on valid pixels; the "
                f"result declares the classes {list(values)}"
            )
        for before, other in enumerate(codes, 1):
            same = int((valid & (code == other)).sum())
            if same:
                raise ValueError(
                    f"{name}: ranks {before} and {r} name one class on {same} "
                    "valid pixels, which no top-k field holds"
                )
        above = int((valid & (raw > raws[-1])).sum()) if raws else 0
        if above:
            raise ValueError(
                f"{name}: rank {r}'s score is above rank {r - 1}'s on {above} "
                "valid pixels, which no top-k field holds"
            )
        prob = (raw.astype(np.float32) - 1.0) / 254.0
        prob[raw == 0] = 0.0
        for channel in range(plan.channels):
            hit = valid & (code == channel)
            out[channel][hit] = prob[hit]
        raw_zero.append(int((valid & (raw == 0)).sum()))
        raw_255.append(int((valid & (raw == 255)).sum()))
        codes.append(code)
        raws.append(raw)
        probs.append(prob)
    first = np.bincount(codes[0][valid].astype(np.int64), minlength=plan.channels)
    class_counts = {plan.labels[c]: int(first[c]) for c in range(plan.channels)}
    counts: dict[str, Any] = {
        "valid_pixels": n_valid,
        "nodata_pixels": height * width - n_valid,
        "class_counts": class_counts,
        "raw_zero_scores_by_rank": raw_zero,
        "raw_255_scores_by_rank": raw_255,
    }
    if plan.channels == 2 and k == 2:
        total = raws[0].astype(np.int16) + raws[1].astype(np.int16)
        summed = (probs[0] + probs[1])[valid]
        counts["two_class_check"] = {
            "raw_sum_254_pixels": int((valid & (total == 254)).sum()),
            "decoded_sum_min": float(summed.min()),
            "decoded_sum_max": float(summed.max()),
            "rank1_below_half_pixels": int((valid & (probs[0] < 0.5)).sum()),
        }
    return out, counts


def decode_unit(
    np: Any, src: Any, plan: FilesPlan, name: str, clip: bool
) -> tuple[Any, dict[str, Any]]:
    """A zone's ``[0, 1]`` score as a ``(1, H, W)`` float32 raster, NaN no-data.

    A pixel is valid where its value is finite and not the file's no-data
    value. Values below 0 or above 1 are refused unless ``clip``, and then
    clipped into ``[0, 1]`` and counted.

    Raises
    ------
    ValueError
        For a file that is not one float band, no valid pixel, or values
        outside ``[0, 1]`` without ``clip``.
    """
    dtype = np.dtype(src.dtypes[0])
    if src.count != 1 or dtype.kind != "f":
        raise ValueError(
            f"{name} holds {src.count} band(s) of {dtype.name}; a [0, 1] score's "
            "file holds one float band"
        )
    height, width = int(src.height), int(src.width)
    _check_values(name, plan.channels, height, width)
    band = src.read(1).astype(np.float32)
    valid = np.isfinite(band)
    nodata = src.nodata
    if nodata is not None and not math.isnan(nodata):
        valid &= band != np.float32(nodata)
    values = band[valid]
    if values.size == 0:
        raise ValueError(f"{name} has no valid pixel")
    below, above = int((values < 0.0).sum()), int((values > 1.0).sum())
    low, high = float(values.min()), float(values.max())
    if (below or above) and not clip:
        raise ValueError(
            f"{name}: {below} valid pixels lie below 0 and {above} above 1 (values "
            f"run {low:.6g} to {high:.6g}); a score read as a probability lies in "
            "[0, 1], so nothing was written. clip_out_of_range=true clips them "
            "into [0, 1], and the result counts them"
        )
    np.clip(band, 0.0, 1.0, out=band)
    band[~valid] = np.nan
    positive = int((valid & (band >= 0.5)).sum())
    n_valid = int(values.size)
    names = plan.class_names
    counts: dict[str, Any] = {
        "valid_pixels": n_valid,
        "nodata_pixels": height * width - n_valid,
        "class_counts": {names["0"]: n_valid - positive, names["1"]: positive},
        "class_counts_cut": "the positive class at a score of 0.5 or more",
        "values_range": [low, high],
        "below_0_pixels": below,
        "above_1_pixels": above,
        "clipped": bool(clip and (below or above)),
    }
    return band[None], counts


def _reusable(zone_dir: Path, digest: str) -> bool:
    """Whether ``zone_dir`` holds a complete zone written from this native file.

    Its manifest (written last) records the native file's sha256, and its
    scores raster is the one the manifest records.
    """
    manifest_path, raster = zone_dir / scores_file.MANIFEST, zone_dir / RASTER
    if not (manifest_path.is_file() and raster.is_file()):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    source = manifest.get("source") if isinstance(manifest, dict) else None
    scores = manifest.get("scores") if isinstance(manifest, dict) else None
    if not (isinstance(source, dict) and isinstance(scores, dict)):
        return False
    if source.get("sha256") != digest:
        return False
    return bool(scores.get("sha256") == scores_file._sha256(raster))


def _write_zone(
    rasterio: Any,
    src: Any,
    array: Any,
    zone_dir: Path,
    manifest: dict[str, Any],
    names: tuple[str, ...],
) -> None:
    """Write a zone's scores raster, then its manifest, each by an atomic rename.

    The raster is on the native file's grid (its CRS and transform), NaN
    no-data, deflate-compressed; the manifest records its sha256 and shape.
    """
    zone_dir.mkdir(parents=True, exist_ok=True)
    channels, height, width = (int(s) for s in array.shape)
    stamp = uuid.uuid4().hex[:8]
    part = zone_dir / f".{RASTER}.{stamp}.part"
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": channels,
        "dtype": "float32",
        "crs": src.crs,
        "transform": src.transform,
        "nodata": float("nan"),
        "compress": "deflate",
    }
    with rasterio.open(part, "w", **profile) as dst:
        dst.write(array)
        for band, label in enumerate(names, 1):
            dst.set_band_description(band, label)
    raster = zone_dir / RASTER
    os.replace(part, raster)
    manifest["scores"] = {
        **manifest["scores"],
        "sha256": scores_file._sha256(raster),
        "shape": [channels, height, width],
    }
    text = zone_dir / f".{scores_file.MANIFEST}.{stamp}.part"
    text.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(text, zone_dir / scores_file.MANIFEST)


def _manifest(
    info: ResultInfo, plan: FilesPlan, native: str, digest: str
) -> dict[str, Any]:
    """A zone's manifest, as ``olmoearth_scores_from_file`` reads one."""
    # Studio exposes no model revision: the result id stands for one.
    model: dict[str, Any] = {"id": info.model_id, "revision": info.result_id}
    scores = {"file": RASTER, "values": "probabilities", "nodata": "NaN"}
    studio: dict[str, Any] = {"kind": plan.kind, "property_name": plan.property_name}
    manifest: dict[str, Any] = {
        "what": "a Studio result's native-resolution file, as probabilities",
        "model": model,
        "scores": scores,
        "classes": plan.class_names,
        "source": {
            "kind": scores_file.STUDIO_FILES,
            "result_id": info.result_id,
            "prediction_id": info.prediction_id,
            "file": native,
            "sha256": digest,
        },
        "studio": studio,
    }
    if plan.kind == TOPK:
        # A class value (what the class bands hold) -> its channel.
        manifest["channel_of_class_value"] = {
            str(v): c for c, v in enumerate(plan.values)
        }
        studio.update(k=plan.k, decode=TOPK_DECODE)
    else:
        studio["assumption"] = UNIT_ASSUMPTION_SENTENCE
    if info.date_window:
        manifest["date_window"] = info.date_window
    return manifest


def read_zones(
    data: bytes, plan: FilesPlan, info: ResultInfo, clip: bool, np: Any, rasterio: Any
) -> tuple[list[Zone], int, bool]:
    """Read a files zip into run directories, one per zone: the blocking part.

    Each zone is decoded (:func:`decode_topk`, :func:`decode_unit`), so a
    refusal is the same on every call, and written unless a complete zone
    from the same native file is already there (:func:`_reusable`).

    Returns
    -------
    tuple
        The zones, the entries ignored, and whether the README calls raw 0
        reserved and unused (``False`` for a ``[0, 1]`` score, which has none).
    """
    from rasterio.io import MemoryFile

    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ValueError("the files answer is not a zip; nothing was read") from None
    with zf:
        entries = zip_entries(zf, plan)
        unused = check_readme(zf, entries.readmes) if plan.kind == TOPK else False
        folder = f"{slug(info.name, 40)}-{slug(info.result_id, 8)}"
        names = plan.labels if plan.kind == TOPK else (plan.property_name,)
        zones: list[Zone] = []
        for entry in entries.tifs:
            raw = _read_entry(zf, entry.info)
            digest = hashlib.sha256(raw).hexdigest()
            zone = f"epsg{entry.epsg}_{entry.index}-{digest[:8]}"
            zone_dir = safe_path(f"{STUDIO_DIR}/{folder}/{zone}", root=write_root())
            with MemoryFile(raw) as memory:
                try:
                    src = memory.open()
                except rasterio.errors.RasterioIOError as exc:
                    raise ValueError(
                        f"{entry.file}: not a GeoTIFF rasterio can read ({exc})"
                    ) from None
                with src:
                    if plan.kind == TOPK:
                        array, counts = decode_topk(np, src, plan, entry.file)
                    else:
                        array, counts = decode_unit(np, src, plan, entry.file, clip)
                    reused = _reusable(zone_dir, digest)
                    if not reused:
                        manifest = _manifest(info, plan, entry.file, digest)
                        _write_zone(rasterio, src, array, zone_dir, manifest, names)
                    del array
            zones.append(Zone(entry.file, zone_dir, digest, reused, counts))
    return zones, entries.ignored, unused


# --------------------------------------------------------------------------- the result


def _raw_zero(file: str, rank: int, pixels: int, said: str) -> dict[str, Any]:
    """The fact of a rank's raw 0 scores on valid pixels."""
    return rules.fact(
        "studio_raw_zero",
        f"{file}: rank {rank}'s score is raw 0 on {pixels} valid pixels, read as "
        f"0.0; {said}.",
        file=file,
        rank=rank,
        pixels=pixels,
    )


def _zone_facts(zone: Zone, plan: FilesPlan, unused: bool) -> list[dict[str, Any]]:
    """The facts one zone's counts give (never a coordinate)."""
    counts, file = zone.counts, zone.file
    n_valid = counts["valid_pixels"]
    total = n_valid + counts["nodata_pixels"]
    shown = ", ".join(f"{label} {n}" for label, n in counts["class_counts"].items())
    head = "rank-1 classes" if plan.kind == TOPK else "classes at a cut of 0.5"
    zone_fact = rules.fact(
        "studio_zone",
        f"{file}: {n_valid} valid pixels of {total}; {head}: {shown}.",
        file=file,
        valid_pixels=n_valid,
        class_counts=counts["class_counts"],
    )
    out = [zone_fact]
    if plan.kind == TOPK:
        said = "the README does not call value 0 unused"
        if unused:
            said = 'the README calls value 0 "reserved and unused"'
        zeros = counts["raw_zero_scores_by_rank"]
        out += [_raw_zero(file, r, n, said) for r, n in enumerate(zeros, 1) if n]
        full = counts["raw_255_scores_by_rank"]
        if any(full):
            ones = rules.fact(
                "studio_raw_255",
                f"{file}: a score band holds 255, the no-data value, on "
                f"{sum(full)} valid pixels (by rank {full}), read as 1.0: "
                "validity is read from the rank-1 class band only.",
                file=file,
                pixels_by_rank=full,
            )
            out.append(ones)
        check = counts.get("two_class_check")
        if check:
            low, high = check["decoded_sum_min"], check["decoded_sum_max"]
            summed = rules.fact(
                "studio_two_class_sum",
                f"{file}: the raw rank-1 and rank-2 scores sum to 254 on "
                f"{check['raw_sum_254_pixels']} of {n_valid} valid pixels; read as "
                f"(value - 1) / 254 they sum to {low:.4f} to {high:.4f}, and the "
                f"rank-1 score is below 0.5 on {check['rank1_below_half_pixels']} "
                "pixels.",
                file=file,
                **check,
            )
            out.append(summed)
    elif counts["clipped"]:
        low, high = counts["values_range"]
        clipped = rules.fact(
            "studio_clipped",
            f"{file}: {counts['below_0_pixels']} valid pixels below 0 and "
            f"{counts['above_1_pixels']} above 1 (values {low:.6g} to {high:.6g}) "
            "were clipped into [0, 1] on request.",
            file=file,
            below_0_pixels=counts["below_0_pixels"],
            above_1_pixels=counts["above_1_pixels"],
        )
        out.append(clipped)
    return out


def _classes_listed(plan: FilesPlan) -> list[dict[str, Any]]:
    """The classes as the result states them: each channel's, or a score's two."""
    if plan.kind == TOPK:
        return [
            {"channel": c, "value": value, "label": label}
            for c, (value, label) in enumerate(zip(plan.values, plan.labels))
        ]
    names = plan.class_names
    return [{"class": 0, "label": names["0"]}, {"class": 1, "label": names["1"]}]


async def _model_id(ctx: ToolContext, prediction_id: str | None) -> str | None:
    """The prediction's ``model_id``, best effort; nothing else of its record."""
    if not prediction_id:
        return None
    try:
        record = await ctx.studio.get_prediction(prediction_id)
    except Exception:  # noqa: BLE001 - a model id is optional; it never fails the tool
        return None
    model_id = record.get("model_id") if isinstance(record, dict) else None
    return model_id if isinstance(model_id, str) and model_id else None


def _period(record: dict[str, Any]) -> dict[str, str] | None:
    """The result's period from ``result_metadata`` (start and end), or ``None``."""
    meta = record.get("result_metadata") or {}
    start, end = meta.get("start_datetime"), meta.get("end_datetime")
    if isinstance(start, str) and isinstance(end, str):
        return {"start": start, "end": end}
    return None


async def _scores_from_studio(
    args: dict[str, Any], ctx: ToolContext
) -> dict[str, Any]:
    """Handler for ``olmoearth_scores_from_studio``.

    The result's record first (its metadata decides the kind, or refuses it),
    then the extra, then the download, so nothing is downloaded that cannot
    be read; the token goes to the client alone.
    """
    result_id = str(args["result_id"]).strip()
    if not result_id:
        raise ValueError("result_id is empty; name the Studio result to read")
    patch = int(args.get("patch", scores_file.DEFAULT_PATCH))
    if patch < 1:
        raise ValueError(f"patch must be a positive number of pixels, got {patch}")
    max_mb = int(args.get("max_mb", DEFAULT_MAX_MB))
    if max_mb < 1:
        raise ValueError(f"max_mb must be 1 or more, got {max_mb}")
    clip = bool(args.get("clip_out_of_range", False))

    record = await ctx.studio.get_prediction_result(result_id)
    plan = files_plan(record)
    try:
        # The modules olmoearth_scores_from_file's handler loads, checked
        # before the download: without them nothing downloaded can be read.
        inferencex.load("review.assess")
        inferencex.load("common.raster")
    except inferencex.InferencexMissingError as exc:
        return scores_file._missing(f"olmoearth-inferencex: {exc}")
    try:
        rasterio = importlib.import_module("rasterio")
        importlib.import_module("rasterio.io")
        np = importlib.import_module("numpy")
    except ImportError as exc:
        return scores_file._missing(f"rasterio: {exc}")

    token = record.get("download_token")
    if not isinstance(token, str) or not token:
        raise ValueError(
            "Studio's record of this result carries no download token, so its "
            "files cannot be downloaded with this key"
        )
    found = record.get("prediction_id")
    prediction_id = found if isinstance(found, str) and found else None
    info = ResultInfo(
        result_id=result_id,
        prediction_id=prediction_id,
        name=str(record.get("prediction_name") or record.get("name") or "result"),
        model_id=await _model_id(ctx, prediction_id),
        date_window=_period(record),
    )
    max_bytes = min(max_mb, MAX_MB) * 2**20
    data = await ctx.studio.prediction_result_files(token, max_bytes=max_bytes)
    del token
    # Unzipping, decoding and writing run in a worker thread, as the
    # certificate route's package calls do: the event loop keeps serving.
    zones, ignored, unused = await asyncio.to_thread(
        read_zones, data, plan, info, clip, np, rasterio
    )
    del data

    facts: list[dict[str, Any]] = []
    listed: list[dict[str, Any]] = []
    for zone in zones:
        try:
            pooled = await scores_file._scores_from_file(
                {"run_dir": str(zone.run_dir), "patch": patch}, ctx
            )
        except ValueError as exc:
            raise ValueError(
                f"{zone.file} (written to {zone.run_dir}): {exc}"
            ) from None
        facts += _zone_facts(zone, plan, unused)
        summary = {k: v for k, v in pooled.items() if k not in _POOLED_LEFT_OUT}
        entry: dict[str, Any] = {
            "file": zone.file,
            "run_dir": str(zone.run_dir),
            "scores_path": pooled.get("scores_path"),
            "reused": zone.reused,
            **zone.counts,
            "pooled": summary,
        }
        listed.append(entry)
    if ignored:
        skipped = rules.fact(
            "studio_zip_ignored",
            f"The files zip held {ignored} other entries, which were not read.",
            ignored_entries=ignored,
        )
        facts.append(skipped)
    out: dict[str, Any] = {
        "available": True,
        "result_id": result_id,
        "prediction_id": prediction_id,
        "kind": plan.kind,
        "property_name": plan.property_name,
        "provenance": scores_file.STUDIO_PROVENANCE,
        "classes": _classes_listed(plan),
        "n_zones": len(listed),
        "zones": listed,
        "ignored_entries": ignored,
        "date_window": info.date_window,
        "model_id": info.model_id,
    }
    if plan.kind == TOPK:
        out["k"] = plan.k
        out["decode"] = TOPK_DECODE
        must = [TOPK_SCORES_SENTENCE]
        if plan.k < plan.channels:
            must.append(UNRETAINED_SENTENCE)
    else:
        out["scores_raster"] = "one channel: the positive class's score"
        out["assumption"] = UNIT_ASSUMPTION_SENTENCE
        must = [UNIT_ASSUMPTION_SENTENCE]
        if any(zone.counts.get("clipped") for zone in zones):
            must.append(CLIPPED_SENTENCE)
    must.append(NATIVE_SENTENCE)
    out["next_step"] = (
        "Pass a zone's scores_path to olmoearth_review_set (which windows to "
        "check first) or olmoearth_plan_label_sample (design='random' with "
        "seed_from_map for a certified zone that can be packed into a "
        "certificate). Each UTM zone is its own map and its own sample."
    )
    return rules.add_contract(out, facts=facts, must_state=must)


def build_studio_scores_tools() -> list[RegisteredTool]:
    """Return the Studio-files scores tool (skill #18; the ``inferencex`` extra)."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_scores_from_studio",
                description=(
                    "Read a STUDIO result's native-resolution files (one GeoTIFF "
                    "per UTM zone) into scores files, one per zone: a class map "
                    "with top-k scores, or a [0, 1] score band; a class map "
                    "without top-k scores is refused. Each zone is pooled as "
                    "olmoearth_scores_from_file pools a run; pass its scores_path "
                    "to olmoearth_review_set or olmoearth_plan_label_sample. "
                    "Returns each zone's valid pixels and classes; no "
                    "coordinates. Needs the inferencex extra."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "result_id": {"type": "string"},
                        "patch": {
                            "type": "integer",
                            "default": scores_file.DEFAULT_PATCH,
                            "description": "Window side in pixels.",
                        },
                        "clip_out_of_range": {
                            "type": "boolean",
                            "default": False,
                            "description": "Clip a [0, 1] score's values outside "
                            "[0, 1] instead of refusing.",
                        },
                        "max_mb": {
                            "type": "integer",
                            "default": DEFAULT_MAX_MB,
                            "description": f"Largest download, at most {MAX_MB}.",
                        },
                    },
                    "required": ["result_id"],
                },
            ),
            handler=_scores_from_studio,
            capability=Capability(
                does="a Studio result's native files (top-k scores or a [0, 1] "
                "score) as scores files, one per zone",
                cannot=("read a class map with no top-k scores",),
            ),
        ),
    ]
