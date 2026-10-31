# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The scores provider: a model run's full output as review-set input (skill #18).

Studio's API returns map tiles and a point lookup, never per-class scores. A
model run outside Studio (for instance ``scripts/score_area.py`` of the
research record, ``2imi9/olmoearth_inferenceX-record``, on a GPU cluster)
writes what Studio does not: a
georeferenced ``(C, H, W)`` float32 raster of logits or probabilities, NaN
where there is no prediction, and a ``manifest.json`` that names the model,
its revision, the classes and whether the values are logits.

It also reads what ``oe-inferencex from-olmoearth DATASET --out RUN`` writes
from the rslearn dataset an OlmoEarth model run leaves
(``oe_inferencex.readers.olmoearth.read_output``): ``scores_<EPSG>.tif``, the
softmax probabilities of every window pasted onto one grid per CRS, NaN where
no window predicted, and ``olmoearth_output.json``, the summary that names
them (no model, revision or class names: the classes are the channels). The
tool takes RUN, the summary, or one ``scores_<EPSG>.tif`` when the run wrote
one per CRS. A run that wrote class ids only (rslearn's default output, which
``from-olmoearth`` refuses) is refused here too, with how to get
probabilities: ``output_probs: true``, a float32 output layer and the
fine-tuned checkpoint, in the package's words; so is the rslearn dataset
itself, with the ``from-olmoearth`` command that reads it.

``olmoearth_scores_from_file`` reads such a directory under
``OLMOEARTH_SCORES_ROOT``, pools the raster to windows exactly as the
package's ``oe-inferencex assess`` does (``oe_inferencex.review.assess.
assess_prediction`` with ``is_logit`` from the manifest and the no-data mask of
the package's own GeoTIFF reader, ``oe_inferencex.common.raster.read_raster``),
and writes one scores file that
``olmoearth_review_set``, ``olmoearth_plan_label_sample``,
``olmoearth_estimate_map_error`` and ``olmoearth_compare_review`` read.

A window's row carries the package's window confidence ``m`` at the window's
majority class and 0 at every other class (the mapping exp64's agent arm and
exp86's fixture F4 use), so a row's top-1 minus top-2 is ``m`` and its
arg-max is the window's class: the agent's review set is then the package's
(exactly tied windows apart, which the package orders by raster position).
The file also carries each window's top-1 probability (``p1``, pooled as the
package's ``sample`` pools it), which the confidence design allocates by, and
its class, which a zero margin cannot carry in a row.

A channel the manifest marks as the label fill value (``score_area.py`` names
it ``nodata_value_<c>``; AWF's channel 9, never a training target) is left out
of the margin and of the class vote unless ``include_fill_class`` is set.

Reading the GeoTIFF needs the optional ``inferencex`` extra (numpy and
rasterio through ``olmoearth-inferencex[geo]``); without it the tool answers
with the install line. A raster the package's reader cannot read is a refusal
(it exits a command-line run; here its message becomes the tool's error). No
coordinates in the result (rule §3.1): windows are a
grid ``(row, col)`` and index; each valid window's location stays in the file,
for the labelling sheet.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from olmoearth_agent.analysis.review_set import MAX_WINDOWS
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools import inferencex
from olmoearth_agent.tools.registry import Capability, RegisteredTool, ToolContext
from olmoearth_agent.tools.review_set import (
    load_json_file,
    read_roots,
    slug,
    write_json_file,
    write_root,
)

#: The manifest a model run writes beside its raster.
MANIFEST = "manifest.json"

#: The summary ``oe-inferencex from-olmoearth`` writes beside its rasters
#: (``oe_inferencex.readers.olmoearth.SCORES_JSON``).
SUMMARY = "olmoearth_output.json"

#: How a run's files name the record that describes them, by kind.
RECORDS = {"manifest": "the manifest", "from-olmoearth": f"the run's {SUMMARY}"}

#: Said with a refusal of class ids, after the package's own fix: the
#: package's Usage on the checkpoint ("From an OlmoEarth model run with
#: rslearn").
CHECKPOINT_NOTE = (
    "Run prediction with the fine-tuned checkpoint (rslearn model predict "
    "--ckpt_path, or the config's management_dir restoring the best one): "
    "without it the model predicts from its initial weights, and its "
    "probabilities look plausible with no error. Then read the dataset with "
    "oe-inferencex from-olmoearth DATASET --out RUN and pass RUN here"
)

#: The raster read when the manifest names none.
DEFAULT_RASTER = "scores.tif"

#: Window side in pixels (``oe-inferencex assess --patch``'s default).
DEFAULT_PATCH = 4

#: Largest raster read, in values (channels x pixels): about 256 MB of float32.
MAX_VALUES = 64_000_000

#: ``score_kind`` of a scores file this tool writes.
SCORE_KIND = "window_confidence"

#: What the rows are, written into the file.
ROW_ENCODING = (
    "each row holds the window's confidence at its majority class and 0 at "
    "every other class (exp64's mapping), so a row's top-1 minus top-2 is the "
    "package's window confidence and its arg-max the window's class"
)

#: Where the scores come from, stated in every result.
PROVENANCE = (
    "full model output from a direct run: every pixel's pre-argmax scores, "
    "read from the run's raster and pooled to windows by olmoearth-inferencex; "
    "not Studio point samples"
)

_FILL_NAME = re.compile(r"nodata_value_(\d+)")

#: The per-pixel confidence ``assess_prediction`` pools, by (is_logit, multi-class).
_PIXEL_CONFIDENCE = {
    (True, True): "the top-1 minus top-2 logit margin",
    (True, False): "the absolute logit (a binary map)",
    (False, True): "the top-1 probability (the package's confidence for "
    "probabilities; the top-1 minus top-2 margin needs logits)",
    (False, False): "twice the probability's distance from 0.5 (a binary map)",
}


def _missing(detail: str) -> dict[str, Any]:
    """The answer when the extra (or rasterio under it) is not importable.

    A release older than ``inferencex.MIN_VERSION`` has no ``review.assess``
    or ``common.raster`` (before 1.19.0), or ``inferencex.load`` refuses it
    (1.19.0 to 1.20.2): the reason names its version and the upgrade line.
    """
    status, install = inferencex.unavailable()
    return {
        "available": False,
        "reason": (
            "Reading a model run's scores raster needs the optional package "
            f"olmoearth-inferencex (>= {inferencex.MIN_VERSION}) with rasterio "
            f"(its geo extra), {status}. Nothing was read; do not rank the map's "
            "windows by hand."
        ),
        "missing": detail,
        "install": install,
    }


def _inside(real: str, roots: list[str]) -> bool:
    """Whether a resolved path sits inside one of ``roots``."""
    return any(os.path.commonpath([root, real]) == root for root in roots)


def resolve_run(raw: str) -> tuple[Path, str | None]:
    """The run directory a model-chosen ``run_dir`` names, and the raster it names.

    A relative path is taken under the scores root (``OLMOEARTH_SCORES_ROOT``,
    else the workspace root); the resolved directory must sit inside a read
    root (the rule scores files already follow). A path to the directory's
    manifest, ``from-olmoearth`` summary or raster names the directory; a
    raster's file name is returned too, which picks one of a
    ``from-olmoearth`` run's rasters (one per CRS).
    """
    text = str(raw).strip()
    if not text:
        raise ValueError("run_dir is empty; name the model run's directory")
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = write_root() / candidate
    real = os.path.realpath(candidate)
    roots = read_roots()
    if not _inside(real, roots):
        msg = f"run_dir must be a directory under {' or '.join(roots)}"
        raise ValueError(msg)
    named: str | None = None
    if os.path.isfile(real) and real.endswith((".tif", ".tiff")):
        named = os.path.basename(real)
        real = os.path.dirname(real)
    elif os.path.isfile(real) and os.path.basename(real) in (MANIFEST, SUMMARY):
        real = os.path.dirname(real)
    if not os.path.isdir(real):
        raise ValueError(f"run_dir {text!r} is not a directory under the scores root")
    return Path(real), named


def resolve_run_dir(raw: str) -> Path:
    """The run directory ``run_dir`` names, or raise (:func:`resolve_run`)."""
    return resolve_run(raw)[0]


def _raster_path(run_dir: Path, manifest: dict[str, Any]) -> Path:
    """The raster the manifest names: a plain .tif name inside the run directory."""
    name = str((manifest.get("scores") or {}).get("file") or DEFAULT_RASTER)
    if name != os.path.basename(name) or not name.endswith((".tif", ".tiff")):
        raise ValueError(
            f"the manifest names the raster {name!r}; it must be a .tif file "
            "name inside the run directory"
        )
    real = os.path.realpath(run_dir / name)
    if not _inside(real, read_roots()) or not os.path.isfile(real):
        raise ValueError(f"the run directory has no raster {name!r}")
    return Path(real)


def _sha256(path: Path) -> str:
    """Hex sha256 of a file, read in 1 MB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _values_kind(manifest: dict[str, Any]) -> str:
    """``"logits"`` or ``"probabilities"`` from the manifest; anything else refused."""
    kind = str((manifest.get("scores") or {}).get("values") or "").strip().lower()
    if kind in ("logit", "logits"):
        return "logits"
    if kind in ("probability", "probabilities"):
        return "probabilities"
    raise ValueError(
        "the manifest does not say whether the scores are logits or "
        "probabilities (scores.values); the tool does not guess"
    )


def _class_names(manifest: dict[str, Any], n_channels: int) -> dict[int, str]:
    """Channel -> class name from the manifest's ``classes`` (checked against the raster)."""
    raw = manifest.get("classes") or {}
    if not isinstance(raw, dict) or not raw:
        return {c: f"class_{c}" for c in range(n_channels)}
    names = {int(k): str(v) for k, v in raw.items()}
    if sorted(names) != list(range(n_channels)):
        raise ValueError(
            f"the manifest names classes {sorted(names)} and the raster has "
            f"{n_channels} channels; they must be the same channels"
        )
    return names


def fill_channels(manifest: dict[str, Any], names: dict[int, str]) -> list[int]:
    """Channels the manifest marks as the label fill value, never a training target.

    ``score_area.py`` names such a channel ``nodata_value_<c>`` (its task
    card's ``nodata_value`` with no class name); the task card's
    ``nodata_value`` marks it too when that channel carries no other name.
    """
    marked = {
        c
        for c, name in names.items()
        if (m := _FILL_NAME.fullmatch(name)) and int(m.group(1)) == c
    }
    task = (manifest.get("task_card") or {}).get("task") or {}
    value = task.get("nodata_value")
    if isinstance(value, int) and value in names:
        if names[value].startswith("nodata_value") or names[value] == f"class_{value}":
            marked.add(value)
    return sorted(marked)


def _model_block(manifest: dict[str, Any]) -> dict[str, Any]:
    """The model and revision from the manifest (plain strings and flags only)."""
    model = manifest.get("model")
    if not isinstance(model, dict):
        return {"repo": None, "revision": None}
    out: dict[str, Any] = {}
    for key in ("repo", "id", "file", "revision", "revision_matches_record"):
        value = model.get(key)
        if value is None or isinstance(value, (str, bool)):
            if value is not None or key in ("repo", "revision"):
                out[key] = value
    return out


def model_and_raster(
    model: dict[str, Any],
    digest: str,
    scores_file: str,
    record: str = RECORDS["manifest"],
) -> str:
    """One sentence that keeps the model's revision and the raster's hash apart.

    exp86 round 7 (brief 4, cluster) gave the raster's sha256 prefix
    ``a7c40be9``, which is also in the scores file's name, as the model's
    revision ``a347b15``. ``record`` is what describes the run's files: its
    manifest, or the summary ``from-olmoearth`` writes, which names no model.
    """
    name = model.get("repo") or model.get("id") or f"not named by {record},"
    revision = model.get("revision")
    at = (
        f"at revision {str(revision)[:7]}"
        if isinstance(revision, str) and revision
        else f"at a revision {record} does not record"
    )
    in_name = (
        f"; the {digest[:8]} in the scores file's name is that hash, not a revision"
        if digest[:8] in scores_file
        else ""
    )
    return (
        f"model {name} {at}; raster sha256 {digest[:8]} (a hash of the scores "
        f"raster, not of the model){in_name}."
    )


def _date_window(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """The run's date window (start and end only), for olmoearth_compare_review."""
    window = manifest.get("date_window")
    if not isinstance(window, dict):
        return None
    start, end = window.get("start"), window.get("end")
    if not (isinstance(start, str) and isinstance(end, str)):
        return None
    return {"start": start, "end": end}


def _top1(np: Any, scores: Any, is_logit: bool) -> Any:
    """Per-pixel top-1 probability, as ``oe-inferencex sample`` computes it."""
    x = np.asarray(scores, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        if x.ndim == 2:
            p = 1.0 / (1.0 + np.exp(-x)) if is_logit else x
            return np.maximum(p, 1.0 - p)
        if is_logit:
            z = np.exp(x - x.max(0, keepdims=True))
            return z.max(0) / z.sum(0)
        return x.max(0)


def _pool_mean(np: Any, a: Any, patch: int, all_valid: bool) -> Any:
    """Window mean over whole windows, over valid pixels only when any are not."""
    h, w = a.shape[0] // patch * patch, a.shape[1] // patch * patch
    blocks = a[:h, :w].reshape(h // patch, patch, w // patch, patch)
    if all_valid:
        return blocks.mean(axis=(1, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(blocks, axis=(1, 3))


def _centres(np: Any, geo: Any, idx: Any, cols: int, patch: int) -> list[Any] | None:
    """Each window's centre as [lon, lat], for the file only; ``None`` without a georeference."""
    if not geo or geo.get("crs") is None:
        return None
    try:
        from rasterio.warp import transform as warp

        rows, cs = np.divmod(idx, cols)
        px, py = cs * patch + patch / 2, rows * patch + patch / 2
        t = geo["transform"]
        xs, ys = t.a * px + t.b * py + t.c, t.d * px + t.e * py + t.f
        lon, lat = warp(geo["crs"], "EPSG:4326", xs.tolist(), ys.tolist())
    except Exception:  # noqa: BLE001 - a location is optional; it never fails the tool
        return None
    return [[float(a), float(b)] for a, b in zip(lon, lat)]


def _check_size(rasterio: Any, path: Path, patch: int) -> None:
    """Refuse a raster too large to read, or with more windows than a ranking takes.

    A file rasterio cannot open is refused in the words of the package's own
    reader (``read_raster``), which this check runs before.
    """
    try:
        src = rasterio.open(path)
    except rasterio.errors.RasterioIOError as exc:
        raise ValueError(
            f"{path.name}: not a raster rasterio can read ({exc})"
        ) from None
    with src:
        count, height, width = src.count, src.height, src.width
    if count * height * width > MAX_VALUES:
        raise ValueError(
            f"the raster holds {count} x {height} x {width} values, more than "
            f"{MAX_VALUES:,} this tool reads; score a smaller area"
        )
    if patch > min(height, width):
        raise ValueError(f"patch {patch} px is larger than the {height} x {width} map")
    n = (height // patch) * (width // patch)
    if n > MAX_WINDOWS:
        smallest = patch
        while (height // smallest) * (width // smallest) > MAX_WINDOWS:
            smallest += 1
        raise ValueError(
            f"{n:,} windows of {patch} px exceed the {MAX_WINDOWS:,} a ranking "
            f"takes; pass patch >= {smallest}"
        )


def read_scores_raster(raster_io: Any, path: Path) -> tuple[Any, Any, Any]:
    """The raster, its valid-pixel mask and its georeference, read by the package.

    ``raster_io`` is ``oe_inferencex.common.raster``. Its ``read_raster`` marks a
    pixel no-data where any channel is not finite, where every channel holds
    the raster's no-data value, or where GDAL's mask says so. It is the
    command line's reader and ends a bad file with ``SystemExit`` ("no such
    file", "not a raster rasterio can read"); that is a refusal here, raised as
    ``ValueError`` so the registry answers with it and the agent keeps running.

    Parameters
    ----------
    raster_io
        The package's raster module, as :func:`inferencex.load` returns it.
    path
        The scores raster.

    Returns
    -------
    tuple
        ``(array, valid, geo)`` as ``read_raster`` returns them.

    Examples
    --------
    >>> read_scores_raster(raster_io, Path("run/scores.tif"))  # doctest: +SKIP
    """
    try:
        array, valid, geo = raster_io.read_raster(str(path))
    except SystemExit as exc:
        reason = str(exc) or f"{path.name}: the package's reader could not read it"
        raise ValueError(f"the scores raster was not read: {reason}") from None
    return array, valid, geo


@dataclass
class RunFiles:
    """What a run directory holds, read before its raster.

    ``manifest`` is score_area.py's ``manifest.json``, or for a
    ``from-olmoearth`` run the same fields made from its summary (the
    raster, ``probabilities``, no classes or model); ``kind`` is
    ``"manifest"`` or ``"from-olmoearth"``; ``files`` is what the result
    states of them; ``shape_hw`` and ``bands`` are the summary's grid shape
    and band count, checked against the raster read.
    """

    kind: str
    manifest: dict[str, Any]
    raster: Path
    files: dict[str, Any]
    shape_hw: list[int] | None = None
    bands: int | None = None


def _class_ids(rasterio: Any, np: Any, path: Path) -> str | None:
    """What ``path`` holds when it is a class map, not scores; ``None`` otherwise.

    As the package's ``from-olmoearth`` reads an output layer
    (``readers.olmoearth._probabilities``): bands of an integer type (one
    band of class ids, or C bands that truncate probabilities to 0 and 1), or
    one float band of whole numbers beyond 1. Read from the raster's first
    block of at most 512 x 512 pixels.
    """
    from rasterio.windows import Window

    try:
        src = rasterio.open(path)
    except rasterio.errors.RasterioIOError:
        return None
    with src:
        count, dtype = src.count, np.dtype(src.dtypes[0])
        window = Window(0, 0, min(512, src.width), min(512, src.height))
        read = src.read(1, window=window, masked=True)  # the no-data value left out
        band = read.astype(np.float64).filled(np.nan)
    if count > 1:
        if dtype.kind in "biu":
            return (
                f"{count} bands of {dtype.name}: an integer layer truncates the "
                "probabilities to 0 and 1, so no confidence survives"
            )
        return None
    values = np.unique(band[np.isfinite(band)])
    whole = values.size > 0 and bool(np.array_equal(values, np.rint(values)))
    if dtype.kind in "biu" or (whole and float(values.max()) > 1.0):
        shown = ", ".join(f"{v:g}" for v in values[:6].tolist())
        more = " ..." if values.size > 6 else ""
        return (
            f"one band of class ids ({shown}{more}), the argmax rslearn's "
            "SegmentationTask writes by default, with no confidence to rank"
        )
    return None


def class_id_refusal(what: str, reason: str, fix: str) -> ValueError:
    """The refusal of a run that wrote class ids, with how to get probabilities.

    ``fix`` is the package's ``readers.olmoearth.FIX_PROBABILITIES``
    (``output_probs: true``, a float32 output layer, the rerun), followed by
    :data:`CHECKPOINT_NOTE` (the fine-tuned checkpoint).
    """
    return ValueError(
        f"{what} holds {reason}; this tool reads probabilities or logits, as "
        f"oe-inferencex from-olmoearth does. {fix.rstrip('.')}. {CHECKPOINT_NOTE}."
    )


def _no_record(run_dir: Path, rasterio: Any, np: Any, fix: str) -> ValueError:
    """Why a directory with neither a manifest nor a summary is refused.

    An rslearn dataset (it holds ``windows/``) is read by ``from-olmoearth``
    first; when its first completed ``output`` layer holds class ids, the
    refusal says how to get probabilities. A directory of class-id GeoTIFFs
    is refused the same way. Only a raster inside a read root is opened: a
    link out of the scores root is passed over, so none of its values reach
    the refusal.
    """
    roots = read_roots()

    def inside(path: Path) -> bool:
        """Whether ``path`` resolves inside a read root."""
        return _inside(os.path.realpath(path), roots)

    if (run_dir / "windows").is_dir():
        first = next(run_dir.glob("windows/*/*/layers/output/*/geotiff.tif"), None)
        reason = None
        if first is not None and inside(first):
            reason = _class_ids(rasterio, np, first)
        if reason is not None:
            return class_id_refusal(
                f"the rslearn dataset {run_dir.name!r} (its output layer)", reason, fix
            )
        return ValueError(
            f"run_dir {run_dir.name!r} is an rslearn dataset (it holds windows/), "
            "not a run's scores: read it first with oe-inferencex from-olmoearth "
            f"{run_dir.name} --out RUN, under the scores root, and pass RUN. "
            "from-olmoearth reads an output layer of probabilities and refuses "
            "class ids, saying how to get probabilities"
        )
    for tif in sorted(run_dir.glob("*.tif")):
        reason = _class_ids(rasterio, np, tif) if inside(tif) else None
        if reason is not None:
            return class_id_refusal(f"{tif.name} in {run_dir.name!r}", reason, fix)
    return ValueError(
        f"run_dir {run_dir.name!r} holds neither {MANIFEST} (a direct run, e.g. "
        f"score_area.py) nor {SUMMARY} (oe-inferencex from-olmoearth DATASET "
        "--out RUN); name the directory that holds one"
    )


def _summary_grid(summary: dict[str, Any], named: str | None) -> dict[str, Any]:
    """The grid of a ``from-olmoearth`` summary whose raster this read takes.

    One grid, or the one ``named`` (``scores_<label>.tif``); a run of several
    CRSs is refused without a name, listing its rasters. A per-window
    classification (``window_probs.npy``) has no grid to pool and is refused.
    """
    grids = [g for g in summary.get("grids") or [] if isinstance(g, dict)]
    if not grids:
        if str(summary.get("kind")) == "per-window probabilities":
            raise ValueError(
                "the run is a per-window classification (window_probs.npy, one "
                "column per window): its columns are not neighbours on a grid, "
                "which this tool pools into windows; rank it with oe-inferencex "
                "assess window_probs.npy --patch 1"
            )
        raise ValueError(f"the run's {SUMMARY} lists no scores raster (grids)")
    names = [f"scores_{g.get('label')}.tif" for g in grids]
    if named is not None:
        if named not in names:
            raise ValueError(
                f"{named} is not a raster the run's {SUMMARY} lists "
                f"({', '.join(names)})"
            )
        return grids[names.index(named)]
    if len(grids) > 1:
        raise ValueError(
            f"the run wrote one scores raster per CRS ({', '.join(names)}); pass "
            "run_dir as the path to one of them"
        )
    return grids[0]


def _read_record(path: Path, what: str, again: str) -> Any:
    """A run's JSON record, or a refusal that names it and how to write it again."""
    try:
        return load_json_file(str(path), what=what)
    except json.JSONDecodeError:
        raise ValueError(f"{what} is not valid JSON; {again}") from None


def read_run_files(
    run_dir: Path, named: str | None, rasterio: Any, np: Any, fix: str
) -> RunFiles:
    """The run directory's record and the raster it names, or a refusal.

    score_area.py's ``manifest.json`` first; else the summary ``from-olmoearth``
    writes (``olmoearth_output.json``), whose raster ``scores_<label>.tif``
    is looked up by name inside ``run_dir`` (the summary's paths are where
    the command ran) and refused when it holds class ids; else a refusal
    that says what the directory is (:func:`_no_record`). ``fix`` is the
    package's text on how to write probabilities.

    Parameters
    ----------
    run_dir
        The run's directory, inside a read root (:func:`resolve_run`).
    named
        A raster file the caller named, or ``None``.
    rasterio, np
        The modules, imported by the handler.
    fix
        ``oe_inferencex.readers.olmoearth.FIX_PROBABILITIES``.

    Returns
    -------
    RunFiles
        The record, the raster and what the result states of them.

    Examples
    --------
    >>> read_run_files(Path("run"), None, rasterio, np, fix)  # doctest: +SKIP
    """
    if (run_dir / MANIFEST).is_file():
        manifest = _read_record(
            run_dir / MANIFEST,
            "the run's manifest.json",
            "write it again with the run (score_area.py)",
        )
        if not isinstance(manifest, dict):
            raise ValueError("the run's manifest.json is not a JSON object")
        raster = _raster_path(run_dir, manifest)
        if named is not None and named != raster.name:
            raise ValueError(
                f"run_dir names {named}; the run's manifest.json names {raster.name}"
            )
        files: dict[str, Any] = {"record": MANIFEST, "raster": raster.name}
        return RunFiles("manifest", manifest, raster, files)
    if not (run_dir / SUMMARY).is_file():
        raise _no_record(run_dir, rasterio, np, fix)
    summary = _read_record(
        run_dir / SUMMARY,
        f"the run's {SUMMARY}",
        "run oe-inferencex from-olmoearth DATASET --out RUN again",
    )
    if not isinstance(summary, dict):
        raise ValueError(f"the run's {SUMMARY} is not a JSON object")
    grid = _summary_grid(summary, named)
    name = f"scores_{grid.get('label')}.tif"
    if name != os.path.basename(name) or "\\" in name:
        raise ValueError(f"the run's {SUMMARY} names a raster outside its directory")
    real = os.path.realpath(run_dir / name)
    if not _inside(real, read_roots()) or not os.path.isfile(real):
        raise ValueError(
            f"the run directory has no raster {name!r}, which its {SUMMARY} lists"
        )
    raster = Path(real)
    reason = _class_ids(rasterio, np, raster)
    if reason is not None:
        raise class_id_refusal(name, reason, fix)
    manifest = {"scores": {"file": name, "values": "probabilities"}}
    shape = grid.get("shape")
    skipped = summary.get("windows_skipped")
    files = {
        "record": SUMMARY,
        "written_by": "oe-inferencex from-olmoearth",
        "raster": name,
        "rasters": [f"scores_{g.get('label')}.tif" for g in summary.get("grids") or []],
        "output": summary.get("kind"),
        "layer": summary.get("layer"),
        "windows_read": summary.get("windows_read"),
        "windows_skipped": len(skipped) if isinstance(skipped, list) else None,
        "notes": [str(n) for n in summary.get("notes") or []],
    }
    return RunFiles(
        "from-olmoearth",
        manifest,
        raster,
        files,
        shape_hw=[int(v) for v in shape] if isinstance(shape, list) else None,
        bands=int(summary["bands"]) if isinstance(summary.get("bands"), int) else None,
    )


async def _scores_from_file(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_scores_from_file``."""
    try:
        # The feature-folder paths of olmoearth-inferencex >= 1.19.0. The
        # 1.18.1 path `assess` still resolves through the package's alias
        # finder, and `cli.read_raster` through the CLI's own import of
        # `common.raster`; neither is the module's own.
        assess = inferencex.load("review.assess")
        raster_io = inferencex.load("common.raster")
    except inferencex.InferencexMissingError as exc:
        return _missing(f"olmoearth-inferencex: {exc}")
    try:
        rasterio = importlib.import_module("rasterio")
        np = importlib.import_module("numpy")
        # The package's reader of an OlmoEarth run, for its words on how to
        # write probabilities; it imports rasterio only when it reads.
        reader = inferencex.load("readers.olmoearth")
    except ImportError as exc:
        return _missing(f"rasterio: {exc}")

    patch = int(args.get("patch", DEFAULT_PATCH))
    if patch < 1:
        raise ValueError(f"patch must be a positive number of pixels, got {patch}")
    include_fill = bool(args.get("include_fill_class", False))
    run_dir, named = resolve_run(str(args["run_dir"]))
    run = read_run_files(run_dir, named, rasterio, np, str(reader.FIX_PROBABILITIES))
    manifest, raster = run.manifest, run.raster
    kind = _values_kind(manifest)
    is_logit = kind == "logits"

    digest = _sha256(raster)
    recorded = (manifest.get("scores") or {}).get("sha256")
    if recorded and str(recorded).lower() != digest:
        raise ValueError(
            "the raster's sha256 is not the one its manifest records: the manifest "
            "does not describe this file, so its classes and values cannot be trusted"
        )
    _check_size(rasterio, raster, patch)
    array, valid, geo = read_scores_raster(raster_io, raster)
    shape = list((manifest.get("scores") or {}).get("shape") or [])
    if shape and shape != list(array.shape) and shape[1:] != list(array.shape):
        raise ValueError(
            f"the manifest records shape {shape}; the raster is {list(array.shape)}"
        )
    bands = 1 if array.ndim == 2 else int(array.shape[0])
    shape_hw = [int(s) for s in array.shape[-2:]]
    off_grid = bool(run.shape_hw) and run.shape_hw != shape_hw
    if off_grid or (run.bands is not None and run.bands != bands):
        raise ValueError(
            f"the run's {SUMMARY} records {run.bands} band(s) of {run.shape_hw} "
            f"pixels; {raster.name} holds {bands} of {shape_hw}"
        )

    fill: dict[str, Any] | None = None
    if array.ndim == 2:
        names = (
            _class_names(manifest, 2) if len(manifest.get("classes") or {}) == 2 else {}
        )
        kept = [0, 1]
        class_scores = array
        labels = {i: names.get(i, f"class_{i}") for i in kept}
    else:
        names = _class_names(manifest, int(array.shape[0]))
        marked = fill_channels(manifest, names)
        kept = [c for c in range(array.shape[0]) if include_fill or c not in marked]
        if marked:
            channel = marked[0]
            argmax_all = np.argmax(np.where(valid[None], array, -np.inf), axis=0)
            fill = {
                "channel": channel,
                "name": names[channel],
                "excluded": not include_fill,
                "note": str(manifest.get("class_note") or "the label fill value"),
                "valid_pixels_where_it_is_the_argmax": int(
                    ((argmax_all == channel) & valid).sum()
                ),
                "effect": (
                    "left out of the margin and the class vote: a window is "
                    "ranked on its real classes"
                    if not include_fill
                    else "kept on request: the ranking is the served map's, "
                    "over every channel"
                ),
            }
        if len(kept) < 2:
            raise ValueError(
                "fewer than two class channels remain once the label-fill channel "
                "is left out; a margin needs two"
            )
        class_scores = array[kept]
        labels = {pos: names[c] for pos, c in enumerate(kept)}

    out = assess.assess_prediction(
        class_scores, is_logit=is_logit, patch=patch, nodata_mask=~valid
    )
    arrays = out["arrays"]
    conf, klass, valid_w = (
        arrays["confidence"],
        arrays["pooled_argmax"],
        arrays["valid"],
    )
    rows_n, cols_n = (int(s) for s in conf.shape)
    idx = np.flatnonzero(valid_w.ravel())
    p1_px = np.where(valid, _top1(np, class_scores, is_logit), np.nan)
    p1_w = _pool_mean(np, p1_px, patch, bool(valid.all()))

    width = len(labels)
    margin_v = conf.ravel()[idx]
    class_v = klass.ravel()[idx]
    if idx.size == 0 or (class_v < 0).any():
        raise ValueError("the map has no valid window with a class to rank")
    scores: list[list[float]] = []
    for m, k in zip(margin_v.tolist(), class_v.tolist()):
        row = [0.0] * width
        row[int(k)] = float(m)
        scores.append(row)
    n_valid = int(idx.size)
    signal = (
        "the package's window confidence, the mean over the window's valid pixels of "
        + (_PIXEL_CONFIDENCE[(is_logit, array.ndim == 3)])
    )
    if fill and not include_fill:
        signal += ", label-fill channel left out"
    model = _model_block(manifest)
    stem = slug(str(model.get("repo") or model.get("id") or "run").split("/")[-1], 32)
    if run.kind == "from-olmoearth":
        stem = slug(f"olmoearth-{raster.stem}", 32)
    tag = "_fill" if fill and include_fill else ""
    package_warnings = [str(w) for w in out.get("warnings", [])]
    payload: dict[str, Any] = {
        "format": "olmoearth-agent/scores@1",
        "provenance": PROVENANCE,
        "score_kind": SCORE_KIND,
        "score_type": "logit" if is_logit else "probability",
        "values": kind,
        "signal": signal + " (lower = more suspect)",
        "row_encoding": ROW_ENCODING,
        "grid": [rows_n, cols_n],
        "patch_px": patch,
        "scores": scores,
        "p1": [float(v) for v in p1_w.ravel()[idx].tolist()],
        "map_class": [int(k) for k in class_v.tolist()],
        "classes": {str(pos): name for pos, name in labels.items()},
        "class_channels": kept,
        "fill_class": fill,
        "model": model,
        # The raster's path too: a sample a certificate rests on is drawn from
        # the raster itself (olmoearth_plan_label_sample with seed_from_map).
        "raster": {
            "file": raster.name,
            "path": str(raster),
            "sha256": digest,
            "shape": list(array.shape),
        },
        "run_files": run.files,
        "date_window": _date_window(manifest),
        "package": {
            "name": inferencex.DISTRIBUTION,
            "version": inferencex.version(),
            "call": f"assess_prediction(is_logit={is_logit}, patch={patch})",
        },
        # Carried in the file so every ranking of it states them (exp86 round 6:
        # the provider's multi-class warning never reached a brief-8 answer).
        "package_warnings": package_warnings,
    }
    if n_valid < rows_n * cols_n:
        payload["windows"] = [int(i) for i in idx.tolist()]
    centres = _centres(np, geo, idx, cols_n, patch)
    if centres is not None:
        payload["centres_lon_lat"] = centres
    name = f"scores_run_{stem}_{digest[:8]}_p{patch}{tag}.json"
    scores_path = write_json_file(name, payload)
    identity = model_and_raster(model, digest, name, RECORDS[run.kind])

    result: dict[str, Any] = {
        "available": True,
        "scores_path": scores_path,
        "scores_file": name,
        "run_dir": run_dir.name,
        "run_files": run.files,
        "provenance": PROVENANCE,
        "values": kind,
        "grid": [rows_n, cols_n],
        "patch_px": patch,
        "n_windows": rows_n * cols_n,
        "n_valid": n_valid,
        "n_nodata": rows_n * cols_n - n_valid,
        "n_nodata_pixels": int((~valid).sum()),
        "pixels_outside_window_grid": int(out.get("pixels_outside_window_grid", 0)),
        "classes": payload["classes"],
        "fill_class": fill,
        "model": model,
        "date_window": _date_window(manifest),
        "raster_check": {
            "sha256": digest,
            "matches_manifest": True if recorded else None,
        },
        "model_and_raster": identity,
        "signal": payload["signal"],
        "package": payload["package"],
        "package_warnings": package_warnings,
        "facts": [
            {
                "id": "model_and_raster",
                "sentence": identity,
                "model": model.get("repo") or model.get("id"),
                "revision": model.get("revision"),
                "raster_sha256": digest,
            }
        ],
        "next_step": (
            "Pass scores_path to olmoearth_review_set (which windows to check "
            "first), to olmoearth_plan_label_sample (how wrong the map is; "
            "design='random' with seed_from_map for a certified zone that can "
            "be packed into a certificate), or, with a second run's file, to "
            "olmoearth_compare_review."
        ),
    }
    return result


def build_scores_file_tools() -> list[RegisteredTool]:
    """Return the scores-provider tool (skill #18; the optional ``inferencex`` extra)."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_scores_from_file",
                description=(
                    "Read a map from a DIRECT model run (a scores raster + "
                    "manifest.json, e.g. score_area.py on a GPU cluster, or "
                    "the folder oe-inferencex from-olmoearth writes: "
                    "olmoearth_output.json + scores_<EPSG>.tif, one .tif "
                    "named when there are several), not "
                    "a Studio result: pools every pixel's logits or "
                    "probabilities to windows as olmoearth-inferencex does, "
                    "leaves out a label-fill class the manifest marks, and "
                    "writes a scores file; class ids are refused with how to "
                    "get probabilities. Pass the returned scores_path to "
                    "olmoearth_review_set, olmoearth_plan_label_sample or "
                    "olmoearth_compare_review. Returns the grid, valid and "
                    "no-data windows, classes, model and revision; no "
                    "coordinates. Needs the inferencex extra."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "run_dir": {
                            "type": "string",
                            "description": "The run's directory under "
                            "OLMOEARTH_SCORES_ROOT.",
                        },
                        "patch": {
                            "type": "integer",
                            "default": DEFAULT_PATCH,
                            "description": "Window side in pixels.",
                        },
                        "include_fill_class": {
                            "type": "boolean",
                            "default": False,
                            "description": "Keep the label-fill channel (the "
                            "served map's argmax).",
                        },
                    },
                    "required": ["run_dir"],
                },
            ),
            handler=_scores_from_file,
            capability=Capability(
                does="read a direct model run already made (scores raster and "
                "manifest.json, or from-olmoearth's files) into a scores file; "
                "it runs no model",
                needs=("run_dir under the scores root",),
            ),
        ),
    ]
