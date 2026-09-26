# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The scores provider: a model run's full output as review-set input (skill #18).

Studio's API returns map tiles and a point lookup, never per-class scores. A
model run outside Studio (for instance ``scripts/score_area.py`` of the
companion repository on a GPU cluster) writes what Studio does not: a
georeferenced ``(C, H, W)`` float32 raster of logits or probabilities, NaN
where there is no prediction, and a ``manifest.json`` that names the model,
its revision, the classes and whether the values are logits.

``olmoearth_scores_from_file`` reads such a directory under
``OLMOEARTH_SCORES_ROOT``, pools the raster to windows exactly as the
package's ``oe-inferencex assess`` does (``oe_inferencex.assess.
assess_prediction`` with ``is_logit`` from the manifest and the no-data mask of
the package's own GeoTIFF reader), and writes one scores file that
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
with the install line. No coordinates in the result (rule §3.1): windows are a
grid ``(row, col)`` and index; each valid window's location stays in the file,
for the labelling sheet.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import warnings
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

_INSTALL_REASON = (
    "Reading a model run's scores raster needs the optional package "
    "olmoearth-inferencex (>= 1.3.0) with rasterio (its geo extra), which is "
    "not installed in this environment. Nothing was read; do not rank the "
    "map's windows by hand."
)


def _missing(detail: str) -> dict[str, Any]:
    """The answer when the extra (or rasterio under it) is not importable."""
    return {
        "available": False,
        "reason": _INSTALL_REASON,
        "missing": detail,
        "install": inferencex.INSTALL_HINT,
    }


def _inside(real: str, roots: list[str]) -> bool:
    """Whether a resolved path sits inside one of ``roots``."""
    return any(os.path.commonpath([root, real]) == root for root in roots)


def resolve_run_dir(raw: str) -> Path:
    """The run directory a model-chosen ``run_dir`` names, or raise.

    A relative path is taken under the scores root (``OLMOEARTH_SCORES_ROOT``,
    else the workspace root); the resolved directory must sit inside a read
    root (the rule scores files already follow). A path to the directory's
    manifest or raster names the directory.
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
    if os.path.isfile(real) and (
        os.path.basename(real) == MANIFEST or real.endswith((".tif", ".tiff"))
    ):
        real = os.path.dirname(real)
    if not os.path.isdir(real):
        raise ValueError(f"run_dir {text!r} is not a directory under the scores root")
    return Path(real)


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


def model_and_raster(model: dict[str, Any], digest: str, scores_file: str) -> str:
    """One sentence that keeps the model's revision and the raster's hash apart.

    exp86 round 7 (brief 4, cluster) gave the raster's sha256 prefix
    ``a7c40be9``, which is also in the scores file's name, as the model's
    revision ``a347b15``.
    """
    name = model.get("repo") or model.get("id") or "(unnamed)"
    revision = model.get("revision")
    at = (
        f"at revision {str(revision)[:7]}"
        if isinstance(revision, str) and revision
        else "at a revision the manifest does not record"
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
    """Refuse a raster too large to read, or with more windows than a ranking takes."""
    with rasterio.open(path) as src:
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


async def _scores_from_file(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_scores_from_file``."""
    try:
        assess = inferencex.load("assess")
        cli = inferencex.load("cli")
    except inferencex.InferencexMissingError as exc:
        return _missing(f"olmoearth-inferencex: {exc}")
    try:
        rasterio = importlib.import_module("rasterio")
        np = importlib.import_module("numpy")
    except ImportError as exc:
        return _missing(f"rasterio: {exc}")

    patch = int(args.get("patch", DEFAULT_PATCH))
    if patch < 1:
        raise ValueError(f"patch must be a positive number of pixels, got {patch}")
    include_fill = bool(args.get("include_fill_class", False))
    run_dir = resolve_run_dir(str(args["run_dir"]))
    manifest = load_json_file(str(run_dir / MANIFEST), what="the run's manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("the run's manifest.json is not a JSON object")
    raster = _raster_path(run_dir, manifest)
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
    # The package's own GeoTIFF reader: a pixel is valid when every channel is
    # finite and none equals the raster's no-data value.
    array, valid, geo = cli.read_raster(str(raster))
    shape = list((manifest.get("scores") or {}).get("shape") or [])
    if shape and shape != list(array.shape) and shape[1:] != list(array.shape):
        raise ValueError(
            f"the manifest records shape {shape}; the raster is {list(array.shape)}"
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
        "raster": {"file": raster.name, "sha256": digest, "shape": list(array.shape)},
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
    identity = model_and_raster(model, digest, name)

    result: dict[str, Any] = {
        "available": True,
        "scores_path": scores_path,
        "scores_file": name,
        "run_dir": run_dir.name,
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
            "first), to olmoearth_plan_label_sample (how wrong the map is), or, "
            "with a second run's file, to olmoearth_compare_review."
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
                    "manifest.json, e.g. score_area.py on a GPU cluster), not "
                    "a Studio result: pools every pixel's logits or "
                    "probabilities to windows as olmoearth-inferencex does, "
                    "leaves out a label-fill class the manifest marks, and "
                    "writes a scores file. Pass the returned scores_path to "
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
                "manifest.json) into a scores file; it runs no model",
                needs=("run_dir under the scores root",),
            ),
        ),
    ]
