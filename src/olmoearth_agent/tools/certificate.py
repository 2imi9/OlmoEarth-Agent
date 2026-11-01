# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""A certified zone as a certificate anyone holding the map can check (skill #5).

olmoearth-inferencex packs a certified zone into a bundle that travels with the
map (``oe_inferencex.certify.certificate``, from 1.17.0; the agent requires
1.20.3 or later, :data:`inferencex.MIN_VERSION`, and refuses an older release
installed): ``certificate.json``, a STAC Item whose ``zc:`` properties state
the guarantee; the zone as a GeoTIFF mask on the map's window grid; copies of
the labelled sample, its sidecar and the zone JSON; and digests that bind it
to the map's decoded pixels. Its ``verify`` builds the certificate again from
the map, the bundled labels and the recorded settings, and compares it.

From 1.20.3 the zone mask may carry GDAL's ModelTransformation matrix, so a
map that is south up, rotated or has no georeferencing gets a certificate
(1.17.0 to 1.20.2 refuse that mask), and ``verify`` reads a mask's
georeferencing as written: a negative PixelScale y step (which passed through
1.20.2 while libgeotiff mirrors the zone), a perspective term, a second
tiepoint or a tag given twice fails it.

The route is file-based. ``certificate.build`` reads the zone JSON that
``oe-inferencex certify SAMPLE.csv`` writes, which needs the sheet and sidecar
that ``oe-inferencex sample ... --design random --seed-from-map --alpha A``
writes; the agent's estimation tools compute on arrays and write neither. So:

- ``olmoearth_plan_label_sample`` with ``design='random'`` and
  ``seed_from_map`` draws that sample through the command line, in a working
  folder under ``<scores root>/certificates/``, and checks it is the agent's
  design, window by window (``estimation._map_seeded_draw``);
- ``olmoearth_write_certificate`` writes the reviewer's labels (as
  ``olmoearth_certify_zone`` takes them) into the package's sheet, which
  refuses labels other than the ones first graded on the plan
  (``estimation.fill_sheet``), runs ``oe-inferencex certify`` on it
  (:func:`inferencex.run_cli_async`: this interpreter, no shell, a time
  limit, a failure refused), checks that the package's certify and
  ``olmoearth_certify_zone``'s computation agree on these labels (the same
  windows), and calls ``certificate.build``, which verifies the bundle
  before it hands it over. It writes no certificate when no zone is
  certified (the labels are in the sheet and certify's zone file beside it
  all the same), and none over one already written for the plan;
- ``olmoearth_verify_certificate`` calls ``certificate.verify`` against a copy
  of the map, and ``certificate.point`` for a window given by its
  ``[row, col]`` on the certificate's window grid, never by coordinates
  (rule §3.1).

A zone read from a Studio result's own files (``olmoearth_scores_from_studio``)
is the probability raster the agent decoded from Studio's native file, and
that raster is the map a certificate on it binds: the result names it
(``map_bound``), states as a fact that Studio's native GeoTIFF is not that
map and does not verify, and tells the holder to check against the decoded
raster; a failed check against a copy, of a certificate that names such a
raster, says that Studio's native file fails without any tampering.

A certificate states the zone ``certify`` gives: a re-checkable record, not a
new test. Every result states the package's ``ISSUED_NOTE`` (what delta
counts) and ``ISSUERS_WORD`` (what is taken on the issuer's word: the labels,
the map's date, that the sample was drawn once). Every file stays under the
scores root; a certificate that names a map outside it, or by a relative
path, is checked only against a copy under it, given as a plain GeoTIFF or
``.npy`` (GDAL would follow the paths inside a VRT). The tools need nothing
beyond the ``inferencex`` extra (rasterio comes with it), so they are default
tools.

The package's calls run in a worker thread (``asyncio.to_thread``), as
:func:`inferencex.run_cli_async` runs its command line: ``certificate.build``
and ``certificate.verify`` read and digest the map and draw the sample and
certify again, ``certificate.point`` reads the certificate (the zone is the
one verify built again), and ``certify_zone`` orders every window of the
map; run inline, each would hold the event loop that serves the web
bridge's other sessions for the whole call. An exception raised in the
thread reaches the handler as it would inline, so every refusal is the
same. Since another session's call can run meanwhile, ``certificate.build``
runs under a lock that first checks no certificate was written for the plan
(:func:`_build`): the package's build replaces a bundle, and a certificate
is written once.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import stat
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools import inferencex, scores_file
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.estimation import (
    _DESIGN_PATH,
    _LABELS_PATH,
    _WRONG,
    CERT_BUNDLE,
    CERT_SAMPLE,
    _jsonable,
    _labels,
    _load_design,
    _margin,
    _whole_map_estimate,
    fill_sheet,
    labels_sha256,
    plan_work_dir,
)
from olmoearth_agent.tools.registry import Capability, RegisteredTool, ToolContext
from olmoearth_agent.tools.review_set import (
    file_studio_kind,
    load_scores_file,
    read_roots,
    write_root,
)

#: The bundle's folder inside a plan's working folder.
BUNDLE_DIR = CERT_BUNDLE

#: The zone JSON ``oe-inferencex certify`` writes beside the package's sheet,
#: and its zone mask (the window grid, ``True`` in the zone).
ZONE_JSON = "sample_zone.json"
ZONE_MASK = "sample_zone.npy"

#: Said when map_path is not a plain GeoTIFF or ``.npy`` file.
NOT_PLAIN = (
    "map_path must be the map as a plain GeoTIFF or .npy file (or the scores "
    "file of olmoearth_scores_from_file that names it)"
)

#: What a certificate proves, stated with every one written or checked.
WHAT_IT_PROVES = (
    "Anyone holding the map can build the certificate again from the map, the "
    "bundled labels and the recorded settings, and compare it "
    "(olmoearth-inferencex's verify: the seed taken from the map, the windows "
    "drawn again, certify run again, the statement and the zone mask). It "
    "shows that the sample and certify's settings were fixed before the "
    "labels. It is the zone certify gives, not a new test, and it does not "
    "show that the labels are right or that the sample was drawn only once."
)

#: The refusal of a design not drawn for a certificate.
NOT_FROM_MAP = (
    "design_path was not drawn for a certificate: a certificate needs a random "
    "sample whose seed anyone holding the map can derive again, with alpha, "
    "delta and the rule fixed before any label. Plan one with "
    "olmoearth_plan_label_sample(design='random', seed_from_map=true, alpha=A) "
    "on a direct run's scores file, and label its windows"
)

#: Said of a written certificate (the contract's ``must_state``).
WHOLE_ZONE = (
    "It certifies the zone's windows taken together, not each pixel; outside "
    "the zone nothing is certified."
)
ISSUERS_LABELS = (
    "The labels, and that the sample was drawn once, are the issuer's word, "
    "which the certificate does not show."
)

#: Said when a check against a copy fails and the certificate names a raster
#: decoded from Studio's files (the contract's ``must_state``).
STUDIO_COPY_NOTE = (
    "This certificate binds a raster decoded from a Studio file; a check "
    "against Studio's native file fails without any tampering."
)


def _studio_map(design: dict[str, Any], raster: str) -> dict[str, Any] | None:
    """What a certificate on a zone read from Studio's own files binds, or ``None``.

    ``olmoearth_scores_from_studio`` writes each zone as ``scores.tif``, a
    float32 probability raster the agent decoded from Studio's native file
    (a class map's uint8 top-k bands by the files README's formula, or a
    ``[0, 1]`` score band). That raster is the map the plan drew its seed
    from and the certificate binds (``map_sha256``), not Studio's file: a
    holder of Studio's native GeoTIFF does not hold the certified map. A
    design pooled from such a zone keeps the manifest's ``studio`` block, or
    Studio's provenance, in its population's source
    (``review_set.file_studio_kind``); the native file's name, its sha256
    and the result id's first 8 characters come from the manifest beside the
    raster, when it is there. ``None`` for a direct run's map.
    """
    population = design.get("population")
    found = population.get("source") if isinstance(population, dict) else None
    source: dict[str, Any] = found if isinstance(found, dict) else {}
    if file_studio_kind(source) is None:
        return None
    block = source.get("studio")
    studio: dict[str, Any] = block if isinstance(block, dict) else {}
    decode = studio.get("decode") or studio.get("assumption")
    out: dict[str, Any] = {
        "map_file": raster,
        "decode": decode if isinstance(decode, str) else None,
        "studio_file": None,
        "studio_file_sha256": None,
        "result_id_prefix": None,
    }
    try:
        text = (Path(raster).parent / scores_file.MANIFEST).read_text(encoding="utf-8")
        manifest = json.loads(text)
    except (OSError, ValueError):
        manifest = None
    native = scores_file.studio_source(manifest) if isinstance(manifest, dict) else None
    if native is not None:
        for key, kept in (("file", "studio_file"), ("sha256", "studio_file_sha256")):
            if isinstance(native.get(key), str):
                out[kept] = native[key]
        if isinstance(native.get("result_id"), str):
            out["result_id_prefix"] = native["result_id"][:8]
    return out


def _decoded_from_studio(named: str) -> bool:
    """Whether a map a certificate names is a zone written from Studio's files.

    Read from the manifest beside it, as ``olmoearth_scores_from_studio``
    writes it (``source.kind`` ``studio_files``); a path that is not a plain
    local one, or has no readable manifest, is not.
    """
    if named.startswith("/vsi") or "://" in named or not os.path.isabs(named):
        return False
    try:
        text = (Path(named).parent / scores_file.MANIFEST).read_text(encoding="utf-8")
        manifest = json.loads(text)
    except (OSError, ValueError):
        return False
    return (
        isinstance(manifest, dict) and scores_file.studio_source(manifest) is not None
    )


def _studio_map_fact(bound: dict[str, Any]) -> dict[str, Any]:
    """The fact that the certificate binds the decoded raster, not Studio's file."""
    native = bound.get("studio_file") or "native file"
    result = bound.get("result_id_prefix")
    of = f"Studio result {result}'s {native}" if result else f"Studio's {native}"
    how = f" ({bound['decode']})" if bound.get("decode") else ""
    sentence = (
        f"The map this certificate binds is the agent's decoded raster "
        f"{bound['map_file']}{how}, written from {of}; Studio's native GeoTIFF "
        "is not that map and does not verify. Checking needs this raster, or "
        "the one olmoearth_scores_from_studio writes again from the same "
        "result, decoded the same way."
    )
    return rules.fact(
        "certificate_map_is_decoded",
        sentence,
        map_file=bound["map_file"],
        studio_file=bound.get("studio_file"),
        studio_file_sha256=bound.get("studio_file_sha256"),
    )


def _under_roots(raw: Any, what: str) -> Path:
    """``raw`` resolved inside a read root (a relative path under the scores root)."""
    text = str(raw or "").strip()
    if not text:
        raise ValueError(f"{what} is empty")
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = write_root() / candidate
    real = os.path.realpath(candidate)
    roots = read_roots()
    if not any(os.path.commonpath([r, real]) == r for r in roots):
        raise ValueError(f"{what} must be under {' or '.join(roots)}")
    return Path(real)


def _map_date(args: dict[str, Any], block: dict[str, Any]) -> str:
    """The date or period the certificate holds for: the caller's, else the run's.

    A ``score_area.py`` manifest records the run's date window; a
    ``from-olmoearth`` run records none, so its certificate needs ``date``.
    """
    if args.get("date"):
        return str(args["date"]).strip()
    window = block.get("date_window")
    if isinstance(window, dict) and window.get("start") and window.get("end"):
        return f"{window['start']}/{window['end']}"
    msg = (
        "pass date: the date or period the map describes (YYYY-MM-DD or "
        "YYYY-MM-DD/YYYY-MM-DD); the run's files record none"
    )
    raise ValueError(msg)


def _same_zone(np: Any, work: Path, mine: dict[str, Any]) -> bool:
    """Whether ``certify``'s zone mask holds the windows of the agent's zone.

    The counts can agree while the windows differ: the zone orders every
    window of the map by confidence, and the plan checked the two readings of
    the raster only at the sampled windows.
    """
    try:
        mask = np.load(work / ZONE_MASK, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"oe-inferencex certify left no zone mask: {exc}") from None
    theirs = {int(i) for i in np.flatnonzero(mask).tolist()}
    order = mine.get("zone_indices_in_order")
    ours = set() if order is None else {int(i) for i in order}
    return theirs == ours


def _agents_zone(
    estimate: Any,
    design: dict[str, Any],
    indices: list[int],
    wrong: list[int],
    plan: tuple[float, float, str],
) -> dict[str, Any]:
    """``olmoearth_certify_zone``'s computation on these labels (run off the loop).

    ``certify_zone`` orders every window of the map by confidence, so it
    runs in a worker thread, the stored margin read back with it.
    """
    alpha, delta, rule = plan
    return dict(
        estimate.certify_zone(
            _margin(design["population"]["margin"]),
            indices,
            wrong,
            alpha,
            delta=delta,
            rule=rule,
        )
    )


#: One lock per bundle folder, held while its certificate is built
#: (:func:`_build`): the check that none was written for the plan and the
#: package's build are one step. A build of another plan does not wait.
_BUILD_LOCKS: dict[Path, threading.Lock] = {}
_BUILD_LOCKS_GUARD = threading.Lock()


def _build_lock(bundle: Path) -> threading.Lock:
    """The lock of this bundle folder (the plan's, resolved by ``plan_work_dir``)."""
    with _BUILD_LOCKS_GUARD:
        return _BUILD_LOCKS.setdefault(bundle, threading.Lock())


def _already_written(bundle: Path) -> str:
    """The refusal of a second certificate on one plan's sample."""
    return (
        "a certificate on this sample is already written, at "
        f"{bundle / 'certificate.json'}; it is written once, with the labels "
        "first given, and never replaced. Check it with "
        "olmoearth_verify_certificate"
    )


def _build(cert: Any, zone_path: Path, bundle: Path, date: str) -> dict[str, Any]:
    """``certificate.build`` (run off the loop); a refusal of it as the tool's.

    The package verifies the bundle before it hands it over, reading and
    digesting the map, so this runs in a worker thread; what it raises is
    the refusal the tool gives. The handler checks that no certificate was
    written before it runs ``certify``, but another session's call on the
    same plan can pass that check while this one awaits, and the package's
    build replaces a bundle ("a bundle written again keeps nothing of the
    earlier one"). So the check is made again here, under the bundle's lock
    (:func:`_build_lock`) with the build: of two calls, the second is
    refused as the early check refuses it, and the first certificate stays.
    """
    with _build_lock(bundle):
        if (bundle / "certificate.json").exists():
            raise ValueError(_already_written(bundle))
        try:
            item: dict[str, Any] = cert.build(str(zone_path), str(bundle), date)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            reason = cert.printable(str(exc))
            raise ValueError(
                f"olmoearth-inferencex wrote no certificate: {reason}"
            ) from None
    return item


async def _write_certificate(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    """Handler for ``olmoearth_write_certificate``."""
    try:
        cert = inferencex.load("certify.certificate")
        estimate = inferencex.load("estimate")
        zone_module = inferencex.load("certify.zone")
        np = importlib.import_module("numpy")
    except (inferencex.InferencexMissingError, ImportError):
        return inferencex.missing_extra("Writing a certificate")
    design_path = str(args["design_path"])
    design = _load_design(design_path)
    block = design.get("certificate")
    if not isinstance(block, dict) or block.get("seed_rule") != "from_map":
        raise ValueError(NOT_FROM_MAP)
    work = plan_work_dir(block)
    sheet = work / CERT_SAMPLE
    if not sheet.is_file():
        raise ValueError(f"the plan's sheet {CERT_SAMPLE} is gone; plan again")
    bundle = work / BUNDLE_DIR
    if (bundle / "certificate.json").exists():
        # the fast refusal; _build checks again under its lock
        raise ValueError(_already_written(bundle))
    indices = [int(i) for i in design["sample"]["indices"]]
    wrong, _reference = _labels(args, indices)
    date = _map_date(args, block)  # asked before anything runs
    alpha, delta, rule = float(block["alpha"]), float(block["delta"]), block["rule"]
    # The labels first graded on this plan are fixed in its sheet: other ones
    # are refused here, before certify runs.
    fill_sheet(sheet, indices, wrong)
    # certify reads alpha, delta and the rule the sample fixed from its sidecar
    await inferencex.run_cli_async(["certify", str(sheet)], cwd=work)
    zone_path = work / ZONE_JSON
    zone = json.loads(zone_path.read_text(encoding="utf-8"))
    mine = await asyncio.to_thread(
        _agents_zone, estimate, design, indices, wrong, (alpha, delta, rule)
    )
    ours = (mine.get("n_zone"), mine.get("coverage"))
    theirs = (zone.get("n_zone"), zone.get("coverage"))
    # The zone mask is on the map's window grid: compared off the loop too.
    same = ours == theirs and (
        zone.get("coverage") is None
        or await asyncio.to_thread(_same_zone, np, work, mine)
    )
    if not same:
        msg = (
            "the package's certify on the plan's sheet gives another zone "
            f"({zone.get('n_zone')} windows) than olmoearth_certify_zone on these "
            f"labels ({mine.get('n_zone')} windows), or other windows: the scores "
            "file is not the package's reading of the raster; no certificate was "
            "written"
        )
        raise ValueError(msg)
    issued = str(zone_module.ISSUED_NOTE)
    # The whole-map estimate from the same labels, as olmoearth_certify_zone
    # gives it: the report states it, and tells two label sets apart by it.
    sampled = await asyncio.to_thread(
        estimate.estimate_error_rate, design["sample"], wrong
    )
    # Nothing certified: as olmoearth_certify_zone says it, the whole-map rate
    # is not the most confident part's, and the evidence on that part is the
    # smallest zone tested (agent trial 12, B6; the step 11 review found this
    # result without it).
    certified = zone.get("coverage") is not None
    levels = list(_jsonable(mine.get("levels") or []))
    # This result has no levels_tested: the package's note says why no level
    # was tested (the step 11 review).
    why_field = "note" if zone.get("note") else None
    whole = _whole_map_estimate(
        _jsonable(sampled),
        certify=True,
        uncertified=None if certified else levels,
        why_field=why_field,
    )
    out: dict[str, Any] = {
        "available": True,
        "design_path": design_path,
        "design": "random",
        "seed": block["seed"],
        "seed_rule": "from_map",
        "alpha": alpha,
        "delta": delta,
        "rule": rule,
        "n_labelled": zone.get("n_labelled"),
        "n_population": zone.get("n_population"),
        "labels_sha256": labels_sha256(wrong),
        "certified": certified,
    }
    if not out["certified"]:
        out.update(
            {
                "certificate_written": False,
                "reason": (
                    f"No zone is certified at alpha={alpha:g}, delta={delta:g} "
                    f"under rule '{rule}' on these labels, so there is nothing to "
                    "put in a certificate."
                ),
                "note": zone.get("note"),
                "labels_held_in": str(sheet),
            }
        )
        return rules.add_contract(
            out,
            facts=[whole],
            must_state=[
                "No zone was certified on these labels; no certificate was written."
            ],
            forbidden_claims=[
                rules.post_hoc_alpha(alpha),
                rules.whole_map_rate_for_a_zone(
                    alpha, levels[0] if levels else None, why_field
                ),
            ],
        )
    item = await asyncio.to_thread(_build, cert, zone_path, bundle, date)
    props = item["properties"]
    path = str(bundle / "certificate.json")
    out.update(
        {
            "certificate_written": True,
            "certificate_path": path,
            "bundle": sorted(p.name for p in bundle.iterdir() if p.is_file()),
            "date": date,
            "n_zone": zone.get("n_zone"),
            "coverage": zone.get("coverage"),
            "upper_bound": zone.get("upper_bound"),
            "map_sha256": props["zc:map"]["map_sha256"],
            "statement": cert.printable(props["zc:statement"]),
            "verified_before_written": True,
            "what_it_proves": WHAT_IT_PROVES,
            "issued_note": issued,
            "issuers_word": str(cert.ISSUERS_WORD),
            "next_steps": [
                "Anyone holding the map checks it: olmoearth_verify_certificate "
                "with certificate_path and map_path (their copy of the map), or "
                f"oe-inferencex verify {path} --scores <their copy of the map>.",
                "Report the zone as olmoearth_certify_zone states it, with "
                "issued_note; the certificate adds a record anyone can check, "
                "not a stronger guarantee.",
            ],
        }
    )
    sentence = (
        f"The certified zone ({out['n_zone']} windows, coverage "
        f"{out['coverage']:g}) is packed into a certificate at {path}, which "
        "olmoearth-inferencex verified before writing it."
    )
    facts = [
        whole,
        rules.fact(
            "certificate",
            sentence,
            certificate_path=path,
            seed=block["seed"],
            map_sha256=out["map_sha256"],
        ),
    ]
    # A zone read from Studio's own files: the map bound is the agent's
    # decoded raster, which a holder of Studio's native file does not hold.
    # Stated as a fact (must_state is full) and in the first next step.
    bound = _studio_map(design, str(block.get("raster") or ""))
    if bound is not None:
        out["map_bound"] = bound
        out["next_steps"][0] = (
            "Anyone holding a copy of that decoded raster (map_bound.map_file), "
            "not Studio's native file, checks it: olmoearth_verify_certificate "
            "with certificate_path and map_path (their copy of the raster), or "
            f"oe-inferencex verify {path} --scores <their copy of the raster>. "
            "Someone holding only the Studio result gets the raster with "
            "olmoearth_scores_from_studio on the same result."
        )
        # No forbidden claim: the contract's ids are fixed
        # (statistical_rules.FIXED_IDS), so the fact carries it.
        facts.append(_studio_map_fact(bound))
    return rules.add_contract(
        out,
        facts=facts,
        must_state=[
            f"The certificate holds for the map of {date}, as the issuer states, "
            "not for other dates or versions of the map.",
            WHOLE_ZONE,
            ISSUERS_LABELS,
        ],
        forbidden_claims=[rules.post_hoc_alpha(alpha)],
    )


def _certificate_path(raw: Any) -> Path:
    """``certificate.json`` under the scores root: the file, or its folder's."""
    real = _under_roots(raw, "certificate_path")
    if real.is_dir():
        real = real / "certificate.json"
    if real.suffix != ".json" or not real.is_file():
        raise ValueError("certificate_path must name a certificate.json or its folder")
    return real


def _map_copy(raw: Any, magic: Mapping[str, tuple[bytes, ...]]) -> Path:
    """The map a certificate is checked against: a raster, or a scores file's raster.

    A scores file from ``olmoearth_scores_from_file`` names the raster it was
    read from (``raster.path``); either way the file must sit under the scores
    root. ``verify`` reads a map given to it with GDAL as it is, sidecar files
    and the paths inside a VRT included, so the file must be a regular,
    plain GeoTIFF or ``.npy`` by its name and its first bytes (``magic``, the
    package's ``RASTER_MAGIC``), as the package checks a map a certificate
    names.
    """
    real = _under_roots(raw, "map_path")
    if real.suffix == ".json" and real.is_file():
        try:
            raster = load_scores_file(str(real)).meta.get("raster")
        except (KeyError, TypeError, ValueError):
            msg = (
                "map_path is a .json that is not a scores file from "
                "olmoearth_scores_from_file; pass the map's GeoTIFF or its "
                "scores file"
            )
            raise ValueError(msg) from None
        named = raster.get("path") if isinstance(raster, dict) else None
        if not named:
            raise ValueError("map_path is a scores file that names no raster")
        real = _under_roots(named, "the scores file's raster")
    if not real.is_file():
        raise ValueError("map_path must name the map's file")
    heads = magic.get(real.suffix.lower())
    if not heads or not stat.S_ISREG(os.lstat(real).st_mode):
        raise ValueError(NOT_PLAIN)
    with open(real, "rb") as fh:
        head = fh.read(8)
    if not any(head.startswith(m) for m in heads):
        raise ValueError(NOT_PLAIN)
    return real


def _same_file(a: Path, b: str) -> bool:
    """Whether ``a`` is the file ``b`` names (``b`` read as ``verify`` reads it)."""
    if b.startswith("/vsi") or "://" in b:
        return False
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _named_maps(path: Path) -> list[str]:
    """The map (and a product's band) the certificate names, as written."""
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
        grid = item["properties"]["zc:map"]
    except (OSError, ValueError, KeyError, TypeError):
        return []  # certificate.verify says why it cannot be read
    if not isinstance(grid, dict):
        return []
    return [str(p) for p in (grid.get("path"), grid.get("confidence_path")) if p]


def window_lookup(
    cert: Any, path: Path, checked: dict[str, Any], window: list[int]
) -> dict[str, Any]:
    """Whether a window of the certificate's grid is in the zone ``verify`` built again.

    ``window`` is ``[row, col]`` on the window grid; the package's
    ``certificate.point`` takes a pixel, the window's first, ``row * patch``
    and ``col * patch``. Nothing is looked up when the zone could not be built
    again; a window outside the map, or in no whole window, is said so.
    ``olmoearth_verify_certificate`` runs it in a worker thread (it reads the
    certificate; the zone mask is verify's, passed in).

    Parameters
    ----------
    cert
        ``oe_inferencex.certify.certificate``.
    path
        The certificate.
    checked
        ``certificate.verify``'s result.
    window
        ``[row, col]``.

    Returns
    -------
    dict
        ``window``, ``inside`` (``None`` when not looked up or not in the
        map) and ``where``, the package's words.

    Examples
    --------
    >>> window_lookup(cert, Path("certificate.json"), res, [2, 3])  # doctest: +SKIP
    """
    if len(window) != 2:
        raise ValueError("window is [row, col] on the certificate's window grid")
    row, col = (int(v) for v in window)
    if checked.get("mask") is None:
        return {
            "window": [row, col],
            "inside": None,
            "where": "not looked up: the zone could not be built again",
        }
    patch = (checked.get("certificate") or {}).get("properties", {}).get("zc:patch")
    if type(patch) is not int or patch < 1:
        msg = "not looked up: the certificate's window size is not a positive integer"
        return {"window": [row, col], "inside": None, "where": msg}
    try:
        found = cert.point(
            str(path), row=row * patch, col=col * patch, mask=checked["mask"]
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        where = f"not looked up: {cert.printable(str(exc))}"
        return {"window": [row, col], "inside": None, "where": where}
    where = str(found.get("where"))
    if not checked.get("verified"):
        where += " (in the zone certify gives again; the certificate is not verified)"
    return {"window": [row, col], "inside": found.get("inside"), "where": where}


async def _verify_certificate(
    args: dict[str, Any], _ctx: ToolContext
) -> dict[str, Any]:
    """Handler for ``olmoearth_verify_certificate``."""
    try:
        cert = inferencex.load("certify.certificate")
        zone_module = inferencex.load("certify.zone")
    except inferencex.InferencexMissingError:
        return inferencex.missing_extra("Checking a certificate")
    path = _certificate_path(args["certificate_path"])
    window = args.get("window")
    if window is not None and (not isinstance(window, list) or len(window) != 2):
        raise ValueError("window is [row, col] on the certificate's window grid")
    named_maps = _named_maps(path)
    map_path: Path | None = None
    if args.get("map_path"):
        map_path = _map_copy(args["map_path"], cert.RASTER_MAGIC)
    else:
        for named in named_maps:
            if named.startswith("/vsi") or "://" in named:
                continue  # verify reads plain local files only, and says so
            if not os.path.isabs(named):
                # verify opens it from the agent's working directory, not
                # the scores root this check would read it under
                msg = (
                    "the certificate names its map by a relative path, which "
                    "this agent does not read: pass map_path, your copy of the "
                    "map under the scores root"
                )
                raise ValueError(msg)
            try:
                _under_roots(named, "the map the certificate names")
            except ValueError:
                msg = (
                    "the certificate names a map outside the scores root, which "
                    "this agent does not read: pass map_path, your copy of the "
                    "map under the scores root"
                )
                raise ValueError(msg) from None
    # A map_path that is the very file the certificate names (a scores file
    # names the run's raster, which the certificate names too) is no copy:
    # the check then shows only that the certificate agrees with that file.
    own_copy = map_path is not None and not any(
        _same_file(map_path, named) for named in named_maps
    )
    # verify digests the map and draws the sample and certify again: in a
    # worker thread, and its refusal reaches this handler as it would inline.
    try:
        res = await asyncio.to_thread(
            cert.verify, str(path), None if map_path is None else str(map_path)
        )
    except cert.Unreadable as exc:
        raise ValueError(cert.printable(str(exc))) from None
    checks = {
        str(name): {"ok": c.get("ok"), "detail": cert.printable(str(c.get("detail")))}
        for name, c in res["checks"].items()
    }
    failed = [name for name, c in checks.items() if c["ok"] is not True]
    verified = bool(res["verified"])
    props = (res.get("certificate") or {}).get("properties") or {}
    seed = props.get("zc:seed")
    out: dict[str, Any] = {
        "available": True,
        "verified": verified,
        "certificate_path": str(path),
        "map_read": res.get("map_read"),
        "map_given": own_copy,
        "checks": checks,
        "failed_checks": failed,
        "statement": cert.printable(res["statement"]) if verified else None,
        "notes": [cert.printable(str(n)) for n in res.get("notes") or []],
        "design": props.get("zc:design"),
        "seed": seed if isinstance(seed, int) and not isinstance(seed, bool) else None,
        "alpha": props.get("zc:alpha"),
        "delta": props.get("zc:delta"),
        "rule": props.get("zc:rule"),
        "what_it_proves": WHAT_IT_PROVES,
        "issued_note": str(zone_module.ISSUED_NOTE),
    }
    if window is not None:
        out["window_lookup"] = await asyncio.to_thread(
            window_lookup, cert, path, res, list(window)
        )
    where = f"the map read ({out['map_read']})" if out["map_read"] else "the map"
    if verified:
        sentence = (
            f"The certificate at {path} verifies against {where}: built again "
            "from the map, its labels and its settings, it is the one written."
        )
        must = [
            "The certificate verifies against this map: built again from the "
            "map, its labels and settings, it is the one written.",
            WHOLE_ZONE,
        ]
    else:
        sentence = (
            f"The certificate at {path} does not verify against {where}: "
            f"{', '.join(failed)} failed or could not be checked."
        )
        must = [
            f"The certificate does not verify against this map ({', '.join(failed)}); "
            "its statement is not shown to hold."
        ]
    if map_path is None:
        must.append(
            "The map was read from the path the certificate names; pass map_path, "
            "your own copy, to check it is that map."
        )
    elif not own_copy:
        must.append(
            "map_path is the file the certificate names, so this shows only that "
            "they agree; pass your own copy to check it is that map."
        )
    elif not verified and any(_decoded_from_studio(n) for n in named_maps):
        # The certificate binds the raster decoded from a Studio zone's file:
        # Studio's own file, given as the copy, fails with nothing changed.
        must.append(STUDIO_COPY_NOTE)
    return rules.add_contract(
        out,
        facts=[
            rules.fact(
                "certificate_check",
                sentence,
                verified=verified,
                failed_checks=failed,
                certificate_path=str(path),
            )
        ],
        must_state=must,
    )


def build_certificate_tools() -> list[RegisteredTool]:
    """Return the certificate tools (skill #5; the optional ``inferencex`` extra)."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_write_certificate",
                description=(
                    "Pack a certified zone into a certificate anyone holding the "
                    "map can check (olmoearth-inferencex's: a STAC Item, a "
                    "GeoTIFF zone mask, the labels, digests binding it to the "
                    "map). Needs a plan from olmoearth_plan_label_sample with "
                    "seed_from_map and its labels; writes no certificate when no "
                    "zone is certified. Needs the inferencex extra."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "design_path": _DESIGN_PATH,
                        "wrong": _WRONG,
                        "labels_path": _LABELS_PATH,
                        "date": {
                            "type": "string",
                            "description": "The map's date, YYYY-MM-DD or "
                            "YYYY-MM-DD/YYYY-MM-DD; default a score_area.py "
                            "run's date window.",
                        },
                    },
                    "required": ["design_path"],
                },
            ),
            handler=_write_certificate,
            capability=Capability(
                does="pack a certified zone into a certificate anyone holding the "
                "map can check",
                needs=("a plan drawn with seed_from_map, and its labels",),
            ),
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_verify_certificate",
                description=(
                    "Check a certificate.json against the map: "
                    "olmoearth-inferencex builds it again from the map, its "
                    "labels and settings, and says which checks pass; with "
                    "window [row, col], whether that window is certified. "
                    "Needs the inferencex extra."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "certificate_path": {
                            "type": "string",
                            "description": "certificate.json or its folder, "
                            "under OLMOEARTH_SCORES_ROOT.",
                        },
                        "map_path": {
                            "type": "string",
                            "description": "Your copy of the map, or its scores "
                            "file.",
                        },
                        "window": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "[row, col] on the window grid.",
                        },
                    },
                    "required": ["certificate_path"],
                },
            ),
            handler=_verify_certificate,
            capability=Capability(
                does="check a certificate against the map, and whether a window "
                "is certified",
                needs=("certificate_path and map_path under the scores root",),
            ),
        ),
    ]
