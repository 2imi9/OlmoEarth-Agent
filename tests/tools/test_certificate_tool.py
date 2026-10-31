# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""A direct OlmoEarth run's certified zone as a certificate, written and checked.

The run is what ``oe-inferencex from-olmoearth`` writes from an rslearn
dataset (``oe_inferencex.readers.olmoearth.read_output``, run here on a
dataset this test writes with rasterio); one test's run is instead a
score_area.py raster that is south up, which olmoearth-inferencex certifies
from 1.20.3 on. ``olmoearth_plan_label_sample`` with
``seed_from_map`` draws through the package's command line (``sample
--seed-from-map``), the labels are fixed, ``olmoearth_certify_zone``
certifies, ``olmoearth_write_certificate`` packs the zone and
``olmoearth_verify_certificate`` checks it, as the package's own ``verify``
does from the command line. Every command runs for real, as a subprocess of
this interpreter; only the failure tests replace the subprocess call. The
package's calls run off the event loop: tests wrap them to record the thread
each ran on, and run two sessions' calls on one plan at once
(``asyncio.gather``) to show the labels and the certificate are fixed once.
"""

from __future__ import annotations

import asyncio
import csv
import json
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.provenance.log import labels_digest
from olmoearth_agent.tools import estimation, inferencex
from olmoearth_agent.tools.certificate import build_certificate_tools
from olmoearth_agent.tools.estimation import build_estimation_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.scores_file import build_scores_file_tools

try:
    import numpy as np
    import rasterio
    from oe_inferencex import cli
    from oe_inferencex import estimate as pkg_estimate
    from oe_inferencex.certify import certificate as pkg_certificate
    from oe_inferencex.certify import zone as pkg_zone
    from oe_inferencex.readers import olmoearth as oe_reader
    from rasterio.transform import Affine

    HAVE_EXTRA = True
except ImportError:  # pragma: no cover - the core install has neither
    HAVE_EXTRA = False

pytestmark = pytest.mark.skipif(
    not HAVE_EXTRA, reason="needs the inferencex extra (numpy, rasterio)"
)

X0, Y0 = 256_330.0, 9_718_100.0  # UTM 37S
BUDGET = 60
ALPHA = 0.2
#: certify's options as `sample --seed-from-map --alpha 0.2` fixes them.
PLAN = {"alpha": ALPHA, "delta": 0.1, "rule": "prefix", "level_cut": "standard"}
_BANNED_KEYS = {"lon", "lat", "bbox", "bounds", "geometry", "coordinates", "crs"}


def _ctx() -> ToolContext:
    return ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_all(build_scores_file_tools())
    registry.register_all(build_estimation_tools())
    registry.register_all(build_certificate_tools())
    return registry


async def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return await _registry().dispatch(
        ToolCall(id="1", name=name, arguments=args), _ctx()
    )


async def _ok(name: str, args: dict[str, Any]) -> dict[str, Any]:
    out = await _call(name, args)
    assert out["ok"] is True, out
    return out["result"]


@pytest.fixture(autouse=True)
def _scores_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "scores_root"
    root.mkdir()
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(root))
    return root


def _softmax(seed: int, h: int, w: int) -> Any:
    """Three bands of float32 class probabilities, as output_probs writes them."""
    z = np.random.default_rng(seed).normal(0.0, 1.5, (3, h, w))
    e = np.exp(z - z.max(0, keepdims=True))
    return (e / e.sum(0, keepdims=True)).astype(np.float32)


async def _direct_run(root: Path) -> dict[str, Any]:
    """A direct run of two 32 x 32 windows, read by from-olmoearth and the tool.

    The windows lie side by side, so the scores raster is 32 x 64 pixels:
    128 windows of 4 px.
    """
    for k in range(2):
        layer = root / "ds" / "windows" / "default" / f"w{k}" / "layers" / "output"
        (layer / "probs").mkdir(parents=True)
        profile = {
            "driver": "GTiff",
            "height": 32,
            "width": 32,
            "count": 3,
            "dtype": "float32",
            "crs": "EPSG:32737",
            "transform": Affine(10.0, 0.0, X0 + 320.0 * k, 0.0, -10.0, Y0),
        }
        with rasterio.open(layer / "probs" / "geotiff.tif", "w", **profile) as dst:
            dst.write(_softmax(30 + k, 32, 32))
        (layer / "completed").write_text("", encoding="utf-8")
    oe_reader.read_output(str(root / "ds"), str(root / "run"))
    return await _ok("olmoearth_scores_from_file", {"run_dir": "run"})


#: The south-up run's folder under the scores root, and its raster.
SOUTH_UP, SOUTH_UP_RASTER = "south_up", "scores.tif"


async def _south_up_run(root: Path) -> dict[str, Any]:
    """A direct run whose scores raster is south up, read by the tool.

    A score_area.py run (its raster and a ``manifest.json`` that says the
    values are probabilities) of 32 x 64 pixels, 128 windows of 4 px, whose
    row 0 is the south edge: the y step is positive. GDAL georeferences such
    a grid, and the zone mask on its windows, with a ModelTransformation
    matrix, which olmoearth-inferencex refused in a zone mask from 1.17.0 to
    1.20.2.
    """
    run = root / SOUTH_UP
    run.mkdir()
    profile = {
        "driver": "GTiff",
        "height": 32,
        "width": 64,
        "count": 3,
        "dtype": "float32",
        "crs": "EPSG:32737",
        "transform": Affine(10.0, 0.0, X0, 0.0, 10.0, Y0 - 320.0),
    }
    with rasterio.open(run / SOUTH_UP_RASTER, "w", **profile) as dst:
        dst.write(_softmax(40, 32, 64))
    manifest = {"scores": {"file": SOUTH_UP_RASTER, "values": "probabilities"}}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return await _ok("olmoearth_scores_from_file", {"run_dir": SOUTH_UP})


async def _map_plan(scores_path: str) -> dict[str, Any]:
    return await _ok(
        "olmoearth_plan_label_sample",
        {
            "scores_path": scores_path,
            "budget": BUDGET,
            "design": "random",
            "seed_from_map": True,
            "alpha": ALPHA,
        },
    )


def _keys(obj: Any) -> set[str]:
    if isinstance(obj, dict):
        return set(obj).union(*(_keys(v) for v in obj.values()))
    if isinstance(obj, list):
        return set().union(*(_keys(v) for v in obj))
    return set()


@pytest.mark.asyncio
async def test_a_certificate_round_trip_reproduces(_scores_root: Path) -> None:
    """Sample with the map's seed, fixed labels, certify, certificate, verify."""
    run = await _direct_run(_scores_root)
    raster = _scores_root / "run" / "scores_32737.tif"
    plan = await _map_plan(run["scores_path"])

    # The seed is the map's: the package derives the same one from the raster.
    drawn_seed = pkg_certificate.draw_seed(
        str(raster),
        nodata=None,
        logits=False,
        confidence_path=None,
        confidence_range=None,
        condition_path=None,
        design="random",
        budget=BUDGET,
        patch=4,
        target=None,
        anchor=None,
        plan=PLAN,
    )
    expected = drawn_seed["seed"]
    assert plan["seed"] == expected == plan["seed_from_map"]["seed"]
    assert plan["seed_from_map"]["alpha"] == ALPHA
    assert plan["seed_from_map"]["map_sha256"] == drawn_seed["map_sha256"]
    design = json.loads(Path(plan["design_path"]).read_text(encoding="utf-8"))
    sheet = Path(plan["seed_from_map"]["package_sheet"])
    with open(sheet, encoding="utf-8", newline="") as fh:
        drawn = [int(row["index"]) for row in csv.DictReader(fh)]
    assert design["sample"]["indices"] == drawn
    listed = [w["window_index"] for w in plan["windows"]]
    assert listed == drawn[: len(listed)]
    assert design["certificate"]["seed_rule"] == "from_map"
    assert sheet.parent.parent.name == "certificates"
    # A certificate's plan needs every row of its sheet labelled (agent trial
    # 12, N2/cluster run 23 said "A partial labelling is fine" of one).
    prefix = next(f for f in plan["facts"] if f["id"] == "prefix_is_random_sample")
    needs = "drawn with seed_from_map for a certificate, which needs every row"
    assert needs in prefix["sentence"]
    (why,) = [
        c["why"]
        for c in plan["forbidden_claims"]
        if c["id"] == "subset_labelling_sufficient"
    ]
    assert needs in why
    # Its routes are all three, and a prefix gets neither the zone nor the
    # certificate (the plain random plan's sentence names no certificate).
    assert prefix["sentence"].startswith(
        "With design_path, olmoearth_estimate_map_error, olmoearth_certify_zone "
        "and olmoearth_write_certificate need every one of the sheet's "
    )
    assert prefix["sentence"].endswith("they give no certified zone or certificate.")
    assert why.endswith("and no certified zone or certificate")

    wrong = [0] * BUDGET
    zone = await _ok(
        "olmoearth_certify_zone",
        {"design_path": plan["design_path"], "wrong": wrong, "alpha": ALPHA},
    )
    assert zone["certified"] is True
    written = await _ok(
        "olmoearth_write_certificate",
        {"design_path": plan["design_path"], "wrong": wrong, "date": "2024-06-01"},
    )
    assert written["certificate_written"] is True and written["certified"] is True
    assert written["n_zone"] == zone["n_zone"]
    assert written["coverage"] == zone["coverage"]
    assert written["seed"] == expected and written["date"] == "2024-06-01"
    assert written["issued_note"] == pkg_zone.ISSUED_NOTE == zone["issued_note"]
    assert written["issuers_word"] == pkg_certificate.ISSUERS_WORD
    assert written["statement"].startswith("The ")
    cert_path = Path(written["certificate_path"])
    assert cert_path.is_file() and cert_path.parent.parent == sheet.parent
    assert {"certificate.json", "labels.csv", "zone_mask.tif"} <= set(written["bundle"])
    assert not (_keys(written) & _BANNED_KEYS)
    # The whole-map estimate from the same labels, as the zone states it, and
    # the labels' digest: the report tells two label sets apart by them.
    assert [f["id"] for f in written["facts"]] == ["whole_map_estimate", "certificate"]
    whole = {f["id"]: f for f in zone["facts"]}["whole_map_estimate"]
    assert written["facts"][0] == whole
    assert written["labels_sha256"] == zone["labels_sha256"] == labels_digest(wrong)

    # Checked against a copy of the map, by the tool and by the package's CLI.
    copy = _scores_root / "my_copy.tif"
    shutil.copyfile(raster, copy)
    mask = np.load(sheet.parent / "sample_zone.npy")
    checked = await _ok(
        "olmoearth_verify_certificate",
        {
            "certificate_path": str(cert_path.parent),
            "map_path": str(copy),
            "window": [0, 0],
        },
    )
    assert checked["verified"] is True and checked["failed_checks"] == []
    assert all(c["ok"] is True for c in checked["checks"].values())
    assert checked["map_given"] is True and checked["seed"] == expected
    assert checked["statement"] and checked["issued_note"] == pkg_zone.ISSUED_NOTE
    assert checked["window_lookup"]["inside"] is bool(mask[0, 0])
    assert pkg_certificate.ISSUERS_WORD in checked["notes"]
    assert not (_keys(checked) & _BANNED_KEYS)
    assert cli.main(["verify", str(cert_path), "--scores", str(copy)]) == 0
    # The scores file names the raster it was read from, which is the file
    # the certificate names: no copy, so the check is said to show only that
    # the certificate agrees with that file.
    by_scores = await _ok(
        "olmoearth_verify_certificate",
        {"certificate_path": str(cert_path), "map_path": run["scores_path"]},
    )
    assert by_scores["verified"] is True and by_scores["map_given"] is False
    assert any(
        s.startswith("map_path is the file the certificate names")
        for s in by_scores["must_state"]
    )
    assert checked["map_given"] is True
    assert not any("file the certificate names" in s for s in checked["must_state"])
    outside = await _ok(
        "olmoearth_verify_certificate",
        {"certificate_path": str(cert_path), "window": [99, 0]},
    )
    assert outside["verified"] is True and outside["map_given"] is False
    assert outside["window_lookup"] == {
        "window": [99, 0],
        "inside": None,
        "where": "outside the map",
    }
    assert any("path the certificate names" in s for s in outside["must_state"])


@pytest.mark.asyncio
async def test_a_south_up_map_gets_a_certificate_that_verifies(
    _scores_root: Path,
) -> None:
    """A scores raster whose y step is positive (row 0 at the south edge) is
    planned, certified, given a certificate and checked against a copy of
    the map, by the tool and by the package's CLI. Its zone mask lies on the
    map's south-up window grid, which GDAL writes as a ModelTransformation
    matrix: olmoearth-inferencex 1.17.0 to 1.20.2 refused that mask, so the
    write failed, and 1.20.3, the release the agent requires, writes it."""
    run = await _south_up_run(_scores_root)
    raster = _scores_root / SOUTH_UP / SOUTH_UP_RASTER
    with rasterio.open(raster) as src:
        grid = tuple(src.transform)[:6]
    assert grid == (10.0, 0.0, X0, 0.0, 10.0, Y0 - 320.0)  # south up, as written
    plan = await _map_plan(run["scores_path"])
    assert plan["seed_from_map"]["alpha"] == ALPHA
    wrong = [0] * BUDGET
    zone = await _ok(
        "olmoearth_certify_zone",
        {"design_path": plan["design_path"], "wrong": wrong, "alpha": ALPHA},
    )
    assert zone["certified"] is True
    written = await _ok(
        "olmoearth_write_certificate",
        {"design_path": plan["design_path"], "wrong": wrong, "date": "2024-06-01"},
    )
    assert written["certificate_written"] is True and written["certified"] is True
    assert written["verified_before_written"] is True
    certified = (zone["n_zone"], zone["coverage"])
    assert (written["n_zone"], written["coverage"]) == certified
    cert_path = Path(written["certificate_path"])
    with rasterio.open(cert_path.parent / "zone_mask.tif") as mask_src:
        mask_grid = tuple(mask_src.transform)[:6]
    # The mask's grid is the map's in windows of 4 px: south up too.
    assert mask_grid == (40.0, 0.0, X0, 0.0, 40.0, Y0 - 320.0)

    copy = _scores_root / "my_south_up_copy.tif"
    shutil.copyfile(raster, copy)
    mask = np.load(cert_path.parent.parent / "sample_zone.npy")
    checked = await _ok(
        "olmoearth_verify_certificate",
        {
            "certificate_path": str(cert_path.parent),
            "map_path": str(copy),
            "window": [0, 0],
        },
    )
    assert checked["verified"] is True and checked["failed_checks"] == []
    assert all(c["ok"] is True for c in checked["checks"].values())
    assert checked["map_given"] is True
    assert checked["window_lookup"]["inside"] is bool(mask[0, 0])
    assert cli.main(["verify", str(cert_path), "--scores", str(copy)]) == 0


@pytest.mark.asyncio
async def test_a_tampered_certificate_does_not_verify(_scores_root: Path) -> None:
    """One label changed in a copy of the bundle: the copy fails, the original not."""
    run = await _direct_run(_scores_root)
    plan = await _map_plan(run["scores_path"])
    written = await _ok(
        "olmoearth_write_certificate",
        {
            "design_path": plan["design_path"],
            "wrong": [0] * BUDGET,
            "date": "2024-06-01",
        },
    )
    bundle = Path(written["certificate_path"]).parent
    tampered = _scores_root / "tampered"
    shutil.copytree(bundle, tampered)
    labels = tampered / "labels.csv"
    with open(labels, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    at = rows[0].index("wrong")
    rows[1][at] = "1"
    with open(labels, "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)
    copy = _scores_root / "my_copy.tif"
    shutil.copyfile(_scores_root / "run" / "scores_32737.tif", copy)
    checked = await _ok(
        "olmoearth_verify_certificate",
        {
            "certificate_path": str(tampered / "certificate.json"),
            "map_path": str(copy),
        },
    )
    assert checked["verified"] is False and checked["statement"] is None
    assert "bundle" in checked["failed_checks"]
    assert checked["must_state"][0].startswith(
        "The certificate does not verify against this map"
    )
    original = await _ok(
        "olmoearth_verify_certificate",
        {"certificate_path": str(bundle), "map_path": str(copy)},
    )
    assert original["verified"] is True


@pytest.mark.asyncio
async def test_a_command_line_failure_is_a_refusal(
    _scores_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The package's CLI ends a refusal with a message and a non-zero status;
    a run past the time limit is stopped. Either is the tool's refusal, never
    a crash, and nothing is written."""
    run = await _direct_run(_scores_root)
    plan = await _map_plan(run["scores_path"])
    seen: list[dict[str, Any]] = []

    def refuse(command: list[str], **kwargs: Any) -> Any:
        seen.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            command, 1, stdout="", stderr="Traceback ...\ncertify: the sheet is gone\n"
        )

    monkeypatch.setattr(inferencex.subprocess, "run", refuse)
    out = await _call(
        "olmoearth_write_certificate",
        {
            "design_path": plan["design_path"],
            "wrong": [0] * BUDGET,
            "date": "2024-06-01",
        },
    )
    assert out["ok"] is False
    assert (
        "oe-inferencex certify refused (exit status 1): certify: the sheet is gone"
        in out["error"]
    )
    (call,) = seen
    assert call["command"][:4] == [sys.executable, "-P", "-m", "oe_inferencex.cli"]
    assert call["command"][4] == "certify" and "shell" not in call
    assert call["timeout"] == inferencex.CLI_TIMEOUT and call["check"] is False
    assert Path(call["cwd"]) == Path(plan["seed_from_map"]["package_sheet"]).parent
    assert "OLMOEARTH_SCORES_ROOT" not in call["env"]
    assert not (Path(call["cwd"]) / "certificate").exists()

    def slow(command: list[str], **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(inferencex.subprocess, "run", slow)
    # Another budget: the first plan's sheet holds its labels now, and the
    # same plan is not drawn again over them.
    out = await _call(
        "olmoearth_plan_label_sample",
        {
            "scores_path": run["scores_path"],
            "budget": BUDGET // 2,
            "design": "random",
            "seed_from_map": True,
            "alpha": ALPHA,
        },
    )
    assert out["ok"] is False
    assert "oe-inferencex sample did not finish within 600 s" in out["error"]


def test_run_cli_runs_the_packages_command_line(tmp_path: Path) -> None:
    """The real CLI: its output on success, its own last line on a refusal."""
    assert "certify" in inferencex.run_cli(["certify", "--help"], cwd=tmp_path)
    with pytest.raises(ValueError, match=r"oe-inferencex certify refused \(exit"):
        inferencex.run_cli(["certify", str(tmp_path / "missing.csv")], cwd=tmp_path)


@pytest.mark.asyncio
async def test_a_plan_keeps_its_settings_and_needs_its_map_seed(
    _scores_root: Path,
) -> None:
    """A seed_from_map plan fixed alpha, delta and the rule; a plan without it
    gets no certificate; seed_from_map needs a random design and alpha."""
    run = await _direct_run(_scores_root)
    plan = await _map_plan(run["scores_path"])
    other = await _call(
        "olmoearth_certify_zone",
        {"design_path": plan["design_path"], "wrong": [0] * BUDGET, "alpha": 0.1},
    )
    assert other["ok"] is False
    assert "fixed alpha 0.2 before any label" in other["error"]
    plain = await _ok(
        "olmoearth_plan_label_sample",
        {"scores_path": run["scores_path"], "budget": BUDGET, "design": "random"},
    )
    refused = await _call(
        "olmoearth_write_certificate",
        {"design_path": plain["design_path"], "wrong": [0] * BUDGET},
    )
    assert refused["ok"] is False
    assert "was not drawn for a certificate" in refused["error"]
    for args, said in (
        ({"design": "confidence"}, "pass design='random'"),
        ({"design": "random", "seed": 3}, "give seed or seed_from_map"),
        ({"design": "random"}, "pass alpha"),
    ):
        out = await _call(
            "olmoearth_plan_label_sample",
            {
                "scores_path": run["scores_path"],
                "budget": BUDGET,
                "seed_from_map": True,
                **args,
            },
        )
        assert out["ok"] is False and said in out["error"], (args, out)


@pytest.mark.asyncio
async def test_nothing_certified_writes_no_certificate(_scores_root: Path) -> None:
    run = await _direct_run(_scores_root)
    plan = await _map_plan(run["scores_path"])
    out = await _ok(
        "olmoearth_write_certificate",
        {
            "design_path": plan["design_path"],
            "wrong": [1] * BUDGET,
            "date": "2024-06-01",
        },
    )
    assert out["certified"] is False and out["certificate_written"] is False
    assert "nothing to put in a certificate" in out["reason"]
    # As olmoearth_certify_zone says it (agent trial 12, B6; the step 11
    # review): the whole-map rate describes the whole map, not its most
    # confident part, and the result forbids reading it against alpha as the
    # reason nothing was certified.
    whole = next(f for f in out["facts"] if f["id"] == "whole_map_estimate")
    sentence = whole["sentence"]
    assert "It describes the whole map, not its most confident part: " in sentence
    assert "smallest_zone_tested" in whole
    claims = {c["id"]: c["why"] for c in out["forbidden_claims"]}
    assert set(claims) == {"post_hoc_alpha", "whole_map_rate_for_a_zone"}
    assert f"alpha={ALPHA:g}" in claims["whole_map_rate_for_a_zone"]
    # This result has no levels_tested, so nothing points at one (the step 11
    # review); with no level tested, its note says why.
    assert "levels_tested" not in out
    assert "levels_tested" not in sentence
    assert "levels_tested" not in claims["whole_map_rate_for_a_zone"]
    sheet = Path(plan["seed_from_map"]["package_sheet"])
    assert not (sheet.parent / "certificate").exists()
    with open(sheet, encoding="utf-8", newline="") as fh:
        assert {row["wrong"] for row in csv.DictReader(fh)} == {"1"}


def _sheet_wrong(sheet: Path) -> list[str]:
    with open(sheet, encoding="utf-8", newline="") as fh:
        return [row["wrong"] for row in csv.DictReader(fh)]


@pytest.mark.asyncio
async def test_a_certificate_rests_on_the_labels_first_given(
    _scores_root: Path,
) -> None:
    """The labels first graded on a seed_from_map plan are fixed in its sheet:
    afterwards a zone, an estimate or a certificate on other labels is
    refused, and so is the same plan drawn again over them. A certificate is
    written once: a second write, with the same labels or others, is refused
    and the first stays byte for byte as written."""
    run = await _direct_run(_scores_root)
    plan = await _map_plan(run["scores_path"])
    sheet = Path(plan["seed_from_map"]["package_sheet"])
    first = [0] * BUDGET
    other = [1] + [0] * (BUDGET - 1)
    zone = await _ok(
        "olmoearth_certify_zone",
        {"design_path": plan["design_path"], "wrong": first, "alpha": ALPHA},
    )
    assert zone["certified"] is True and _sheet_wrong(sheet) == ["0"] * BUDGET
    for tool, extra in (
        ("olmoearth_certify_zone", {"alpha": ALPHA}),
        ("olmoearth_estimate_map_error", {}),
        ("olmoearth_write_certificate", {"date": "2024-06-01"}),
    ):
        out = await _call(
            tool, {"design_path": plan["design_path"], "wrong": other, **extra}
        )
        assert out["ok"] is False, tool
        assert f"already holds other labels on 1 of {BUDGET} windows" in out["error"]
        assert "a certificate rests on the labels first given" in out["error"]
    assert _sheet_wrong(sheet) == ["0"] * BUDGET
    assert not (sheet.parent / "certificate").exists()

    written = await _ok(
        "olmoearth_write_certificate",
        {"design_path": plan["design_path"], "wrong": first, "date": "2024-06-01"},
    )
    bundle = Path(written["certificate_path"]).parent
    before = {p.name: p.read_bytes() for p in bundle.iterdir()}
    for labels in (first, other):
        out = await _call(
            "olmoearth_write_certificate",
            {"design_path": plan["design_path"], "wrong": labels, "date": "2024-07"},
        )
        assert out["ok"] is False
        assert "a certificate on this sample is already written" in out["error"]
    assert {p.name: p.read_bytes() for p in bundle.iterdir()} == before

    # The draw is the map's and the settings': drawn again, it would clear the
    # labels, so the plan names its design file instead.
    again = await _call(
        "olmoearth_plan_label_sample",
        {
            "scores_path": run["scores_path"],
            "budget": BUDGET,
            "design": "random",
            "seed_from_map": True,
            "alpha": ALPHA,
        },
    )
    assert again["ok"] is False and "was drawn before" in again["error"]
    assert plan["design_path"] in again["error"]
    assert _sheet_wrong(sheet) == ["0"] * BUDGET


@pytest.mark.asyncio
async def test_a_design_file_naming_another_folder_is_refused(
    _scores_root: Path,
) -> None:
    """A design file whose certificate block names a folder the plan tool did
    not make (one of the user's own sheets, or the certificates folder
    itself) is refused before anything is written there."""
    run = await _direct_run(_scores_root)
    plan = await _map_plan(run["scores_path"])
    sheet = Path(plan["seed_from_map"]["package_sheet"])
    theirs = _scores_root / "my_sheets"
    theirs.mkdir()
    shutil.copyfile(sheet, theirs / "sample.csv")
    shutil.copyfile(sheet.with_suffix(".json"), theirs / "sample.json")
    before = (theirs / "sample.csv").read_bytes()
    design = json.loads(Path(plan["design_path"]).read_text(encoding="utf-8"))
    for folder in (theirs, sheet.parent.parent):
        design["certificate"]["work_dir"] = str(folder)
        forged = _scores_root / "forged_design.json"
        forged.write_text(json.dumps(design), encoding="utf-8")
        for tool, extra in (
            ("olmoearth_write_certificate", {"date": "2024-06-01"}),
            ("olmoearth_certify_zone", {"alpha": ALPHA}),
        ):
            out = await _call(
                tool, {"design_path": str(forged), "wrong": [1] * BUDGET, **extra}
            )
            assert out["ok"] is False, (folder, tool)
            assert "not one olmoearth_plan_label_sample wrote" in out["error"]
    assert (theirs / "sample.csv").read_bytes() == before
    assert not (theirs / "sample_zone.json").exists()
    assert _sheet_wrong(sheet) == [""] * BUDGET


@pytest.mark.asyncio
async def test_a_check_reads_only_a_plain_map_under_the_root(
    _scores_root: Path,
) -> None:
    """verify reads a map given to it with GDAL as it is, so map_path must be
    a plain GeoTIFF or .npy (XML named .tif is what GDAL would follow); a
    .json that is not a scores file, a certificate naming its map by a
    relative path, a certificate.json that is not JSON and a window that is
    not [row, col] are each refused, saying what to pass."""
    run = await _direct_run(_scores_root)
    plan = await _map_plan(run["scores_path"])
    written = await _ok(
        "olmoearth_write_certificate",
        {
            "design_path": plan["design_path"],
            "wrong": [0] * BUDGET,
            "date": "2024-06-01",
        },
    )
    cert_path = Path(written["certificate_path"])
    vrt = _scores_root / "map.tif"
    vrt.write_text(
        '<VRTDataset rasterXSize="1" rasterYSize="1"></VRTDataset>', encoding="utf-8"
    )
    summary = _scores_root / "run" / "olmoearth_output.json"
    for map_path, said in (
        (vrt, "map_path must be the map as a plain GeoTIFF or .npy file"),
        (cert_path, "map_path is a .json that is not a scores file"),
        (summary, "map_path is a .json that is not a scores file"),
    ):
        out = await _call(
            "olmoearth_verify_certificate",
            {"certificate_path": str(cert_path), "map_path": str(map_path)},
        )
        assert out["ok"] is False and said in out["error"], (map_path, out)

    relative = _scores_root / "relative"
    shutil.copytree(cert_path.parent, relative)
    item = json.loads((relative / "certificate.json").read_text(encoding="utf-8"))
    item["properties"]["zc:map"]["path"] = "../x.tif"
    (relative / "certificate.json").write_text(json.dumps(item), encoding="utf-8")
    out = await _call(
        "olmoearth_verify_certificate", {"certificate_path": str(relative)}
    )
    assert out["ok"] is False and "names its map by a relative path" in out["error"]
    elsewhere = _scores_root.parent / "elsewhere.tif"
    shutil.copyfile(_scores_root / "run" / "scores_32737.tif", elsewhere)
    item["properties"]["zc:map"]["path"] = str(elsewhere)
    (relative / "certificate.json").write_text(json.dumps(item), encoding="utf-8")
    out = await _call(
        "olmoearth_verify_certificate", {"certificate_path": str(relative)}
    )
    assert out["ok"] is False
    assert "names a map outside the scores root" in out["error"]
    assert "pass map_path, your copy of the map" in out["error"]

    unreadable = _scores_root / "unreadable"
    unreadable.mkdir()
    (unreadable / "certificate.json").write_text("not json", encoding="utf-8")
    out = await _call(
        "olmoearth_verify_certificate", {"certificate_path": str(unreadable)}
    )
    assert out["ok"] is False
    assert "is not a certificate this package can read" in out["error"]

    out = await _call(
        "olmoearth_verify_certificate",
        {"certificate_path": str(cert_path), "window": [1]},
    )
    assert out["ok"] is False and "window is [row, col]" in out["error"]


@pytest.mark.asyncio
async def test_a_map_seeded_plan_needs_the_runs_own_raster(_scores_root: Path) -> None:
    """seed_from_map draws from the run's raster, so it needs the scores file
    olmoearth_scores_from_file wrote, naming a raster that is still there
    under the root; alpha, delta and rule given without it are refused, as
    oe-inferencex sample refuses them without --seed-from-map (the defaults,
    echoed, change nothing)."""
    run = await _direct_run(_scores_root)
    saved = json.loads(Path(run["scores_path"]).read_text(encoding="utf-8"))
    base = {"budget": BUDGET, "design": "random", "seed_from_map": True, "alpha": 0.2}
    out = await _call(
        "olmoearth_plan_label_sample", {**base, "scores": saved["scores"]}
    )
    assert out["ok"] is False and "draws from the map itself" in out["error"]
    raster = saved["raster"]
    for edit, said in (
        ({k: v for k, v in raster.items() if k != "path"}, "records no raster path"),
        ({**raster, "path": str(_scores_root / "gone.tif")}, "is gone"),
        ({**raster, "path": str(_scores_root.parent / "x.tif")}, "is not under"),
    ):
        edited = _scores_root / "edited_scores.json"
        edited.write_text(json.dumps({**saved, "raster": edit}), encoding="utf-8")
        out = await _call(
            "olmoearth_plan_label_sample", {**base, "scores_path": str(edited)}
        )
        assert out["ok"] is False and said in out["error"], (said, out)
    assert not (_scores_root / "certificates").exists()

    plain = {"scores_path": run["scores_path"], "budget": BUDGET, "design": "random"}
    for extra, said in (
        ({"alpha": 0.05}, "alpha given without seed_from_map"),
        ({"delta": 0.2, "rule": "bonferroni"}, "delta and rule given without"),
    ):
        out = await _call("olmoearth_plan_label_sample", {**plain, **extra})
        assert out["ok"] is False and said in out["error"], (extra, out)
    echoed = await _ok(
        "olmoearth_plan_label_sample", {**plain, "delta": 0.1, "rule": "prefix"}
    )
    design = json.loads(Path(echoed["design_path"]).read_text(encoding="utf-8"))
    assert "certificate" not in design and "seed_from_map" not in echoed


@pytest.mark.asyncio
async def test_a_sheet_that_lists_other_windows_is_refused(_scores_root: Path) -> None:
    """The package's sheet must list the plan's windows in the plan's order:
    one edited by hand (two rows swapped) is refused, by the certificate and
    by the zone, and left as it was."""
    run = await _direct_run(_scores_root)
    plan = await _map_plan(run["scores_path"])
    sheet = Path(plan["seed_from_map"]["package_sheet"])
    with open(sheet, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    rows[1], rows[2] = rows[2], rows[1]
    with open(sheet, "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)
    before = sheet.read_bytes()
    for tool, extra in (
        ("olmoearth_write_certificate", {"date": "2024-06-01"}),
        ("olmoearth_certify_zone", {"alpha": ALPHA}),
    ):
        out = await _call(
            tool, {"design_path": plan["design_path"], "wrong": [0] * BUDGET, **extra}
        )
        assert out["ok"] is False, tool
        assert "does not list the plan's windows in its order" in out["error"]
    assert sheet.read_bytes() == before


# --------------------------------------------------------------------------- off the event loop

#: The thread each wrapped package call ran on, by the call's name.
_Threads = list[tuple[str, threading.Thread]]


def _recorded(threads: _Threads, name: str, real: Callable[..., Any]) -> Any:
    """``real``, recording under ``name`` the thread it ran on."""

    def call(*args: Any, **kwargs: Any) -> Any:
        threads.append((name, threading.current_thread()))
        return real(*args, **kwargs)

    return call


def _raising(threads: _Threads, name: str, exc: Exception) -> Any:
    """A stand-in recording under ``name`` the thread it ran on, then raising."""

    def call(*args: Any, **kwargs: Any) -> Any:
        threads.append((name, threading.current_thread()))
        raise exc

    return call


def _off_the_loop(threads: _Threads, loop_thread: threading.Thread) -> None:
    """Every recorded call ran, and none on the event loop's own thread."""
    assert threads
    assert all(t is not loop_thread for _name, t in threads), threads


async def _written(root: Path) -> tuple[dict[str, Any], Path]:
    """A direct run's certificate on labels that certify, and a copy of its map."""
    run = await _direct_run(root)
    plan = await _map_plan(run["scores_path"])
    written = await _ok(
        "olmoearth_write_certificate",
        {
            "design_path": plan["design_path"],
            "wrong": [0] * BUDGET,
            "date": "2024-06-01",
        },
    )
    copy = root / "my_copy.tif"
    shutil.copyfile(root / "run" / "scores_32737.tif", copy)
    return written, copy


@pytest.mark.asyncio
async def test_build_runs_off_the_event_loop(
    _scores_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """certificate.build reads and digests the map and verifies the bundle: it
    runs in a worker thread, never on the handler's own thread (the event
    loop's, which serves the web bridge's other sessions), and its refusal
    and its certificate come back as they did inline."""
    run = await _direct_run(_scores_root)
    plan = await _map_plan(run["scores_path"])
    labels = {"design_path": plan["design_path"], "wrong": [0] * BUDGET}
    args = {**labels, "date": "2024-06-01"}
    loop_thread = threading.current_thread()
    threads: _Threads = []
    real_build = pkg_certificate.build
    refusal = ValueError("the zone file is not one certify wrote")
    monkeypatch.setattr(pkg_certificate, "build", _raising(threads, "build", refusal))
    out = await _call("olmoearth_write_certificate", args)
    assert out["ok"] is False
    assert (
        "olmoearth-inferencex wrote no certificate: the zone file is not one "
        "certify wrote" in out["error"]
    )
    work = Path(plan["seed_from_map"]["package_sheet"]).parent
    assert not (work / "certificate" / "certificate.json").exists()

    monkeypatch.setattr(
        pkg_certificate, "build", _recorded(threads, "build", real_build)
    )
    written = await _ok("olmoearth_write_certificate", args)
    assert written["certificate_written"] is True
    assert written["verified_before_written"] is True
    assert Path(written["certificate_path"]).is_file()
    zone = await _ok("olmoearth_certify_zone", {**labels, "alpha": ALPHA})
    certified = (zone["n_zone"], zone["coverage"])
    assert (written["n_zone"], written["coverage"]) == certified
    assert [name for name, _t in threads] == ["build", "build"]
    _off_the_loop(threads, loop_thread)


@pytest.mark.asyncio
async def test_verify_runs_off_the_event_loop(
    _scores_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """certificate.verify digests the map and draws the sample and certify
    again: it runs in a worker thread; the result is the package's, check by
    check, and an unreadable certificate is the same refusal as inline."""
    written, copy = await _written(_scores_root)
    cert_path = str(written["certificate_path"])
    expected = pkg_certificate.verify(cert_path, str(copy))
    loop_thread = threading.current_thread()
    threads: _Threads = []
    real_verify = pkg_certificate.verify
    monkeypatch.setattr(
        pkg_certificate, "verify", _recorded(threads, "verify", real_verify)
    )
    args = {"certificate_path": cert_path, "map_path": str(copy)}
    checked = await _ok("olmoearth_verify_certificate", args)
    assert checked["verified"] is True is expected["verified"]
    assert {n: c["ok"] for n, c in checked["checks"].items()} == {
        n: c["ok"] for n, c in expected["checks"].items()
    }
    assert checked["statement"] == pkg_certificate.printable(expected["statement"])
    assert checked["map_read"] == expected["map_read"]

    unreadable = pkg_certificate.Unreadable(
        f"{cert_path} is not a certificate this package can read: forged"
    )
    monkeypatch.setattr(
        pkg_certificate, "verify", _raising(threads, "verify", unreadable)
    )
    out = await _call("olmoearth_verify_certificate", args)
    assert out["ok"] is False
    assert "is not a certificate this package can read: forged" in out["error"]
    assert [name for name, _t in threads] == ["verify", "verify"]
    _off_the_loop(threads, loop_thread)


@pytest.mark.asyncio
async def test_a_window_lookup_runs_off_the_event_loop(
    _scores_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """certificate.point, for the window asked about, runs in a worker thread
    too; its answer is the zone verify built again, and an error of it is the
    lookup's "not looked up", as inline, the check itself unchanged."""
    written, copy = await _written(_scores_root)
    cert_path = Path(written["certificate_path"])
    inside = bool(np.load(cert_path.parent.parent / "sample_zone.npy")[0, 0])
    loop_thread = threading.current_thread()
    threads: _Threads = []
    real_point = pkg_certificate.point
    monkeypatch.setattr(
        pkg_certificate, "point", _recorded(threads, "point", real_point)
    )
    args = {
        "certificate_path": str(cert_path),
        "map_path": str(copy),
        "window": [0, 0],
    }
    checked = await _ok("olmoearth_verify_certificate", args)
    assert checked["verified"] is True
    assert checked["window_lookup"] == {
        "window": [0, 0],
        "inside": inside,
        "where": (
            "inside the certified zone" if inside else "outside the certified zone"
        ),
    }

    broken = ValueError("the zone mask is unreadable")
    monkeypatch.setattr(pkg_certificate, "point", _raising(threads, "point", broken))
    again = await _ok("olmoearth_verify_certificate", args)
    assert again["verified"] is True
    assert again["window_lookup"] == {
        "window": [0, 0],
        "inside": None,
        "where": "not looked up: the zone mask is unreadable",
    }
    assert [name for name, _t in threads] == ["point", "point"]
    _off_the_loop(threads, loop_thread)


@pytest.mark.asyncio
async def test_the_map_seeded_route_computes_off_the_event_loop(
    _scores_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The draw a certificate rests on, its estimate, its zone, and the
    certificate's own zone and whole-map estimate: each of the package's
    computations runs in a worker thread, and the route ends as before (the
    certificate packs the zone olmoearth_certify_zone certified)."""
    run = await _direct_run(_scores_root)
    loop_thread = threading.current_thread()
    threads: _Threads = []
    names = ("sample_for_estimation", "certify_zone", "estimate_error_rate")
    for name in names:
        real = getattr(pkg_estimate, name)
        monkeypatch.setattr(pkg_estimate, name, _recorded(threads, name, real))
    plan = await _map_plan(run["scores_path"])
    labels = {"design_path": plan["design_path"], "wrong": [0] * BUDGET}
    estimated = await _ok("olmoearth_estimate_map_error", labels)
    assert estimated["n_labelled"] == BUDGET
    zone = await _ok("olmoearth_certify_zone", {**labels, "alpha": ALPHA})
    dated = {**labels, "date": "2024-06-01"}
    written = await _ok("olmoearth_write_certificate", dated)
    assert zone["certified"] is True and written["certificate_written"] is True
    certified = (zone["n_zone"], zone["coverage"])
    assert (written["n_zone"], written["coverage"]) == certified
    assert {name for name, _t in threads} == set(names)
    _off_the_loop(threads, loop_thread)


@pytest.mark.asyncio
async def test_an_estimate_fixes_a_map_seeded_plans_labels_once(
    _scores_root: Path,
) -> None:
    """The estimate of a seed_from_map plan reports the interval valid at
    every sample size first, as every design's does, and still writes the
    labels it grades into the plan's sheet the first time: other labels
    afterwards are refused, by the estimate and the zone alike, and the
    sheet keeps the first."""
    run = await _direct_run(_scores_root)
    plan = await _map_plan(run["scores_path"])
    sheet = Path(plan["seed_from_map"]["package_sheet"])
    first = [1] + [0] * (BUDGET - 1)
    other = [0] * BUDGET
    labels = {"design_path": plan["design_path"], "wrong": first}
    estimated = await _ok("olmoearth_estimate_map_error", labels)
    assert _sheet_wrong(sheet) == [str(w) for w in first]
    valid, design = estimated["intervals"]
    assert (valid["interval"], design["interval"]) == ("valid", "design")
    assert estimated["plan_design"] == "random"
    assert estimated["valid_note"] == pkg_estimate.VALID_NOTE
    # Under a random sample the two are the same exact interval (VALID_NOTE).
    assert (valid["low"], valid["high"]) == (design["low"], design["high"])
    again = await _ok("olmoearth_estimate_map_error", labels)
    assert again["valid_interval"] == estimated["valid_interval"]
    for tool, extra in (
        ("olmoearth_estimate_map_error", {}),
        ("olmoearth_certify_zone", {"alpha": ALPHA}),
    ):
        out = await _call(
            tool, {"design_path": plan["design_path"], "wrong": other, **extra}
        )
        assert out["ok"] is False, tool
        assert f"already holds other labels on 1 of {BUDGET} windows" in out["error"]
        assert "a certificate rests on the labels first given" in out["error"]
    assert _sheet_wrong(sheet) == [str(w) for w in first]


# --------------------------------------------------------------------------- two sessions at once


class _SlowSheetReads:
    """The csv module as ``estimation`` reads a sheet, each read held a moment.

    Two sessions grading one plan at once then both read the sheet before
    either writes it, unless the read, the check and the write are one step.
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def __getattr__(self, name: str) -> Any:
        return getattr(csv, name)

    def reader(self, *args: Any, **kwargs: Any) -> Any:
        rows = list(csv.reader(*args, **kwargs))
        time.sleep(self.seconds)
        return iter(rows)


@pytest.mark.asyncio
async def test_two_sessions_grading_one_plan_fix_one_set_of_labels(
    _scores_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The estimate fixes a seed_from_map plan's labels from a worker thread.
    Two sessions grade the plan at once with different labels, each read of
    the sheet held a moment (without the lock both read it empty and both
    write): exactly one estimate is given, the other is refused as other
    labels than the ones fixed, and the sheet holds the labels of the one
    given, with no file left beside it."""
    run = await _direct_run(_scores_root)
    plan = await _map_plan(run["scores_path"])
    sheet = Path(plan["seed_from_map"]["package_sheet"])
    first = [1] + [0] * (BUDGET - 1)
    other = [0] * BUDGET
    monkeypatch.setattr(estimation, "csv", _SlowSheetReads(0.3))
    outs = await asyncio.gather(
        *(
            _call(
                "olmoearth_estimate_map_error",
                {"design_path": plan["design_path"], "wrong": labels},
            )
            for labels in (first, other)
        )
    )
    given = [k for k, out in enumerate(outs) if out["ok"] is True]
    assert len(given) == 1, outs
    (k,) = given
    refused = outs[1 - k]
    assert refused["ok"] is False
    assert f"already holds other labels on 1 of {BUDGET} windows" in refused["error"]
    assert "a certificate rests on the labels first given" in refused["error"]
    fixed = (first, other)[k]
    assert _sheet_wrong(sheet) == [str(w) for w in fixed]
    assert sorted(p.name for p in sheet.parent.glob("*.tmp")) == []
    # The estimate given is the one on the labels fixed: graded again, the same.
    again = await _ok(
        "olmoearth_estimate_map_error",
        {"design_path": plan["design_path"], "wrong": fixed},
    )
    assert again["valid_interval"] == outs[k]["result"]["valid_interval"]


def test_a_sheet_is_replaced_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fill_sheet writes the labels to a file beside the sheet and moves it
    over the sheet, so a reader never finds it cut short: a write that fails
    part way leaves the sheet byte for byte as it was, with no file beside
    it, and one that succeeds leaves the sheet alone in its folder."""
    folder = tmp_path / "plan"
    folder.mkdir()
    sheet = folder / "sample.csv"
    sheet.write_bytes(b"index,confidence,map_class,wrong\r\n4,0.9,1,\r\n9,0.4,0,\r\n")
    before = sheet.read_bytes()

    class _HalfWriter:
        def __init__(self, fh: Any) -> None:
            self.inner = csv.writer(fh)

        def writerow(self, row: list[str]) -> Any:
            return self.inner.writerow(row)

        def writerows(self, rows: list[list[str]]) -> None:
            raise OSError("the disk is full")

    class _FailingCsv:
        def __getattr__(self, name: str) -> Any:
            return getattr(csv, name)

        def writer(self, fh: Any, *args: Any, **kwargs: Any) -> _HalfWriter:
            return _HalfWriter(fh)

    monkeypatch.setattr(estimation, "csv", _FailingCsv())
    with pytest.raises(OSError, match="the disk is full"):
        estimation.fill_sheet(sheet, [4, 9], [0, 1])
    assert sheet.read_bytes() == before
    assert [p.name for p in folder.iterdir()] == ["sample.csv"]
    monkeypatch.setattr(estimation, "csv", csv)
    estimation.fill_sheet(sheet, [4, 9], [0, 1])
    assert _sheet_wrong(sheet) == ["0", "1"]
    assert [p.name for p in folder.iterdir()] == ["sample.csv"]


@pytest.mark.asyncio
async def test_two_certificates_on_one_plan_at_once_write_one(
    _scores_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two sessions write a certificate on one plan at once, with the same
    labels: both pass the early check before either builds, and the
    package's build replaces a bundle. The check is made again with the
    build, under one lock: one certificate is written, the other call is
    refused as the early check refuses it. (certify runs once here, so that
    the second run does not rewrite the zone files the first is reading.)"""
    run = await _direct_run(_scores_root)
    plan = await _map_plan(run["scores_path"])
    work = Path(plan["seed_from_map"]["package_sheet"]).parent
    real_cli = inferencex.run_cli_async
    ran: list[list[str]] = []
    one_at_a_time = asyncio.Lock()

    async def certify_once(argv: list[str], **kwargs: Any) -> str:
        async with one_at_a_time:
            if ran:
                return ""
            ran.append(argv)
            return await real_cli(argv, **kwargs)

    built: list[str] = []

    def stub_build(zone_path: str, out_dir: str, date: str, **_kw: Any) -> Any:
        time.sleep(0.3)  # long enough for the other call to reach its build
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "certificate.json").write_text("{}", encoding="utf-8")
        built.append(date)
        return {
            "properties": {
                "zc:map": {"map_sha256": "0" * 64},
                "zc:statement": "The stub's statement.",
            }
        }

    monkeypatch.setattr(inferencex, "run_cli_async", certify_once)
    monkeypatch.setattr(pkg_certificate, "build", stub_build)
    labels = {"design_path": plan["design_path"], "wrong": [0] * BUDGET}
    outs = await asyncio.gather(
        *(
            _call("olmoearth_write_certificate", {**labels, "date": date})
            for date in ("2024-06-01", "2024-07")
        )
    )
    written = [out for out in outs if out["ok"] is True]
    refused = [out for out in outs if out["ok"] is False]
    assert len(written) == 1 and len(refused) == 1, outs
    assert written[0]["result"]["certificate_written"] is True
    assert "a certificate on this sample is already written" in refused[0]["error"]
    assert "never replaced" in refused[0]["error"]
    assert len(built) == 1 and len(ran) == 1
    assert written[0]["result"]["date"] == built[0]
    assert (work / "certificate" / "certificate.json").is_file()


# --------------------------------------------------------------------------- the zones disagree


@pytest.mark.asyncio
async def test_a_zone_on_other_windows_writes_no_certificate(
    _scores_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The package's certify on the plan's sheet and olmoearth_certify_zone's
    computation must give the same zone. Its count differing, or the same
    count on other windows (the zone mask compared off the loop), is refused
    with the same message, and no certificate is written; certify_zone runs
    off the event loop's thread each time."""
    run = await _direct_run(_scores_root)
    plan = await _map_plan(run["scores_path"])
    work = Path(plan["seed_from_map"]["package_sheet"]).parent
    args = {
        "design_path": plan["design_path"],
        "wrong": [0] * BUDGET,
        "date": "2024-06-01",
    }
    loop_thread = threading.current_thread()
    threads: _Threads = []
    real_zone = pkg_estimate.certify_zone

    def other_windows(*a: Any, **kw: Any) -> Any:
        threads.append(("certify_zone", threading.current_thread()))
        zone = dict(real_zone(*a, **kw))
        order = [int(i) for i in zone["zone_indices_in_order"]]
        order[-1] = max(order) + 1  # one window swapped: the counts agree
        zone["zone_indices_in_order"] = order
        return zone

    def one_window_fewer(*a: Any, **kw: Any) -> Any:
        threads.append(("certify_zone", threading.current_thread()))
        zone = dict(real_zone(*a, **kw))
        zone["n_zone"] = int(zone["n_zone"]) - 1
        return zone

    for stand_in in (other_windows, one_window_fewer):
        monkeypatch.setattr(pkg_estimate, "certify_zone", stand_in)
        out = await _call("olmoearth_write_certificate", args)
        assert out["ok"] is False, stand_in
        assert (
            "the package's certify on the plan's sheet gives another zone"
            in out["error"]
        )
        assert "no certificate was written" in out["error"]
        assert not (work / "certificate" / "certificate.json").exists()
    assert len(threads) == 2
    _off_the_loop(threads, loop_thread)


@pytest.mark.asyncio
async def test_certify_leaving_no_zone_mask_writes_no_certificate(
    _scores_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """certify certified a zone but its zone mask is gone: the comparison of
    the windows, run off the loop, is refused with what is missing, and no
    certificate is written."""
    run = await _direct_run(_scores_root)
    plan = await _map_plan(run["scores_path"])
    work = Path(plan["seed_from_map"]["package_sheet"]).parent
    real_cli = inferencex.run_cli_async

    async def without_mask(argv: list[str], **kwargs: Any) -> str:
        out = await real_cli(argv, **kwargs)
        if argv[0] == "certify":
            (Path(kwargs["cwd"]) / "sample_zone.npy").unlink()
        return out

    monkeypatch.setattr(inferencex, "run_cli_async", without_mask)
    out = await _call(
        "olmoearth_write_certificate",
        {
            "design_path": plan["design_path"],
            "wrong": [0] * BUDGET,
            "date": "2024-06-01",
        },
    )
    assert out["ok"] is False
    assert "oe-inferencex certify left no zone mask" in out["error"]
    zone = json.loads((work / "sample_zone.json").read_text(encoding="utf-8"))
    assert zone["coverage"] is not None  # certify did certify a zone
    assert not (work / "certificate" / "certificate.json").exists()
