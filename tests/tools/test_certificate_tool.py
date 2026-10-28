# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""A direct OlmoEarth run's certified zone as a certificate, written and checked.

The run is what ``oe-inferencex from-olmoearth`` writes from an rslearn
dataset (``oe_inferencex.readers.olmoearth.read_output``, run here on a
dataset this test writes with rasterio). ``olmoearth_plan_label_sample`` with
``seed_from_map`` draws through the package's command line (``sample
--seed-from-map``), the labels are fixed, ``olmoearth_certify_zone``
certifies, ``olmoearth_write_certificate`` packs the zone and
``olmoearth_verify_certificate`` checks it, as the package's own ``verify``
does from the command line. Every command runs for real, as a subprocess of
this interpreter; only the failure tests replace the subprocess call.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.provenance.log import labels_digest
from olmoearth_agent.tools import inferencex
from olmoearth_agent.tools.certificate import build_certificate_tools
from olmoearth_agent.tools.estimation import build_estimation_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.scores_file import build_scores_file_tools

try:
    import numpy as np
    import rasterio
    from oe_inferencex import cli
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
