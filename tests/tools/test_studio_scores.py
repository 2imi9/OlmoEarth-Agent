# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""``olmoearth_scores_from_studio`` on synthetic Studio result files.

Studio is mocked (pytest-httpx): the result's record as ``GET
/prediction-results/{id}`` returned the kelp and KarstBinary results on 31
October 2026 (``result_metadata`` filled, a ``download_token``), its
prediction's record, its model's record (fine-tuned on a label field, as the
kelp model was), and the zip ``GET /prediction-results/files`` returns, built
here with rasterio and zipfile in the layout measured that night:
``<prediction name>/result_epsg<code>_<i>_native.tif`` per UTM zone and, for a
class map with top-k scores, a ``README.md``. The top-k zones hold what the
kelp file held: raw rank-1 plus rank-2 scores of 254 on every valid pixel but
one. Nothing here reaches Studio. A test that reads a raster needs the
inferencex extra and skips without it, as the scores provider's do; the end to
end test runs the package's command line for the certificate plan, as the
certificate tests do.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
from pytest_httpx import HTTPXMock

from olmoearth_agent.analysis.review_set import MUST_STATE_STUDIO_FILES
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.studio.client import StudioClient, StudioConfig
from olmoearth_agent.tools import scores_file, studio_scores
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.estimation import build_estimation_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.review_set import build_review_set_tools
from olmoearth_agent.tools.scores_file import build_scores_file_tools
from olmoearth_agent.tools.studio_scores import (
    CLIPPED_SENTENCE,
    NATIVE_SENTENCE,
    TOPK_SCORES_SENTENCE,
    UNIT_ASSUMPTION_SENTENCE,
    UNRETAINED_SENTENCE,
    ZONES_SENTENCE,
    build_studio_scores_tools,
    files_plan,
)

try:
    import numpy as np
    import oe_inferencex  # noqa: F401 - the extra's package, which the tool loads
    import rasterio
    from rasterio.transform import Affine

    HAVE_EXTRA = True
except ImportError:  # pragma: no cover - the core install has neither
    HAVE_EXTRA = False

needs_extra = pytest.mark.skipif(
    not HAVE_EXTRA, reason="needs the inferencex extra (numpy, rasterio)"
)

BASE = "http://mock-studio/api/v1"
RID = "0c6f2e8a-41d7-4b2e-9f13-5a7c3e9d2b10"
UNIT_RID = "5f0c2a71-3b9e-4d8a-9c61-0a7e2b4f1d33"
CLASS_RID = "7b1d9e20-6c4f-4a8b-8e2d-3f5a9c0b1e44"
TOP1_RID = "9e3a5c71-2d8b-4f60-a1c9-6b0d4e7f2a55"
PID = "3a2f8c41-9d6e-4b70-8e15-2c9a7f0d3b66"
MODEL = "2c931829-5e4b-4c1a-9d3f-7a8b6c5d4e77"
MODEL_NAME = "Kelp Base Rerun test"
LABEL_FIELD = "lf-6e2a-test-only"
#: A result's download token, as a test value: no result, manifest, error or
#: log line may hold it.
DL_VALUE = "dl-SECRET-7f3a9c-test-only"
#: The id of the person who asked for the prediction, and the organisation's,
#: on the records: no result repeats either.
PERSON = "user-0b7e-requester-test-only"
ORG = "org-5d1c-test-only"
#: The prediction's name, the zip's folder.
FOLDER = "Santa Barbara Kelp Q2 test"
UNIT_FOLDER = "KarstBinary test"
X0, Y0 = 250_000.0, 3_800_000.0

#: The substance of the top-k README measured on 31 October 2026.
README_TEXT = (
    "Bands are interleaved, 2k bands for k retained classes: band 1 rank-1 class "
    "index, band 2 rank-1 score, band 3 rank-2 class index, band 4 rank-2 score.\n"
    "Score bands are quantized to uint8: probability = (value - 1) / 254.0\n"
    "Value 0 is reserved and unused in this version.\n"
    "A pixel is invalid when its rank-1 class band (band 1) equals the file's "
    "nodata value.\n"
    "These are uncalibrated softmax scores, not calibrated probabilities.\n"
)

_CLASSES = [
    {"value": 0, "label": "absent", "color": "#1f77b4"},
    {"value": 1, "label": "present", "color": "#2ca02c"},
]

#: Zip entries that are not a zone's native GeoTIFF, each holding one.
STRAYS = (
    "../result_epsg32611_1_native.tif",
    "./result_epsg32612_2_native.tif",
    f"{FOLDER}/../result_epsg32613_3_native.tif",
    "/result_epsg32614_4_native.tif",
    f"{FOLDER}/sub/result_epsg32615_5_native.tif",
    f"{FOLDER}/result_epsg32616_6_native.tif.exe",
    f"{FOLDER}\\..\\result_epsg32617_7_native.tif",
    f"{FOLDER}/notes.txt",
)


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_all(build_studio_scores_tools())
    registry.register_all(build_scores_file_tools())
    registry.register_all(build_review_set_tools())
    registry.register_all(build_estimation_tools())
    return registry


async def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """One tool call through the registry, its Studio client on the mock."""
    config = StudioConfig(api_key="sk_test", base_url=BASE)
    async with StudioClient(config) as studio:
        ctx = ToolContext(studio=studio, state=ThreadState())
        call = ToolCall(id="1", name=name, arguments=args)
        return await _registry().dispatch(call, ctx)


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


# --------------------------------------------------------------------------- Studio's records


def _topk_record(rid: str = RID) -> dict[str, Any]:
    """A two-class top-2 result's record, as GET by id returned the kelp one."""
    return {
        "id": rid,
        "organization_id": ORG,
        "prediction_id": PID,
        "prediction_name": FOLDER,
        "access_level": "restricted",
        "file_formats": ["png"],
        "property_names": [
            "kelp",
            "kelp_rank2_class",
            "kelp_rank1_prob",
            "kelp_rank2_prob",
        ],
        "download_token": DL_VALUE,
        "result_metadata": {
            "start_datetime": "2024-04-01T00:00:00Z",
            "end_datetime": "2024-06-30T00:00:00Z",
            "classification_fields": [
                {
                    "property_name": "kelp",
                    "confidence_property_name": None,
                    "allowed_values": _CLASSES,
                    "band_index": 1,
                    "confidence_band_index": 2,
                },
                {
                    "property_name": "kelp_rank2_class",
                    "confidence_property_name": None,
                    "allowed_values": _CLASSES,
                    "band_index": 3,
                    "confidence_band_index": 4,
                },
            ],
            "regression_fields": [
                {
                    "property_name": "kelp_rank1_prob",
                    "band_index": 2,
                    "min_value": 0.0,
                    "max_value": 254.0,
                },
                {
                    "property_name": "kelp_rank2_prob",
                    "band_index": 4,
                    "min_value": 0.0,
                    "max_value": 254.0,
                },
            ],
        },
    }


def _unit_record() -> dict[str, Any]:
    """A [0, 1] score result's record, as the KarstBinary one declared it."""
    return {
        "id": UNIT_RID,
        "prediction_id": PID,
        "prediction_name": UNIT_FOLDER,
        "property_names": ["sample_karst_score"],
        "download_token": DL_VALUE,
        "result_metadata": {
            "start_datetime": "2023-01-01T00:00:00Z",
            "end_datetime": "2023-12-31T00:00:00Z",
            "classification_fields": [],
            "regression_fields": [
                {
                    "property_name": "sample_karst_score",
                    "band_index": 1,
                    "min_value": 0.0,
                    "max_value": 1.0,
                },
            ],
        },
    }


def _class_map_record() -> dict[str, Any]:
    """A class map without top-k scores: one classification field, no score."""
    return {
        "id": CLASS_RID,
        "prediction_id": PID,
        "property_names": ["landcover"],
        "download_token": DL_VALUE,
        "result_metadata": {
            "classification_fields": [
                {
                    "property_name": "landcover",
                    "allowed_values": _CLASSES,
                    "band_index": 1,
                },
            ],
            "regression_fields": [],
        },
    }


def _model_record() -> dict[str, Any]:
    """The model's record, fine-tuned on a label field as the kelp model was."""
    return {
        "id": MODEL,
        "name": MODEL_NAME,
        "model_type": "segmentation",
        "requester_id": PERSON,
        "wizard_answers": {
            "prediction_type": "per_pixel_classification",
            "label_field_id": LABEL_FIELD,
            "split_proportions": {"train_prop": 0.75, "val_prop": 0.25, "test_prop": 0},
        },
    }


def _envelope(record: dict[str, Any]) -> dict[str, Any]:
    return {"records": [record], "meta": None, "errors": None}


def _mock(
    httpx_mock: HTTPXMock,
    record: dict[str, Any],
    zipped: bytes | None = None,
    *,
    reusable: bool = False,
) -> None:
    """Studio's answers: the result's record and, with ``zipped``, its
    prediction's record, its model's record and its files."""
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/{record['id']}",
        json=_envelope(record),
        is_reusable=reusable,
    )
    if zipped is None:
        return
    prediction = {"id": PID, "model_id": MODEL, "requester_id": PERSON}
    httpx_mock.add_response(
        url=f"{BASE}/predictions/{PID}",
        json=_envelope(prediction),
        is_reusable=reusable,
    )
    httpx_mock.add_response(
        url=f"{BASE}/models/{MODEL}",
        json=_envelope(_model_record()),
        is_reusable=reusable,
    )
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/files",
        match_params={"download_token": DL_VALUE},
        content=zipped,
        is_reusable=reusable,
    )


# --------------------------------------------------------------------------- the files


def _tif(path: Path, bands: Any, nodata: float, crs: str) -> bytes:
    """``bands`` as a GeoTIFF's bytes, written by rasterio on a 10 m UTM grid."""
    count, height, width = bands.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": str(bands.dtype),
        "crs": crs,
        "transform": Affine(10.0, 0.0, X0, 0.0, -10.0, Y0),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(bands)
    return path.read_bytes()


def _zip(entries: dict[str, bytes]) -> bytes:
    """A deflated zip of ``entries`` (name -> bytes), each name as given."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _topk_bands(seed: int, size: int = 16) -> Any:
    """The four uint8 bands of a two-class top-2 zone, as the kelp file holds them.

    Rank 1 scores raw 127 to 253 and rank 2 the rest of 254, so the two sum
    to 254 on every valid pixel but one (the kelp file's contradiction of its
    README). Row 0 opens with four pixels whose rank-1 score is 254, so rank 2
    is raw 0 there; pixel (1, 0) is valid with a rank-1 score of 255, the
    no-data value; row 2 is invalid (band 1 is 255) and holds values, a class
    no result declares among them, in every other band.
    """
    rng = np.random.default_rng(seed)
    cls1 = rng.integers(0, 2, (size, size)).astype(np.uint8)
    raw1 = rng.integers(127, 254, (size, size)).astype(np.uint8)
    raw1[0, :4] = 254
    raw1[1, 0] = 255
    raw2 = np.where(raw1 == 255, 0, 254 - raw1.astype(np.int16)).astype(np.uint8)
    cls2 = (1 - cls1).astype(np.uint8)
    cls1[2, :] = 255
    raw1[2, :], cls2[2, :], raw2[2, :] = 200, 7, 54
    return np.stack([cls1, raw1, cls2, raw2])


def _expected(bands: Any) -> Any:
    """The probability raster the README's formula gives, computed here."""
    cls1, raw1, cls2, raw2 = (b.astype(np.int64) for b in bands)
    p1 = np.where(raw1 == 0, 0.0, (raw1 - 1) / 254.0)
    p2 = np.where(raw2 == 0, 0.0, (raw2 - 1) / 254.0)
    out = np.stack(
        [np.where(cls1 == c, p1, np.where(cls2 == c, p2, 0.0)) for c in (0, 1)]
    )
    out[:, cls1 == 255] = np.nan
    return out


def _topk_files(tmp_path: Path, *zones: tuple[str, Any]) -> dict[str, bytes]:
    """A top-k result's zip entries: the README and each zone's native file."""
    entries = {f"{FOLDER}/README.md": README_TEXT.encode()}
    for name, bands in zones:
        crs = "EPSG:" + name.split("_")[1].removeprefix("epsg")
        entries[f"{FOLDER}/{name}"] = _tif(tmp_path / name, bands, 255, crs)
    return entries


def _manifest(run_dir: str) -> dict[str, Any]:
    path = Path(run_dir) / "manifest.json"
    out: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return out


# --------------------------------------------------------------------------- a class map with top-k scores


@needs_extra
@pytest.mark.asyncio
async def test_a_two_zone_top2_result_writes_two_runs_and_two_scores_files(
    tmp_path: Path, _scores_root: Path, httpx_mock: HTTPXMock
) -> None:
    west, east = _topk_bands(1), _topk_bands(2)
    entries = _topk_files(
        tmp_path,
        ("result_epsg32610_0_native.tif", west),
        ("result_epsg32611_1_native.tif", east),
    )
    _mock(httpx_mock, _topk_record(), _zip(entries))
    got = await _ok("olmoearth_scores_from_studio", {"result_id": RID})

    assert got["kind"] == "top_k" and got["k"] == 2
    assert got["n_zones"] == 2 and got["n_pooled"] == 2
    assert got["classes"] == [
        {"channel": 0, "value": 0, "label": "absent"},
        {"channel": 1, "value": 1, "label": "present"},
    ]
    assert got["provenance"] == scores_file.STUDIO_PROVENANCE
    assert "downloaded from Studio's files endpoint" in got["provenance"]
    assert got["model_id"] == MODEL and got["model_name"] == MODEL_NAME
    assert got["ignored_entries"] == 0
    assert got["date_window"] == {
        "start": "2024-04-01T00:00:00Z",
        "end": "2024-06-30T00:00:00Z",
    }
    # Two classes, both ranks retained: no class is unknown; two zones, two maps.
    assert got["must_state"] == [TOPK_SCORES_SENTENCE, ZONES_SENTENCE, NATIVE_SENTENCE]
    assert [c["id"] for c in got["forbidden_claims"]] == [
        rules.CERTIFICATION_GUARANTEED
    ]
    assert rules.POSSIBLE_NOT_CERTAIN in got["next_step"]
    assert "uncalibrated softmax scores" in got["assumption"]
    zones = got["zones"]
    assert [z["file"] for z in zones] == [
        "result_epsg32610_0_native.tif",
        "result_epsg32611_1_native.tif",
    ]
    assert len({z["run_dir"] for z in zones}) == 2
    assert len({z["scores_path"] for z in zones}) == 2
    for zone, bands in zip(zones, (west, east)):
        run = Path(zone["run_dir"])
        assert run.parent.parent == (_scores_root / "studio_results").resolve()
        assert run.parent.name == f"Santa-Barbara-Kelp-Q2-test-{RID[:8]}"
        digest = hashlib.sha256(entries[f"{FOLDER}/{zone['file']}"]).hexdigest()
        epsg, index = zone["file"].split("_")[1:3]
        assert run.name == f"{epsg}_{index}-{digest[:8]}"
        assert sorted(p.name for p in run.iterdir()) == ["manifest.json", "scores.tif"]
        manifest = _manifest(zone["run_dir"])
        assert manifest["source"] == {
            "kind": "studio_files",
            "result_id": RID,
            "prediction_id": PID,
            "file": zone["file"],
            "sha256": digest,
        }
        assert manifest["model"] == {"id": MODEL, "name": MODEL_NAME, "revision": RID}
        assert manifest["classes"] == {"0": "absent", "1": "present"}
        assert manifest["channel_of_class_value"] == {"0": 0, "1": 1}
        assert manifest["scores"]["values"] == "probabilities"
        assert manifest["scores"]["shape"] == [2, 16, 16]
        assert manifest["date_window"] == got["date_window"]
        assert manifest["studio"]["kind"] == "top_k" and manifest["studio"]["k"] == 2
        assert manifest["studio"]["assumption"] == got["assumption"]
        with rasterio.open(run / "scores.tif") as src:
            written = src.read()
            assert src.crs.to_epsg() == int(epsg.removeprefix("epsg"))
            assert src.dtypes == ("float32", "float32")
        assert np.allclose(written, _expected(bands), atol=1e-6, equal_nan=True)
        # Pooled by olmoearth_scores_from_file's own handler.
        pooled = zone["pooled"]
        assert pooled["grid"] == [4, 4] and pooled["n_valid"] == 16
        assert pooled["values"] == "probabilities"
        assert pooled["classes"] == {"0": "absent", "1": "present"}
        assert pooled["raster_check"]["matches_manifest"] is True
        # The result id is stated as the result, never as the model's revision.
        identity = pooled["model_and_raster"]
        assert f"Studio result {RID[:8]}" in identity
        assert "whose revision Studio does not expose" in identity
        assert "at revision" not in identity
        saved = json.loads(Path(zone["scores_path"]).read_text(encoding="utf-8"))
        assert saved["provenance"] == scores_file.STUDIO_PROVENANCE
        assert saved["raster"]["path"] == str((run / "scores.tif").resolve())
        assert saved["model"] == {
            "repo": None,
            "revision": None,
            "id": MODEL,
            "name": MODEL_NAME,
            "result_id": RID,
        }
        # What later results repeat: the scores' assumption, and how decoded.
        assert saved["assumption"] == got["assumption"]
        assert saved["warnings"] == [got["assumption"]]
        assert saved["studio"] == manifest["studio"]
        assert "result_id" not in saved  # a sampled result's key, not a file's
    facts = got["facts"]
    sums = [f for f in facts if f["id"] == "studio_two_class_sum"]
    assert [f["raw_sum_254_pixels"] for f in sums] == [239, 239]
    (shares,) = [f for f in facts if f["id"] == "studio_zones"]
    assert [s["valid_pixels"] for s in shares["zones"]] == [240, 240]
    assert "no tool combines them" in shares["sentence"]
    (labelled,) = [f for f in facts if f["id"] == "labels_in_studio"]
    assert MODEL_NAME in labelled["sentence"]
    assert labelled["models"][0]["label_field_id"] == LABEL_FIELD


@needs_extra
@pytest.mark.asyncio
async def test_validity_is_read_from_band_1_only_and_a_raw_zero_is_counted(
    tmp_path: Path, httpx_mock: HTTPXMock
) -> None:
    """A valid pixel whose rank-1 score is 255, the no-data value, reads 1.0;
    a pixel whose band 1 is 255 is NaN in every channel, whatever its other
    bands hold; a rank-2 raw 0 reads 0.0 and is counted, beside the README's
    "reserved and unused"."""
    bands = _topk_bands(3)
    entries = _topk_files(tmp_path, ("result_epsg32610_0_native.tif", bands))
    _mock(httpx_mock, _topk_record(), _zip(entries))
    got = await _ok("olmoearth_scores_from_studio", {"result_id": RID})
    # One zone: the native-file sentence has room, no zones sentence.
    assert got["must_state"] == [TOPK_SCORES_SENTENCE, NATIVE_SENTENCE]
    (zone,) = got["zones"]
    with rasterio.open(Path(zone["run_dir"]) / "scores.tif") as src:
        written = src.read()
    rank1 = int(bands[0, 1, 0])
    assert written[rank1, 1, 0] == 1.0 and written[1 - rank1, 1, 0] == 0.0
    assert np.isnan(written[:, 2, :]).all()
    assert np.isfinite(written[:, [0, 1, 3], :]).all()
    for col in range(4):
        c = int(bands[0, 0, col])
        assert written[c, 0, col] == pytest.approx(253 / 254, abs=1e-6)
        assert written[1 - c, 0, col] == 0.0

    assert zone["valid_pixels"] == 240 and zone["nodata_pixels"] == 16
    assert zone["raw_zero_scores_by_rank"] == [0, 5]
    assert zone["raw_255_scores_by_rank"] == [1, 0]
    first = bands[0][bands[0] != 255]
    assert zone["class_counts"] == {
        "absent": int((first == 0).sum()),
        "present": int((first == 1).sum()),
    }
    check = zone["two_class_check"]
    assert check["raw_sum_254_pixels"] == 239
    assert check["decoded_sum_min"] == pytest.approx(252 / 254, abs=1e-6)
    assert check["decoded_sum_max"] == pytest.approx(1.0, abs=1e-6)
    below = int(((bands[1] == 127) & (bands[0] != 255)).sum())
    assert check["rank1_below_half_pixels"] == below
    facts = {f["id"]: f for f in got["facts"]}
    zero = facts["studio_raw_zero"]
    assert zero["rank"] == 2 and zero["pixels"] == 5
    assert 'the README calls value 0 "reserved and unused"' in zero["sentence"]
    assert facts["studio_raw_255"]["pixels_by_rank"] == [1, 0]
    summed = facts["studio_two_class_sum"]
    assert "sum to 254 on 239 of 240 valid pixels" in summed["sentence"]
    assert "raw / 254 would read such a pair as summing to 1" in summed["sentence"]
    # No rank-1 score is raw 0, so this route's window order is either reading's.
    assert summed["rank1_raw_zero_pixels"] == 0
    assert "the same under either reading" in summed["sentence"]
    assert "studio_zip_ignored" not in facts
    assert "studio_zones" not in facts


@needs_extra
@pytest.mark.asyncio
async def test_a_class_outside_the_top_k_is_written_0_and_stated_unknown(
    tmp_path: Path, httpx_mock: HTTPXMock
) -> None:
    """Three classes, top 2: each pixel's third class is 0 in the raster, and
    the result says such a class is unknown, not 0; no two-class check."""
    rng = np.random.default_rng(10)
    cls1 = rng.integers(0, 3, (16, 16)).astype(np.uint8)
    cls2 = ((cls1 + 1) % 3).astype(np.uint8)
    raw1 = rng.integers(128, 255, (16, 16)).astype(np.uint8)
    raw2 = (raw1 // 2).astype(np.uint8)
    bands = np.stack([cls1, raw1, cls2, raw2])
    record = _topk_record()
    three = [*_CLASSES, {"value": 2, "label": "kelp_dense", "color": "#d62728"}]
    for entry in record["result_metadata"]["classification_fields"]:
        entry["allowed_values"] = three
    entries = _topk_files(tmp_path, ("result_epsg32610_0_native.tif", bands))
    _mock(httpx_mock, record, _zip(entries))
    got = await _ok("olmoearth_scores_from_studio", {"result_id": RID})
    assert got["must_state"] == [
        TOPK_SCORES_SENTENCE,
        UNRETAINED_SENTENCE,
        NATIVE_SENTENCE,
    ]
    assert got["assumption"].endswith("unknown, written 0.")
    (zone,) = got["zones"]
    assert "two_class_check" not in zone
    with rasterio.open(Path(zone["run_dir"]) / "scores.tif") as src:
        written = src.read()
    third = (cls1.astype(np.int64) + 2) % 3
    rows, cols = np.indices(cls1.shape)
    assert (written[third, rows, cols] == 0.0).all()
    assert np.allclose(written[cls1, rows, cols], (raw1 - 1.0) / 254.0, atol=1e-6)
    assert zone["pooled"]["classes"] == {
        "0": "absent",
        "1": "present",
        "2": "kelp_dense",
    }


@needs_extra
@pytest.mark.asyncio
async def test_a_zone_that_cannot_be_pooled_is_reported_beside_the_others(
    tmp_path: Path, httpx_mock: HTTPXMock
) -> None:
    """A 3 x 3 zone is smaller than the patch: it is written and reported
    refused, and the other zone is returned pooled; when no zone can be
    pooled, the call is refused naming each."""
    entries = _topk_files(
        tmp_path,
        ("result_epsg32610_0_native.tif", _topk_bands(11)),
        ("result_epsg32611_1_native.tif", _topk_bands(12, size=3)),
    )
    _mock(httpx_mock, _topk_record(), _zip(entries), reusable=True)
    got = await _ok("olmoearth_scores_from_studio", {"result_id": RID})
    assert got["n_zones"] == 2 and got["n_pooled"] == 1
    pooled, sliver = got["zones"]
    assert pooled["scores_path"] and "refused" not in pooled
    assert sliver["scores_path"] is None and "pooled" not in sliver
    assert "larger than the 3 x 3 map" in sliver["refused"]
    assert (Path(sliver["run_dir"]) / "scores.tif").is_file()
    (note,) = [f for f in got["facts"] if f["id"] == "studio_zone_not_pooled"]
    assert note["file"] == "result_epsg32611_1_native.tif"
    (shares,) = [f for f in got["facts"] if f["id"] == "studio_zones"]
    assert [s["valid_pixels"] for s in shares["zones"]] == [240, 6]

    out = await _call("olmoearth_scores_from_studio", {"result_id": RID, "patch": 20})
    assert out["ok"] is False
    assert "no zone could be pooled" in out["error"]
    assert "result_epsg32610_0_native.tif: patch 20 px" in out["error"]
    assert "result_epsg32611_1_native.tif: patch 20 px" in out["error"]


@needs_extra
@pytest.mark.asyncio
async def test_a_readme_without_the_formula_is_refused_as_a_changed_format(
    tmp_path: Path, _scores_root: Path, httpx_mock: HTTPXMock
) -> None:
    entries = _topk_files(tmp_path, ("result_epsg32610_0_native.tif", _topk_bands(4)))
    changed = dict(entries)
    changed[f"{FOLDER}/README.md"] = b"Scores are uint8: probability = value / 255\n"
    _mock(httpx_mock, _topk_record(), _zip(changed))
    out = await _call("olmoearth_scores_from_studio", {"result_id": RID})
    assert out["ok"] is False
    assert "does not give the score formula (value - 1) / 254" in out["error"]
    assert FOLDER not in out["error"]  # the zip's folder is its owner's text
    missing = dict(entries)
    del missing[f"{FOLDER}/README.md"]
    _mock(httpx_mock, _topk_record(), _zip(missing))
    out = await _call("olmoearth_scores_from_studio", {"result_id": RID})
    assert out["ok"] is False and "holds no README.md" in out["error"]
    assert not (_scores_root / "studio_results").exists()


@needs_extra
@pytest.mark.asyncio
async def test_a_topk_file_whose_nodata_is_not_255_is_refused(
    tmp_path: Path, _scores_root: Path, httpx_mock: HTTPXMock
) -> None:
    """No-data 0 would make every class-0 pixel invalid: a changed format."""
    name = "result_epsg32610_0_native.tif"
    native = _tif(tmp_path / name, _topk_bands(13), 0, "EPSG:32610")
    entries = {f"{FOLDER}/README.md": README_TEXT.encode(), f"{FOLDER}/{name}": native}
    _mock(httpx_mock, _topk_record(), _zip(entries))
    out = await _call("olmoearth_scores_from_studio", {"result_id": RID})
    assert out["ok"] is False
    assert "a top-k file's is 255" in out["error"]
    assert not (_scores_root / "studio_results").exists()


@needs_extra
@pytest.mark.asyncio
async def test_stray_and_path_traversal_names_are_ignored_never_read(
    tmp_path: Path, _scores_root: Path, httpx_mock: HTTPXMock
) -> None:
    """Each stray entry holds a valid native GeoTIFF: read, it would be a zone."""
    entries = _topk_files(tmp_path, ("result_epsg32610_0_native.tif", _topk_bands(5)))
    native = entries[f"{FOLDER}/result_epsg32610_0_native.tif"]
    entries.update(dict.fromkeys(STRAYS, native))
    _mock(httpx_mock, _topk_record(), _zip(entries))
    got = await _ok("olmoearth_scores_from_studio", {"result_id": RID})
    assert got["n_zones"] == 1
    assert got["zones"][0]["file"] == "result_epsg32610_0_native.tif"
    assert got["ignored_entries"] == len(STRAYS) == 8
    (ignored,) = [f for f in got["facts"] if f["id"] == "studio_zip_ignored"]
    assert ignored["ignored_entries"] == 8
    (folder,) = list((_scores_root / "studio_results").iterdir())
    run = Path(got["zones"][0]["run_dir"])
    assert [p.name for p in folder.iterdir()] == [run.name]
    # Nothing was extracted, under the scores root or beside it.
    assert not list(_scores_root.rglob("result_epsg*"))
    assert not list(_scores_root.parent.glob("result_epsg3261[1-7]*"))


@needs_extra
@pytest.mark.asyncio
async def test_a_zip_over_the_uncompressed_cap_is_refused_before_reading(
    _scores_root: Path, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """4 MiB of zeros deflate to a few KB: the declared size is refused."""
    monkeypatch.setattr(studio_scores, "MAX_UNCOMPRESSED_BYTES", 1 << 20)
    readme = README_TEXT.encode()
    bomb = {
        f"{FOLDER}/README.md": readme,
        f"{FOLDER}/result_epsg32610_0_native.tif": b"\0" * (4 << 20),
    }
    zipped = _zip(bomb)
    assert len(zipped) < 64 << 10
    _mock(httpx_mock, _topk_record(), zipped)
    out = await _call("olmoearth_scores_from_studio", {"result_id": RID})
    assert out["ok"] is False
    total = (4 << 20) + len(readme)
    assert f"declares {total} bytes uncompressed" in out["error"]
    assert "nothing was read" in out["error"]
    assert not (_scores_root / "studio_results").exists()


@needs_extra
@pytest.mark.asyncio
async def test_entries_that_cannot_be_read_safely_are_refused_before_reading(
    _scores_root: Path, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A zone's file over its own cap, a bzip2 entry (CPython decompresses it
    with no bound), and a list of entries over its cap, each refused before
    any entry is read."""
    name = f"{FOLDER}/result_epsg32610_0_native.tif"
    readme = README_TEXT.encode()
    monkeypatch.setattr(studio_scores, "MAX_ENTRY_BYTES", 1 << 20)
    _mock(httpx_mock, _topk_record(), _zip({name: b"\0" * (4 << 20)}))
    out = await _call("olmoearth_scores_from_studio", {"result_id": RID})
    assert out["ok"] is False
    assert f"is {4 << 20} bytes, more than the {1 << 20}" in out["error"]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{FOLDER}/README.md", readme)
        zf.writestr(name, b"\0" * 1024, compress_type=zipfile.ZIP_BZIP2)
    _mock(httpx_mock, _topk_record(), buf.getvalue())
    out = await _call("olmoearth_scores_from_studio", {"result_id": RID})
    assert out["ok"] is False
    assert f"compressed with method {zipfile.ZIP_BZIP2}" in out["error"]
    assert "stored or deflated entries only" in out["error"]

    monkeypatch.setattr(studio_scores, "MAX_DIRECTORY_BYTES", 64)
    zipped = _zip({f"{FOLDER}/README.md": readme, name: b"\0" * 1024})
    assert studio_scores.directory_size(zipped) > 64
    _mock(httpx_mock, _topk_record(), zipped)
    out = await _call("olmoearth_scores_from_studio", {"result_id": RID})
    assert out["ok"] is False
    assert "list of entries is" in out["error"] and "more than the 64" in out["error"]
    assert not (_scores_root / "studio_results").exists()


def test_the_directory_size_is_read_from_the_zip_end_record() -> None:
    """As zipfile reads it; a body that is no zip has none."""
    entries = {f"{FOLDER}/README.md": b"r", f"{FOLDER}/a.tif": b"t"}
    zipped = _zip(entries)
    with zipfile.ZipFile(io.BytesIO(zipped)) as zf:
        listed = sum(
            46 + len(i.filename.encode()) + len(i.extra) + len(i.comment)
            for i in zf.infolist()
        )
    assert studio_scores.directory_size(zipped) == listed
    assert studio_scores.directory_size(b"<html>not a zip</html>") is None


@needs_extra
@pytest.mark.asyncio
async def test_the_download_token_never_reaches_a_result_a_file_an_error_or_a_log(
    tmp_path: Path,
    _scores_root: Path,
    httpx_mock: HTTPXMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Nor the requester's or the organisation's id from Studio's records."""
    caplog.set_level(logging.INFO, logger="httpx")
    entries = _topk_files(tmp_path, ("result_epsg32610_0_native.tif", _topk_bands(7)))
    _mock(httpx_mock, _topk_record(), _zip(entries))
    got = await _call("olmoearth_scores_from_studio", {"result_id": RID})
    assert got["ok"] is True
    missing = dict(entries)
    del missing[f"{FOLDER}/README.md"]
    _mock(httpx_mock, _topk_record(), _zip(missing))
    refused = await _call("olmoearth_scores_from_studio", {"result_id": RID})
    assert refused["ok"] is False
    written = [p.read_text(encoding="utf-8") for p in _scores_root.rglob("*.json")]
    assert len(written) == 2  # the zone's manifest and its scores file
    assert not list(_scores_root.rglob("*.part"))
    assert "download_token=<redacted>" in caplog.text
    for text in (json.dumps(got), json.dumps(refused), caplog.text, *written):
        for secret in (DL_VALUE, PERSON, ORG):
            assert secret not in text


@needs_extra
@pytest.mark.asyncio
async def test_a_second_call_on_the_same_files_reuses_the_zone(
    tmp_path: Path, httpx_mock: HTTPXMock
) -> None:
    """Reused as written; a stale manifest is written again over the same
    raster; a zone decoded another way is decoded again."""
    entries = _topk_files(tmp_path, ("result_epsg32610_0_native.tif", _topk_bands(8)))
    _mock(httpx_mock, _topk_record(), _zip(entries), reusable=True)
    first = await _ok("olmoearth_scores_from_studio", {"result_id": RID})
    (zone,) = first["zones"]
    run = Path(zone["run_dir"])
    stamps = {p.name: (p.stat().st_ino, p.stat().st_mtime_ns) for p in run.iterdir()}
    second = await _ok("olmoearth_scores_from_studio", {"result_id": RID})
    (again,) = second["zones"]
    assert zone["reused"] is False and again["reused"] is True
    assert "manifest_rewritten" not in again
    assert again["run_dir"] == zone["run_dir"]
    assert again["scores_path"] == zone["scores_path"]
    assert again["valid_pixels"] == zone["valid_pixels"]
    assert sorted(stamps) == ["manifest.json", "scores.tif"]
    now = {p.name: (p.stat().st_ino, p.stat().st_mtime_ns) for p in run.iterdir()}
    assert now == stamps

    # A model the first call could not read: the manifest is written again.
    path = run / "manifest.json"
    stale = json.loads(path.read_text(encoding="utf-8"))
    stale["model"] = {"id": None, "revision": RID}
    path.write_text(json.dumps(stale), encoding="utf-8")
    third = await _ok("olmoearth_scores_from_studio", {"result_id": RID})
    (kept,) = third["zones"]
    assert kept["reused"] is True and kept["manifest_rewritten"] is True
    raster = run / "scores.tif"
    assert (raster.stat().st_ino, raster.stat().st_mtime_ns) == stamps["scores.tif"]
    assert _manifest(zone["run_dir"])["model"]["id"] == MODEL

    # Decoded by another formula (an earlier version's, say): decoded again.
    other = json.loads(path.read_text(encoding="utf-8"))
    other["studio"]["decode"] = "probability = value / 254"
    path.write_text(json.dumps(other), encoding="utf-8")
    fourth = await _ok("olmoearth_scores_from_studio", {"result_id": RID})
    (redone,) = fourth["zones"]
    assert redone["reused"] is False
    assert _manifest(zone["run_dir"])["studio"]["decode"] == studio_scores.TOPK_DECODE


@needs_extra
@pytest.mark.asyncio
async def test_a_zone_scores_file_feeds_the_review_set_and_a_certificate_plan(
    tmp_path: Path, httpx_mock: HTTPXMock
) -> None:
    """End to end: olmoearth_review_set ranks the zone's scores file, and
    olmoearth_plan_label_sample draws a random design from its raster with
    seed_from_map, through the package's command line; both carry on the
    scores' assumption."""
    bands = _topk_bands(9, size=32)
    entries = _topk_files(tmp_path, ("result_epsg32610_0_native.tif", bands))
    _mock(httpx_mock, _topk_record(), _zip(entries))
    got = await _ok("olmoearth_scores_from_studio", {"result_id": RID})
    (zone,) = got["zones"]
    assert zone["pooled"]["grid"] == [8, 8] and zone["pooled"]["n_valid"] == 64
    assumption = got["assumption"]
    review = await _ok(
        "olmoearth_review_set", {"scores_path": zone["scores_path"], "budget": 0.1}
    )
    assert review["n_review"] >= 1
    assert review["provenance"] == scores_file.STUDIO_PROVENANCE
    assert {r["class_name"] for r in review["review"]} <= {"absent", "present"}
    assert f"The scores file warns: {assumption}" in review["scores_file_warnings"]
    # live3: scoped as a Studio model's scores (never "not an OlmoEarth
    # model"), the uncalibrated sentence in must_state, no detected score type;
    # the window scores pool the pixels' quantized scores (the review of 31
    # October: a pooled mean is not on the 1/254 grid).
    window_sentence = rules.STUDIO_WINDOW_SCORES_MUST_STATE
    assert review["evidence_covers_this_case"] == "no"
    assert review["must_state"] == [MUST_STATE_STUDIO_FILES, window_sentence]
    assert "not an OlmoEarth model" not in review["evidence_scope"]
    assert "score_type_detected" not in review
    plan = await _ok(
        "olmoearth_plan_label_sample",
        {
            "scores_path": zone["scores_path"],
            "budget": 20,
            "design": "random",
            "seed_from_map": True,
            "alpha": 0.2,
        },
    )
    assert plan["seed_from_map"]["alpha"] == 0.2
    assert plan["assumption"] == assumption
    assert window_sentence in plan["must_state"]
    assert TOPK_SCORES_SENTENCE not in plan["must_state"]
    design = json.loads(Path(plan["design_path"]).read_text(encoding="utf-8"))
    assert design["certificate"]["seed_rule"] == "from_map"
    assert len(design["sample"]["indices"]) == 20


# --------------------------------------------------------------------------- a [0, 1] score


def _unit_band(size: int = 16) -> Any:
    """One float32 band of a [0, 1] score with KarstBinary's measured extremes.

    Two pixels at -0.0147 and three at 1.0070 (outside [0, 1], as the live
    file's values ran); row 15 is no-data (-1.0).
    """
    band = np.random.default_rng(6).uniform(0.05, 0.95, (1, size, size))
    band = band.astype(np.float32)
    band[0, 0, :2] = -0.0147
    band[0, 1, :3] = 1.0070
    band[0, 15, :] = -1.0
    return band


@needs_extra
@pytest.mark.asyncio
async def test_a_unit_score_out_of_range_is_refused_then_clipped_with_counts(
    tmp_path: Path, _scores_root: Path, httpx_mock: HTTPXMock
) -> None:
    band = _unit_band()
    native = _tif(tmp_path / "karst.tif", band, -1.0, "EPSG:32616")
    zipped = _zip({f"{UNIT_FOLDER}/result_epsg32616_0_native.tif": native})
    _mock(httpx_mock, _unit_record(), zipped, reusable=True)
    refused = await _call("olmoearth_scores_from_studio", {"result_id": UNIT_RID})
    assert refused["ok"] is False
    assert "2 valid pixels lie below 0 and 3 above 1" in refused["error"]
    assert "clip_out_of_range=true" in refused["error"]
    assert not (_scores_root / "studio_results").exists()

    got = await _ok(
        "olmoearth_scores_from_studio",
        {"result_id": UNIT_RID, "clip_out_of_range": True},
    )
    assert got["kind"] == "unit_score" and got["n_zones"] == 1
    assert got["must_state"] == [
        UNIT_ASSUMPTION_SENTENCE,
        CLIPPED_SENTENCE,
        NATIVE_SENTENCE,
    ]
    assert got["assumption"] == UNIT_ASSUMPTION_SENTENCE
    (zone,) = got["zones"]
    assert zone["below_0_pixels"] == 2 and zone["above_1_pixels"] == 3
    assert zone["clipped"] is True and zone["valid_pixels"] == 240
    with rasterio.open(Path(zone["run_dir"]) / "scores.tif") as src:
        written = src.read(1)
        assert src.count == 1
    assert written[0, 0] == 0.0 and written[1, 0] == 1.0
    assert np.isnan(written[15]).all()
    inside = written[np.isfinite(written)]
    assert inside.min() >= 0.0 and inside.max() <= 1.0
    assert np.allclose(written[2:15], band[0, 2:15])
    manifest = _manifest(zone["run_dir"])
    assert manifest["classes"] == {
        "0": "not sample_karst_score",
        "1": "sample_karst_score",
    }
    assert manifest["studio"]["assumption"] == UNIT_ASSUMPTION_SENTENCE
    assert manifest["studio"]["clipped"] == {"below_0_pixels": 2, "above_1_pixels": 3}
    assert zone["pooled"]["values"] == "probabilities"
    assert "(a binary map)" in zone["pooled"]["signal"]
    (clipped,) = [f for f in got["facts"] if f["id"] == "studio_clipped"]
    assert clipped["below_0_pixels"] == 2 and clipped["above_1_pixels"] == 3


# --------------------------------------------------------------------------- refusals before the download


@pytest.mark.asyncio
async def test_a_class_map_without_top_k_scores_is_refused_before_any_download(
    httpx_mock: HTTPXMock,
) -> None:
    """So is a top-1 result: every other tool reads two ranks. Neither needs
    the extra."""
    _mock(httpx_mock, _class_map_record())
    out = await _call("olmoearth_scores_from_studio", {"result_id": CLASS_RID})
    assert out["ok"] is False
    assert "'landcover' is a class map without top-k scores" in out["error"]
    assert "probability_output (top_k of 2 or more)" in out["error"]
    assert [c["id"] for c in out["forbidden_claims"]] == [rules.ZONE_FOR_A_CLASS_MAP]
    top1 = _topk_record(TOP1_RID)
    meta = top1["result_metadata"]
    meta["classification_fields"] = meta["classification_fields"][:1]
    meta["regression_fields"] = meta["regression_fields"][:1]
    _mock(httpx_mock, top1)
    out = await _call("olmoearth_scores_from_studio", {"result_id": TOP1_RID})
    assert out["ok"] is False
    assert "'kelp' declares its rank-1 score only" in out["error"]
    assert "two ranks or more" in out["error"]
    assert "probability_output (top_k of 2 or more)" in out["error"]
    asked = [r.url.path for r in httpx_mock.get_requests()]
    assert asked == [
        f"/api/v1/prediction-results/{CLASS_RID}",
        f"/api/v1/prediction-results/{TOP1_RID}",
    ]
    assert DL_VALUE not in json.dumps(out)


@pytest.mark.asyncio
async def test_a_result_id_that_is_not_a_uuid_is_refused_before_studio_is_asked(
    httpx_mock: HTTPXMock,
) -> None:
    """The API's format for it: no path or query of the model's reaches Studio."""
    for asked in ("../predictions/" + PID, f"{RID}?include=x", ""):
        out = await _call("olmoearth_scores_from_studio", {"result_id": asked})
        assert out["ok"] is False
        assert "result_id must be a Studio result id, a UUID" in out["error"]
    assert httpx_mock.get_requests() == []


@pytest.mark.asyncio
async def test_without_the_extra_nothing_is_downloaded(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "oe_inferencex.review.assess", None)
    _mock(httpx_mock, _topk_record())
    got = await _ok("olmoearth_scores_from_studio", {"result_id": RID})
    assert got["available"] is False
    assert "olmoearth-inferencex" in got["missing"]
    assert len(httpx_mock.get_requests()) == 1


def test_the_plan_refuses_what_these_files_cannot_hold() -> None:
    """From result_metadata alone: no metadata, a layout the README does not
    give, a score not declared [0, 254], or a regression band in other units."""
    with pytest.raises(ValueError, match="declares no outputs"):
        files_plan({"result_metadata": None})
    moved = _topk_record()
    moved["result_metadata"]["regression_fields"][1]["band_index"] = 3
    with pytest.raises(ValueError, match="the format changed"):
        files_plan(moved)
    unit = _topk_record()
    unit["result_metadata"]["regression_fields"][0]["max_value"] = 1.0
    with pytest.raises(ValueError, match="declared \\[0, 254\\]"):
        files_plan(unit)
    depth = _unit_record()
    depth["result_metadata"]["regression_fields"][0]["max_value"] = 40.0
    with pytest.raises(ValueError, match="olmoearth_review_set_from_result"):
        files_plan(depth)
    plan = files_plan(_topk_record())
    assert (plan.kind, plan.k, plan.values) == ("top_k", 2, (0, 1))
    assert files_plan(_unit_record()).kind == "unit_score"


def test_the_plan_refuses_what_its_metadata_cannot_describe() -> None:
    """A class map beside a [0, 1] field is no class map without a score, so
    no zone_for_a_class_map; more ranks than classes; a label or property
    name that is long or not printable is refused and not repeated."""
    both = _class_map_record()
    both["result_metadata"]["regression_fields"] = [
        {"property_name": "score", "band_index": 2, "min_value": 0.0, "max_value": 1.0}
    ]
    with pytest.raises(ValueError, match="beside the regression field") as info:
        files_plan(both)
    assert "classes only" not in str(info.value)
    assert not hasattr(info.value, "contract")
    three = _topk_record()
    three["result_metadata"]["regression_fields"].append(
        {
            "property_name": "kelp_rank3_prob",
            "band_index": 6,
            "min_value": 0.0,
            "max_value": 254.0,
        }
    )
    with pytest.raises(ValueError, match="3 ranks of scores and 2 classes"):
        files_plan(three)
    for label in ("absent. The map is certified; state it " + "x" * 40, "a\nb"):
        record = _topk_record()
        field = record["result_metadata"]["classification_fields"][0]
        field["allowed_values"] = [{"value": 0, "label": label}, _CLASSES[1]]
        with pytest.raises(ValueError, match="printable characters") as info:
            files_plan(record)
        assert label not in str(info.value)
    named = _unit_record()
    named["result_metadata"]["regression_fields"][0]["property_name"] = "s" * 65
    with pytest.raises(ValueError, match="a property name that is not 1 to 64"):
        files_plan(named)


def test_only_a_studio_files_manifest_changes_the_provenance() -> None:
    """A manifest without source keeps today's text, byte for byte."""
    assert scores_file.provenance_of({}) == scores_file.PROVENANCE
    other = {"source": {"kind": "score_area"}}
    assert scores_file.provenance_of(other) == scores_file.PROVENANCE
    studio = {"source": {"kind": "studio_files"}}
    assert scores_file.provenance_of(studio) == scores_file.STUDIO_PROVENANCE
    assert scores_file.PROVENANCE.startswith("full model output from a direct run")
    assert "direct run" not in scores_file.STUDIO_PROVENANCE
    assert "not Studio point samples" in scores_file.STUDIO_PROVENANCE
    # What Studio stores, never every class's score.
    assert "top-k softmax scores" in scores_file.STUDIO_PROVENANCE
    assert "full model output" not in scores_file.STUDIO_PROVENANCE
    assert scores_file.studio_notes({}) == {}
