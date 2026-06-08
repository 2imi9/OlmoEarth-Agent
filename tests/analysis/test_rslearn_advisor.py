# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the rslearn config advisor + validator (skill #17)."""

from __future__ import annotations

from olmoearth_agent.analysis.rslearn_advisor import (
    EMBEDDING_SIZES,
    compose,
    diagnose,
    recommend,
    validate_config,
)

# --------------------------------------------------------------------------- #
# recommend
# --------------------------------------------------------------------------- #


def test_recommend_routes_segmentation_from_plain_language() -> None:
    rec = recommend(goal="map land cover at every pixel", num_classes=10)
    assert rec["task"] == "segmentation"
    assert rec["data"]["label_layer_type"] == "raster"
    # OlmoEarth dense decoder is Upsample -> Conv (the proven data-prep pattern)
    assert "Conv" in " ".join(rec["model"]["decoder"])
    assert rec["model"]["head"].endswith("SegmentationHead")
    assert rec["model"]["out_channels"] == 10  # = num_classes


def test_recommend_detection_reserves_background_class() -> None:
    rec = recommend(goal="detect and count wind turbines", num_classes=3)
    assert rec["task"] == "detection"
    # Faster R-CNN reserves class 0 for background -> out_channels = num_classes + 1
    assert rec["model"]["out_channels"] == 4
    assert "FasterRCNN" in " ".join(rec["model"]["decoder"])
    assert rec["data"]["label_layer_type"] == "vector"


def test_recommend_per_pixel_regression_scale_factor_from_range() -> None:
    rec = recommend(goal="estimate biomass value at each pixel", label_range=[0, 100])
    assert rec["task"] == "per_pixel_regression"
    assert rec["task_config"]["scale_factor"] == 0.01  # 1/100 -> train on ~[0,1]
    assert rec["model"]["head"].endswith("PerPixelRegressionHead")


def test_recommend_classification_uses_pooling_and_feature_vector() -> None:
    rec = recommend(goal="classify the whole scene image into a type", num_classes=3)
    assert rec["task"] == "classification"
    assert rec["model"]["feature_type"] == "FeatureVector"
    assert "PoolingDecoder" in " ".join(rec["model"]["decoder"])


def test_recommend_model_size_scales_with_samples() -> None:
    assert recommend(goal="segment crops", num_classes=4, num_samples=50)["model"]["encoder"]["model_size"] == "tiny"
    big = recommend(goal="segment crops", num_classes=4, num_samples=5000)["model"]["encoder"]
    assert big["model_size"] == "base"
    assert big["embedding_dim"] == EMBEDDING_SIZES["base"]


def test_recommend_unknown_goal_asks_for_clarity() -> None:
    rec = recommend(goal="do something cool with satellites")
    assert rec["task"] is None
    assert "tasks" in rec  # offers the menu of tasks instead of guessing


def test_recommend_missing_num_classes_is_asked_for() -> None:
    rec = recommend(goal="segment land cover")  # no num_classes
    assert "num_classes" in rec["ask_for"]


# --------------------------------------------------------------------------- #
# validate_config
# --------------------------------------------------------------------------- #


def _good_dataset() -> dict:
    return {
        "layers": {
            "sentinel2": {
                "type": "raster",
                "band_sets": [{"dtype": "uint8", "bands": ["B04", "B03", "B02"]}],
                "data_source": {
                    "class_path": "rslearn.data_sources.gcp_public_data.Sentinel2",
                    "init_args": {"sort_by": "cloud_cover"},
                },
            },
            "label": {
                "type": "raster",
                "band_sets": [{"dtype": "int32", "bands": ["classes"]}],
                "is_target": True,
            },
        }
    }


def _model(in_channels: int = 128, out_channels: int = 10, num_classes: int = 10) -> dict:
    return {
        "model": {
            "init_args": {
                "encoder": [
                    {"class_path": "rslearn.models.olmoearth_pretrain.model.OlmoEarth", "init_args": {"model_id": "OLMOEARTH_V1_NANO"}}
                ],
                "decoder": [
                    {"class_path": "rslearn.models.unet.UNetDecoder", "init_args": {"in_channels": in_channels, "out_channels": out_channels}},
                    {"class_path": "rslearn.train.tasks.segmentation.SegmentationHead"},
                ],
                "task": {"class_path": "rslearn.train.tasks.segmentation.SegmentationTask", "init_args": {"num_classes": num_classes}},
                "inputs": {"image": {"layers": ["sentinel2"], "bands": ["B04", "B03", "B02"]}},
            }
        }
    }


def test_validate_clean_config_is_ok() -> None:
    res = validate_config(_good_dataset(), _model())
    assert res["ok"] is True
    assert res["errors"] == []
    assert "encoder->decoder channel match" in res["checks"]


def test_validate_catches_encoder_decoder_channel_mismatch() -> None:
    # Nano emits 128 but the decoder declares in_channels=768.
    res = validate_config(_good_dataset(), _model(in_channels=768))
    assert res["ok"] is False
    assert any("128" in e and "in_channels" in e for e in res["errors"])


def test_validate_catches_out_channels_vs_num_classes() -> None:
    res = validate_config(_good_dataset(), _model(out_channels=5, num_classes=10))
    assert res["ok"] is False
    assert any("out_channels" in e and "10 classes" in e for e in res["errors"])


def test_validate_catches_task_label_type_mismatch() -> None:
    ds = _good_dataset()
    ds["layers"]["label"]["type"] = "vector"  # but SegmentationTask needs raster
    res = validate_config(ds, _model())
    assert res["ok"] is False
    assert any("raster target" in e for e in res["errors"])


def test_validate_catches_band_not_in_layer() -> None:
    m = _model()
    m["model"]["init_args"]["inputs"]["image"]["bands"] = ["R", "G", "B"]  # not in sentinel2
    res = validate_config(_good_dataset(), m)
    assert res["ok"] is False
    assert any("does not exist in dataset layer" in e for e in res["errors"])


def test_validate_warns_on_sort_by_typo() -> None:
    ds = _good_dataset()
    ds["layers"]["sentinel2"]["data_source"]["init_args"]["sort_by"] = "cloudcover"  # typo
    res = validate_config(ds, _model())
    assert any("sort_by" in w for w in res["warnings"])


def test_validate_detection_background_class_warning() -> None:
    m = _model(out_channels=3, num_classes=3)
    m["model"]["init_args"]["task"]["class_path"] = "rslearn.train.tasks.detection.DetectionTask"
    # detection labels are vector
    ds = _good_dataset()
    ds["layers"]["label"]["type"] = "vector"
    res = validate_config(ds, m)
    assert any("background" in w for w in res["warnings"])


def test_validate_empty_input_is_a_warning_not_a_crash() -> None:
    res = validate_config(None, None)
    assert res["ok"] is True
    assert res["warnings"]  # nudges to pass configs


# --------------------------------------------------------------------------- #
# compose
# --------------------------------------------------------------------------- #


def test_compose_segmentation_emits_valid_yaml() -> None:
    import yaml

    res = compose(task="segmentation", model_size="base", num_classes=7)
    assert res["ok"] is True
    # the emitted model.yaml round-trips through a YAML parser
    cfg = yaml.safe_load(res["yaml"])
    assert cfg["model"]["init_args"]["model"]["class_path"].endswith("MultiTaskModel")
    decs = cfg["model"]["init_args"]["model"]["init_args"]["decoders"]["segment"]
    paths = [d["class_path"] for d in decs]
    assert any(p.endswith("Upsample") for p in paths)
    conv = next(d for d in decs if d["class_path"].endswith("Conv"))
    assert conv["init_args"]["in_channels"] == EMBEDDING_SIZES["base"]
    assert conv["init_args"]["out_channels"] == 7  # = num_classes
    assert cfg["data"]["init_args"]["task"]["init_args"]["tasks"]["segment"]["class_path"].endswith("SegmentationTask")


def test_compose_classification_pools_and_uses_vector_label() -> None:
    res = compose(task="classification", num_classes=3, class_names=["a", "b", "c"])
    cfg = res["config"]
    decs = cfg["model"]["init_args"]["model"]["init_args"]["decoders"]["classify"]
    assert any(d["class_path"].endswith("PoolingDecoder") for d in decs)
    assert cfg["data"]["init_args"]["inputs"]["label"]["data_type"] == "vector"


def test_compose_detection_is_guided_not_emitted() -> None:
    res = compose(task="detection", num_classes=3)
    assert res["ok"] is False
    assert "Faster R-CNN" in res["note"]


def test_compose_regression_without_scale_factor_asks() -> None:
    res = compose(task="regression")
    assert any("scale_factor" in a for a in res["ask_for"])


# --------------------------------------------------------------------------- #
# diagnose
# --------------------------------------------------------------------------- #


def test_diagnose_zero_windows() -> None:
    res = diagnose(summary={"layers": {"sentinel2": {"windows_prepared": 0, "windows_rejected": 0}}})
    assert res["ok"] is False
    assert any("0 windows" in d for d in res["diagnosis"])
    assert any("--start" in f or "time range" in f for f in res["fixes"])


def test_diagnose_no_scenes_from_log() -> None:
    res = diagnose(log_text="ERROR: no scenes found for window seattle in 2024-06..2024-06")
    assert any("no scenes" in d.lower() for d in res["diagnosis"])


def test_diagnose_crs_hint() -> None:
    res = diagnose(log_text="windows materialized off-target; EPSG:4326 vs UTM zone mismatch")
    assert any("CRS" in d or "crs" in d.lower() for d in res["diagnosis"])
    assert any("src_crs" in f for f in res["fixes"])


def test_diagnose_clean_returns_no_known_pattern() -> None:
    res = diagnose(log_text="prepare complete: 128 windows prepared, 0 rejected")
    assert res["ok"] is True
    assert any("No known failure" in d for d in res["diagnosis"])
