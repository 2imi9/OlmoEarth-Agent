# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The property glyph's fields: months in UTC, words, steps, and a model's
output, size, sources and months, never a guess and never a whole record."""

from __future__ import annotations

from olmoearth_agent.mcp_app import glyph

APR_TO_JUN = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]


def test_months_cover_any_month_the_range_touches_in_utc() -> None:
    assert (
        glyph.months_covered("2024-04-01T00:00:00Z", "2024-06-30T00:00:00Z")
        == APR_TO_JUN
    )
    # An end at 00:00 on the first of a month is exclusive.
    assert (
        glyph.months_covered("2024-04-01T00:00:00Z", "2024-07-01T00:00:00Z")
        == APR_TO_JUN
    )
    # A zone offset is read in UTC: 23:00 at -02:00 is 01:00 the next day.
    assert glyph.months_covered("2024-03-31T23:00:00-02:00", "2024-04-02") == [
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    ]  # fmt: skip
    # Across the year end, and twelve months or more.
    assert glyph.months_covered("2023-11-15", "2024-02-10") == [
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
    ]  # fmt: skip
    assert glyph.months_covered("2019-03-01", "2024-02-01") == [1] * 12


def test_unreadable_ranges_are_not_recorded_not_empty() -> None:
    for start, end in [
        (None, "2024-01-01"),
        ("2024-05-01", "2024-04-01"),
        ("soon", "later"),
    ]:
        assert glyph.months_covered(start, end) is None
        assert glyph.year_text(start, end) is None
        assert glyph.span_text(start, end) is None


def test_years_and_span_in_words() -> None:
    assert glyph.year_text("2024-04-01", "2024-07-01") == "2024"
    assert glyph.year_text("2023-11-01", "2024-03-01") == "2023-24"
    assert glyph.year_text("2019-01-01", "2024-06-01") == "2019-2024"
    assert glyph.span_text("2024-04-03", "2024-04-20") == "Apr 2024"
    assert glyph.span_text("2024-04-01", "2024-07-01") == "Apr to Jun 2024"
    assert glyph.span_text("2023-11-01", "2024-03-01") == "Nov 2023 to Feb 2024"


def test_prediction_glyph_reads_its_requested_range_only() -> None:
    out = glyph.prediction_glyph(
        {"start_time": "2024-04-01T00:00:00Z", "end_time": "2024-07-01T00:00:00Z",
         "requester_id": "u1", "result": {"download_token": "t"}}
    )  # fmt: skip
    assert out == {"months": APR_TO_JUN, "years": "2024", "span": "Apr to Jun 2024"}


def test_project_ring_steps_are_fixed() -> None:
    def pred(start: str, end: str) -> dict[str, str]:
        return {"start_time": start, "end_time": end}

    records = [pred("2024-01-01", "2024-02-01")]  # Jan: 1
    records += [pred("2024-03-01", "2024-04-01")] * 3  # Mar: 3 -> step 2
    records += [pred("2023-05-01", "2023-06-01")] * 5  # May: 5 -> step 3
    records += [{"start_time": None}]  # undated: counted, not drawn
    out = glyph.project_glyph(records, limit=200)
    assert out["months"] == [1, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0]
    assert (out["count"], out["dated"], out["capped"]) == (10, 9, False)
    assert glyph.project_glyph(records, limit=10)["capped"] is True
    # Predictions with no readable dates: "not recorded", not "covers no month".
    undated = glyph.project_glyph([{"start_time": None}, {}], limit=200)
    assert undated["months"] is None and (undated["count"], undated["dated"]) == (2, 0)
    assert glyph.project_glyph([], limit=200)["months"] == [0] * 12


def test_model_glyph_from_wizard_answers() -> None:
    record = {
        "id": "m1",
        "name": "Crops",
        "model_type": "fine_tuned",
        "requester_id": "u1",
        "checkpoint_token": "secret",
        "data_selectors": [{"intersects_geometry": {"type": "Point"}}],
        "wizard_answers": {
            "prediction_type": "per_pixel_classification",
            "encoder_variant": "tiny",
            "foundation_model_version": "1.2",
            "imagery_sources": [
                "landsat_8_9_c2_l1",
                "sentinel1_rtc",
                "sentinel2_l2a",
                "nisar",
            ],
            "temporal_context": "period",
            "period_duration": 3,
            "start_month_constraints": ["Apr"],
        },
    }
    out = glyph.model_glyph(record)
    assert set(out) == {
        "id", "name", "grain", "kind", "size", "version", "sensors",
        "context_mode", "months", "context",
    }  # fmt: skip
    assert (out["grain"], out["kind"], out["size"], out["version"]) == (
        "pixel",
        "classes",
        "Tiny",
        "v1.2",
    )
    assert out["sensors"] == ["S2", "S1", "L8/9"]  # the badge order; NISAR gets none
    assert out["months"] == APR_TO_JUN
    assert out["context"] == "3-month period, starts Apr"


def test_model_glyph_never_guesses() -> None:
    out = glyph.model_glyph({"id": "m2", "wizard_answers": {"encoder_variant": "huge"}})
    assert (out["grain"], out["kind"], out["size"], out["version"]) == (
        None,
        None,
        None,
        None,
    )
    assert out["sensors"] == [] and out["months"] is None and out["context"] is None
    embeddings = glyph.model_glyph({"model_type": "embeddings", "wizard_answers": {}})
    assert embeddings["kind"] == "embeddings"
    window = glyph.model_glyph(
        {"wizard_answers": {"prediction_type": "per_window_regression"}}
    )
    assert (window["grain"], window["kind"]) == ("window", "values")
    # spatial_context is the surrounding area, not the output's grain.
    patch = glyph.model_glyph(
        {
            "wizard_answers": {
                "prediction_type": "regression",
                "spatial_context": "64_pixels",
            }
        }
    )
    assert (patch["grain"], patch["kind"]) == (None, "values")
    assert (
        glyph.model_glyph({"wizard_answers": {"temporal_context": 5}})["context_mode"]
        is None
    )


def test_model_months_follow_the_context() -> None:
    def months(**answers: object) -> dict[str, object]:
        return glyph.model_glyph({"wizard_answers": answers})

    moment = months(temporal_context="single_moment", period_duration=3)
    assert moment["months"] is None and moment["context"] == "single moment"
    assert moment["context_mode"] == "moment"
    anytime = months(
        temporal_context="period", period_duration=2, start_month_constraints=[]
    )
    assert (
        anytime["months"] == [1] * 12
        and anytime["context"] == "2-month period, any start month"
    )
    wrap = months(
        temporal_context="period", period_duration=2, start_month_constraints=["12"]
    )
    assert wrap["months"] == [1] + [0] * 10 + [1]
    # A twelve-month period keeps its start months in words.
    year = months(
        temporal_context="period",
        period_duration=12,
        start_month_constraints=["3", "4"],
    )
    assert (
        year["months"] == [1] * 12
        and year["context"] == "12-month period, starts Mar or Apr"
    )
    # Start months not read, or in a shape not known: no months, said in words.
    for constraint in (["?"], "3,4", {"months": [3, 4]}, 4):
        unread = months(
            temporal_context="period",
            period_duration=3,
            start_month_constraints=constraint,
        )
        assert unread["months"] is None
        assert unread["context"] == "3-month period, start months not recorded"
    # Exact spellings only: a word that merely starts like a month is not read.
    for word in (["Junk"], ["Mayor"], ["13"], ["\N{SUPERSCRIPT TWO}"]):
        odd = months(
            temporal_context="period", period_duration=3, start_month_constraints=word
        )
        assert odd["months"] is None, word
    spelled = months(
        temporal_context="period",
        period_duration=1,
        start_month_constraints=["june", "04"],
    )
    assert spelled["context"] == "1-month period, starts Apr or Jun"
    nothing = months(
        temporal_context="period", period_duration=3, start_month_constraints=None
    )
    assert nothing["context"] == "3-month period, any start month"
    for duration in (None, 0, 13, True):
        out = months(temporal_context="period", period_duration=duration)
        assert (
            out["months"] is None
            and out["context"] == "a period of a length not recorded"
        )
