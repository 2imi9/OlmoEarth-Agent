# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-a (17 Nov 2026), track CHECK, on the MCP App, as
specs/r5-a-round.md gives it. The owner's first live run picked a result that
is one score from 0 to 1 in a zone over the size limit, and the Check card
answered with a raw error.

- K1: a 0-to-1 score zone over the old limit gets a Check card (the real
  tools on a mocked Studio, the limits set small so the zone is read in
  strips): one part, counts offered, the rule gate, the cut line;
- K2: the card reads the result with ``clip_out_of_range``; its Details say
  how many pixels were set into 0 to 1, per part, and the model's
  ``must_state`` keeps the tool's sentence on it; a score whose values run
  to -1 or to 2 is not read as a 0-to-1 score and gets no offer,
  however few its pixels, and one within them is offered however many;
- K3: a score map says its cut: one read-only line on the face, the same
  line among the Details' notes and first in the model's ``must_state``;
  "above 0.5", as the pooling counts it; no other map gets the line;
- K4: a card's error has no exception name (a ``KeyError``, whose own text
  is only a key, is put in words), and the hint written for the model
  reaches a card's view data only with the sign-in; the model's text and
  the stored draw record keep the name (``check.FINAL_ERRORS`` reads it).

The server's words are read through its own functions and the views as text,
as ``views.view_html()`` serves them (no test here runs the scripts).
"""

from __future__ import annotations

import copy
import io
import json
import zipfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from pytest_httpx import HTTPXMock

pytest.importorskip("mcp")
from mcp import types  # noqa: E402

from olmoearth_agent.mcp_app import auth, check, server, views  # noqa: E402
from olmoearth_agent.tools import scores_file, studio_scores  # noqa: E402

try:
    import numpy as np
    import oe_inferencex  # noqa: F401 - the extra's package, which the tools load
    import rasterio
    from rasterio.transform import Affine

    HAVE_EXTRA = True
except ImportError:  # pragma: no cover - the core install has neither
    HAVE_EXTRA = False

needs_extra = pytest.mark.skipif(
    not HAVE_EXTRA, reason="needs the inferencex extra (numpy, rasterio)"
)

BASE = "http://mock-studio/api/v1"
#: A Studio key, as a test value.
KEY = "sk_test_r5a_key_789"
#: A result's download token, as a test value.
DOWNLOAD = "dl-test-value-r5a"
R = "11111111-2222-4333-8444-555555555555"
P = "99999999-8888-4777-8666-555555555555"
MODEL = "2c931829-5e4b-4c1a-9d3f-7a8b6c5d4e77"
ZONE = "result_epsg32616_0_native.tif"
ZONES = ("result_epsg32610_0_native.tif", "result_epsg32733_1_native.tif")
#: The score's property name, as Studio's record declares it.
NAME = "sample_karst_score"

#: K3: the cut line, for a score named :data:`NAME`.
CUT = (
    "This map gives each pixel a score from 0 to 1, not a yes or no. This check "
    'counts a pixel as "sample_karst_score" when its score is above 0.5.'
)

#: K2: the Details line for five pixels set into 0 to 1.
FIVE_CLIPPED = (
    "5 pixels had a score outside 0 to 1 and were set to the nearest end, 0 or 1."
)

#: The registry's hint on a failed tool, written for the model.
MODEL_HINT = (
    "The tool ran but failed. Check that referenced ids exist (discover them "
    "with search/list tools) and that argument values are valid; do not retry "
    "with identical arguments."
)

CANNED = {
    "offer": [
        {"labels": 20, "width": 0.2, "sentence": "Twenty labels.", "rate_note": "r50"},
        {"labels": 40, "width": 0.1, "sentence": "Forty labels.", "rate_note": "r50"},
    ],
    "refused": None,
}

Plan = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@pytest.fixture(autouse=True)
def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The write root (scores, outputs and the check records): the test's own."""
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(out))
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(out))
    monkeypatch.delenv(auth.KEY_ENV, raising=False)
    monkeypatch.setenv(auth.KEY_FILE_ENV, str(tmp_path / "cfg" / "studio.env"))
    return out


@pytest.fixture
def canned_offers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Counts of 20 and 40 offered for every zone, without the package."""

    def fake(n_valid: int, widths: Any = check.CHECK_WIDTHS) -> dict[str, Any]:
        """A fresh copy of :data:`CANNED`."""
        return copy.deepcopy(CANNED)

    monkeypatch.setattr(check, "offers", fake)


def _app() -> server.AppServer:
    return server.AppServer(None, keys=auth.KeyStore(), base_url=BASE)


def _view(result: types.CallToolResult) -> dict[str, Any]:
    view: dict[str, Any] = (result.meta or {})[server.VIEW_DATA_KEY]
    return view


def _model(result: types.CallToolResult) -> dict[str, Any]:
    """The model's copy of a Check card's result: its last text block."""
    texts = [c.text for c in result.content if isinstance(c, types.TextContent)]
    body: dict[str, Any] = json.loads(texts[-1])
    return body


def _zone_entry(root: Path, index: int, file: str) -> dict[str, Any]:
    """One zone of ``olmoearth_scores_from_studio``'s result, its scores file
    written under the root."""
    path = root.resolve() / f"zone{index}_scores.json"
    path.write_text(json.dumps({"scores": [[0.0, 0.4]] * 300}), "utf-8")
    return {
        "file": file,
        "scores_path": str(path),
        "patch": 4,
        "valid_pixels": 4800,
        "pooled": {
            "n_valid": 300,
            "n_windows": 300,
            "raster_check": {"sha256": str(index) * 64},
        },
    }


def _score_answer(
    root: Path,
    *,
    below: int = 2,
    above: int = 3,
    values_range: tuple[float, float] | None = (-0.0147, 1.007),
) -> dict[str, Any]:
    """``olmoearth_scores_from_studio``'s envelope for a result that is one
    0-to-1 score, in one zone, ``below`` and ``above`` of its pixels clipped;
    ``values_range`` is the zone's lowest and highest value (``None``: the
    zone gives none)."""
    clipped = bool(below or above)
    zone = {
        **_zone_entry(root, 0, ZONE),
        "class_counts": {f"not {NAME}": 4000, NAME: 800},
        "below_0_pixels": below,
        "above_1_pixels": above,
        "clipped": clipped,
    }
    if values_range is not None:
        zone["values_range"] = list(values_range)
    must = [studio_scores.UNIT_ASSUMPTION_SENTENCE]
    if clipped:
        must.append(studio_scores.CLIPPED_SENTENCE)
    must.append(studio_scores.NATIVE_SENTENCE)
    inner = {
        "available": True,
        "result_id": R,
        "prediction_id": P,
        "kind": studio_scores.UNIT,
        "property_name": NAME,
        "classes": [
            {"class": 0, "label": f"not {NAME}"},
            {"class": 1, "label": NAME},
        ],
        "date_window": {"start": "2023-01-01T00:00:00Z", "end": "2023-12-31T00:00:00Z"},
        "n_zones": 1,
        "zones": [zone],
        "facts": [],
        "must_state": must,
        "forbidden_claims": [{"id": "certified", "why": "Nothing certifies it."}],
    }
    return {"ok": True, "result": inner}


def _topk_answer(root: Path) -> dict[str, Any]:
    """The envelope for a class map with top-k scores in two zones, of presence
    and absence: the owner's labelling rule is set, so a press needs none."""
    zones = [_zone_entry(root, i, file) for i, file in enumerate(ZONES)]
    shares = [{"file": f, "share": s} for f, s in zip(ZONES, (0.6, 0.4))]
    inner = {
        "available": True,
        "result_id": R,
        "prediction_id": P,
        "kind": studio_scores.TOPK,
        "property_name": "kelp",
        "classes": [
            {"channel": 0, "value": 0, "label": "absent"},
            {"channel": 1, "value": 1, "label": "present"},
        ],
        "date_window": {"start": "2024-04-01T00:00:00Z", "end": "2024-06-30T00:00:00Z"},
        "n_zones": 2,
        "zones": zones,
        "facts": [{"id": "studio_zones", "sentence": "Two.", "zones": shares}],
        "must_state": [
            studio_scores.TOPK_SCORES_SENTENCE,
            studio_scores.ZONES_SENTENCE,
        ],
        "forbidden_claims": [{"id": "certified", "why": "Nothing certifies it."}],
    }
    return {"ok": True, "result": inner}


def _fake_tools(
    app: server.AppServer,
    monkeypatch: pytest.MonkeyPatch,
    studio: dict[str, Any],
    plan: Plan | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """``app.run_agent_tool`` answering ``studio`` for the Studio files and
    ``plan`` for a plan; the calls made, in order."""
    calls: list[tuple[str, dict[str, Any]]] = []

    async def run(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """The tool's envelope, as run_agent_tool gives it."""
        calls.append((name, dict(arguments)))
        if name == server.SCORES_TOOL:
            return copy.deepcopy(studio)
        assert plan is not None, name
        return await plan(dict(arguments))

    monkeypatch.setattr(app, "run_agent_tool", run)
    return calls


def _failing(error: str) -> Plan:
    """A plan that always fails with ``error``, as the registry writes it."""

    async def plan(args: dict[str, Any]) -> dict[str, Any]:
        """The failed envelope."""
        return {"ok": False, "error": error, "hint": MODEL_HINT}

    return plan


def _press(*draws: tuple[str, int]) -> dict[str, Any]:
    """The press's arguments: ``(check_key, budget)`` per zone."""
    return {"draws": [{"check_key": k, "budget": b} for k, b in draws]}


# --------------------------------------------------------------------------- K2, K3: a score map's card


@pytest.mark.asyncio
async def test_k2_the_card_reads_the_result_with_values_outside_0_to_1_clipped(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """The one call the card makes for the result's files asks for values
    outside 0 to 1 to be clipped, whatever the result's kind turns out to
    be (the flag changes nothing for a class map with top-k scores)."""
    app = _app()
    calls = _fake_tools(app, monkeypatch, _score_answer(_roots))
    await app.check_map_tool({"result_id": f" {R} "}, views=True)
    assert calls == [(server.SCORES_TOOL, {"result_id": R, "clip_out_of_range": True})]
    await app.aclose()


@pytest.mark.asyncio
async def test_k3_a_score_map_gets_one_part_its_counts_the_rule_gate_and_its_cut(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """A result that is one score from 0 to 1: one part with its counts and
    none picked; no rule of the owner's, so the card's gate asks for one; the
    cut line as the view's ``cut``, first among its notes and first in the
    model's ``must_state``, with the tool's own sentences after it (the
    clipping among them); the part's Details line on its five clipped pixels;
    and no exception name anywhere in the view."""
    app = _app()
    studio = _score_answer(_roots)
    _fake_tools(app, monkeypatch, studio)
    result = await app.check_map_tool({"result_id": R}, views=True)
    assert result.isError is False
    view = _view(result)
    assert (view["kind"], view["n_parts"]) == ("unit_score", 1)
    (zone,) = view["zones"]
    assert [offer["labels"] for offer in zone["offer"]] == [20, 40]
    assert (zone["state"], zone["part"], zone["n_valid"]) == ("to draw", None, 300)
    assert view["rule"] == {
        "set": False,
        "text": None,
        "source": None,
        "scope": None,
        "replaceable": False,
    }
    assert view["cut"] == CUT
    assert server.CHECK_CUT_LINE.format(name=NAME) == CUT
    tool_must = studio["result"]["must_state"]
    assert view["must_state"] == [CUT, *tool_must]
    model = _model(result)
    assert model["must_state"] == [CUT, *tool_must]
    assert studio_scores.CLIPPED_SENTENCE in model["must_state"]
    assert model["kind"] == "unit_score" and model["labelling_rule"] is None
    assert server.CHECK_NO_RULE_NOTE in model["note"]
    assert zone["clipped"] == FIVE_CLIPPED
    assert model["zones"][0]["clipped_pixels"] == 5
    assert "Error" not in json.dumps(view)
    await app.aclose()


@pytest.mark.asyncio
async def test_k2_a_score_inside_0_to_1_and_a_class_map_are_as_they_were(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """A score with no pixel outside 0 to 1 gets the cut line and no clipped
    line. A class map with top-k scores gets neither: its view and the
    model's ``must_state`` hold what the tool gave, with no key added."""
    app = _app()
    inside = _score_answer(_roots, below=0, above=0)
    _fake_tools(app, monkeypatch, inside)
    result = await app.check_map_tool({"result_id": R}, views=True)
    (zone,) = _view(result)["zones"]
    assert _view(result)["cut"] == CUT and "clipped" not in zone
    assert "clipped_pixels" not in _model(result)["zones"][0]

    topk = _topk_answer(_roots)
    _fake_tools(app, monkeypatch, topk)
    result = await app.check_map_tool({"result_id": R}, views=True)
    view, model = _view(result), _model(result)
    assert "cut" not in view
    assert view["must_state"] == topk["result"]["must_state"]
    assert model["must_state"] == topk["result"]["must_state"]
    assert len(view["zones"]) == 2
    for shown, told in zip(view["zones"], model["zones"]):
        assert "clipped" not in shown and "clipped_pixels" not in told
    assert view["rule"]["source"] == "owner"
    await app.aclose()


@pytest.mark.asyncio
async def test_k2_a_score_far_outside_0_to_1_gets_no_offer(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """A map declared a 0-to-1 score whose values reach -1 or 2 (a
    no-data fill its file does not declare, a 0-to-100 scale) is not one:
    clipped, each such pixel would count at full confidence. The card is
    refused whole, in plain words with both numbers, before any offer is
    written; Claude reads the same sentence."""
    offered: list[Any] = []
    offer_zones = check.offer_zones

    def offer_spy(*given: Any, **named: Any) -> Any:
        """``check.offer_zones``, noted."""
        offered.append(given)
        return offer_zones(*given, **named)

    monkeypatch.setattr(check, "offer_zones", offer_spy)
    app = _app()
    for span, low, high in (
        ((-1.0, 1.007), "-1", "1.007"),
        ((0.0, 100.0), "0", "100"),
        ((-0.01, 2.0), "-0.01", "2"),
    ):
        _fake_tools(app, monkeypatch, _score_answer(_roots, values_range=span))
        result = await app.check_map_tool({"result_id": R}, views=True)
        assert result.isError is True, span
        words = (
            f"This map's scores run from {low} to {high}, so it is not read as a "
            "score from 0 to 1."
        )
        assert words == server.CHECK_NOT_A_SCORE.format(low=low, high=high)
        view = _view(result)
        assert view["error"] == words and "zones" not in view
        assert view["hint"] is None and view["needs_sign_in"] is False
        assert "Error" not in json.dumps(view)
        assert _model(result) == {"ok": False, "error": words}
    assert offered == []
    await app.aclose()


@pytest.mark.asyncio
async def test_k2_the_bound_is_on_how_far_never_on_how_many(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """A score above -1 and below 2 is offered as before however many of its
    pixels were clipped (2,000 of 4,800 here: a model trained on 0 and 1 can
    put many pixels just past an end), with the values of a real result
    (-0.0865 to 1.781), and when the zone gives no range of its values."""
    app = _app()
    for answer in (
        _score_answer(_roots, below=1500, above=500),
        _score_answer(_roots, below=1500, above=500, values_range=(-0.0865, 1.781)),
        _score_answer(_roots, below=1500, above=500, values_range=None),
    ):
        _fake_tools(app, monkeypatch, answer)
        result = await app.check_map_tool({"result_id": R}, views=True)
        assert result.isError is False
        (zone,) = _view(result)["zones"]
        assert [offer["labels"] for offer in zone["offer"]] == [20, 40]
        assert zone["clipped"] == (
            "2,000 pixels had a score outside 0 to 1 and were set to the nearest "
            "end, 0 or 1."
        )
    await app.aclose()


def test_k2_only_a_score_with_a_range_of_two_numbers_is_held_to_the_bound() -> None:
    """The lowest and highest value over the zones that give a range; a
    range that is not two finite numbers is passed over, and a result of
    another kind is never refused here."""
    refuse = server._not_a_score
    assert (server.CHECK_SCORE_LOWEST, server.CHECK_SCORE_HIGHEST) == (-1.0, 2.0)
    two = {
        "kind": "unit_score",
        "zones": [{"values_range": [0.0, 1.0]}, {"values_range": [-9999.0, 0.9]}],
    }
    assert refuse(two) == server.CHECK_NOT_A_SCORE.format(low="-9999", high="1")
    for span in ([None, 2.0], ["a", "b"], [float("nan"), 3.0], [1.0], None, [True, 9]):
        odd = {"kind": "unit_score", "zones": [{"values_range": span}, "not a zone"]}
        assert refuse(odd) is None, span
    assert refuse({"kind": "unit_score"}) is None
    assert refuse({"kind": "top_k", "zones": [{"values_range": [0.0, 254.0]}]}) is None


def test_k3_the_cut_line_says_above_and_only_a_score_map_has_one() -> None:
    """The words as the round gives them: "above 0.5", which is what the
    pooling counts, never "0.5 or more". The line names class 1 of the
    result's classes, else its property name; no other kind of result, and
    no result without a name, has one."""
    assert server.CHECK_CUT_LINE == (
        "This map gives each pixel a score from 0 to 1, not a yes or no. This "
        'check counts a pixel as "{name}" when its score is above 0.5.'
    )
    assert "0.5 or more" not in server.CHECK_CUT_LINE
    assert server.CHECK_UNIT_KIND == studio_scores.UNIT
    classes = [{"class": 0, "label": "not karst"}, {"class": 1, "label": "karst"}]
    unit = {"kind": "unit_score", "classes": classes, "property_name": "p"}
    assert server._check_cut(unit) == server.CHECK_CUT_LINE.format(name="karst")
    named = {"kind": "unit_score", "property_name": "depth to karst"}
    assert server._check_cut(named) == server.CHECK_CUT_LINE.format(
        name="depth to karst"
    )
    assert server._check_cut({"kind": "unit_score"}) is None
    assert server._check_cut({"kind": "top_k", "classes": classes}) is None
    assert server._check_cut({"classes": classes, "property_name": "p"}) is None


def test_k2_the_clipped_line_counts_the_pixels_set_into_0_to_1() -> None:
    """One pixel, then several with thousands separators; none, and a zone
    that was not clipped or does not say, give no line."""
    one = {"clipped": True, "below_0_pixels": 1, "above_1_pixels": 0}
    assert server._clipped_pixels(one) == 1
    assert server._clipped_line(1) == (
        "1 pixel had a score outside 0 to 1 and was set to the nearest end, 0 or 1."
    )
    many = {"clipped": True, "below_0_pixels": 1200, "above_1_pixels": 34}
    assert server._clipped_pixels(many) == 1234
    assert server._clipped_line(1234) == (
        "1,234 pixels had a score outside 0 to 1 and were set to the nearest end, "
        "0 or 1."
    )
    assert server._clipped_view(1234) == {"clipped": server._clipped_line(1234)}
    assert server._clipped_line(0) is None and server._clipped_view(0) == {}
    for entry in (
        {"clipped": False, "below_0_pixels": 3, "above_1_pixels": 4},
        {"below_0_pixels": 3},
        {"clipped": True, "below_0_pixels": True, "above_1_pixels": "7"},
        None,
    ):
        assert server._clipped_pixels(entry) == 0


def test_k3_the_page_shows_the_cut_over_the_gate_and_the_clipped_line_in_details() -> (
    None
):
    """The Check card's page: the view's ``cut``, as text, right over the
    gate (whose own code is as it was); a part's ``clipped`` line among its
    Details lines. The words are the server's: the page holds no copy."""
    page = views.view_html("check")
    wanted = (
        "  const cut = typeof data.cut === 'string' && data.cut ? data.cut : null;\n",
        "  if (cut) root.append(OE.el('div', { id: 'oe-check-cut' }, cut));\n"
        "  root.append(gate);\n",
        "    gate.hidden = mode === 'none';\n",
        "    clipped: typeof z.clipped === 'string' && z.clipped ? z.clipped "
        ": null,\n",
        "      if (m.clipped) lines.push(OE.el('div', {}, m.clipped));\n",
    )
    for line in wanted:
        assert line in page, line
    assert "above 0.5" not in page and "nearest end, 0 or 1" not in page


@needs_extra
def test_k3_a_pixel_at_exactly_half_is_not_the_named_class() -> None:
    """What the line says is what the package's one-band rule does: a score
    of exactly 0.5 is the other class, a score above it the named one. The
    agent's own path (``olmoearth_scores_from_file``, whole and in strips)
    is pinned in ``tests/tools/test_scores_file_strips.py``."""
    from oe_inferencex.review import assess

    scores = np.array([[0.5, 0.5 + 1e-6], [0.25, 0.75]])
    out = assess.assess_prediction(scores, is_logit=False, patch=1)
    assert out["arrays"]["pooled_argmax"].tolist() == [[0, 1], [0, 1]]


# --------------------------------------------------------------------------- K4: a card's error in plain words


def test_k4_only_an_exception_name_at_the_start_is_dropped() -> None:
    """``"<Name>Error: "`` and ``"<Name>Exception: "`` at the very start, as
    the registry writes a failure, and after it an ``OSError``'s number
    (round r5-d R11); any other text is as it was. A ``KeyError``'s own text
    is only the missing key's repr, so it is put in plain words around it."""
    plain = server._plain_error
    assert plain("ValueError: the raster holds too much") == "the raster holds too much"
    assert plain("OSError: [Errno 28] No space left on device") == (
        "No space left on device"
    )
    assert plain("RuntimeError: GET /x failed (ReadTimeout)") == (
        "GET /x failed (ReadTimeout)"
    )
    assert plain("StudioAccessError: Studio refused it") == "Studio refused it"
    assert plain("KeyError: 'labels'") == (
        "the tool did not find an entry it needs ('labels')"
    )
    assert plain("KeyError: 3") == "the tool did not find an entry it needs (3)"
    assert plain("KeyError: ") == ""
    assert server.MISSING_ENTRY == "the tool did not find an entry it needs ({key})"
    assert server._refusal_words("KeyError: 'labels'") == (
        "The tool did not find an entry it needs ('labels')"
    )
    assert plain("ReplayedToolException: replayed") == "replayed"
    for kept in (
        "missing result_id",
        "result_epsg32610_0_native.tif: its probability raster holds too much",
        "The plan failed: ValueError: inside the sentence",
        "ValueError:no space after the colon",
        "Error: no name before it",
        "",
    ):
        assert plain(kept) == kept
    assert server._card_error(None) is None
    assert server._card_error("ValueError: x") == "x"
    # The class-map refusal's words are the same helper's, with a capital.
    assert server._refusal_words(" ValueError: the map") == "The map"


@pytest.mark.asyncio
async def test_k4_a_failed_check_card_has_no_exception_name_and_no_model_hint(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The files could not be downloaded: the card gets the words without
    "RuntimeError: " and no hint (it is written for the model, who reads the
    error and the hint as the tool gave them). Signed out, the hint comes
    with the sign-in."""
    failed = {
        "ok": False,
        "error": "RuntimeError: GET /prediction-results/files failed (ReadTimeout); "
        "no file was read",
        "tool": server.SCORES_TOOL,
        "hint": MODEL_HINT,
    }
    app = _app()
    _fake_tools(app, monkeypatch, failed)
    result = await app.check_map_tool({"result_id": R}, views=True)
    assert result.isError is True
    view = _view(result)
    assert view["error"] == (
        "GET /prediction-results/files failed (ReadTimeout); no file was read"
    )
    assert view["hint"] is None and view["needs_sign_in"] is False
    text = json.dumps(view)
    assert "Error" not in text and "search/list tools" not in text
    model = _model(result)
    assert (model["error"], model["hint"]) == (failed["error"], MODEL_HINT)

    signed_out = {
        "ok": False,
        "error": "NotSignedIn: " + auth.NOT_SIGNED_IN_MARK,
        "needs_sign_in": True,
        "hint": auth.NOT_SIGNED_IN,
    }
    _fake_tools(app, monkeypatch, signed_out)
    out = _view(await app.check_map_tool({"result_id": R}, views=True))
    assert out["needs_sign_in"] is True and out["hint"] == auth.NOT_SIGNED_IN
    await app.aclose()


def test_k4_a_result_cards_error_is_plain_and_its_hint_comes_with_the_sign_in() -> None:
    """The result view's data (every other card): the error without its
    exception's name, the hint only when the card offers the sign-in, and a
    result that did not fail as it was."""
    app = _app()
    plan = "olmoearth_plan_label_sample"
    failed = {"ok": False, "error": "ValueError: budget 9 is too small", "hint": "H."}
    data = app._view_data(plan, failed, None)
    assert (data["ok"], data["error"]) == (False, "budget 9 is too small")
    assert data["hint"] is None and data["needs_sign_in"] is False
    signed_out = {**failed, "needs_sign_in": True}
    data = app._view_data(plan, signed_out, None)
    assert (data["hint"], data["needs_sign_in"]) == ("H.", True)
    fine = app._view_data(plan, {"ok": True, "result": {}}, None)
    assert (fine["ok"], fine["error"], fine["hint"]) == (True, None, None)


def test_k4_a_failed_parts_error_is_plain_on_the_card_only() -> None:
    """A zone whose draw failed, and one zone of a press's reply: the card's
    copy has no exception name; the zone and the reply given are not
    changed, and one with no error is passed through as it is."""
    zone = {"state": "failed", "error": "ValueError: this sheet holds labels"}
    shown = server._check_zone_for_view(zone)
    assert shown == {"state": "failed", "error": "this sheet holds labels"}
    assert zone["error"] == "ValueError: this sheet holds labels"
    drawn = {"state": "drawn", "draw": {"budget": 20, "facts": [{"id": "f"}]}}
    assert server._check_zone_for_view(drawn) == {
        "state": "drawn",
        "draw": {"budget": 20},
    }
    reply = {"check_key": "k", "error": "OSError: [Errno 28] No space left on device"}
    # Round r5-d R11: the OSError's number goes from the card's copy too.
    assert server._draw_for_view(reply)["error"] == "No space left on device"
    assert reply["error"].startswith("OSError: [Errno 28] ")
    quiet = {"check_key": "k", "error": None}
    assert server._draw_for_view(quiet) is quiet


@pytest.mark.asyncio
async def test_k4_a_failed_pick_is_plain_on_the_card_and_keeps_its_name_in_the_record(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """A plan refused for good and one that may pass: the press's reply and
    the card opened again show the error without its name; the draw record
    keeps the name, so the first is not drawn again and the second is
    retried, as before; the model's zone keeps the error as the plan gave
    it."""
    for error, plain, retry in (
        ("ValueError: this sheet holds labels", "this sheet holds labels", False),
        ("MemoryError: out of memory", "out of memory", True),
        # A KeyError is put in words on the card; its record keeps its name.
        ("KeyError: 'labels'", server.MISSING_ENTRY.format(key="'labels'"), False),
    ):
        app = _app()
        calls = _fake_tools(app, monkeypatch, _topk_answer(_roots), _failing(error))
        card = _view(await app.check_map_tool({"result_id": R}, views=True))
        key = card["zones"][0]["check_key"]
        pressed = await app.app_tool("olmoearth_app_check_draw", _press((key, 20)))
        (reply,) = pressed["draws"]
        assert (reply["state"], reply["error"], reply["retry"]) == (
            "failed",
            plain,
            retry,
        )
        path = _roots / "checks" / f"{key}.draw.json"
        record = json.loads(path.read_text("utf-8"))
        assert (record["error"], record["retry"]) == (error, retry)
        later = await app.check_map_tool({"result_id": R}, views=True)
        shown = _view(later)["zones"][0]
        assert (shown["state"], shown["error"], shown["retry"]) == (
            "failed",
            plain,
            retry,
        )
        told = _model(later)["zones"][0]
        assert (told["error"], told["failed_for_good"]) == (error, not retry)
        again = await app.app_tool("olmoearth_app_check_draw", _press((key, 20)))
        assert again["draws"][0]["error"] == plain
        plans = [name for name, _args in calls if name == server.PLAN_TOOL]
        assert len(plans) == (2 if retry else 1)
        path.unlink()
        await app.aclose()


# --------------------------------------------------------------------------- K1: a score zone over the old limit


def _unit_record() -> dict[str, Any]:
    """A [0, 1] score result's record, as Studio's GET by id gives one."""
    return {
        "id": R,
        "prediction_id": P,
        "prediction_name": "KarstBinary test",
        "property_names": [NAME],
        "download_token": DOWNLOAD,
        "result_metadata": {
            "start_datetime": "2023-01-01T00:00:00Z",
            "end_datetime": "2023-12-31T00:00:00Z",
            "classification_fields": [],
            "regression_fields": [
                {
                    "property_name": NAME,
                    "band_index": 1,
                    "min_value": 0.0,
                    "max_value": 1.0,
                },
            ],
        },
    }


def _wide_zip(tmp_path: Path) -> bytes:
    """The files zip of a score zone 12 x 2100 pixels: two pixels below 0 and
    three above 1 (as the live file's values ran), an invalid block and a
    last row of no-data (-1.0)."""
    band = np.random.default_rng(21).uniform(0.05, 0.95, (1, 12, 2100))
    band = band.astype(np.float32)
    band[0, 0, :2] = -0.0147
    band[0, 5, 10] = 1.0070
    band[0, 9, 11:13] = 1.0070
    band[0, 4:8, :8] = -1.0
    band[0, 11, :] = -1.0
    path = tmp_path / "native.tif"
    profile = {
        "driver": "GTiff",
        "height": 12,
        "width": 2100,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32616",
        "transform": Affine(10.0, 0.0, 250_000.0, 0.0, -10.0, 3_800_000.0),
        "nodata": -1.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(band)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"KarstBinary test/{ZONE}", path.read_bytes())
    return buf.getvalue()


def _mock_studio(httpx_mock: HTTPXMock, zipped: bytes) -> None:
    """Studio's answers: the result's record, its prediction's (with its
    name), its model's and its files."""

    def envelope(record: dict[str, Any]) -> dict[str, Any]:
        """Studio's envelope around one record."""
        return {"records": [record], "meta": None, "errors": None}

    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/{R}",
        json=envelope(_unit_record()),
        is_reusable=True,
    )
    httpx_mock.add_response(
        url=f"{BASE}/predictions/{P}",
        json=envelope({"id": P, "model_id": MODEL, "name": "Karst 2023"}),
        is_reusable=True,
    )
    httpx_mock.add_response(
        url=f"{BASE}/models/{MODEL}",
        json=envelope({"id": MODEL, "name": "Karst model", "model_type": "x"}),
        is_reusable=True,
    )
    httpx_mock.add_response(
        url=f"{BASE}/prediction-results/files",
        match_params={"download_token": DOWNLOAD},
        content=zipped,
        is_reusable=True,
    )


@needs_extra
@pytest.mark.asyncio
async def test_k1_a_unit_score_zone_over_the_old_limit_gets_a_check_card(
    tmp_path: Path,
    _roots: Path,
    httpx_mock: HTTPXMock,
    monkeypatch: pytest.MonkeyPatch,
    canned_offers: None,
) -> None:
    """The real tools on a mocked Studio, the limits set small so the zone is
    over ``MAX_VALUES``: it is counted and written in strips, pooled in
    strips, and the card shows one part with its counts, the rule gate, the
    cut line and the part's clipped pixels, where the same zone was refused
    with "ValueError: ... more than ... this tool reads" before."""
    monkeypatch.setenv(auth.KEY_ENV, KEY)
    monkeypatch.setattr(scores_file, "MAX_VALUES", 1000)
    monkeypatch.setattr(scores_file, "BLOCK_VALUES", 8400)
    used: list[str] = []
    scan_unit, pool_strips = studio_scores.scan_unit, scores_file._pool_strips

    def scan_spy(*given: Any) -> Any:
        """The strip decode, noted."""
        used.append("scan")
        return scan_unit(*given)

    def pool_spy(*given: Any) -> Any:
        """The strip pooling, noted."""
        used.append("pool")
        return pool_strips(*given)

    monkeypatch.setattr(studio_scores, "scan_unit", scan_spy)
    monkeypatch.setattr(scores_file, "_pool_strips", pool_spy)
    _mock_studio(httpx_mock, _wide_zip(tmp_path))
    app = _app()
    result = await app.check_map_tool({"result_id": R}, views=True)
    assert result.isError is False, result.content
    assert used == ["scan", "pool"]
    view = _view(result)
    assert (view["kind"], view["n_parts"]) == ("unit_score", 1)
    (zone,) = view["zones"]
    assert [offer["labels"] for offer in zone["offer"]] == [20, 40]
    # 3 x 525 squares of 4 px; the two of the invalid block hold no pixel.
    assert (zone["state"], zone["n_valid"]) == ("to draw", 1573)
    assert zone["clipped"] == FIVE_CLIPPED
    assert view["rule"]["set"] is False
    assert view["cut"] == CUT and view["must_state"][0] == CUT
    assert view["map"]["name"] == "Karst 2023"
    model = _model(result)
    assert model["must_state"][0] == CUT
    assert studio_scores.CLIPPED_SENTENCE in model["must_state"]
    assert model["zones"][0]["clipped_pixels"] == 5
    assert "Error" not in json.dumps(view)
    await app.aclose()
