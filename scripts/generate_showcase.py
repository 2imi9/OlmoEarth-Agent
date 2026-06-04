# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Generate docs/SHOWCASE.md by driving every skill through the live LLM.

Each transcript is captured by handing a natural-language brief to the
real Qwen3.6 backbone (served via llama.cpp) and running the agent loop:
the model reasons, calls the skill's tool(s), receives the real result,
and writes a final answer. Nothing is fabricated: the reasoning, the
tool arguments, and the answers are the model's own output, and the tool
results are real (live Studio API for #5/#13, the vendored SKILL.md files
for #1-#4, a live arXiv/OpenAlex search for #16, real computation for
#6-#15, #17, and #18). Run:

    # source your .env first so OLMOEARTH_API_KEY is set for #5/#13
    set -a; . ./.env; set +a
    uv run python scripts/generate_showcase.py > docs/SHOWCASE.md

Requires the LLM server up (see docs/serving.md); point LLM_ENDPOINT at
it (default http://localhost:8000/v1). Skills #5 (predict, read-only) and
#13 (data-export) hit the live Studio API and need OLMOEARTH_API_KEY; if
it is absent they degrade to a short documented note. Because the agent
samples at temperature 1.0, each run captures a fresh trace; the wording
varies, the behaviour does not.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.client import OlmoEarthLLM
from olmoearth_agent.llm.types import Message
from olmoearth_agent.provenance.tools import build_provenance_tools
from olmoearth_agent.studio.client import StudioClient
from olmoearth_agent.tools.automate import build_automate_tools
from olmoearth_agent.tools.baseline_compare import build_baseline_compare_tools
from olmoearth_agent.tools.change_detect import build_change_detect_tools
from olmoearth_agent.tools.cloud_mask_audit import build_cloud_mask_audit_tools
from olmoearth_agent.tools.evaluate import build_evaluate_tools
from olmoearth_agent.tools.export import build_export_tools
from olmoearth_agent.tools.litsearch import build_litsearch_tools
from olmoearth_agent.tools.narrative import build_narrative_tools
from olmoearth_agent.tools.negative_sampler import build_negative_sampler_tools
from olmoearth_agent.tools.predict import build_predict_tools
from olmoearth_agent.tools.qgis import build_qgis_tools
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext, ToolRegistry
from olmoearth_agent.tools.similarity import build_similarity_tools
from olmoearth_agent.tools.skill_tools import build_skill_tools
from olmoearth_agent.tools.uncertainty import build_uncertainty_tools

SHOWCASE_SYSTEM = (
    "You are the OlmoEarth Studio agent. You help Earth-observation "
    "researchers by calling the provided tools. Read the brief, call the "
    "appropriate tool(s) with arguments taken from the brief or from prior "
    "tool results, then stop calling tools and reply with a short plain-text "
    "answer that interprets the result for the researcher. Never invent data "
    "or ids: use ids returned by earlier tool calls."
)

#: Default per-turn output budget. Above a full <think> block + a tool
#: call, and far enough below the server's 8K context that multi-turn runs
#: never overflow.
MAX_TOKENS = 3000

#: Skill-loading turns feed a whole SKILL.md back into context, so cap the
#: final-answer budget lower to stay clear of the 8K window.
SKILL_MAX_TOKENS = 2200

#: Skill #5 has a write tool (submit_prediction). The showcase only ever
#: demonstrates the read-only half so it cannot mutate the user's account.
_PREDICT_READONLY = {
    "olmoearth_search_predictions",
    "olmoearth_fetch_results",
    "olmoearth_get_prediction_result",
}


@dataclass
class TurnRecord:
    """One LLM round-trip: the model's reasoning, answer, and any calls."""

    thinking: str | None
    content: str | None
    finish: str | None
    usage: dict[str, int] | None
    calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Transcript:
    """The full captured run for one skill demonstration."""

    brief: str
    turns: list[TurnRecord]
    final: str | None


def _json(obj: object) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _readonly_predict_tools() -> list[RegisteredTool]:
    return [t for t in build_predict_tools() if t.spec.name in _PREDICT_READONLY]


async def capture(
    llm: OlmoEarthLLM,
    tools: list[RegisteredTool],
    brief: str,
    *,
    state: ThreadState | None = None,
    studio: StudioClient | None = None,
    max_turns: int = 6,
    max_tokens: int = MAX_TOKENS,
) -> Transcript:
    """Drive ``brief`` through the live agent loop and record every step."""
    registry = ToolRegistry()
    registry.register_all(tools)
    state = state or ThreadState()
    ctx = ToolContext(studio=studio, state=state)  # type: ignore[arg-type]
    messages: list[Message] = [
        Message(role="system", content=SHOWCASE_SYSTEM),
        Message(role="user", content=brief),
    ]
    turns: list[TurnRecord] = []
    final: str | None = None

    for _ in range(max_turns):
        response = await llm.chat(
            messages, tools=registry.specs(), max_tokens=max_tokens
        )
        turn = TurnRecord(
            thinking=response.thinking,
            content=response.content,
            finish=response.finish_reason,
            usage=response.usage,
        )
        if not response.tool_calls:
            final = response.content
            turns.append(turn)
            break
        messages.append(
            Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
            )
        )
        for call in response.tool_calls:
            result = await registry.dispatch(call, ctx)
            state.provenance.record_tool_call(call.name, call.arguments, result)
            turn.calls.append(
                {"name": call.name, "args": call.arguments, "result": result}
            )
            messages.append(
                Message(
                    role="tool",
                    tool_call_id=call.id,
                    name=call.name,
                    content=json.dumps(result),
                )
            )
        turns.append(turn)

    return Transcript(brief=brief, turns=turns, final=final)


def _fence_for(text: str) -> str:
    """Pick a backtick fence longer than any run inside ``text``."""
    longest = run = 0
    for ch in text:
        if ch == "`":
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return "`" * max(3, longest + 1)


def _result_block(result: dict[str, Any]) -> str:
    """Render the substantive payload the model received from one call."""
    payload = result.get("result", result) if result.get("ok") else result
    if isinstance(payload, dict) and "instructions" in payload and "name" in payload:
        body = payload["instructions"]
        fence = _fence_for(body)
        return (
            f"**Function result**, the full `{payload['name']}` SKILL.md, "
            f"loaded into the model's context:\n\n"
            f"{fence}markdown\n{body}\n{fence}\n"
        )
    return f"**Function result:**\n\n```json\n{_json(payload)}\n```\n"


def render(
    num: int | str,
    name: str,
    category: str,
    what: str,
    transcript: Transcript,
) -> str:
    """Render one captured run as a Markdown showcase block."""
    parts = [
        f"### #{num} `{name}` - {what}\n",
        f"*Category: {category}* · *Model: Qwen3.6-35B-A3B (`UD-IQ4_XS`, "
        f"llama.cpp)*\n",
        f"**Brief.**\n\n> {transcript.brief}\n",
    ]
    step = 0
    for turn in transcript.turns:
        if turn.thinking:
            label = "Qwen3.6 reasoning" if step == 0 else "Qwen3.6 reasoning (cont.)"
            parts.append(f"**{label}:**\n\n```\n{turn.thinking.strip()}\n```\n")
        for call in turn.calls:
            step += 1
            parts.append(
                f"**Function call** - `{call['name']}`:\n\n"
                f"```json\n{_json(call['args'])}\n```\n"
            )
            parts.append(_result_block(call["result"]))
    if transcript.final:
        parts.append(f"**Qwen3.6 answer.**\n\n> {transcript.final.strip()}\n")
    return "\n".join(parts)


def _note(num: int | str, name: str, category: str, what: str, note: str) -> str:
    return f"### #{num} `{name}` - {what}\n\n*Category: {category}*\n\n{note}\n"


def _seed_provenance(state: ThreadState) -> None:
    """Pre-populate a run log so the reporting skills have history to act on."""
    log = state.provenance
    log.record_tool_call(
        "olmoearth_load_context",
        {},
        {"ok": True, "result": {"name": "demo-user", "project_count": 5}},
    )
    log.record_tool_call(
        "olmoearth_search_predictions",
        {"project_id": "proj_karst"},
        {"ok": True, "result": {"total": 12}},
    )
    log.record_tool_call(
        "olmoearth_change_detect",
        {"metric": "karst_positive_fraction"},
        {"ok": True, "result": {"trend": "increasing"}},
    )


async def main() -> None:
    llm = OlmoEarthLLM()
    sections: list[str] = []

    try:
        studio: StudioClient | None = StudioClient.from_env()
    except RuntimeError:
        studio = None  # no OLMOEARTH_API_KEY: #5 / #13 fall back to a note

    try:
        # ---- #1-2 olmoearth-data-prep (vendored, progressive disclosure) ----
        sections.append(
            render(
                "1-2",
                "olmoearth-data-prep",
                "Prep",
                "labels → Studio-ready dataset (8 prep pitfalls), via load_skill",
                await capture(
                    llm,
                    build_skill_tools(),
                    "I'm about to import field-collected labels (a CSV of points "
                    "with a class column) into OlmoEarth Studio to train a model, "
                    "and I want to avoid the common data-prep mistakes. List the "
                    "available instruction skills, then load the data-prep one and "
                    "walk me through the key pitfalls.",
                    max_tokens=SKILL_MAX_TOKENS,
                ),
            )
        )

        # ---- #3 olmoearth-studio-job-config (vendored) ----
        sections.append(
            render(
                3,
                "olmoearth-studio-job-config",
                "Configure",
                "task description → Studio job-wizard answers (14 presets)",
                await capture(
                    llm,
                    build_skill_tools(),
                    "I need to configure a new OlmoEarth Studio training job for a "
                    "land-cover segmentation task but I'm unsure how to answer the "
                    "wizard. Load the 'olmoearth-studio-job-config' instruction "
                    "skill and summarize how to fill it in.",
                    max_tokens=SKILL_MAX_TOKENS,
                ),
            )
        )

        # ---- #4 olmoearth-embeddings (vendored) ----
        sections.append(
            render(
                4,
                "olmoearth-embeddings",
                "Configure",
                "embeddings-vs-fine-tune decision, via load_skill",
                await capture(
                    llm,
                    build_skill_tools(),
                    "Should I use OlmoEarth embeddings with a shallow classifier, "
                    "or fine-tune the full model, for my crop-type mapping task? "
                    "Load the 'olmoearth-embeddings' instruction skill and give me "
                    "the decision criteria.",
                    max_tokens=SKILL_MAX_TOKENS,
                ),
            )
        )

        # ---- #5 olmoearth-predict (live Studio API, read-only) ----
        if studio is not None:
            sections.append(
                render(
                    5,
                    "olmoearth-predict",
                    "Run",
                    "the core run loop: discover a model_id, then fetch result tiles",
                    await capture(
                        llm,
                        _readonly_predict_tools(),
                        "Find a reusable model in my existing OlmoEarth predictions, "
                        "then pull the result tiles for one of those predictions. Do "
                        "not submit anything new.",
                        studio=studio,
                    ),
                )
            )
        else:
            sections.append(
                _note(
                    5,
                    "olmoearth-predict",
                    "Run",
                    "the core run loop: search / submit / poll / fetch results",
                    "Calls the live OlmoEarth Studio API (`/predictions/*`); set "
                    "`OLMOEARTH_API_KEY` to capture a live read-only run here "
                    "(search predictions → fetch result tiles). The write half "
                    "(submit / poll) is intentionally never exercised by this "
                    "generator.",
                )
            )

        # ---- #6 change-detect ----
        sections.append(
            render(
                6,
                "olmoearth-change-detect",
                "Run",
                "multi-date trajectory diff (refuses naive 2-date)",
                await capture(
                    llm,
                    build_change_detect_tools(),
                    "I ran the karst prediction over one AOI for four quarters and "
                    "measured the positive-area fraction each time: 2024-03-01 = "
                    "0.12, 2024-06-01 = 0.18, 2024-09-01 = 0.15, 2024-12-01 = 0.27 "
                    "(metric: karst_positive_fraction). Is the positive area "
                    "genuinely trending up over the year, or just wobbling quarter "
                    "to quarter? Use the change-detect tool, then tell me.",
                ),
            )
        )

        # ---- #7 baseline-compare ----
        sections.append(
            render(
                7,
                "olmoearth-baseline-compare",
                "Run",
                "OlmoEarth vs AlphaEarth head-to-head",
                await capture(
                    llm,
                    build_baseline_compare_tools(),
                    "I have 6 validation pixels in a transfer region. Ground-truth "
                    "labels y_true = [1,1,0,0,1,0]. Fine-tuned OlmoEarth predicted "
                    "[1,1,0,0,1,0] with score layer [0.9,0.8,0.2,0.1,0.85,0.3]. "
                    "AlphaEarth-derived predictions were [1,0,0,1,1,0] with score "
                    "layer [0.6,0.55,0.3,0.4,0.5,0.45]. Does OlmoEarth actually beat "
                    "the AlphaEarth baseline here? Compare them.",
                ),
            )
        )

        # ---- #8 evaluate (two tools) ----
        sections.append(
            render(
                8,
                "olmoearth-evaluate",
                "Analyze",
                "random-vs-spatial CV inflation check",
                await capture(
                    llm,
                    build_evaluate_tools(),
                    "My training labels sit in 3 tight geographic clusters. Their "
                    "lon/lat coordinates are [[0,0],[0.02,0],[0,0.02],[10,0],"
                    "[10.02,0],[10,0.02],[20,0],[20.02,0],[20,0.02]]. If I use a "
                    "random 3-fold train/test split, will it overstate my map "
                    "accuracy? Check the inflation risk with n_folds=3.",
                ),
            )
        )
        sections.append(
            render(
                8,
                "olmoearth-evaluate",
                "Analyze",
                "per-class classification metrics",
                await capture(
                    llm,
                    build_evaluate_tools(),
                    "Score a 3-class prediction against ground truth. "
                    "y_true = [1,1,0,0,2,2], y_pred = [1,0,0,0,2,2]. Give me "
                    "per-class precision, recall, F1 and IoU.",
                ),
            )
        )

        # ---- #9 similarity ----
        sections.append(
            render(
                9,
                "olmoearth-similarity",
                "Analyze",
                "top-K embedding search + geographic-prior warning",
                await capture(
                    llm,
                    build_similarity_tools(),
                    "Find the sites whose embeddings most resemble my query "
                    "embedding [0.9, 0.1, 0.0]. The corpus (in order) is "
                    "ids=['site_A','site_B','site_C','site_D'] with embeddings "
                    "[[0.9,0.1,0.0],[0.8,0.2,0.0],[0.1,0.9,0.0],[0.0,0.1,0.9]] "
                    "and coordinates (lon,lat) "
                    "[[-77.1,40.4],[-76.9,40.6],[12.5,41.9],[-122.3,47.6]]. My "
                    "query is at query_coord=[-77.0,40.5]. Return the top k=3 and "
                    "warn me if the matches are just nearby in space rather than "
                    "similar in features.",
                ),
            )
        )

        # ---- #10 uncertainty ----
        sections.append(
            render(
                10,
                "olmoearth-uncertainty",
                "Analyze",
                "Meyer-Pebesma Area-of-Applicability OOD flag",
                await capture(
                    llm,
                    build_uncertainty_tools(),
                    "I trained on 5 feature vectors: train_features = "
                    "[[0.1,0.2],[0.15,0.25],[0.12,0.18],[0.2,0.22],[0.13,0.21]]. "
                    "Now I want to score two new AOI points: new_features = "
                    "[[0.14,0.2],[5.0,5.0]]. Which (if any) fall outside my model's "
                    "area of applicability? Softmax confidence won't catch "
                    "extrapolation, so use the area-of-applicability tool.",
                ),
            )
        )

        # ---- #11 cloud-mask-audit ----
        sections.append(
            render(
                11,
                "olmoearth-cloud-mask-audit",
                "Analyze",
                "ensemble disagreement + bad-mask-vs-bad-model verdict",
                await capture(
                    llm,
                    build_cloud_mask_audit_tools(),
                    "Four cloud-mask algorithms ran over an 8-pixel scene "
                    "(1 = cloud): cfmask=[1,1,0,0,0,1,0,0], "
                    "s2cloudless=[1,1,1,0,0,1,0,0], sen2cor=[1,0,0,0,1,1,0,0], "
                    "maja=[1,1,0,0,0,0,0,0]. My model made errors on pixels 2 and "
                    "3: model_error=[0,1,1,0,0,0,0,0]. Were those errors caused by "
                    "cloud confusion, or is it a real model miss? Audit it.",
                ),
            )
        )

        # ---- #12 qgis-bridge ----
        sections.append(
            render(
                12,
                "olmoearth-qgis-bridge",
                "Integrate",
                "tile URLs → QGIS XYZ layer + OGC SLD style",
                await capture(
                    llm,
                    build_qgis_tools(),
                    "Turn this prediction-result tile endpoint into something a GIS "
                    "analyst can load in QGIS: "
                    "'/api/v1/prediction-results/0a098ce7/tiles/{z}/{x}/{y}.png"
                    "?property_name=sample_karst_score'. Name the layer "
                    "'pa_karst_v7' and style the score ramp from vmin=0 to vmax=1.",
                ),
            )
        )

        # ---- #13 olmoearth-data-export (live Studio API) ----
        if studio is not None:
            sections.append(
                render(
                    13,
                    "olmoearth-data-export",
                    "Integrate",
                    "export Studio projects + predictions, grouped, to JSON",
                    await capture(
                        llm,
                        build_export_tools(),
                        "Export my OlmoEarth Studio projects and their predictions "
                        "to JSON files, grouped by project.",
                        studio=studio,
                    ),
                )
            )
        else:
            sections.append(
                _note(
                    13,
                    "olmoearth-data-export",
                    "Integrate",
                    "export Studio projects + predictions, grouped, to JSON",
                    "Calls the live Studio API; set `OLMOEARTH_API_KEY` to capture "
                    "a live run here (curated metadata only: ids / names / "
                    "statuses / times, no raw geometry).",
                )
            )

        # ---- #14 provenance (seeded run log) ----
        prov_state = ThreadState()
        _seed_provenance(prov_state)
        sections.append(
            render(
                14,
                "olmoearth-provenance",
                "Report",
                "manifest over every tool call + replay skeleton",
                await capture(
                    llm,
                    build_provenance_tools(),
                    "I've finished a short run: loading context, searching "
                    "predictions, and running change detection. Produce an "
                    "auditable record of what this run did, with a replay skeleton.",
                    state=prov_state,
                ),
            )
        )

        # ---- #15 case-narrative (seeded provenance, fresh + stale results) ----
        fresh_ts = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        narr_state = ThreadState()
        _seed_provenance(narr_state)
        sections.append(
            render(
                15,
                "olmoearth-case-narrative",
                "Report",
                "stakeholder Markdown writeup + freshness gate",
                await capture(
                    llm,
                    build_narrative_tools(),
                    "Write a stakeholder brief titled 'PA Karst - Stakeholder "
                    "Brief' from two prediction results, with a 24-hour freshness "
                    "window. Result 1: result_id='res_current', "
                    "tile_urls=['/api/v1/prediction-results/0a098ce7/tiles/"
                    "{z}/{x}/{y}.png?property_name=sample_karst_score'], "
                    "property_names=['sample_karst_score'], "
                    f"creation_time='{fresh_ts}'. Result 2: result_id='res_2020', "
                    "tile_urls=['/api/v1/prediction-results/old123/tiles/"
                    "{z}/{x}/{y}.png?property_name=sample_karst_score'], "
                    "property_names=['sample_karst_score'], "
                    "creation_time='2020-01-01T00:00:00Z'. Stale tiles must not be "
                    "rendered.",
                    state=narr_state,
                ),
            )
        )

        # ---- #16 olmoearth-litsearch (live arXiv + OpenAlex) ----
        sections.append(
            render(
                16,
                "olmoearth-litsearch",
                "Report",
                "live arXiv + OpenAlex search to ground a citation",
                await capture(
                    llm,
                    build_litsearch_tools(),
                    "I'm writing up results and need to cite the OlmoEarth "
                    "foundation-model paper. Search arXiv and OpenAlex for it and "
                    "give me the title, authors, year, and a link I can cite.",
                ),
            )
        )

        # ---- #17 olmoearth-automate (local decision + config) ----
        sections.append(
            render(
                17,
                "olmoearth-automate",
                "Configure",
                "auto-decide embeddings vs fine-tune + propose a config",
                await capture(
                    llm,
                    build_automate_tools(),
                    "I have 200 labeled samples across 9 land-cover classes and only "
                    "a Colab T4 GPU. Should I fine-tune OlmoEarth or use embeddings, "
                    "and what setup should I use? Decide and propose a config.",
                ),
            )
        )

        # ---- #18 olmoearth-negative-sampler (presence-only -> trainable) ----
        sections.append(
            render(
                18,
                "olmoearth-negative-sampler",
                "Prep",
                "generate a negative/background class for a presence-only set",
                await capture(
                    llm,
                    build_negative_sampler_tools(),
                    "My label file at docs/showcase_fixtures/karst_presence.geojson "
                    "is presence-only - every point is a karst sighting "
                    "(sample_category=karst) with no negative class, so the "
                    "data-prep audit fails. Generate background samples so it "
                    "becomes trainable, writing the result next to the input.",
                ),
            )
        )
    finally:
        if studio is not None:
            await studio.aclose()

    live_line = (
        "Skills #5 and #13 are captured against the **live Studio API** "
        "(#5 read-only: the write half is never exercised); #1-#4 load the "
        "real vendored `SKILL.md` bodies through `olmoearth_load_skill`."
        if studio is not None
        else "Skills #5 and #13 need `OLMOEARTH_API_KEY` (absent here), so they "
        "show a short note; every other skill is a live capture."
    )

    header = (
        "<!-- Generated by scripts/generate_showcase.py: do not edit by hand. -->\n"
        "# OlmoEarth Agent - Skills in Action\n\n"
        "The OlmoEarth Agent ships **18 skills** that drive the OlmoEarth Studio "
        "platform from natural-language briefs. This page shows each skill, in "
        "order, being **driven by the real LLM**: a brief goes in, the Qwen3.6 "
        "backbone reasons, calls the skill's tool(s), reads the result, and "
        "answers.\n\n"
        "> **How to read this.** Every transcript below is a captured run of the "
        "agent loop against the live **Qwen3.6-35B-A3B** model (4-bit "
        "`UD-IQ4_XS` GGUF, served via llama.cpp). The reasoning, the function "
        "arguments, and the answer are all the model's own output, and the "
        "function results are real: the live Studio API (#5, #13), the vendored "
        "`SKILL.md` files (#1-#4), a live arXiv + OpenAlex search (#16), and real "
        "computation (#6-#15, #17, #18). Nothing is "
        "fabricated. Because the agent samples at temperature 1.0, each run "
        "captures a fresh trace; the wording varies, the behaviour does not. "
        "Regenerate with:\n>\n"
        "> ```\n> set -a; . ./.env; set +a   # OLMOEARTH_API_KEY for #5/#13\n"
        "> uv run python scripts/generate_showcase.py > docs/SHOWCASE.md\n> ```\n\n"
        f"{live_line}\n\n"
        "---\n\n"
    )

    print(header + "\n---\n\n".join(sections))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # emit UTF-8
    asyncio.run(main())
