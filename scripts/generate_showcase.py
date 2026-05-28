# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Generate docs/SHOWCASE.md by running each skill's real tool.

Every "Actual output" in the showcase is captured by invoking the skill's
tool handler against an illustrative input — no fabricated results. Run:

    uv run python scripts/generate_showcase.py > docs/SHOWCASE.md

Skills #5 (predict) and #13 (data-export) call the live Studio API, and
#1-#4 are vendored SKILL.md packages, so they are documented (with their
verified-live results) rather than re-run offline here.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.provenance.tools import build_provenance_tools
from olmoearth_agent.tools.baseline_compare import build_baseline_compare_tools
from olmoearth_agent.tools.change_detect import build_change_detect_tools
from olmoearth_agent.tools.cloud_mask_audit import build_cloud_mask_audit_tools
from olmoearth_agent.tools.evaluate import build_evaluate_tools
from olmoearth_agent.tools.narrative import build_narrative_tools
from olmoearth_agent.tools.qgis import build_qgis_tools
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext
from olmoearth_agent.tools.similarity import build_similarity_tools
from olmoearth_agent.tools.uncertainty import build_uncertainty_tools


def _tool(bundle: list[RegisteredTool], name: str) -> RegisteredTool:
    return next(t for t in bundle if t.spec.name == name)


def _ctx() -> ToolContext:
    return ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]


def _json(obj: object) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _block(
    num: int | str,
    name: str,
    category: str,
    what: str,
    scenario: str,
    tool: str,
    args: dict[str, Any],
    result: dict[str, Any],
) -> str:
    return (
        f"### #{num} `{name}` — {what}\n\n"
        f"*Category: {category}*\n\n"
        f"**Scenario.** {scenario}\n\n"
        f"**Tool call** — `{tool}`:\n\n"
        f"```json\n{_json(args)}\n```\n\n"
        f"**Actual output:**\n\n"
        f"```json\n{_json(result)}\n```\n"
    )


def _note(num: int | str, name: str, category: str, what: str, note: str) -> str:
    return f"### #{num} `{name}` — {what}\n\n" f"*Category: {category}*\n\n" f"{note}\n"


async def main() -> None:
    sections: list[str] = []

    # --- #6 change-detect ---
    cd = _tool(build_change_detect_tools(), "olmoearth_change_detect")
    args: dict[str, Any] = {
        "series": [
            {"date": "2024-03-01", "value": 0.12},
            {"date": "2024-06-01", "value": 0.18},
            {"date": "2024-09-01", "value": 0.15},
            {"date": "2024-12-01", "value": 0.27},
        ],
        "metric": "karst_positive_fraction",
    }
    sections.append(
        _block(
            6,
            "olmoearth-change-detect",
            "Run",
            "multi-date trajectory diff (refuses naive 2-date)",
            "Four quarterly karst-prediction snapshots over one AOI — does the "
            "positive area trend up, or just wobble?",
            "olmoearth_change_detect",
            args,
            await cd.handler(args, _ctx()),
        )
    )

    # --- #7 baseline-compare ---
    bc = _tool(build_baseline_compare_tools(), "olmoearth_baseline_compare")
    args = {
        "y_true": [1, 1, 0, 0, 1, 0],
        "olmoearth_pred": [1, 1, 0, 0, 1, 0],
        "alphaearth_pred": [1, 0, 0, 1, 1, 0],
        "olmoearth_layer": [0.9, 0.8, 0.2, 0.1, 0.85, 0.3],
        "alphaearth_layer": [0.6, 0.55, 0.3, 0.4, 0.5, 0.45],
    }
    sections.append(
        _block(
            7,
            "olmoearth-baseline-compare",
            "Run",
            "OlmoEarth vs AlphaEarth head-to-head",
            "Same labels in a transfer region; does fine-tuned OlmoEarth beat "
            "AlphaEarth-derived predictions?",
            "olmoearth_baseline_compare",
            args,
            await bc.handler(args, _ctx()),
        )
    )

    # --- #8 evaluate (two tools) ---
    cv = _tool(build_evaluate_tools(), "olmoearth_cv_inflation_check")
    args = {
        "points": [
            [0.0, 0.0],
            [0.02, 0.0],
            [0.0, 0.02],
            [10.0, 0.0],
            [10.02, 0.0],
            [10.0, 0.02],
            [20.0, 0.0],
            [20.02, 0.0],
            [20.0, 0.02],
        ],
        "n_folds": 3,
    }
    sections.append(
        _block(
            8,
            "olmoearth-evaluate",
            "Analyze",
            "random-vs-spatial CV inflation check",
            "Labels sit in 3 tight clusters — would a random train/test split "
            "overstate map accuracy?",
            "olmoearth_cv_inflation_check",
            args,
            await cv.handler(args, _ctx()),
        )
    )
    cm = _tool(build_evaluate_tools(), "olmoearth_classification_metrics")
    args = {"y_true": [1, 1, 0, 0, 2, 2], "y_pred": [1, 0, 0, 0, 2, 2]}
    sections.append(
        _block(
            8,
            "olmoearth-evaluate",
            "Analyze",
            "per-class classification metrics",
            "Score a 3-class prediction against ground truth.",
            "olmoearth_classification_metrics",
            args,
            await cm.handler(args, _ctx()),
        )
    )

    # --- #9 similarity ---
    sim = _tool(build_similarity_tools(), "olmoearth_similarity_search")
    args = {
        "query": [0.9, 0.1, 0.0],
        "corpus": [
            [0.88, 0.12, 0.01],
            [0.1, 0.9, 0.0],
            [0.85, 0.15, 0.05],
            [0.0, 0.1, 0.95],
        ],
        "ids": ["site_A", "site_B", "site_C", "site_D"],
        "k": 3,
        "query_coord": [-77.0, 40.5],
        "coords": [[-77.1, 40.4], [-76.9, 40.6], [12.5, 41.9], [-122.3, 47.6]],
    }
    sections.append(
        _block(
            9,
            "olmoearth-similarity",
            "Analyze",
            "top-K embedding search + geographic-prior warning",
            "Find sites whose embeddings resemble a query site — and warn if the "
            "matches just cluster nearby (location, not features).",
            "olmoearth_similarity_search",
            args,
            await sim.handler(args, _ctx()),
        )
    )

    # --- #10 uncertainty ---
    aoa = _tool(build_uncertainty_tools(), "olmoearth_area_of_applicability")
    args = {
        "train_features": [
            [0.1, 0.2],
            [0.15, 0.25],
            [0.12, 0.18],
            [0.2, 0.22],
            [0.13, 0.21],
        ],
        "new_features": [[0.14, 0.2], [5.0, 5.0]],
    }
    sections.append(
        _block(
            10,
            "olmoearth-uncertainty",
            "Analyze",
            "Meyer-Pebesma Area-of-Applicability OOD flag",
            "Two AOI points to score: one resembles the training data, one is far "
            "outside it. Softmax confidence wouldn't catch the second.",
            "olmoearth_area_of_applicability",
            args,
            await aoa.handler(args, _ctx()),
        )
    )

    # --- #11 cloud-mask-audit ---
    cma = _tool(build_cloud_mask_audit_tools(), "olmoearth_cloud_mask_audit")
    args = {
        "masks": {
            "cfmask": [1, 1, 0, 0, 0, 1, 0, 0],
            "s2cloudless": [1, 1, 1, 0, 0, 1, 0, 0],
            "sen2cor": [1, 0, 0, 0, 1, 1, 0, 0],
            "maja": [1, 1, 0, 0, 0, 0, 0, 0],
        },
        "model_error": [0, 1, 1, 0, 0, 0, 0, 0],
    }
    sections.append(
        _block(
            11,
            "olmoearth-cloud-mask-audit",
            "Analyze",
            "ensemble disagreement + bad-mask-vs-bad-model verdict",
            "Four cloud algorithms over one scene; the model erred on 2 pixels — "
            "was it cloud confusion or a real model miss?",
            "olmoearth_cloud_mask_audit",
            args,
            await cma.handler(args, _ctx()),
        )
    )

    # --- #12 qgis-bridge ---
    qg = _tool(build_qgis_tools(), "olmoearth_qgis_bridge")
    args = {
        "tile_urls": [
            "/api/v1/prediction-results/0a098ce7/tiles/{z}/{x}/{y}.png?property_name=sample_karst_score"
        ],
        "layer_name": "pa_karst_v7",
        "vmin": 0.0,
        "vmax": 1.0,
    }
    sections.append(
        _block(
            12,
            "olmoearth-qgis-bridge",
            "Integrate",
            "tile URLs -> QGIS XYZ layer + OGC SLD style",
            "Hand a GIS analyst a desktop-loadable layer from a prediction result.",
            "olmoearth_qgis_bridge",
            args,
            await qg.handler(args, _ctx()),
        )
    )

    # --- #14 provenance (needs a populated run log) ---
    prov_ctx = _ctx()
    log = prov_ctx.state.provenance
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
        "olmoearth_change_detect", args, {"ok": True, "result": {"trend": "increasing"}}
    )
    prov = _tool(build_provenance_tools(), "olmoearth_provenance_summary")
    sections.append(
        _block(
            14,
            "olmoearth-provenance",
            "Report",
            "manifest over every tool call + replay skeleton",
            "After a 3-step run, produce an auditable record of what the agent did.",
            "olmoearth_provenance_summary",
            {},
            await prov.handler({}, prov_ctx),
        )
    )

    # --- #15 case-narrative (fresh + stale results, with provenance) ---
    fresh_ts = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    narr = _tool(build_narrative_tools(), "olmoearth_case_narrative")
    args = {
        "title": "PA Karst — Stakeholder Brief",
        "results": [
            {
                "result_id": "res_current",
                "tile_urls": [
                    "/api/v1/prediction-results/0a098ce7/tiles/{z}/{x}/{y}.png?property_name=sample_karst_score"
                ],
                "property_names": ["sample_karst_score"],
                "creation_time": fresh_ts,
            },
            {
                "result_id": "res_2020",
                "tile_urls": [
                    "/api/v1/prediction-results/old123/tiles/{z}/{x}/{y}.png?property_name=sample_karst_score"
                ],
                "property_names": ["sample_karst_score"],
                "creation_time": "2020-01-01T00:00:00Z",
            },
        ],
        "freshness_window_hours": 24,
    }
    # share the provenance from the #14 demo so the report has an audit section
    narr_ctx = ToolContext(studio=None, state=prov_ctx.state)  # type: ignore[arg-type]
    sections.append(
        _block(
            15,
            "olmoearth-case-narrative",
            "Report",
            "stakeholder Markdown writeup + freshness gate",
            "Assemble a brief from two results — one current, one from 2020. The "
            "stale tile must NOT be rendered.",
            "olmoearth_case_narrative",
            args,
            await narr.handler(args, narr_ctx),
        )
    )

    # --- documented (not re-run offline) ---
    documented = [
        _note(
            5,
            "olmoearth-predict",
            "Run",
            "the core run loop: search / submit / poll / fetch results",
            "Calls the live OlmoEarth Studio API (`/predictions/*`). **Verified "
            "live 2026-05-28**: searched 12 real PA Karst predictions (each with "
            "a reusable `model_id`) and fetched result tile URLs like "
            "`/api/v1/prediction-results/{id}/tiles/{z}/{x}/{y}.png?property_name="
            "sample_karst_score`. Requires `OLMOEARTH_API_KEY`, so it is not "
            "re-run in this offline showcase.",
        ),
        _note(
            13,
            "olmoearth-data-export",
            "Integrate",
            "export Studio projects + predictions, grouped, to JSON",
            "Calls the live Studio API. **Verified live 2026-05-28**: exported the "
            "real account to 5 per-project files (12 predictions linked) + 2 "
            "per-status files. Requires `OLMOEARTH_API_KEY`; not re-run here.",
        ),
        _note(
            "1-4",
            "olmoearth-data-prep / studio-job-config / embeddings",
            "Prep / Configure",
            "vendored SKILL.md guidance packages",
            "Skills #1-#4 are vendored from "
            "[`2imi9/OlmoEarth-Skills`](https://github.com/2imi9/OlmoEarth-Skills) "
            "(git submodule) and surfaced to the agent via progressive disclosure "
            "(`olmoearth_list_skills` → name+description index; `olmoearth_load_skill` "
            "→ full SKILL.md body). They are instructional packages (8 data-prep "
            "pitfalls, 14 job-config presets, embeddings-vs-fine-tune decision), not "
            "computational tools, so there is no single JSON output to show.",
        ),
    ]

    header = (
        "<!-- Generated by scripts/generate_showcase.py — do not edit by hand. -->\n"
        "# OlmoEarth Agent — Skills in Action\n\n"
        "The OlmoEarth Agent ships **15 skills** that drive the OlmoEarth Studio "
        "platform from natural-language briefs. This page shows each skill's tool "
        "**producing a real result**.\n\n"
        '> **How to read this.** Every "Actual output" below is captured by '
        "invoking the skill's real tool handler against an illustrative input — "
        "nothing is fabricated. Inputs are representative fixtures, not live Studio "
        "data. Regenerate with:\n>\n"
        "> ```\n> uv run python scripts/generate_showcase.py > docs/SHOWCASE.md\n> ```\n\n"
        "Skills #5 and #13 call the live Studio API and #1-#4 are vendored "
        "guidance packages, so those are documented (with their verified-live "
        "results) rather than re-run offline.\n\n"
        "---\n\n"
    )

    body = "\n---\n\n".join(sections + documented)
    print(header + body)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # emit UTF-8
    asyncio.run(main())
