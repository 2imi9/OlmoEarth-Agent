# OlmoEarth Agent

A tool that drives the [OlmoEarth Studio](https://allenai.org/blog/olmoearth) platform from natural-language briefs. Same shape as Google's Google Earth Agent: a compact catalog of functions (Studio API, EO data fetch, geometry utilities) plus a sandboxed Python interpreter, with operational constraints built in.

## Status

**v0.2 spec.** No runnable code yet. See [`PLAN.md`](PLAN.md) for the tool catalog, harness data classes, operational rules, and the underlying stack.

## What's in this repo

- [`PLAN.md`](PLAN.md) — Tool catalog (CSV-shaped), harness dataclasses, operational rules, underlying stack, roadmap.
- `LICENSE` — Apache 2.0.
- `.gitignore` — Python / EO / agent-state.

## Tool surface (summary)

The agent exposes:

- `system:python` — sandboxed Python interpreter with `pandas`, `geopandas`, `xarray`, `rioxarray`, `shapely`, `pystac_client`, `planetary_computer`, `rslearn`, `olmoearth_projects` preloaded. No `import` statements.
- `system:search`, `system:fetch` — web search and documented-endpoint HTTP GET.
- `olmoearth.*` — Studio API wrappers: `load_context`, `resolve_to_aoi`, `search_dataset_spec`, `get_data_in_locations`, `create_project`/`create_area`/`create_dataset`/`create_labelset`/`upload_labels`, `submit_prediction`/`poll_prediction`/`fetch_results`/`save_view`.
- `eo.*` — STAC search, asset signing, AOI windowing.
- `utils.*` — geometry helpers, equal-frequency binning, spatial cross-validation split.

Full catalog with arguments and return types in [`PLAN.md` §1](PLAN.md).

## Studio API

- Docs: https://docs.olmoearth.allenai.org/
- Auth: https://docs.olmoearth.allenai.org/authentication/ — Bearer token; max 10 keys per account
- Live OpenAPI spec: https://olmoearth.allenai.org/api/v1/openapi.json
- Resources: Areas, Projects, Datasets, Labelsets, Labels, Annotations, Tasks, Predictions, PredictionResults, Users

## Underlying stack (reference only)

| Layer | Reference |
|---|---|
| Harness | [ByteDance DeerFlow v2](https://github.com/bytedance/deer-flow) |
| Skill packaging | [NVIDIA AI-Q](https://docs.nvidia.com/aiq-blueprint/latest/integration/agent-skills.html) on the open [agentskills.io](https://agentskills.io) spec |
| Vision–language model | [Prismatic VLMs](https://arxiv.org/abs/2402.07865) + [unsloth/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4) |
| Geospatial encoder | [OlmoEarth-v1-Large](https://huggingface.co/allenai/OlmoEarth-v1-Large) embeddings |
| Self-improvement | [Stanford CS329A](https://cs329a.stanford.edu/) techniques (Reflexion, Self-Refine, repeated-sampling+verifier, STaR/SWiRL) |

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
