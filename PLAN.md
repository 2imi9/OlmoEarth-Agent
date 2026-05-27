# OlmoEarth Agent

A tool that drives the [OlmoEarth Studio](https://allenai.org/blog/olmoearth) platform from natural-language briefs. It exposes a compact set of functions covering Studio's HTTP API, EO data fetch, and geometry utilities; runs them in a Python sandbox preloaded with the standard geospatial stack; and enforces a small list of operational constraints. The agent's contract is the tool catalog in §1. Everything below it is supporting detail.

**Status:** v0.2 spec, 2026-05-27. No runnable code yet.
**Verification discipline:** every external claim has a real URL. Unverified items are flagged **UNVERIFIED** inline.

---

## 1. Tool catalog

The agent exposes the following functions. Code below the table renders the same content as CSV for easy copy-paste into a sheet.

| Module | Type | Name | Args / Inputs | Returns | Description |
|---|---|---|---|---|---|
| `system` | Tool | `python` | `code: str` | `ExecutionResult` | Run Python in a sandboxed interpreter. Preloaded: `pandas`, `geopandas`, `xarray`, `rioxarray`, `shapely`, `pystac_client`, `planetary_computer`, `rslearn`, `olmoearth_projects`. State persists across turns. **No `import` statements.** |
| `system` | Tool | `search` | `queries: list[str]` | `SearchResponse` | Web search for external references (docs, papers, code examples). |
| `system` | Tool | `fetch` | `url: str`, `headers: dict = None` | `FetchResponse` | HTTP GET on documented endpoints (OlmoEarth Studio, Planetary Computer, NLDI, Earthdata). |
| `olmoearth` | Function | `load_context` | None | `StudioContext` | Returns the user's active Studio project, areas, datasets, recent predictions. |
| `olmoearth` | Function | `resolve_to_aoi` | `locations: list[str]` (max 30) | `AOIWrapper` | Place names, watershed IDs (HUC-8/10/12), county FIPS, or free text → polygon geometries. |
| `olmoearth` | Function | `search_dataset_spec` | `query: str` | `DatasetRetrievalSpecs` | Searches Studio catalog + Planetary Computer STAC for datasets matching the query. Returns both analyzable rasters and visual-only layers. |
| `olmoearth` | Function | `get_data_in_locations` | `spec: DataRetrievalSpec`, `aoi: AOIWrapper`, `time_range: TimeRange` | `DataBundle` | Fetches EO data per spec, filtered to AOIs and time range. |
| `olmoearth` | Function | `create_project` | `name: str`, `description: str` | `ProjectRef` | Wraps `POST /projects`. |
| `olmoearth` | Function | `create_area` | `project: ProjectRef`, `aoi: AOIWrapper` | `AreaRef` | Wraps `POST /areas`. |
| `olmoearth` | Function | `create_dataset` | `project: ProjectRef`, `area: AreaRef`, `bands: list[str]`, `time_range: TimeRange` | `DatasetRef` | Configures a Studio dataset. |
| `olmoearth` | Function | `create_labelset` | `dataset: DatasetRef`, `schema: LabelSchema` | `LabelsetRef` | Creates a labelset; validates schema (`sample_category`, `es_label`, `oe_labels`) before submission. |
| `olmoearth` | Function | `upload_labels` | `labelset: LabelsetRef`, `path: str` | `int` | Imports a normalized GeoJSON / CSV / Shapefile of labels; returns count. |
| `olmoearth` | Function | `submit_prediction` | `kind: Literal["finetune","embed","reference"]`, `project: ProjectRef`, `dataset: DatasetRef`, `config: dict` | `PredictionRef` | Wraps `POST /predictions`. Three modes correspond to the three case-study methodologies. |
| `olmoearth` | Function | `poll_prediction` | `ref: PredictionRef` | `PredictionStatus` | Wraps `GET /predictions/{id}`. Exponential backoff. |
| `olmoearth` | Function | `fetch_results` | `ref: PredictionRef`, `kind: Literal["tiles","vectors","metrics"]` | `ResultBundle` | Wraps `GET /prediction-results/...` (XYZ raster tiles, MVT vector tiles, metrics JSON). |
| `olmoearth` | Function | `save_view` | `bundle: ResultBundle`, `name: str` | `FilePath` | Publishes layer back to the user's Studio project. |
| `eo` | Function | `search_stac` | `collection: str`, `aoi: AOIWrapper`, `time_range: TimeRange`, `cloud_lt: float = 0.2` | `STACItems` | Planetary Computer / Earthdata STAC query. |
| `eo` | Function | `sign_assets` | `items: STACItems` | `STACItems` | Planetary Computer asset signing. |
| `eo` | Function | `window_tile` | `aoi: AOIWrapper`, `size_m: int = 256` | `list[Window]` | Tiles AOI into fixed-size windows for rslearn dataset configs. |
| `utils` | Function | `add_area` | `aoi: AOIWrapper` | `AOIWrapper` | Adds an `area_sq_meters` field. |
| `utils` | Function | `add_buffer` | `aoi: AOIWrapper`, `distance_m: float` | `AOIWrapper` | Buffers geometries outward. |
| `utils` | Function | `to_geodataframe` | `aoi: AOIWrapper` | `geopandas.GeoDataFrame` | Deserialize for complex spatial ops (e.g. `sjoin`). |
| `utils` | Function | `from_geodataframe` | `gdf: geopandas.GeoDataFrame` | `AOIWrapper` | Re-serialize after manipulation. |
| `utils` | Function | `equal_frequency_bins` | `series: pandas.Series`, `n: int` | `pandas.Series` | Equal-frequency binning for class-imbalanced labels. |
| `utils` | Function | `spatial_train_val_split` | `aoi: AOIWrapper`, `k: int = 5` | `tuple[AOIWrapper, AOIWrapper]` | Spatial CV split (refuse random splits on auto-correlated AOIs). |
| `utils` | Function | `organize_save_order` | `bundles: list[tuple[str, ResultBundle]]` | `list[tuple[str, ResultBundle]]` | Deduplicates + orders layers (rasters first, vectors second, POIs last). |

Same catalog as CSV:

```csv
Module,Type,Name,Arguments,Return Type,Description
system,Tool,python,code (str),ExecutionResult,Run Python in a sandboxed interpreter; preloaded pandas/geopandas/xarray/rioxarray/shapely/pystac_client/planetary_computer/rslearn/olmoearth_projects; state persists across turns; no import statements.
system,Tool,search,queries (list[str]),SearchResponse,Web search for external references.
system,Tool,fetch,url (str) | headers (dict),FetchResponse,HTTP GET on documented endpoints.
olmoearth,Function,load_context,(none),StudioContext,Returns the user's active Studio project/areas/datasets/recent predictions.
olmoearth,Function,resolve_to_aoi,locations (list[str] max 30),AOIWrapper,Place names / watershed IDs / county FIPS / free text -> polygon geometries.
olmoearth,Function,search_dataset_spec,query (str),DatasetRetrievalSpecs,Searches Studio catalog + Planetary Computer STAC; returns analyzable rasters + visual-only layers.
olmoearth,Function,get_data_in_locations,spec (DataRetrievalSpec) | aoi (AOIWrapper) | time_range (TimeRange),DataBundle,Fetches EO data per spec filtered to AOIs and time range.
olmoearth,Function,create_project,name (str) | description (str),ProjectRef,POST /projects.
olmoearth,Function,create_area,project (ProjectRef) | aoi (AOIWrapper),AreaRef,POST /areas.
olmoearth,Function,create_dataset,project (ProjectRef) | area (AreaRef) | bands (list[str]) | time_range (TimeRange),DatasetRef,Configures a Studio dataset.
olmoearth,Function,create_labelset,dataset (DatasetRef) | schema (LabelSchema),LabelsetRef,Creates a labelset; validates sample_category/es_label/oe_labels.
olmoearth,Function,upload_labels,labelset (LabelsetRef) | path (str),int,Imports normalized GeoJSON/CSV/Shapefile labels; returns count.
olmoearth,Function,submit_prediction,kind ("finetune"|"embed"|"reference") | project (ProjectRef) | dataset (DatasetRef) | config (dict),PredictionRef,POST /predictions.
olmoearth,Function,poll_prediction,ref (PredictionRef),PredictionStatus,GET /predictions/{id} with exponential backoff.
olmoearth,Function,fetch_results,ref (PredictionRef) | kind ("tiles"|"vectors"|"metrics"),ResultBundle,GET /prediction-results/... .
olmoearth,Function,save_view,bundle (ResultBundle) | name (str),FilePath,Publishes layer back to user's Studio project.
eo,Function,search_stac,collection (str) | aoi (AOIWrapper) | time_range (TimeRange) | cloud_lt (float = 0.2),STACItems,Planetary Computer / Earthdata STAC query.
eo,Function,sign_assets,items (STACItems),STACItems,Planetary Computer asset signing.
eo,Function,window_tile,aoi (AOIWrapper) | size_m (int = 256),list[Window],Tiles AOI into fixed-size windows.
utils,Function,add_area,aoi (AOIWrapper),AOIWrapper,Adds area_sq_meters field.
utils,Function,add_buffer,aoi (AOIWrapper) | distance_m (float),AOIWrapper,Buffers geometries outward.
utils,Function,to_geodataframe,aoi (AOIWrapper),geopandas.GeoDataFrame,Deserialize for complex spatial ops.
utils,Function,from_geodataframe,gdf (geopandas.GeoDataFrame),AOIWrapper,Re-serialize after manipulation.
utils,Function,equal_frequency_bins,series (pandas.Series) | n (int),pandas.Series,Equal-frequency binning for imbalance.
utils,Function,spatial_train_val_split,aoi (AOIWrapper) | k (int = 5),tuple[AOIWrapper AOIWrapper],Spatial CV split.
utils,Function,organize_save_order,bundles (list[tuple[str ResultBundle]]),list[tuple[str ResultBundle]],Deduplicates and orders layers (rasters > vectors > POIs).
```

---

## 2. Harness data classes

Internal types passed between tools. Plain Python dataclasses; no schema dependency.

```python
import dataclasses
from typing import Literal, Union

@dataclasses.dataclass
class ToolExecutionError:
    message: str | None = None
    status_code: int | None = None

@dataclasses.dataclass
class TimeRange:
    """ISO-8601 strings, inclusive."""
    start: str
    end: str

@dataclasses.dataclass
class AOIWrapper:
    """Serialized geometry bundle. Mirrors Google Earth Agent's SGTWrapper."""
    error: Union[ToolExecutionError, None] = None
    count: int | None = None
    serialized_geojson: str | None = None
    crs: str | None = "EPSG:4326"

@dataclasses.dataclass
class StudioContext:
    project: Union["ProjectRef", None] = None
    areas: list["AreaRef"] | None = None
    datasets: list["DatasetRef"] | None = None
    recent_predictions: list["PredictionRef"] | None = None

@dataclasses.dataclass
class ProjectRef:
    id: str
    name: str

@dataclasses.dataclass
class AreaRef:
    id: str
    aoi: AOIWrapper

@dataclasses.dataclass
class DatasetRef:
    id: str
    bands: list[str]
    time_range: TimeRange

@dataclasses.dataclass
class LabelSchema:
    """Studio expects sample_category / es_label / oe_labels keys."""
    sample_category: str
    es_label: str
    oe_labels: list[str]

@dataclasses.dataclass
class LabelsetRef:
    id: str
    schema: LabelSchema

@dataclasses.dataclass
class DataRetrievalSpec:
    """lookup_strategy ∈ {'STUDIO_DATASET', 'PLANETARY_COMPUTER_STAC',
                          'EARTHDATA_CMR', 'NLDI', 'NHD', 'HUC'}."""
    data_type: str
    description: str
    lookup_keys: list[str]
    lookup_strategy: str

@dataclasses.dataclass
class DatasetRetrievalSpecs:
    error: Union[ToolExecutionError, None] = None
    specs: list[DataRetrievalSpec] | None = None

@dataclasses.dataclass
class STACItems:
    error: Union[ToolExecutionError, None] = None
    item_ids: list[str] | None = None
    asset_urls: dict[str, str] | None = None
    signed: bool = False

@dataclasses.dataclass
class Window:
    """Geographic window for tiling AOIs into rslearn dataset rows."""
    bbox: tuple[float, float, float, float]
    size_m: int
    crs: str = "EPSG:4326"

@dataclasses.dataclass
class DataBundle:
    error: Union[ToolExecutionError, None] = None
    items: list[STACItems] | None = None
    local_paths: list[str] | None = None

@dataclasses.dataclass
class PredictionRef:
    id: str
    kind: Literal["finetune", "embed", "reference"]

@dataclasses.dataclass
class PredictionStatus:
    ref: PredictionRef
    state: Literal["queued", "running", "succeeded", "failed"]
    progress: float | None = None
    eta_seconds: int | None = None

@dataclasses.dataclass
class ResultBundle:
    ref: PredictionRef
    tile_template: str | None = None  # XYZ template, e.g. /tiles/{z}/{x}/{y}.png
    vector_url: str | None = None     # MVT or GeoJSON URL
    metrics: dict | None = None

@dataclasses.dataclass
class FilePath:
    error: Union[ToolExecutionError, None] = None
    path: str | None = None
```

---

## 3. Operational rules

Hard constraints the harness enforces — refusals where appropriate, defaults where not.

1. **No raw coordinates in chat.** Never output lat/lon, WKT, or full GeoJSON in conversational responses. Save to a `FilePath` and reference the path. (Lat/lon and partner data are sensitive; traces must redact them.)
2. **Default temporal window.** When the user does not specify a time range, default to the trailing 12 months from today; surface the choice in the response.
3. **Default buffer.** When the user asks for "nearby" without a distance, use 1000 m via `utils.add_buffer`.
4. **Sandbox isolation.** `system:python` does not allow `import` statements; rely on the preloaded libraries. Persistent state across turns is allowed.
5. **Cost guard.** Refuse `submit_prediction(kind="finetune")` if the estimated cost exceeds the session budget. Surface the estimate and ask before proceeding.
6. **Spatial cross-validation.** When AOIs are spatially auto-correlated, refuse random train/val splits — use `utils.spatial_train_val_split`.
7. **Class-balance hygiene.** When label classes are skewed > 10:1, default to `utils.equal_frequency_bins` before submitting fine-tunes. Document in the run trace.
8. **Studio quota awareness.** OlmoEarth Studio caps API keys per account at 10. Cache `load_context` to one call per 5 minutes; do not regenerate keys programmatically.
9. **Async job pattern.** `submit_prediction` returns immediately with a `PredictionRef`. Downstream tools accept the ref and poll. Never block on synchronous training.
10. **Schema-validate labels before upload.** `create_labelset` rejects schemas missing the `sample_category` / `es_label` / `oe_labels` keys. MIME type and column types validated on `upload_labels`.
11. **Refuse silent overwrites.** `save_view` with a name that exists in the target project must require an explicit `overwrite=True`.
12. **Trace redaction.** OpenTelemetry traces redact lat/lon, partner-controlled attributes, and full API keys before export.

---

## 4. Underlying stack

The tool catalog above is the contract. Everything in this section is implementation guidance for the runtime that hosts the tools — included for reference, not as part of the agent's public surface.

| Layer | Reference |
|---|---|
| Harness | [ByteDance DeerFlow v2](https://github.com/bytedance/deer-flow) — LangGraph lead agent + subagents-as-tools + middleware chain + MCP-first tools. [`ARCHITECTURE.md`](https://github.com/bytedance/deer-flow/blob/main/backend/docs/ARCHITECTURE.md). |
| Skill packaging | [NVIDIA AI-Q Agent Skills](https://docs.nvidia.com/aiq-blueprint/latest/integration/agent-skills.html) implementing the open [agentskills.io](https://agentskills.io) spec — `SKILL.md` progressive disclosure + signed `skill-card.md`. |
| Vision–language model | [Prismatic VLMs](https://arxiv.org/abs/2402.07865) (Karamcheti et al., ICML 2024) + [unsloth/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4) on Blackwell with LoRA / projectors trained in BF16 via Unsloth. |
| Geospatial encoder stream | [OlmoEarth-v1-Large](https://huggingface.co/allenai/OlmoEarth-v1-Large) embeddings as a parallel input alongside Prismatic vision features. |
| Tooling protocol | MCP for outbound system access (Studio, Planetary Computer, NLDI, Earthdata, HF Hub). |
| Self-improvement | Reflexion / Self-Refine / repeated-sampling-plus-verifier for inference-time; STaR / SWiRL for train-time when traces accumulate. Grounded in [Stanford CS329A](https://cs329a.stanford.edu/). |
| Eval + observability | OpenTelemetry with lat/lon redaction; LangSmith dataset-from-traces; custom EO benchmark on a small frozen question set. |

The Studio API itself:

- Docs: https://docs.olmoearth.allenai.org/
- Auth: https://docs.olmoearth.allenai.org/authentication/ — `Authorization: Bearer <key>`; max 10 keys/account
- Live OpenAPI spec: https://olmoearth.allenai.org/api/v1/openapi.json
- Resources: Areas, Projects, Datasets, Labelsets, Labels, Annotations, Tasks, Predictions, PredictionResults, Users
- No `/models` or `/jobs` resources — async work is `Predictions` (request) + `PredictionResults` (outputs)
- Sample code: https://github.com/allenai/olmoearth_projects
- Foundation weights: https://huggingface.co/allenai/OlmoEarth-v1-Large

**Studio gaps that affect tool design** (unchanged from v0.1):
- Webhook / push-notification mechanism for job completion: **UNVERIFIED** — `poll_prediction` assumes polling.
- How a fine-tuned model is referenced when creating a new `Prediction`: **UNVERIFIED** — needs OpenAPI schema dive.
- Concrete rate limits, payload size caps: **UNVERIFIED** — `cost_guard` rule is precautionary.

---

## 5. End-to-end example

User: *"Map alfalfa fields in the Klamath basin from 2022–2024 Sentinel-2. I have ~300 ground-truth points in `points.geojson`."*

```python
# 1. Resolve the AOI
ctx = olmoearth.load_context()
basin = olmoearth.resolve_to_aoi(["Klamath HUC-8"])

# 2. Prep labels (validate the 8 prep pitfalls before submit)
labels_gdf = utils.to_geodataframe(eo.read_file("points.geojson"))
# ... schema mapping, equal-frequency binning if needed ...
schema = LabelSchema(sample_category="train", es_label="alfalfa", oe_labels=["alfalfa"])

# 3. Studio project + dataset
project   = olmoearth.create_project("Klamath alfalfa", "2022-2024")
area      = olmoearth.create_area(project, basin)
dataset   = olmoearth.create_dataset(project, area,
                bands=["B2","B3","B4","B8","B11","B12"],
                time_range=TimeRange("2022-01-01","2024-12-31"))
labelset  = olmoearth.create_labelset(dataset, schema)
n         = olmoearth.upload_labels(labelset, "points.geojson")

# 4. Pick method (300 labels is below the FT threshold -> embed+LP)
pred = olmoearth.submit_prediction(
    kind="embed", project=project, dataset=dataset,
    config={"head": "linear_probe", "spatial_cv_k": 5})

# 5. Wait, then publish
status = olmoearth.poll_prediction(pred)   # backs off automatically
bundle = olmoearth.fetch_results(pred, kind="tiles")
view   = olmoearth.save_view(bundle, "alfalfa_2022_2024")
```

The interpreter sandbox runs this script as a single `system:python` call. Operational rules 5, 6, 7 apply: cost-guard runs before step 4, spatial-CV is the explicit `spatial_cv_k=5`, class-balance is checked in label prep. Rule 1 means the agent's chat response references `view.path`, not the raw geometries.

---

## 6. Roadmap

| Phase | Duration | Deliverables | Exit criteria |
|---|---|---|---|
| **P0 — Scaffolding** | 2 weeks | Repo layout; `olmoearth` MCP server codegen'd from OpenAPI; Bearer-auth handshake; harness dataclasses (§2) as a `types.py` | One round-trip: `create_project` → `load_context` reflects new project |
| **P1 — Tool surface MVP** | 3 weeks | All `olmoearth.*` and `utils.*` tools implemented + tested against the live Studio API | Klamath example (§5) runs end-to-end with human approval gates |
| **P2 — Sandbox + harness** | 3 weeks | `system:python` interpreter with preloaded libs and persistent state; DeerFlow v2 lead-agent ported; middleware enforces rules §3 | All 12 rules in §3 verifiable from traces |
| **P3 — Model stack** | 4 weeks | Prismatic vision tower + MLP projector + LoRA adapters on `unsloth/Qwen3.6-35B-A3B-NVFP4` via Unsloth on Blackwell | Inference path through TRT-LLM or vLLM |
| **P4 — Geo encoder stream** | 3 weeks | `OlmoEarth-v1-Large` projector wired alongside Prismatic; expert-usage histograms logged | Held-out EO eval not degraded vs vision-only |
| **P5 — Observability** | 2 weeks | OpenTelemetry with redaction; LangSmith dataset-from-traces; small EO question bench | Weekly trendline dashboard live |
| **P6 — Self-improvement** | open | Reflexion + repeated-sampling-plus-verifier at inference; STaR / SWiRL once trace volume justifies | METR-horizon metric improves on the bench |

P0–P2 is the buildable core. P3–P4 adds the multimodal model. P5 turns on the feedback loop. P6 is graduated improvement.

---

## 7. References

**OlmoEarth Studio**
https://docs.olmoearth.allenai.org/ · https://docs.olmoearth.allenai.org/authentication/ · https://olmoearth.allenai.org/api/v1/openapi.json · https://docs.olmoearth.allenai.org/api/ · https://allenai.org/blog/olmoearth · https://allenai.org/blog/olmoearth-v1-1 · https://huggingface.co/allenai/OlmoEarth-v1-Large · https://github.com/allenai/olmoearth_projects · https://github.com/allenai/olmoearth_pretrain

**Harness — DeerFlow v2**
https://github.com/bytedance/deer-flow · https://deerflow.tech/ · https://github.com/bytedance/deer-flow/blob/main/backend/docs/ARCHITECTURE.md · https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/agents/factory.py · https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/agents/lead_agent/agent.py · https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/subagents/executor.py · https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/mcp/tools.py · https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/agents/thread_state.py

**Skills — NVIDIA AI-Q + agentskills.io**
https://docs.nvidia.com/aiq-blueprint/latest/integration/agent-skills.html · https://docs.nvidia.com/aiq-blueprint/latest/customization/mcp-tools.html · https://docs.nvidia.com/aiq-blueprint/latest/deployment/observability.html · https://github.com/NVIDIA-AI-Blueprints/aiq · https://github.com/NVIDIA/skills · https://agentskills.io · https://developer.nvidia.com/blog/add-a-specialized-deep-research-skill-to-agent-harnesses/ · https://developer.nvidia.com/blog/nvidia-verified-agent-skills-provide-capability-governance-for-ai-agents/

**Vision–language stack**
https://arxiv.org/abs/2402.07865 · https://github.com/TRI-ML/prismatic-vlms · https://huggingface.co/Qwen/Qwen3.6-35B-A3B · https://github.com/QwenLM/Qwen3.6 · https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4 · https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ · https://developer.nvidia.com/blog/train-an-llm-on-an-nvidia-blackwell-desktop-with-unsloth-and-scale-it/ · https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf · https://arxiv.org/abs/2310.03744 · https://arxiv.org/abs/2301.12597 · https://arxiv.org/abs/2312.06742 · https://arxiv.org/abs/2404.13046 · https://arxiv.org/abs/2407.12709 · https://arxiv.org/abs/2401.15947

**Self-improvement — Stanford CS329A and papers**
https://cs329a.stanford.edu/ · https://online.stanford.edu/courses/cs329a-self-improving-ai-agents · https://arxiv.org/abs/2303.11366 · https://arxiv.org/abs/2303.17651 · https://arxiv.org/abs/2203.14465 · https://arxiv.org/abs/2305.20050 · https://arxiv.org/abs/2407.21787 · https://arxiv.org/abs/2409.15254 · https://arxiv.org/abs/2310.04406 · https://arxiv.org/abs/2504.04736 · https://arxiv.org/abs/2210.03629 · https://arxiv.org/abs/2503.14499

---

*End of v0.2. Next pass should resolve the §4 unverified Studio gaps and add a CONTRIBUTING.md following [earth2studio](https://github.com/NVIDIA/earth2studio)'s shape.*
