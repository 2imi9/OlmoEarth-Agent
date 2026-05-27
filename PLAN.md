# OlmoEarth Agent

A tool that drives the [OlmoEarth Studio](https://allenai.org/blog/olmoearth) platform from natural-language briefs. It exposes a compact set of functions covering Studio's HTTP API, EO data fetch, and geometry utilities; runs them in a Python sandbox preloaded with the standard geospatial stack; and enforces a small list of operational constraints. The agent's contract is the tool catalog in §1. Everything below it is supporting detail.

**Status:** v0.4 spec, 2026-05-27. No runnable code yet.
**Scope:** Text-only LLM ([unsloth/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4)) with function calling. Multimodal stack (Prismatic + vision adapter + OlmoEarth embedding stream) and the train-time self-improvement loops are parked in §7 Future work — they re-activate only if a skill empirically needs them.
**Skill catalog:** [`SKILLS.md`](SKILLS.md) — 16 skills (Prep / Configure / Run / Analyze / Integrate / Report). The agent ships skills one PR at a time.
**Verification discipline:** every external claim has a real URL. Unverified items are flagged **UNVERIFIED** inline.
**Studio API spec version:** `0.1.0` (per `info.version` of [`openapi.json`](https://olmoearth.allenai.org/api/v1/openapi.json)). Pre-1.0 — fields and enums may change without notice.

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
| `olmoearth` | Function | `create_labelset` | `project: ProjectRef`, `spec: LabelsetSpec` | `LabelsetRef` | Wraps `POST /labelsets`. Creates labelset metadata (`name`, `description`, optional `template_id`); individual label classes are added with `create_label`. |
| `olmoearth` | Function | `create_label` | `labelset: LabelsetRef`, `name: str`, `color: str` | `LabelDef` | Wraps `POST /labels`. Adds one class (e.g. `("alfalfa", "#33CC66")`) to a labelset. |
| `olmoearth` | Function | `upload_labels` | `labelset: LabelsetRef`, `path: str` | `int` | Imports a normalized GeoJSON / CSV / Shapefile of labels; returns count. |
| `olmoearth` | Function | `submit_prediction` | `kind: Literal["finetune","embed","reference"]`, `project: ProjectRef`, `dataset: DatasetRef`, `config: dict` | `PredictionRef` | Wraps `POST /predictions`. Three modes correspond to the three case-study methodologies. |
| `olmoearth` | Function | `poll_prediction` | `ref: PredictionRef` | `PredictionStatus` | Wraps `GET /predictions/{id}`. Exponential backoff. |
| `olmoearth` | Function | `fetch_results` | `ref: PredictionRef`, `kind: Literal["tiles","vectors","metrics"]` | `ResultBundle` | Wraps `GET /prediction-results/...` (XYZ raster tiles, MVT vector tiles, metrics JSON). |
| `olmoearth` | Function | `pixel_value` | `ref: PredictionRef`, `aoi: AOIWrapper` | `list[PixelValueResult]` | Wraps `GET /prediction-results/{id}/pixel-value`. Pointwise model output per coord. Used by skills #5, #9, #10. |
| `olmoearth` | Function | `features_search` | `ref: PredictionRef`, `class_name: str \| None = None`, `limit: int = 100` | `list[FeatureMatch]` | Wraps `POST /prediction-results/features/search`. Returns vector features filtered by class. Used by skill #5. |
| `olmoearth` | Function | `fetch_embedding` | `aoi: AOIWrapper`, `model_size: Literal["nano","tiny","base","large"]`, `time_range: TimeRange` | `EmbeddingVector` | Pulls OlmoEarth embeddings (Nano/Tiny/Base/Large) for an AOI. Used by skills #4, #9. |
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
olmoearth,Function,create_labelset,project (ProjectRef) | spec (LabelsetSpec),LabelsetRef,POST /labelsets — labelset metadata (name/description/template_id).
olmoearth,Function,create_label,labelset (LabelsetRef) | name (str) | color (str),LabelDef,POST /labels — one class within a labelset.
olmoearth,Function,upload_labels,labelset (LabelsetRef) | path (str),int,Imports normalized GeoJSON/CSV/Shapefile labels; returns count.
olmoearth,Function,submit_prediction,kind ("finetune"|"embed"|"reference") | project (ProjectRef) | dataset (DatasetRef) | config (dict),PredictionRef,POST /predictions.
olmoearth,Function,poll_prediction,ref (PredictionRef),PredictionStatus,GET /predictions/{id} with exponential backoff.
olmoearth,Function,fetch_results,ref (PredictionRef) | kind ("tiles"|"vectors"|"metrics"),ResultBundle,GET /prediction-results/... .
olmoearth,Function,pixel_value,ref (PredictionRef) | aoi (AOIWrapper),list[PixelValueResult],GET /prediction-results/{id}/pixel-value (skills 5/9/10).
olmoearth,Function,features_search,ref (PredictionRef) | class_name (str | None) | limit (int = 100),list[FeatureMatch],POST /prediction-results/features/search (skill 5).
olmoearth,Function,fetch_embedding,aoi (AOIWrapper) | model_size ("nano"|"tiny"|"base"|"large") | time_range (TimeRange),EmbeddingVector,Pulls OlmoEarth embeddings (skills 4/9).
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
class LabelsetSpec:
    """Studio `LabelsetWrite` — metadata about a labelset.
    Matches `components.schemas.LabelsetWrite` in openapi.json v0.1.0."""
    name: str
    description: str | None = None
    template_id: str | None = None  # UUID of a Labelset template, if reusing

@dataclasses.dataclass
class LabelDef:
    """Studio `LabelWrite` — one class within a labelset.
    Matches `components.schemas.LabelWrite` in openapi.json v0.1.0."""
    name: str
    color: str           # hex, e.g. "#33CC66"
    labelset_id: str     # UUID back-reference

@dataclasses.dataclass
class LabelsetRef:
    id: str
    spec: LabelsetSpec
    labels: list[LabelDef]

@dataclasses.dataclass
class DataPrepLabelSchema:
    """Field names used by the OlmoEarth dataset-prep / rslearn layer
    (NOT Studio API). Used during label normalization in the
    olmoearth-data-prep skill BEFORE upload_labels hands data through
    to Studio. See PLAN.md §4 for the layer distinction."""
    sample_category: str   # "train" / "val" / "test"
    es_label: str
    oe_labels: list[str]

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
    """Reference to a Studio Prediction.

    `kind` is a CLIENT-SIDE dispatch key that selects the `config` shape
    we send to `submit_prediction`. The Studio API itself does not
    distinguish prediction kinds — every Prediction takes a `model_id`
    UUID and the fine-tune-vs-inference distinction lives in that
    UUID's lineage. See PLAN.md §4 for the model_id provenance gap.
    """
    id: str
    kind: Literal["finetune", "embed", "reference"]

@dataclasses.dataclass
class PredictionStatus:
    """Mirrors `components.schemas.PredictionStatus` in openapi.json v0.1.0.

    The API returns one of five states; the harness exposes `progress`
    as a normalized 0..1 float derived from the API's free-form
    `Prediction.progress` object (which has no documented schema).
    """
    ref: PredictionRef
    state: Literal["pending", "running", "completed", "failed", "cancelled"]
    progress: float | None = None     # normalized 0..1; see docstring
    eta_seconds: int | None = None    # client estimate; not from API

@dataclasses.dataclass
class ResultBundle:
    ref: PredictionRef
    tile_template: str | None = None  # XYZ template, e.g. /tiles/{z}/{x}/{y}.png
    vector_url: str | None = None     # MVT or GeoJSON URL
    metrics: dict | None = None

@dataclasses.dataclass
class PixelValueResult:
    """One row from /prediction-results/{id}/pixel-value."""
    lon: float
    lat: float
    value: float | int | None
    class_name: str | None = None

@dataclasses.dataclass
class FeatureMatch:
    """One match from /prediction-results/features/search."""
    feature_id: str
    class_name: str
    score: float
    bbox: tuple[float, float, float, float] | None = None

@dataclasses.dataclass
class EmbeddingVector:
    """OlmoEarth embedding for an AOI at one time slice."""
    aoi_id: str
    model_size: Literal["nano", "tiny", "base", "large"]
    time: str           # ISO-8601
    dim: int            # vector dimensionality
    values: list[float] # length == dim

@dataclasses.dataclass
class ProvenanceManifest:
    """Append-only manifest written by provenance_middleware (rule §3.13).
    One entry per Studio API call. Replay script reconstructs the
    PredictionRef chain from this."""
    run_id: str
    timestamp: str          # ISO-8601
    api_call: str           # e.g. "POST /predictions"
    request_hash: str       # sha256 of the request payload
    response_summary: dict  # ids, status codes, hashes — no raw geometry
    prediction_id: str | None = None
    model_id: str | None = None
    dataset_hashes: list[str] | None = None

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
13. **Provenance manifest.** Every `olmoearth.*` API call writes a `ProvenanceManifest` entry via the `provenance_middleware` (skill #14 in [`SKILLS.md`](SKILLS.md)). Sessions that disable provenance refuse to call `save_view`. The replay script is the single source of truth for "what did this run do?"

---

## 4. Underlying stack

The tool catalog above is the contract. Everything in this section is implementation guidance for the runtime that hosts the tools — included for reference, not as part of the agent's public surface.

| Layer | Reference |
|---|---|
| LLM | [unsloth/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4) — text-only, function calling. Served on Blackwell via [TensorRT-LLM ≥0.17](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) or vLLM. No fine-tuning in v0.4 scope. |
| Harness | [ByteDance DeerFlow v2](https://github.com/bytedance/deer-flow) — LangGraph lead agent + subagents-as-tools + middleware chain + MCP-first tools. [`ARCHITECTURE.md`](https://github.com/bytedance/deer-flow/blob/main/backend/docs/ARCHITECTURE.md). |
| Skills | 16 skills in [`SKILLS.md`](SKILLS.md), packaged per the open [agentskills.io](https://agentskills.io) spec ([NVIDIA AI-Q](https://docs.nvidia.com/aiq-blueprint/latest/integration/agent-skills.html) implementation reference). Upstream canonical home for skills #1–#4: [`2imi9/OlmoEarth-Skills`](https://github.com/2imi9/OlmoEarth-Skills) — the agent vendors them rather than re-implementing. Trigger-heavy `SKILL.md` frontmatter + signed `skill-card.md`. |
| Tooling protocol | MCP for outbound system access (Studio, Planetary Computer, NLDI, Earthdata, HF Hub, GEE, OSM, USGS Water, NOAA — last four added by skill #13). |
| Eval + observability | OpenTelemetry with lat/lon redaction; LangSmith dataset-from-traces; custom EO benchmark on a small frozen question set. Provenance manifest (rule §3.13) is the audit substrate. |

The Studio API itself:

- Docs: https://docs.olmoearth.allenai.org/
- Auth: https://docs.olmoearth.allenai.org/authentication/ — `Authorization: Bearer <key>`; max 10 keys/account
- Live OpenAPI spec: https://olmoearth.allenai.org/api/v1/openapi.json (v0.1.0)
- Resources: Areas, Projects, Datasets, Labelsets, Labels, Annotations, Tasks, Predictions, PredictionResults, Users
- No `/models` or `/jobs` resources — async work is `Predictions` (request) + `PredictionResults` (outputs)
- Sample code: https://github.com/allenai/olmoearth_projects
- Foundation weights: https://huggingface.co/allenai/OlmoEarth-v1-Large

**Studio gap-closure findings** (resolved 2026-05-27 from an `openapi.json` v0.1.0 schema dive; supersedes the v0.1/v0.2 UNVERIFIED notes):

- **Webhooks / push notifications: CLOSED — none exist.** No `/webhooks`, `/notifications`, `/events`, `/subscriptions` or `/callbacks` paths; no `Webhook` / `Notification` / `Callback` / `Subscription` / `Event` schemas; no top-level `webhooks` key; no `callbacks` field on any operation; no `webhook_url`/`notification_url`/`callback_url` property on `PredictionWrite`. **`GET /api/v1/predictions/{id}` polling is the only completion-detection path.** No documented polling interval or backoff guidance — the client picks its own. Operational rule §3.9 (async-by-ref) stands.
- **Fine-tuned model reference: PARTIALLY CLOSED.** `PredictionWrite.model_id` is a required UUID (`{"type": "string", "format": "uuid", "description": "ID of the model to run"}`). Same field appears on `PredictionRead.model_id` and `PredictionSearchRequest.model_id`. **However:** there is no `/api/v1/models` path, no `Model` / `ModelRead` / `ModelWrite` schema, and the `model-management/` docs page is a content stub. **How a client obtains a `model_id` (especially for a fine-tuned model) is not in the public surface.** Likely paths: (a) the Studio UI hands out the UUID after the user runs a fine-tune flow, (b) a fine-tune is a side-effect of a `Prediction` whose `model_id` references a base model. **Still UNVERIFIED — needs partner conversation with Ai2.**
- **Rate limits / quotas / payload caps: CLOSED — undocumented at the API surface.** No `x-rateLimit-*` extensions, no `429` responses, no `Retry-After` headers, no `quota_*` fields on `UserReadMe` / `Project*`. The only quota documented anywhere is "max 10 API keys per account" from the auth doc. Pagination caps exist but are not rate limits: `PredictionSearchRequest.limit ≤ 10000`, `DatasetSearchRequest.limit ≤ 1000`. `ApiErrorCode` enum is `["not_found_error", "permission_error", "server_error", "unauthorized_error", "validation_error", "not_implemented_error"]` — explicitly **no** `rate_limited_error`, reinforcing this. Operational rule §3.5 (cost guard) is precautionary; we keep it.

**Other findings worth surfacing** (newly verified during the schema dive):

- **API uses Firebase for identity.** `UserReadMe.firebase_user_id` is a real field. Public auth is bearer-token; the backend is Firebase. Affects how we'd ever extend to per-user OAuth.
- **`PredictionResultAccessLevel` enum:** `["private", "organization", "public"]` — visibility scoping at the result level, not the prediction level.
- **`PredictionUpdate` is rename-only.** Only mutable field is `name`. Cannot change `model_id`, `area_id`, or times after creation.
- **`PredictionRead.progress`** is `{type: "object", additionalProperties: true}` — free-form. No documented shape. The harness normalizes this to a `0..1` float on `PredictionStatus.progress`; see §2 docstring.
- **`PredictionWrite` required fields** are `{name, area_id, model_id, start_time, end_time, project_id}`. Note `area_id`, not `dataset_id` — area is the primary geographic anchor for a prediction. `dataset` in our §1 tool catalog is a sibling concept (the band/time-range config). Future PR will reconcile.
- **`TaskStatus` enum** for annotation tasks is different from `PredictionStatus`: `["pending", "in_progress", "completed", "cancelled", "to_be_reviewed", "reviewed"]`. Don't conflate.
- **Tile and pixel-value endpoints exist on prediction results:** `/prediction-results/{id}/tiles/{z}/{x}/{y}.{png|mvt}` and `/pixel-value`. Already reflected in §1 `fetch_results`.
- **`*-management/` doc pages are content stubs.** `/dataset-management/`, `/job-management/`, `/model-management/`, `/prediction-management/` all return real pages but with one-line descriptions and a link to the API browser — no prose docs.

**Remaining UNVERIFIED items:**
- How to obtain a `model_id` for a fine-tuned model from the public API (above).
- Whether the API will gain webhooks / a `/models` listing in a future minor version (spec is `0.1.0` and explicitly pre-1.0).
- Server-side rate limits exist almost certainly; empirical probing is out of scope for this doc-only PR.
- `PredictionResultRead.source` is a free-form string ("Source of the imagery") with no documented enum — value range unknown.

---

## 5. End-to-end example

User: *"Map alfalfa fields in the Klamath basin from 2022–2024 Sentinel-2. I have ~300 ground-truth points in `points.geojson`."*

```python
# 1. Resolve the AOI
ctx = olmoearth.load_context()
basin = olmoearth.resolve_to_aoi(["Klamath HUC-8"])

# 2. Prep labels (data-prep layer; NOT the Studio API schema)
labels_gdf = utils.to_geodataframe(eo.read_file("points.geojson"))
# ... data-prep schema mapping (DataPrepLabelSchema), equal-frequency binning if needed ...

# 3. Studio project + dataset + labelset (Studio API schema)
project   = olmoearth.create_project("Klamath alfalfa", "2022-2024")
area      = olmoearth.create_area(project, basin)
dataset   = olmoearth.create_dataset(project, area,
                bands=["B2","B3","B4","B8","B11","B12"],
                time_range=TimeRange("2022-01-01","2024-12-31"))
labelset  = olmoearth.create_labelset(project,
                LabelsetSpec(name="alfalfa", description="binary"))
alfalfa   = olmoearth.create_label(labelset, name="alfalfa", color="#33CC66")
n         = olmoearth.upload_labels(labelset, "points.geojson")

# 4. Pick method (300 labels is below the FT threshold -> embed+LP)
# Note: every Studio Prediction requires a model_id (UUID). The
# OlmoEarth foundation model's UUID is resolved via load_context().
pred = olmoearth.submit_prediction(
    kind="embed", project=project, dataset=dataset,
    config={"model_id": ctx.foundation_model_id,
            "head": "linear_probe", "spatial_cv_k": 5})

# 5. Wait, then publish
status = olmoearth.poll_prediction(pred)   # backs off automatically
bundle = olmoearth.fetch_results(pred, kind="tiles")
view   = olmoearth.save_view(bundle, "alfalfa_2022_2024")
```

The interpreter sandbox runs this script as a single `system:python` call. Operational rules 5, 6, 7 apply: cost-guard runs before step 4, spatial-CV is the explicit `spatial_cv_k=5`, class-balance is checked in label prep. Rule 1 means the agent's chat response references `view.path`, not the raw geometries.

---

## 6. Roadmap

Skills-first. Each P0–P3 phase is a single PR; each Skill-* row is one PR (per [`SKILLS.md`](SKILLS.md) numbering). Order beyond Skill-5 is driven by the case-study queue, not catalog order.

| Phase | Status | Deliverables | Exit criteria |
|---|---|---|---|
| **P0 — Scaffolding** | done — [PR #2](https://github.com/2imi9/OlmoEarth-Agent/pull/2) | `pyproject.toml`, `.pre-commit-config.yaml`, `CHANGELOG.md`, `.env.example`, empty `src/olmoearth_agent/` package | Scaffold lands; lint/test/type-check pipeline runs |
| **P1 — Studio API gap closure** | done — [PR #3](https://github.com/2imi9/OlmoEarth-Agent/pull/3) | PLAN.md §4 verified findings; dataclass corrections | All three v0.1/v0.2 UNVERIFIED items resolved |
| **P2 — Skills/scope rewrite** | this PR | PLAN.md v0.4 + SKILLS.md (16-skill catalog) | Multimodal parked; skills are the unit of progress |
| **P3 — LLM serving + harness MVP** | next | Qwen3.6-35B-A3B-NVFP4 served via TRT-LLM/vLLM with function calling enabled; DeerFlow v2 lead-agent ported; provenance middleware enforcing rule §3.13 | A trivial function call (e.g. `load_context`) round-trips through the LLM |
| **Skill-5 — `olmoearth-predict`** | after P3 | Core run primitive: submit / poll / pixel-value / features / files | `SKILLS.md` §5 acceptance criteria met against live Studio API |
| **Skill-1 — `olmoearth-studio-upload`** | TBD | MIME / 10K / multi-metric guards | A 12K-row Windows-origin GeoJSON imports clean |
| **Skill-8 — `olmoearth-evaluate`** | TBD | Spatial-block CV + NNDM-LOO | Numbers reproduce Ploton 2020's documented inflation pattern on a held-out test |
| **Skill-14 — `olmoearth-provenance`** | TBD (interleaved early) | Manifest middleware + replay script emitter | Replay script reconstructs a known-good prediction end-to-end |
| **Skills 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 15, 16** | TBD | One PR per skill, ordered by case-study demand | Per-skill exit criteria in [`SKILLS.md`](SKILLS.md) |

P0–P2 are the foundation. P3 + Skill-5 is the first vertical slice where the agent does anything real. Skills 14 + 8 land early because they're cross-cutting (provenance for audit; evaluation for honest metrics). Everything else follows demand.

---

## 7. Future work (parked)

Two tracks were in scope through v0.3 and are now explicitly deferred. They re-activate only if a v0.4-scope skill empirically needs them.

### 7.1 Multimodal stack

The text-only LLM with function calling is sufficient when Studio handles all imagery and returns metrics / GeoJSON / manifest hashes through tool returns. The agent does not need to *see* pixels to do its job. If a future skill (e.g. a successor to skill #11 cloud-mask-audit) cannot be solved by calling existing tools and genuinely needs image-level reasoning the LLM can't fake, this track re-opens.

Parked components:
- [Prismatic VLMs](https://arxiv.org/abs/2402.07865) (Karamcheti et al., ICML 2024) — fused vision encoder (DINOv2 + SigLIP)
- MLP / Q-Former / [MoVA](https://arxiv.org/abs/2404.13046) / [MoME](https://arxiv.org/abs/2407.12709) projector designs
- [OlmoEarth-v1-Large](https://huggingface.co/allenai/OlmoEarth-v1-Large) embedding stream as parallel input
- LoRA / projector training on Blackwell via [Unsloth + NVFP4](https://developer.nvidia.com/blog/train-an-llm-on-an-nvidia-blackwell-desktop-with-unsloth-and-scale-it/)
- Adapter references: [LLaVA-1.5](https://arxiv.org/abs/2310.03744), [BLIP-2](https://arxiv.org/abs/2301.12597), [Honeybee](https://arxiv.org/abs/2312.06742), [MoE-LLaVA](https://arxiv.org/abs/2401.15947)

### 7.2 Train-time self-improvement

Inference-time techniques that don't require weight updates (Reflexion / Self-Refine / repeated-sampling-plus-verifier) may be folded into individual skills as they ship. Train-time techniques (STaR / SWiRL / GRPO / DAPO) are parked until trace volume justifies the investment.

Parked references (from [Stanford CS329A](https://cs329a.stanford.edu/)):
- [Reflexion](https://arxiv.org/abs/2303.11366), [Self-Refine](https://arxiv.org/abs/2303.17651)
- [STaR](https://arxiv.org/abs/2203.14465), [SWiRL](https://arxiv.org/abs/2504.04736)
- [Large Language Monkeys](https://arxiv.org/abs/2407.21787), [Archon](https://arxiv.org/abs/2409.15254)
- [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300), [DAPO](https://arxiv.org/abs/2503.14476)
- Eval references: [METR long-task horizon](https://arxiv.org/abs/2503.14499), [KernelBench](https://arxiv.org/abs/2502.10517)

---

## 8. References

**OlmoEarth Studio**
https://docs.olmoearth.allenai.org/ · https://docs.olmoearth.allenai.org/authentication/ · https://olmoearth.allenai.org/api/v1/openapi.json · https://docs.olmoearth.allenai.org/api/ · https://allenai.org/blog/olmoearth · https://allenai.org/blog/olmoearth-v1-1 · https://huggingface.co/allenai/OlmoEarth-v1-Large · https://github.com/allenai/olmoearth_projects · https://github.com/allenai/olmoearth_pretrain

**Harness — DeerFlow v2**
https://github.com/bytedance/deer-flow · https://deerflow.tech/ · https://github.com/bytedance/deer-flow/blob/main/backend/docs/ARCHITECTURE.md · https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/agents/factory.py · https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/agents/lead_agent/agent.py · https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/subagents/executor.py · https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/mcp/tools.py · https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/agents/thread_state.py

**Skills — NVIDIA AI-Q + agentskills.io**
https://docs.nvidia.com/aiq-blueprint/latest/integration/agent-skills.html · https://docs.nvidia.com/aiq-blueprint/latest/customization/mcp-tools.html · https://docs.nvidia.com/aiq-blueprint/latest/deployment/observability.html · https://github.com/NVIDIA-AI-Blueprints/aiq · https://github.com/NVIDIA/skills · https://agentskills.io · https://developer.nvidia.com/blog/add-a-specialized-deep-research-skill-to-agent-harnesses/ · https://developer.nvidia.com/blog/nvidia-verified-agent-skills-provide-capability-governance-for-ai-agents/

**LLM (in-scope for v0.4)**
https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4 · https://huggingface.co/Qwen/Qwen3.6-35B-A3B · https://github.com/QwenLM/Qwen3.6 · https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/

**Parked tracks** — see §7.1 (multimodal) and §7.2 (self-improvement) above for full reference lists.

**Skill-layer citations** — see [`SKILLS.md`](SKILLS.md) (Ploton 2020, Meyer-Pebesma 2021, Skakun CMIX 2022, WorldCereal lessons, IAMAP, Samuel et al. reproducibility gap, Kedron-Holler 2022, NASA Similarity Search, AlphaEarth transfer literature).

---

*End of v0.4. Multimodal and self-improvement tracks are parked; skills are the unit of progress. Next pass: open Skill-5 PR after P3 (LLM serving + harness MVP) lands.*
