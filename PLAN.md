# OlmoEarth Agent

A tool that drives the [OlmoEarth Studio](https://allenai.org/blog/olmoearth) platform from natural-language briefs and verifies what it returns. It exposes a compact set of functions covering Studio's HTTP API, the verification layer (review sets, label samples, error rates and certified zones, through `olmoearth-inferencex`), and geometry utilities; dispatches them as function-call tools, with an opt-in `system:python` subprocess for light glue code (not a preloaded in-process interpreter); and enforces a small list of operational constraints. The agent's contract is the tool catalog in §1.

**Status:** v1.4.0, 2026-08-14 (unreleased changes in [`CHANGELOG.md`](CHANGELOG.md)). All 18 skills implemented. In the unreleased version the default registry is the Studio run loop plus the verification layer, which includes which of two maps of one time is more accurate (`plan_label_sample` with `design="disagreement"`, then `estimate_map_error`); 15 tools that train models outside Studio, need inputs no Studio run produces, repeat what Ai2's Explorer and Asta do, or name a winner with no test (#6's baseline comparison) are opt-in (`OLMOEARTH_OPT_IN_TOOLS=1`; [`SKILLS.md`](SKILLS.md#opt-in-tools)). Runs live against the Studio API and a local or hosted (Claude/ChatGPT/Gemini) LLM. See §6 and [`CHANGELOG.md`](CHANGELOG.md). (The catalog was consolidated 19 -> 16: data-prep #1/#2, embeddings #4/#17, and change-detection #6/#19 were each one capability split across two rows.)
**Scope:** Text-only LLM with function calling. Local default: [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) (4-bit `UD-IQ4_XS`, via llama.cpp); the bridge also accepts a hosted Claude / ChatGPT / Gemini backend (bring-your-own-key) selected at runtime. Multimodal stack (Prismatic + vision adapter + OlmoEarth embedding stream) and the train-time self-improvement loops are parked in §7 Future work; they re-activate only if a skill empirically needs them. Not built, by decision: data-layer catalogs, basemaps and layer styling, STAC or Earth Engine fetches, embedding fetches, and label-free change maps; Google Earth and Ai2's own tools do these, and this agent verifies their outputs instead.
**Skill catalog:** [`SKILLS.md`](SKILLS.md): 18 skills (Prep / Configure / Run / Analyze / Integrate / Report); skills #6, #8, #10, #15 and #16 are opt-in whole. The JEPA latent-change detector in the separate [`olmoearth-jepa-change`](https://github.com/2imi9/olmoearth-jepa-change) repo is not connected. The agent ships skills one PR at a time.
**Verification discipline:** every external claim has a real URL. Unverified items are flagged **UNVERIFIED** inline.
**Studio API spec version:** `0.1.0` (per `info.version` of [`openapi.json`](https://olmoearth.allenai.org/api/v1/openapi.json)). Pre-1.0: fields and enums may change without notice.

---

## 1. Tool catalog

The agent exposes the following functions. Code below the table renders the same content as CSV for easy copy-paste into a sheet. Rows built as tools today: `python` (`olmoearth_run_python`, opt-in), `load_context`, `create_project`, `submit_prediction`, `poll_prediction` (`olmoearth_get_prediction`), `fetch_results`, `pixel_value`, `compare_results`, `review_set_from_result`, `scores_from_file`, `plan_label_sample`, `estimate_map_error` and `certify_zone`; an area is created by `olmoearth_request_aoi`, which the user draws. The other rows are the design target, not code. The registry also holds tools this table does not list (`SKILLS.md` names them).

| Module | Type | Name | Args / Inputs | Returns | Description |
|---|---|---|---|---|---|
| `system` | Tool | `python` | `code: str` | `ExecutionResult` | Run Python for light glue between calls. **Opt-in** (`OLMOEARTH_RUN_PYTHON=1`; off by default): an isolated subprocess (`python -I`) with a wall-clock timeout + output cap, in a throwaway dir. **State does not persist across calls; use normal `import`s; the heavy geospatial/rslearn/GDAL stack is not guaranteed installed.** Surface long `rslearn` jobs to the user. *(A preloaded, import-free, state-persistent in-process interpreter is the original design target -- see `src/olmoearth_agent/tools/system.py`.)* |
| `system` | Tool | `search` | `queries: list[str]` | `SearchResponse` | Web search for external references (docs, papers, code examples). |
| `system` | Tool | `fetch` | `url: str`, `headers: dict = None` | `FetchResponse` | HTTP GET on documented endpoints (OlmoEarth Studio, NLDI). |
| `olmoearth` | Function | `load_context` | `limit: int = 50`, `offset: int = 0` | `StudioContext` | Returns the user's identity, organization and one page of projects (id, name, creation time) with the total count; `offset` pages. (Areas, datasets and recent predictions are the design target, not yet returned.) |
| `olmoearth` | Function | `resolve_to_aoi` | `locations: list[str]` (max 30) | `AOIWrapper` | Place names, watershed IDs (HUC-8/10/12), county FIPS, or free text -> polygon geometries. |
| `olmoearth` | Function | `create_project` | `name: str`, `description: str` | `ProjectRef` | Wraps `POST /projects`. |
| `olmoearth` | Function | `create_area` | `project: ProjectRef`, `aoi: AOIWrapper` | `AreaRef` | Wraps `POST /areas`. |
| `olmoearth` | Function | `create_dataset` | `project: ProjectRef`, `area: AreaRef`, `bands: list[str]`, `time_range: TimeRange` | `DatasetRef` | Configures a Studio dataset. |
| `olmoearth` | Function | `create_labelset` | `project: ProjectRef`, `spec: LabelsetSpec` | `LabelsetRef` | Wraps `POST /labelsets`. Creates labelset metadata (`name`, `description`, optional `template_id`); individual label classes are added with `create_label`. |
| `olmoearth` | Function | `create_label` | `labelset: LabelsetRef`, `name: str`, `color: str` | `LabelDef` | Wraps `POST /labels`. Adds one class (e.g. `("alfalfa", "#33CC66")`) to a labelset. |
| `olmoearth` | Function | `upload_labels` | `labelset: LabelsetRef`, `path: str` | `int` | Imports a normalized GeoJSON / CSV / Shapefile of labels; returns count. |
| `olmoearth` | Function | `submit_prediction` | `kind: Literal["finetune","embed","reference"]`, `project: ProjectRef`, `dataset: DatasetRef`, `config: dict` | `PredictionRef` | Wraps `POST /predictions`. Three modes correspond to the three case-study methodologies. |
| `olmoearth` | Function | `poll_prediction` | `ref: PredictionRef` | `PredictionStatus` | Wraps `GET /predictions/{id}`. Exponential backoff. |
| `olmoearth` | Function | `fetch_results` | `ref: PredictionRef`, `kind: Literal["tiles","vectors","metrics"]` | `ResultBundle` | Wraps `GET /prediction-results/...` (XYZ raster tiles, MVT vector tiles, metrics JSON). |
| `olmoearth` | Function | `pixel_value` | `ref: PredictionRef`, `aoi: AOIWrapper` | `list[PixelValueResult]` | Wraps `GET /prediction-results/{id}/pixel-value`. Pointwise model output per coord. Used by `compare_results` (skills #4, #5, #9) and `review_set_from_result` (#18). Studio returns no-data as a value (e.g. `-1` on a `[0, 1]` band): every grid sampler drops a value that is missing/NaN, equals the model's `nodata_value`, or lies outside the band's declared range, and reports `n_nodata_dropped`. |
| `olmoearth` | Function | `compare_results` | `result_ids: list[str]` (2-8), `mode: Literal["auto","pair","group","series","ensemble"] = "auto"`, `kind: Literal["cross_model","temporal"] \| None`, `property_name: str \| None`, `grid: int \| None`, `tolerance: float = 0.1`, `value_range`, `value_type`, `allow_different_properties: bool = False` | `Comparison` | How Studio results differ, with no ground truth (skills #4, #5, #9): samples every result on one grid through `pixel_value` and reads a `pair` (difference, RMSE, correlation, agreement; cross-model or temporal), a `group` of 2-6 models (pairwise + consensus), a date-ordered `series` of one model (steps, trajectories, calibrated shift) or an `ensemble` (per-cell spread as a confidence). `auto` picks pair / series / group from the results' models and dates. No-data dropped and counted; different properties refused; cells by `(row, col)`, never coordinates. |
| `olmoearth` | Function | `review_set_from_result` | `result_id: str`, `grid: int = 10` (2-16), `budgets: list[float] = [0.01, 0.05, 0.1]`, `property_name: str \| None`, `threshold: float \| None` | `ReviewSet` | Skill #18 for a Studio result: samples its band on a grid (one pixel-value per window), reads a `[0, 1]` score `s` as `[1 - s, s]` (stated assumption; other ranges need `threshold`, classification refused) and ranks by margin, least decided first. Windows by `(row, col)` and index; locations only in the saved scores file. |
| `olmoearth` | Function | `scores_from_file` | `run_dir: str` (under `OLMOEARTH_SCORES_ROOT`), `patch: int = 4`, `include_fill_class: bool = False` | `ScoresFile` | Skill #18 for a map from a direct model run (a scores raster and its `manifest.json`, e.g. `score_area.py` on a GPU cluster), not Studio: pools every pixel's logits or probabilities to windows with `oe_inferencex.review.assess.assess_prediction` (`is_logit` from the manifest; the raster read by `oe_inferencex.common.raster.read_raster`, a file it cannot read refused; the optional `inferencex` extra, >= 1.20.1, which brings rasterio), leaves out of the margin a label-fill channel the manifest marks, and writes a scores file that `review_set`, `plan_label_sample` and `compare_review` read. Returns the window grid, valid and no-data windows, classes, model and revision; no coordinates. |
| `olmoearth` | Function | `plan_label_sample` | `scores \| scores_path \| result_id`, `budget: int`, `design: Literal["confidence","proportional","random","class","disagreement"] = "confidence"`, `grid: int \| [rows, cols]` (a `result_id`: N or `[N, N]`, 2-16, default 10; two results: at most 12); for `"class"`: `min_per_class: int = 50`; for a class map or `"class"`: `width: float = 0.10`, `expected_error_rate`, `class_share`; for `"disagreement"`: `result_id_b \| scores_path_b`, `date_a`, `date_b`, `difference: float = 0.05` | `FilePath` (design) | Which windows to label for an error-rate estimate (`oe_inferencex.estimate.sample_for_estimation`, optional `inferencex` extra). A review set is not a sample. A Studio result's grid above 16 is capped, and the result's `sampling` states the grid used and the cap. A Studio classification result is a class map: `"random"` or `"class"` (strata by map class, Olofsson et al. 2014) only, with `plan.plan_error_rate`'s and `plan.plan_class`'s label budgets; the ranking designs are refused for it. With `design="disagreement"`, which of two maps is more accurate: two Studio results of the same area (sampled by `compare_results`' pair sampler) or two scores files of one window grid, refused unless `compare.dates_reading` reads them as one time; the windows where they differ, drawn by `estimate.sample_disagreement`, with `plan.plan_which_map`'s budget advice. |
| `olmoearth` | Function | `estimate_map_error` | `design_path: FilePath`, `wrong: list[int] \| labels_path`, `reference: list[int \| str] \| None` | `ErrorEstimate` | Error rate with the interval the design earns and its `method`; per-class accuracy; windows labelled without a design are checked as a random sample (a review set is refused) and returned with `design` `none (checked as a random sample)`, the package's reading of them under `estimated_as`. On a class map's or a `"class"` design, `estimate_error_rate(valid=True)`'s interval valid at every sample size first, then the design's, and `estimate_per_class(valid=True)` per class. On a `"disagreement"` design, the class seen per window (`?`: unjudged) graded by `estimate.compare_from_disagreement`: which map is more accurate, or neither (exact sign test up to a fifth of the differing windows, the interval past that), the 95% interval on the accuracy difference, the counts and the package's notes. Every estimate from a design file returns that file's `seed`. |
| `olmoearth` | Function | `certify_zone` | `design_path: FilePath` (random design), `wrong`, `alpha: float`, `delta: float = 0.1`, `rule: Literal["prefix","bonferroni"]` | `CertifiedZone` | The largest most-confident share of the map with error rate at most `alpha` (`oe_inferencex.estimate.certify_zone`); a stratified design, or a class map (no confidence to rank), is refused. A certified zone's `meaning` ends with the package's `ISSUED_NOTE` (what delta counts), verbatim, and `issued_note` holds it on its own; the result names the design file's design and `seed`. |
| `olmoearth` | Function | `save_view` | `bundle: ResultBundle`, `name: str` | `FilePath` | Publishes layer back to the user's Studio project. |
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
system,Tool,python,code (str),ExecutionResult,Opt-in (OLMOEARTH_RUN_PYTHON=1) light-Python glue; isolated subprocess (python -I) with timeout + output cap; state does NOT persist across calls; normal imports allowed; heavy geospatial stack not guaranteed installed.
system,Tool,search,queries (list[str]),SearchResponse,Web search for external references.
system,Tool,fetch,url (str) | headers (dict),FetchResponse,HTTP GET on documented endpoints (OlmoEarth Studio / NLDI).
olmoearth,Function,load_context,limit (int = 50) | offset (int = 0),StudioContext,Returns identity/organization and one page of projects with the total count (areas/datasets/recent predictions: design target).
olmoearth,Function,resolve_to_aoi,locations (list[str] max 30),AOIWrapper,Place names / watershed IDs / county FIPS / free text -> polygon geometries.
olmoearth,Function,create_project,name (str) | description (str),ProjectRef,POST /projects.
olmoearth,Function,create_area,project (ProjectRef) | aoi (AOIWrapper),AreaRef,POST /areas.
olmoearth,Function,create_dataset,project (ProjectRef) | area (AreaRef) | bands (list[str]) | time_range (TimeRange),DatasetRef,Configures a Studio dataset.
olmoearth,Function,create_labelset,project (ProjectRef) | spec (LabelsetSpec),LabelsetRef,POST /labelsets: labelset metadata (name/description/template_id).
olmoearth,Function,create_label,labelset (LabelsetRef) | name (str) | color (str),LabelDef,POST /labels: one class within a labelset.
olmoearth,Function,upload_labels,labelset (LabelsetRef) | path (str),int,Imports normalized GeoJSON/CSV/Shapefile labels; returns count.
olmoearth,Function,submit_prediction,kind ("finetune"|"embed"|"reference") | project (ProjectRef) | dataset (DatasetRef) | config (dict),PredictionRef,POST /predictions.
olmoearth,Function,poll_prediction,ref (PredictionRef),PredictionStatus,GET /predictions/{id} with exponential backoff.
olmoearth,Function,fetch_results,ref (PredictionRef) | kind ("tiles"|"vectors"|"metrics"),ResultBundle,GET /prediction-results/... .
olmoearth,Function,pixel_value,ref (PredictionRef) | aoi (AOIWrapper),list[PixelValueResult],GET /prediction-results/{id}/pixel-value (compare_results: skills 4/5/9; review_set_from_result: skill 18); no-data values (missing/NaN/model nodata_value/outside the declared range) dropped and counted.
olmoearth,Function,compare_results,result_ids (list[str]; 2-8) | mode ("auto"|"pair"|"group"|"series"|"ensemble") | kind | property_name | grid | tolerance | value_range | value_type | allow_different_properties,Comparison,How Studio results differ with no ground truth: pair / group / date-ordered series / ensemble on one pixel-value grid; no-data dropped; different properties refused; cells by (row col) only.
olmoearth,Function,review_set_from_result,result_id (str) | grid (int = 10) | budgets (list[float]) | property_name (str | None) | threshold (float | None),ReviewSet,Skill 18 for a Studio result: grid-sampled [0 1] score read as [1-s s] and ranked by margin; windows by (row col) and index.
olmoearth,Function,scores_from_file,run_dir (str; under OLMOEARTH_SCORES_ROOT) | patch (int = 4) | include_fill_class (bool = False),ScoresFile,Skill 18 for a direct model run's scores raster + manifest: pooled to windows by oe_inferencex assess (inferencex extra); label-fill channel left out; writes the scores file the review and estimation tools read; no coordinates.
olmoearth,Function,plan_label_sample,scores | scores_path | result_id; budget (int); design ("confidence"|"proportional"|"random"|"class"|"disagreement"); grid (int | [rows cols]; result_id: 2-16; two results: up to 12); class: min_per_class (int = 50); class map or class: width (float = 0.10) | expected_error_rate | class_share; disagreement: result_id_b | scores_path_b; date_a; date_b; difference (float = 0.05),FilePath,Which windows to label for an error-rate estimate (inferencex extra); a review set is not a sample; a capped Studio grid is stated; a Studio class map: random or class design only with plan_error_rate and plan_class budgets; disagreement: the windows where two maps of one time differ (sample_disagreement) with plan_which_map's budget advice; different dates refused.
olmoearth,Function,estimate_map_error,design_path (FilePath) | wrong (list[int]) or labels_path | reference (list[int | str] | None),ErrorEstimate,Error rate with the design's interval and method; per-class accuracy; a review set is refused as a sample; a class map or class design: the valid interval (estimate --valid) first; a disagreement design: which map is more accurate (or neither) by compare_from_disagreement (sign test or interval) and the accuracy difference's interval.
olmoearth,Function,certify_zone,design_path (FilePath; random design) | wrong | alpha (float) | delta (float = 0.1) | rule ("prefix"|"bonferroni"),CertifiedZone,Largest most-confident share with error rate at most alpha; stratified designs and class maps refused.
olmoearth,Function,save_view,bundle (ResultBundle) | name (str),FilePath,Publishes layer back to user's Studio project.
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
    """Studio `LabelsetWrite`: metadata about a labelset.
    Matches `components.schemas.LabelsetWrite` in openapi.json v0.1.0."""
    name: str
    description: str | None = None
    template_id: str | None = None  # UUID of a Labelset template, if reusing

@dataclasses.dataclass
class LabelDef:
    """Studio `LabelWrite`: one class within a labelset.
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
class Window:
    """Geographic window for tiling AOIs into rslearn dataset rows."""
    bbox: tuple[float, float, float, float]
    size_m: int
    crs: str = "EPSG:4326"

@dataclasses.dataclass
class PredictionRef:
    """Reference to a Studio Prediction.

    `kind` is a CLIENT-SIDE dispatch key that selects the `config` shape
    we send to `submit_prediction`. The Studio API itself does not
    distinguish prediction kinds; every Prediction takes a `model_id`
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
class ProvenanceManifest:
    """Append-only manifest written by ProvenanceLog.record_tool_call
    (src/olmoearth_agent/provenance/log.py; rule §3.13). One entry per
    tool call. Replay script reconstructs the PredictionRef chain from this."""
    run_id: str
    timestamp: str          # ISO-8601
    api_call: str           # e.g. "POST /predictions"
    request_hash: str       # sha256 of the request payload
    response_summary: dict  # ids, status codes, hashes, no raw geometry
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

Hard constraints the harness enforces: refusals where appropriate, defaults where not.

1. **No raw coordinates in chat.** Never output lat/lon, WKT, or full GeoJSON in conversational responses. Save to a `FilePath` and reference the path. (Lat/lon and partner data are sensitive; traces must redact them.)
2. **Default temporal window.** When the user does not specify a time range, default to the trailing 12 months from today; surface the choice in the response.
3. **Default buffer.** When the user asks for "nearby" without a distance, use 1000 m via `utils.add_buffer`.
4. **Sandbox isolation.** `system:python` (`olmoearth_run_python`) is **opt-in** (`OLMOEARTH_RUN_PYTHON=1`) and runs in an isolated subprocess: use normal `import`s, **state does not persist across calls**, and the geospatial stack is not guaranteed installed. The child runs with a **credential-scrubbed environment** (Studio/LLM keys removed) -- defence-in-depth, not a network sandbox. Use it for light glue/inspection; surface long `rslearn ingest/materialize/fit` jobs to the user.
   - **Egress guard.** Every outbound endpoint is validated against a per-capability host allowlist + an SSRF block (private/loopback/metadata ranges) before a request leaves the process (`security/egress.py`; `OLMOEARTH_EGRESS=audit` default / `enforce` / `off`, `OLMOEARTH_EGRESS_ALLOW` to extend). Redirects are re-validated against the allowlist (the fetchers do not blindly follow a 3xx off-allowlist). Informed by NVIDIA NemoClaw -- see [`docs/archive/nemoclaw-assessment.md`](docs/archive/nemoclaw-assessment.md).
   - **Workspace path guard.** Model-controlled file paths in tools (`out_dir` / `output_path` / `positives_path` / ...) are confined to a workspace root via `security/paths.py` (`safe_path`), so a `../../` or absolute path cannot read or write outside it. The root is `OLMOEARTH_OUTPUT_ROOT` if set, else `<cwd>/olmoearth_outputs`.
5. **Cost guard.** Refuse `submit_prediction(kind="finetune")` if the estimated cost exceeds the session budget. Surface the estimate and ask before proceeding.
6. **Spatial cross-validation.** When AOIs are spatially auto-correlated, refuse random train/val splits: use `utils.spatial_train_val_split`.
7. **Class-balance hygiene.** When label classes are skewed > 10:1, default to `utils.equal_frequency_bins` before submitting fine-tunes. Document in the run trace.
8. **Studio quota awareness.** OlmoEarth Studio caps API keys per account at 10. Cache `load_context` to one call per 5 minutes; do not regenerate keys programmatically.
9. **Async job pattern.** `submit_prediction` returns immediately with a `PredictionRef`. Downstream tools accept the ref and poll. Never block on synchronous training.
10. **Schema-validate labels before upload.** `create_labelset` rejects schemas missing the `sample_category` / `es_label` / `oe_labels` keys. MIME type and column types validated on `upload_labels`.
11. **Refuse silent overwrites.** `save_view` with a name that exists in the target project must require an explicit `overwrite=True`.
12. **Trace redaction.** OpenTelemetry traces redact lat/lon, partner-controlled attributes, and full API keys before export.
13. **Provenance manifest.** Every `olmoearth.*` API call writes a `ProvenanceManifest` entry via `ProvenanceLog.record_tool_call` (`src/olmoearth_agent/provenance/log.py`; surfaced by skill #14 in [`SKILLS.md`](SKILLS.md)). Sessions that disable provenance refuse to call `save_view`. The replay script is the single source of truth for "what did this run do?" The verification tools' calls (`plan_label_sample`, `estimate_map_error`, `certify_zone`) are also kept whole (`ProvenanceLog.verification`): skill #14's report states their numbers from them, never from figures the model types (a title or result typed with an accuracy figure is refused; a zone tried again on the same labels with another alpha, delta, rule or labels is not stated as certified), and skill #13's summary lists each with its design and seed (inline labels by a digest). The log is one request's: on the web UI each message is a request with a fresh log.

---

## 4. Underlying stack

The tool catalog above is the contract. Everything in this section is implementation guidance for the runtime that hosts the tools, included for reference, not as part of the agent's public surface.

| Layer | Reference |
|---|---|
| LLM | [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) 4-bit `UD-IQ4_XS`: text-only with function calling (model has a native vision encoder; we don't use it -- multimodal is parked, §7.1). Served via **llama.cpp** (`ghcr.io/ggml-org/llama.cpp:server-cuda -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS --jinja -ngl 999 -c 16384 --parallel 1 --no-mmap`); see [`docs/serving.md`](docs/serving.md) and [`docs/CANON.md`](docs/CANON.md). Agent sessions set `chat_template_kwargs.preserve_thinking=True`. NVFP4 dropped (doesn't fit 24 GB; CANON C4/C7). |
| Harness | [ByteDance DeerFlow v2](https://github.com/bytedance/deer-flow): LangGraph lead agent + subagents-as-tools + middleware chain + MCP-first tools. [`ARCHITECTURE.md`](https://github.com/bytedance/deer-flow/blob/main/backend/docs/ARCHITECTURE.md). |
| Skills | 18 skills in [`SKILLS.md`](SKILLS.md) (15 tools opt-in, behind `OLMOEARTH_OPT_IN_TOOLS=1`), packaged per the open [agentskills.io](https://agentskills.io) spec ([NVIDIA AI-Q](https://docs.nvidia.com/aiq-blueprint/latest/integration/agent-skills.html) implementation reference). The `SKILL.md` packages for skills #1-#3 and #17 ship inside the Python package, in [`src/olmoearth_agent/skills/packages/`](src/olmoearth_agent/skills/packages/), moved from the now-frozen [`2imi9/OlmoEarth-Skills`](https://github.com/2imi9/OlmoEarth-Skills). Trigger-heavy `SKILL.md` frontmatter. |
| Tooling protocol | In-process tool registry (`ToolRegistry`); outbound HTTP via `httpx`, gated by the per-capability egress allowlist + SSRF guard in `security/egress.py` (Studio, cloud-LLM endpoints, a local LLM on loopback, and for opt-in tools arXiv/OpenAlex and the HF datasets-server). External MCP connectors are a parked option (§7), not the current transport. |
| Eval + observability | OpenTelemetry with lat/lon redaction; LangSmith dataset-from-traces; custom EO benchmark on a small frozen question set. Provenance manifest (rule §3.13) is the audit substrate. |

The Studio API itself:

- Docs: https://docs.olmoearth.allenai.org/
- Auth: https://docs.olmoearth.allenai.org/authentication/, `Authorization: Bearer <key>`; max 10 keys/account
- Live OpenAPI spec: https://olmoearth.allenai.org/api/v1/openapi.json (v0.1.0)
- Resources: Areas, Projects, Datasets, Labelsets, Labels, Annotations, Tasks, Predictions, PredictionResults, Users
- No `/jobs` resource; `/models` is live but undocumented (`POST /models/search`, `GET /models/{id}`; absent from openapi v0.1.0, verified live 2026-05-30 - carries `model_type` fine_tuned vs embeddings). Async work is `Predictions` (request) + `PredictionResults` (outputs)
- Sample code: https://github.com/allenai/olmoearth_projects
- Foundation weights: https://huggingface.co/allenai/OlmoEarth-v1-Large

**Studio gap-closure findings** (resolved 2026-05-27 from an `openapi.json` v0.1.0 schema dive; supersedes the v0.1/v0.2 UNVERIFIED notes):

- **Webhooks / push notifications: CLOSED, none exist.** No `/webhooks`, `/notifications`, `/events`, `/subscriptions` or `/callbacks` paths; no `Webhook` / `Notification` / `Callback` / `Subscription` / `Event` schemas; no top-level `webhooks` key; no `callbacks` field on any operation; no `webhook_url`/`notification_url`/`callback_url` property on `PredictionWrite`. **`GET /api/v1/predictions/{id}` polling is the only completion-detection path.** No documented polling interval or backoff guidance; the client picks its own. Operational rule §3.9 (async-by-ref) stands.
- **Fine-tuned model reference: CLOSED by live probe, not by the spec.** `PredictionWrite.model_id` is a required UUID (`{"type": "string", "format": "uuid", "description": "ID of the model to run"}`). Same field appears on `PredictionRead.model_id` and `PredictionSearchRequest.model_id`. openapi v0.1.0 has no `/models` path and no `Model` / `ModelRead` / `ModelWrite` schema, and the `model-management/` docs page is a content stub, **but the live API serves `POST /models/search` and `GET /models/{id}`** (`GET /models` with no id is 404; verified 2026-05-30). The code uses both (`studio/client.py`): `StudioClient.search_models` lists the account's models (the web bridge labels the project tree's model level with it), and `StudioClient.get_model` reads one model record (`tools/sampling.model_summary`: name, `model_type` fine-tuned vs embeddings, `prediction_type`, no-data value, and whether a label field was trained on). The agent gets a reusable `model_id` from `olmoearth_search_predictions` (each prediction names its model, with that summary beside it); it never invents one.
- **Rate limits / quotas / payload caps: CLOSED, undocumented at the API surface.** No `x-rateLimit-*` extensions, no `429` responses, no `Retry-After` headers, no `quota_*` fields on `UserReadMe` / `Project*`. The only quota documented anywhere is "max 10 API keys per account" from the auth doc. Pagination caps exist but are not rate limits: `PredictionSearchRequest.limit <= 10000`, `DatasetSearchRequest.limit <= 1000`. `ApiErrorCode` enum is `["not_found_error", "permission_error", "server_error", "unauthorized_error", "validation_error", "not_implemented_error"]`, explicitly **no** `rate_limited_error`, reinforcing this. Operational rule §3.5 (cost guard) is precautionary; we keep it.

**Other findings worth surfacing** (newly verified during the schema dive):

- **API uses Firebase for identity.** `UserReadMe.firebase_user_id` is a real field. Public auth is bearer-token; the backend is Firebase. Affects how we'd ever extend to per-user OAuth.
- **`PredictionResultAccessLevel` enum:** `["private", "organization", "public"]`, visibility scoping at the result level, not the prediction level.
- **`PredictionUpdate` is rename-only.** Only mutable field is `name`. Cannot change `model_id`, `area_id`, or times after creation.
- **`PredictionRead.progress`** is `{type: "object", additionalProperties: true}`: free-form. No documented shape. The harness normalizes this to a `0..1` float on `PredictionStatus.progress`; see §2 docstring.
- **`PredictionWrite` required fields** are `{name, area_id, model_id, start_time, end_time, project_id}`. Note `area_id`, not `dataset_id`: area is the primary geographic anchor for a prediction. `dataset` in our §1 tool catalog is a sibling concept (the band/time-range config). Future PR will reconcile.
- **`TaskStatus` enum** for annotation tasks is different from `PredictionStatus`: `["pending", "in_progress", "completed", "cancelled", "to_be_reviewed", "reviewed"]`. Don't conflate.
- **Tile and pixel-value endpoints exist on prediction results:** `/prediction-results/{id}/tiles/{z}/{x}/{y}.{png|mvt}` and `/pixel-value`. Already reflected in §1 `fetch_results`.
- **`*-management/` doc pages are content stubs.** `/dataset-management/`, `/job-management/`, `/model-management/`, `/prediction-management/` all return real pages but with one-line descriptions and a link to the API browser, no prose docs.

**Remaining UNVERIFIED items:**
- Whether the live `/models` endpoints (above) will enter the documented spec, or change, in a future minor version, and whether the API will gain webhooks (spec is `0.1.0` and explicitly pre-1.0).
- Server-side rate limits exist almost certainly; empirical probing is out of scope for this doc-only PR.
- `PredictionResultRead.source` is a free-form string ("Source of the imagery") with no documented enum; value range unknown.

---

## 5. End-to-end example

User: *"Map alfalfa fields in the Klamath basin from 2022-2024 Sentinel-2. I have ~300 ground-truth points in `points.geojson`."*

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

The §1 calls above are illustrative pseudocode (the design catalog), dispatched as separate function-call tools rather than run inside one `system:python` sandbox. Operational rules 5, 6, 7 apply: cost-guard runs before step 4, spatial-CV is the explicit `spatial_cv_k=5`, class-balance is checked in label prep. Rule 1 means the agent's chat response references `view.path`, not the raw geometries.

---

## 6. Roadmap

Skills-first. Each P0-P3 phase is a single PR; each Skill-* row is one PR (per [`SKILLS.md`](SKILLS.md) numbering). Order beyond Skill-5 is driven by the case-study queue, not catalog order.

| Phase | Status | Deliverables | Exit criteria |
|---|---|---|---|
| **P0 - Scaffolding** | done - [PR #2](https://github.com/2imi9/OlmoEarth-Agent/pull/2) | `pyproject.toml`, `.pre-commit-config.yaml`, `CHANGELOG.md`, `.env.example`, empty `src/olmoearth_agent/` package | Scaffold lands; lint/test/type-check pipeline runs |
| **P1 - Studio API gap closure** | done - [PR #3](https://github.com/2imi9/OlmoEarth-Agent/pull/3) | PLAN.md §4 verified findings; dataclass corrections | All three v0.1/v0.2 UNVERIFIED items resolved |
| **P2 - Skills/scope rewrite** | done | PLAN.md v0.4 + SKILLS.md (18-skill catalog) | Multimodal parked; skills are the unit of progress |
| **P3 - LLM serving + harness MVP** | done - PRs #5/#6 | 4-bit GGUF served via llama.cpp (`docs/serving.md`) with function calling + `preserve_thinking=True`; DeerFlow-style lead-agent + tool registry + provenance middleware (rule §3.13) | A function call round-trips through the LLM, verified live |
| **Skill-5 - `olmoearth-predict`** | done - PRs #8/#11 | search / submit / poll / fetch-results | Verified live against the Studio API |
| **Skill-8 - `olmoearth-evaluate`** | done - PR #10 | Spatial-block CV inflation diagnostic + metrics + NNDM-LOO CV (CAST port) | Reproduces Ploton-2020 inflation on clustered data |
| **Skill-14 - `olmoearth-provenance`** | done - PR #7 | Manifest per tool call + replay skeleton | Recorded live during agent runs |
| **Vendored skills (now #1-#3)** | vendored - PR #9 | submodule `vendor/olmoearth-skills` + SkillLoader (data-prep, studio-job-config, embeddings); the packages have since moved into `src/olmoearth_agent/skills/packages/` and the submodule is gone | Agent loads a SKILL.md via progressive disclosure |
| **Skills #6, #7, #9, #10, #11, #12, #13, #15** | done - PRs #16-#21 | One PR per skill | All shipped; see [`CHANGELOG.md`](CHANGELOG.md) |

P0-P3 are done: the agent talks to the live Studio API, drives the LLM, records provenance, loads vendored skills, and computes honest accuracy. **All 18 skills are now shipped**, and in the unreleased version the default registry is the Studio run loop plus the verification layer, with 15 tools opt-in (see [`CHANGELOG.md`](CHANGELOG.md)); the remaining work is the verification layer on Studio results (class maps' error rates and which of two maps is more accurate are built, unreleased) and live-smoke verification on real data.

---

## 7. Future work (parked)

Two tracks were in scope through v0.3 and are now deferred. Re-activate only if a future skill empirically needs them.

**7.1 Multimodal stack.** Text-only + function calling is sufficient when Studio returns metrics / GeoJSON / manifest hashes through tool returns. Parked: [Prismatic VLMs](https://arxiv.org/abs/2402.07865), [LLaVA-1.5](https://arxiv.org/abs/2310.03744)/[BLIP-2](https://arxiv.org/abs/2301.12597)/[Honeybee](https://arxiv.org/abs/2312.06742)/[MoVA](https://arxiv.org/abs/2404.13046)/[MoE-LLaVA](https://arxiv.org/abs/2401.15947) projector designs, [OlmoEarth-v1-Large](https://huggingface.co/allenai/OlmoEarth-v1-Large) embedding stream, [Unsloth fine-tuning](https://github.com/unslothai/unsloth). Note: the Qwen3.6 checkpoint *does* ship a native vision encoder: re-opening this track means using or replacing that tower, not training one from scratch.

**7.2 Train-time self-improvement.** Inference-time techniques without weight updates ([Reflexion](https://arxiv.org/abs/2303.11366), [Self-Refine](https://arxiv.org/abs/2303.17651), repeated-sampling+verifier per [Monkeys](https://arxiv.org/abs/2407.21787)/[Archon](https://arxiv.org/abs/2409.15254)) may be folded into individual skills as they ship. Train-time ([STaR](https://arxiv.org/abs/2203.14465), [SWiRL](https://arxiv.org/abs/2504.04736), [GRPO](https://arxiv.org/abs/2402.03300), [DAPO](https://arxiv.org/abs/2503.14476)) parked until trace volume justifies. Eval framing: [METR horizon](https://arxiv.org/abs/2503.14499), [KernelBench](https://arxiv.org/abs/2502.10517). Source course: [Stanford CS329A](https://cs329a.stanford.edu/). An adopt-first proposal for the inference-time techniques (grounded per-skill self-check first; RL stays parked) is in [`docs/archive/self-improvement-proposal.md`](docs/archive/self-improvement-proposal.md).

---

## 8. References

**OlmoEarth Studio**
https://docs.olmoearth.allenai.org/ | https://docs.olmoearth.allenai.org/authentication/ | https://olmoearth.allenai.org/api/v1/openapi.json | https://docs.olmoearth.allenai.org/api/ | https://allenai.org/blog/olmoearth | https://allenai.org/blog/olmoearth-v1-1 | https://huggingface.co/allenai/OlmoEarth-v1-Large | https://github.com/allenai/olmoearth_projects | https://github.com/allenai/olmoearth_pretrain

**Harness: DeerFlow v2**
https://github.com/bytedance/deer-flow | https://deerflow.tech/ | https://github.com/bytedance/deer-flow/blob/main/backend/docs/ARCHITECTURE.md | https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/agents/factory.py | https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/agents/lead_agent/agent.py | https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/subagents/executor.py | https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/mcp/tools.py | https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/agents/thread_state.py

**Skills: NVIDIA AI-Q + agentskills.io**
https://docs.nvidia.com/aiq-blueprint/latest/integration/agent-skills.html | https://docs.nvidia.com/aiq-blueprint/latest/customization/mcp-tools.html | https://docs.nvidia.com/aiq-blueprint/latest/deployment/observability.html | https://github.com/NVIDIA-AI-Blueprints/aiq | https://github.com/NVIDIA/skills | https://agentskills.io | https://developer.nvidia.com/blog/add-a-specialized-deep-research-skill-to-agent-harnesses/ | https://developer.nvidia.com/blog/nvidia-verified-agent-skills-provide-capability-governance-for-ai-agents/

**LLM + serving**
https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF | https://huggingface.co/Qwen/Qwen3.6-35B-A3B | https://github.com/QwenLM/Qwen3.6 | https://github.com/ggml-org/llama.cpp | [`docs/serving.md`](docs/serving.md) | [`docs/CANON.md`](docs/CANON.md)

**Parked tracks:** see §7.1 (multimodal) and §7.2 (self-improvement) above for full reference lists.

**Skill-layer citations:** see [`SKILLS.md`](SKILLS.md) (Ploton 2020, Meyer-Pebesma 2021, Skakun CMIX 2022, WorldCereal lessons, IAMAP, Samuel et al. reproducibility gap, Kedron-Holler 2022, NASA Similarity Search, AlphaEarth transfer literature).

---

*Skills are the unit of progress; multimodal and self-improvement tracks are parked (§7). All 18 skills are shipped, 15 tools of them opt-in (see [`CHANGELOG.md`](CHANGELOG.md)); current work is the verification layer, live-smoke verification and refinement.*
