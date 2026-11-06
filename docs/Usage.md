# Usage

This page is the reference for the OlmoEarth Agent. Changes are in [CHANGELOG.md](../CHANGELOG.md).

## Installation

The agent needs Python 3.11 or later and [uv](https://docs.astral.sh/uv/). From a checkout:

```bash
git clone https://github.com/2imi9/OlmoEarth-Agent.git
cd OlmoEarth-Agent
uv sync --all-extras          # what `make setup` runs
```

The core package needs only `openai` and `httpx`. The extras are `serve` (FastAPI and uvicorn, for the web UI),
`claude` (`anthropic`, for Claude as the web UI's backend), `inferencex` (`olmoearth-inferencex[geo]` 1.20.3 or later,
which brings numpy and rasterio, on Python 3.11 to 3.13) and `dev` (pytest, ruff, black, mypy, interrogate,
pre-commit). `inferencex` is for skill #5's `olmoearth_scores_from_file`, `olmoearth_scores_from_studio`,
`olmoearth_plan_label_sample`, `olmoearth_estimate_map_error`, `olmoearth_certify_zone`, `olmoearth_write_certificate`,
`olmoearth_verify_certificate` and `olmoearth_compare_review`'s reading of dates. Without it they compute nothing and
give the install line, `pip install 'olmoearth-agent[inferencex]'`; an older olmoearth-inferencex is refused with the
upgrade line. `olmoearth_review_set`, `olmoearth_review_set_from_result`, `olmoearth_review_budget_ceiling` and the
rest of `olmoearth_compare_review` need no extra. You also need a Studio API key (Studio UI -> profile ->
**API Keys**). The local model needs Docker and an NVIDIA GPU ([LLM backends and serving](#llm-backends-and-serving));
the cloud-API path needs neither.

## Quick start

Pick the LLM behind the agent; both drive the same web UI.

```bash
make setup      # uv sync --all-extras
make bridge     # cloud API: the live web UI on http://localhost:8088, no local model
make up         # or a local model: start the LLM (the first run downloads about 17.7 GB), then the web UI
```

Open http://localhost:8088 and paste your Studio key. For a cloud API, open the status chip's menu (foot of the
sidebar), **Agent settings -> Model backend**, pick Claude, ChatGPT or Gemini and paste its key. Send a brief. From the
terminal, against the local model: set `OLMOEARTH_API_KEY`, then `make serve && make agent` (a sample brief; your own
with `Q="..."`).

| Target | What it does |
|---|---|
| `make setup` | `uv sync --all-extras` |
| `make serve` | `scripts/serve-llm.sh`: starts the llama.cpp container and waits up to 600 s for `/health` |
| `make down` | stops the LLM (`docker compose -f docker/llama.compose.yml down`) |
| `make up` | setup, serve, then bridge; Ctrl-C stops the UI and leaves the LLM running |
| `make agent` | `uv run olmoearth-agent "$(Q)"`; `Q` defaults to "How many OlmoEarth Studio projects do I have?" |
| `make bridge` | `uv run olmoearth-agent-serve --port 8088`: the live web UI |
| `make web` | the static demo web UI on port 8080 (`python -m http.server`): sample data, no backend |
| `make verify-demo` | the verification tools on olmoearth-inferencex's expert-annotated Dynamic World tile, with the `inferencex` extra and no Studio key or LLM; writes [VERIFICATION_DEMO.md](VERIFICATION_DEMO.md) (`ARGS="--work DIR --docs DIR"`) |

`make help` lists them. Without `make` (Windows with Git Bash, say), run what they run: `uv sync --all-extras`,
`./scripts/serve-llm.sh`, then `uv run olmoearth-agent-serve --port 8088` or `uv run olmoearth-agent "<brief>"`.
`olmoearth-agent BRIEF` runs one brief and prints the answer; `--max-turns N` caps the model round trips (default 8)
and `--show-trace` prints the tool calls to stderr. It always uses the model the `LLM_*` variables name and asks it for
short answers, as for the local model.

The settings ([.env.example](../.env.example) lists the main ones):

| Variable | Default | What |
|---|---|---|
| `OLMOEARTH_API_KEY` | none | the Studio key of `olmoearth-agent` and `olmoearth-agent-bridge`; the web UI sends its own with each request |
| `OLMOEARTH_BASE_URL` | `https://olmoearth.allenai.org/api/v1` | the Studio API |
| `LLM_ENDPOINT` | `http://localhost:8000/v1` | the OpenAI-compatible LLM server |
| `LLM_MODEL` | `unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS` | the served model id |
| `LLM_API_KEY` | `EMPTY` | the LLM server's key; a local llama.cpp needs none |
| `OLMOEARTH_READ_ONLY` | off | `1`, `true`, `yes` or `on`: a read-only Studio key ([Tools](#tools)) |
| `OLMOEARTH_OUTPUT_ROOT` | `<cwd>/olmoearth_outputs` | the workspace root: spilled results and the files the tools write |
| `OLMOEARTH_SCORES_ROOT` | unset | where scores, design, label and certificate files are read and written |
| `OLMOEARTH_EGRESS` | `audit` | the egress guard: `audit`, `enforce` or `off` (rule 4) |
| `OLMOEARTH_EGRESS_ALLOW` | none | more allowed hosts, comma-separated (a self-hosted Studio or LLM) |
| `OLMOEARTH_CHECK_NUMBERS` | on | `0` turns the answer's number check off |
| `OLMOEARTH_CHECK_ANSWER` | on | `0` turns the answer's other checks off |
| `OLMOEARTH_CHECK_CLAIMS` | off | `1` turns the claim check on |
| `OLMOEARTH_SOUL_PATH` | the packaged `harness/soul.md` | another soul file; a missing one falls back to the packaged soul |
| `OLMOEARTH_TOOL_RESULT_SPILL_BYTES` | `20000` | a larger tool result goes to a workspace file; `0` never spills |
| `OLMOEARTH_SKILLS_DIR` | the packaged `skills/packages/` | where instruction skills are found |
| `OLMOEARTH_WEBUI_DIR` | the repository's `webui/` | what the bridge serves |
| `HF_CACHE_DIR`, `HF_PROXY`, `HF_TOKEN` | see [serving](#llm-backends-and-serving) | the local LLM container's download |

## What it checks

Tool names below drop the `olmoearth_` prefix; [Skills](#skills) and [Tools](#tools) give the details.

| Question | Tools | Needs | Gives |
|---|---|---|---|
| Which windows to check first | `review_set`, `review_set_from_result` | scores, no labels | the windows to open first |
| How wrong one map is | `plan_label_sample`, `estimate_map_error` | labels on a planned sample | error rate, valid interval, per-class accuracy |
| Which part to trust | `certify_zone` | a random sample | a certified zone |
| A Studio class map without top-k scores | `plan_label_sample`, `estimate_map_error` | a random or class sample | error rate over its grid points |
| A Studio result's own files | `scores_from_studio` | top-k scores or a `[0, 1]` band | a scores file per UTM zone |
| A direct run, and its certificate | `scores_from_file`, `write_certificate`, `verify_certificate` | the run's scores | every window; a certificate |
| Which of two maps is more accurate | `plan_label_sample`, `estimate_map_error` | labels where they differ | which map, or neither |
| What labels Studio holds | `studio_label_inventory` | a prediction, model or project | counts by status, split and class; no error rate (no random design) |
| How Studio results differ | `compare_results` | 2 to 8 result ids | differences, no ground truth |

A Studio result is sampled on at most 16 x 16 points; its own files (`scores_from_studio`) cover every valid window.
Maps of different times are measured each on its own sample. Not built, by decision: data catalogs, basemaps and layer
styling, STAC or Earth Engine fetches, embedding fetches and label-free change maps.

The review set (both tools), the one-map plan and estimate, the certified zone, and the two-map plan and grade each
write a figure of the determination beside their files (`figure_path`, an SVG with no plotting dependency) and return
its legend (`figure_caption`), written from the result's own numbers. Panel (a) is the window grid by row and column,
with no coordinates, and what the determination used marked on it: the review set, the drawn windows, each label right
or wrong, the certified zone, or where two maps differ and which side each label took. Panel (b), where the answer rests
on one, is the error rate with its intervals, each level's upper bound as the rule decided it (with delta and the rule
stated), or the labels' counts. Tied confidences share a shade, a class map without scores is drawn flat (it has no
confidence), and a grid over 420 windows a side is drawn in blocks. The web UI shows the figure and caption with the
tool's result, above the answer (the SVG goes to the browser, not to the model). A chat reopened from the history shows
its figures again from the browser's IndexedDB, which keeps the most recently viewed figures (about 50 MB of figure
text, or 300) and is cleared with the chat history; deleting a chat deletes the figures no other chat shows; where a figure is no longer there (or the browser keeps no IndexedDB, as in some
private windows) it shows the caption and where the figure was saved. Demo mode shows the figures `make verify-demo`
recorded. The command line prints the path and the
caption after the answer, and so does the replay bridge. The answer leads with the result and what it assumes and does
not restate the figure. No figure: a review set ordered boundary-first, an estimate from `window_indices` without a
design, or a figure that could not be drawn or written (the determination stands).

## Skills

The agent ships 5 skills, by workflow stage. The first is an instruction package that `olmoearth_load_skill` loads on
demand; the others are tools.

| # | Stage | Skill | Example brief |
|---|---|---|---|
| 1 | Configure | [`olmoearth-studio-job-config`](#1-olmoearth-studio-job-config) | "I want per-pixel mangrove classification from Sentinel-2: fill in the Studio job wizard." |
| 2 | Run | [`olmoearth-predict`](#2-olmoearth-predict) | "Run a flood-extent prediction over this area for last month." / "Where do my three runs of this model disagree most?" |
| 3 | Report | [`olmoearth-provenance`](#3-olmoearth-provenance) | "List every call this run made, with each label plan's design and seed." |
| 4 | Report | [`olmoearth-case-narrative`](#4-olmoearth-case-narrative) | "Write a stakeholder brief for this result with the live map tiles." |
| 5 | Analyze | [`olmoearth-review-set`](#5-olmoearth-review-set) | "I can check 200 of these 4,000 windows: which 200?" / "These two land-cover results cover the same month: which is more accurate? I can label 100 windows." |

### 1. `olmoearth-studio-job-config`

**In:** a task description. **Out:** the answers to Studio's "new model" wizard: model name, model type (per-pixel,
window, detection or embeddings), foundation model (Nano, Tiny or Base), label field, training data (all or a
proportion), data split (spatial, recommended, or a metadata field), temporal context (a period, a single moment with
context, or a single moment), image sources (Sentinel-2 alone or with Sentinel-1) and surrounding area (160, 320, 640
by default, or 1280 m). It has 14 presets (crop type, mangrove, land cover, soil moisture, tree height, biomass,
ecosystem type, vessel detection, solar array detection, oil slick, flood, drought, burn scar, embeddings) and checks
the cross-field traps: detection with a small patch fails silently, Landsat is not selectable, embeddings mode does not
go with the label steps, and random splits leak spatial neighbours. Studio trains on Ai2's compute: the skill never
asks about the user's hardware or reroutes to local training. The package's `scripts/recommend.py` (`recommend`,
`validate`) is a helper for the user's own machine; the agent does not run it.

### 2. `olmoearth-predict`

**In:** a project, an area, a model id and a time range. **Out:** the prediction's id, status and results.
`olmoearth_submit_prediction` sends the area as `target_area_ids` and an optional `probability_top_k` as Studio's
`probability_output`, so a segmentation model's class map carries its top-k class scores (from the spec; not verified
with a read-only key). `olmoearth_get_prediction` polls; `olmoearth_fetch_results` returns tile URLs, properties,
declared outputs, period and `file_formats`. `olmoearth_pixel_value` reads one result at one lon/lat the user gives:
the model's output, not validated truth. A prediction needs an area: `olmoearth_request_aoi` shows a map in the web
UI, the user draws a rectangle or polygon, it is saved as a Studio area, and its `area_id` comes back next turn.

`olmoearth_compare_results` says how Studio results differ, with no ground truth. It samples 2 to 8 results on one
grid of windows over their shared extent (`pixel-value` per point). `pair`: difference, RMSE, correlation and
agreement, as `cross_model` (agreement, not accuracy) or `temporal` (net change, later minus earlier). `group` (2 to 6
models): a pairwise matrix, the consensus and the most contested cells. `auto` picks pair, series or group from the
results' models and dates. Every mode drops a missing or NaN value, the model's `nodata_value` and a value outside the
band's declared range (Studio returns no-data as a value, such as `-1` on a `[0, 1]` score), and counts them. Results
of different properties are refused unless `allow_different_properties`. Each correlation comes with its 95% interval
(Fisher's z). Cells are named by `(row, col)`, never coordinates. Each result costs grid² slow `pixel-value` calls, so
grids are capped.

#### Change over time (series mode)

`mode="series"` takes 3 to 8 results of one model over one area. It orders them by their predictions' `start_time`
(a missing date falls back to the given order, said loudly, since a wrong order flips every sign) and returns per-step
statistics, per-point trajectories (net change, slope, the largest step and its dates) and shift sizes calibrated to
the field's range (`value_range` from the user, else the observed range, named in `calibration.source`). Class maps
get class-transition counts. It is movement between run dates, not verified change on the ground.

#### Ensemble disagreement (ensemble mode)

`mode="ensemble"` reads two or more distinct results as an ensemble: per-point spread (standard deviation, coefficient
of variation and range; for classes, vote shares, normalized entropy and margin) folded into a confidence in [0, 1]
(`analysis/uncertainty.py`). It is model self-consistency, not a calibrated chance of being right. For a class map
without scores it is the one confidence signal Studio gives, but it does not rank which windows are wrong: measured
against the model's own margin on expert labels, it did not win. When scores exist, use #5.

### 3. `olmoearth-provenance`

Every tool call gets a manifest entry (rule 13): the tool, a sha256 of its arguments, the time and an id-only summary
of the result (never geometry). The replay skeleton lists the calls in order with their argument hashes; the arguments
are not stored, so it records the run and does not re-run it. `olmoearth_provenance_summary` lists each verification
call with what repeats its sample: design, budget and seed (for a `seed_from_map` plan, draw it again with that flag
and its alpha, delta and rule), the maps or scores file, the labels (inline ones counted, with a sha256 digest), the
zone's settings, and a certificate's date, map copy and verdict.

### 4. `olmoearth-case-narrative`

`olmoearth_case_narrative` writes a stakeholder Markdown report with live map tiles, a provenance section and a
freshness gate that strikes through tiles older than `freshness_window_hours` (24 by default). Its section "How
accurate is this map" reads the verification results from the same request's log, never from figures the model types.
On the web UI each message is a request with a fresh log, so an earlier message's estimate is stated only when made
again. It states, as returned: each error rate with every interval named by method and level, the labels, design and
seed; each certified zone and certificate with the package's `ISSUED_NOTE` and `ISSUERS_WORD` word for word; each
which-map verdict with its test; labels changed after a result was seen; and the tools' `must_state` limits and the
package's notes. A zone tried again on one design file with other settings or labels is never stated as certified.
With no estimate it says none was made, and an accuracy figure is forbidden; a title or result typed with an accuracy
figure is refused.

### 5. `olmoearth-review-set`

**In:** per-window scores (logits or probabilities), a direct run's files, or a Studio result with scores; a review
budget; for an estimate, a label budget and then the labels. **Out:** the review set and its ceiling, an error rate,
per-class accuracy, a certified zone and certificate, or which of two maps is more accurate. The verification steps
wrap [olmoearth-inferencex](https://github.com/2imi9/olmoearth_inferenceX); every number is the package's.

**The review set.** Windows are ranked by the model's top-1 minus top-2 margin, least confident first, optionally
boundary-first (how many of the 8 neighbours have another class); exact ties go to the higher index first, as in the
package's `assess.review_order`. A long list is saved whole to a CSV, without coordinates. A margin orders a review;
it is not the chance a window is wrong. `olmoearth_review_budget_ceiling` guards capture figures: no rule catches more
than `min(1, budget / error_rate)` of the errors, so 0.45 at a 10% budget and a 20% error rate is 90% of what was
reachable. Each result carries `facts`, `must_state` (at most three limits of at most 25 words), `forbidden_claims`
and an `evidence_scope` sentence (the full text in the file `evidence_detail_path` names). `olmoearth_compare_review`
says where two inferences of the same windows differ (10 inline, all in a file), not which is right; given `date_a`,
`date_b` and `labels_date`, it adds the package's reading across dates.

#### Where the scores come from

Studio exposes no logits, and a margin cannot be had from a hard class. Bring your own (rslearn `model predict`, an
embeddings probe, any exported softmax), or:

- **A Studio `[0, 1]` score** of a two-class task: `olmoearth_review_set_from_result` samples it on a grid (one
  `pixel-value` per window, no-data dropped and counted) and reads `s` as `[1 - s, s]`, for ranking only, as the
  result says. Another regression band needs a `threshold`; a class map without top-k scores is refused.
- **A Studio class map with top-k scores** (submitted with `probability_top_k` of 2 or more): each rank's softmax
  score (`P_rank1_prob`, ...) is a band of 0 to 254; windows rank by rank-1 minus rank-2. The result says the scores
  are uncalibrated, quantized to 1/254 and graded by no recorded experiment. Every label design works on it, and a
  random one can certify a zone; `olmoearth_compare_review` refuses its scores file.
- **A Studio result's own files**: `olmoearth_scores_from_studio` downloads the native output of a top-k class map or
  a `[0, 1]` score band (one GeoTIFF per UTM zone), its kind decided from `result_metadata` first. Each zone is its own
  map, a run directory under `<scores root>/studio_results/`, pooled as below; with no `patch`, the smallest side from
  4 up whose windows a ranking takes (at most 200,000). Top-k scores are decoded by Studio's files README,
  `(value - 1) / 254`; on the kelp result `raw / 254` fit better, and the order is the same while no rank-1 score is
  raw 0. A certificate binds the decoded `scores.tif`, not Studio's file. The zip is untrusted input: only the
  GeoTIFFs and README are read, never extracted by name, sizes capped (`max_mb`, 1024). Assumed, not verified: the
  README's layout for k above 2, and the result's model as its prediction's `model_id`.
- **A direct model run**: `olmoearth_scores_from_file` reads, under the scores root, what
  `oe-inferencex from-olmoearth DATASET --out RUN` writes (`scores_<EPSG>.tif` and `olmoearth_output.json`, which
  names no model or classes, so classes are `class_0` and on) or what the research record's
  [`score_area.py`](https://github.com/2imi9/olmoearth_inferenceX-record/blob/main/scripts/score_area.py) writes
  (`scores.tif`, `manifest.json`). It pools to windows of `patch` pixels (4 by default; a window counts when half its
  pixels are valid; pixels past the last whole window are left out and counted as `pixels_outside_window_grid`) with
  the package's `assess_prediction`, so the review set and label plans match the package's.
  A window's confidence is its mean top-1 minus top-2 margin for logits, its mean top-1 probability for probabilities.
  A label-fill channel the manifest marks (AWF's channel 9) is left out unless `include_fill_class`. Refused: a link or
  manifest pointing outside the root, a raster whose sha256 is not the manifest's, class ids (with how to get
  probabilities: `output_probs: true`, a float32 layer, the fine-tuned checkpoint) and the rslearn dataset itself.

Windows are named by `(row, col)` and index; locations stay in the saved scores file.

#### How wrong the map is

A review set is chosen to hold errors, so it is not a sample: labelling it and dividing overstates the error rate (1.8
to 5.8 times on olmoearth-inferencex's exp78). `olmoearth_plan_label_sample` draws a design and saves a labelling
sheet: `confidence` (margin strata, Neyman allocation; the default), `proportional`, `random` (needed for a certified
zone), `class` or `disagreement`. Label every row; a random sheet's first rows serve only when labelling stops for a
reason unrelated to the labels, and then give no zone. `olmoearth_estimate_map_error` gives the interval valid at every
sample size first, then the design's, with the package's `VALID_NOTE` word for word. The design's is exact under a
random design and nominal under the confidence and proportional ones (Wilson's; 91.3% coverage at 100 labels on a
200-window map, exp105). The answer must state the valid interval and the design. With reference classes, per-class
accuracy comes the same way: producer's beside user's accuracy, and the classes drawn too little or too much. Windows
labelled without a design are checked as a possible random sample (a review set is refused).

`olmoearth_certify_zone` gives the largest most-confident share with error rate at most `alpha` (`rule` `prefix` or
`bonferroni`; `delta` 0.1 by default), on random designs only. Its meaning ends with the package's `ISSUED_NOTE`: delta
counts every sample, those that certify nothing included, so a zone is not a 1 - delta chance of being right.

A Studio class map without top-k scores has nothing to rank: no review set, no zone, no ranking design. Labels still
measure it over its grid points with `design='random'` or `'class'` (Olofsson et al. 2014: strata by map class, at
least `min_per_class` each, 50 by default). The plan gives the package's budgets: `plan.plan_error_rate` for an
interval no wider than `width` (0.10), and `plan.plan_class` for a `class_share`. The reviewer writes the class seen,
blind under the class design. A `reference_class` filled on only some rows is refused.

#### Which of two maps is more accurate

Where two maps of the same windows and time agree, both are right or both wrong, so labels where they differ decide,
with far fewer labels than a whole-map sample and no scores. `design='disagreement'` takes two Studio results
(`result_id`, `result_id_b`; at most 12 points a side; a `[0, 1]` score cut at 0.5, another regression band only with
a `threshold`) or two scores files of one grid (`scores_path`, `scores_path_b`). Dates come from the predictions or
`date_a` and `date_b`: different, overlapping or one-sided dates are refused, since a difference can be change on the
ground. Missing dates are noted: a window where the maps differ is an error in one map only if both show one time. It
draws windows at random where the maps differ, with `plan.plan_which_map`'s budget for a `difference`
(0.05). The reviewer writes the class seen, or `?`. The estimate names the more accurate map or neither, by the exact
sign test while the labels are at most a fifth of the differing windows and by the interval past that, with the 95%
interval on a minus b. The answer must state the test or interval and the labels; naming a map without its grading is
forbidden, and the check reads a map named by its letter. It says which map is better on these windows, not either
map's accuracy, nor which model is better on other areas, dates or training runs.

#### A certificate anyone with the map can check

A certified zone of a direct run's map can travel as olmoearth-inferencex's certificate: a STAC Item stating the
guarantee, the zone as a GeoTIFF mask, the labels and digests binding them to the map.

1. `olmoearth_plan_label_sample` with `design='random'`, `seed_from_map` and `alpha` (and `delta`, `rule`), on a
   scores file `olmoearth_scores_from_file` wrote from a GeoTIFF over every channel, runs `oe-inferencex sample
   --design random --seed-from-map` in `<scores root>/certificates/`. The seed comes from the map's digest and the
   settings, so anyone with the map draws the same windows. Inline scores, a Studio grid sample and `alpha`, `delta` or
   `rule` without `seed_from_map` are refused.
2. The labels first graded on the plan are fixed in its sheet; other labels later are refused (rule 11).
3. `olmoearth_write_certificate`, with those labels and the map's `date` (default: a `score_area.py` manifest's date
   window; a `from-olmoearth` run records none, so pass `date`), runs `oe-inferencex certify`, checks it gives the
   windows `olmoearth_certify_zone` gives, and builds and verifies the bundle. No zone, no certificate; never a second
   one.
4. `olmoearth_verify_certificate` checks it against a copy of the map under the scores root (`map_path` when the
   certificate names its map elsewhere); against the very file it names, it shows only that the two agree. `window`
   `[row, col]` asks whether a window is in the zone.

Every result states that the labels, the map's date and a single draw are the issuer's word. A certificate restates
the zone and adds no guarantee.

#### Evidence and limits

On Ai2's published embedding suite the margin's AUROC for errors is above 0.5 in all 3,560 cells of 16 encoders,
OlmoEarth's four sizes among them (exp79). With OlmoEarth Base, the 10% of units with the lowest margin hold a median
21.4% of the errors on the suite's 24 tasks, against 10% for a random 10% (exp70). Ensemble, cross-model and two-view
disagreement and feature-space typicality each ranked errors no better than the margin on expert labels
([the record](https://github.com/2imi9/olmoearth_inferenceX-record)). Limits: the evidence is for the OlmoEarth family
at window level against seven references; the suite's no-model controls are near chance; on one Sen1Floods11 flood
event a no-model NDWI control matched confidence; and scores that order a review well can be badly calibrated.

## Tools

The agent registers 25 tools by default (`skills/registry.py`). All 25 are sent on every turn.

| Tool | What |
|---|---|
| `olmoearth_load_context` | The user, organization and one page of projects (`limit` 50, `offset`) with the total |
| `olmoearth_create_project` | `POST /projects` (writes) |
| `olmoearth_request_aoi` | Asks the user to draw an area in the web UI; saved as a Studio area |
| `olmoearth_search_predictions` | Predictions with each model's name, type and `prediction_type`: where a `model_id` comes from |
| `olmoearth_submit_prediction` | `POST /predictions`, with `probability_top_k` (writes) |
| `olmoearth_get_prediction` | A prediction's status and `model_id` |
| `olmoearth_fetch_results` | A prediction's results (`scan_limit` at most 1000) |
| `olmoearth_get_prediction_result` | One result by id, without its geometry |
| `olmoearth_pixel_value` | One result's output at one lon/lat the user gives |
| `olmoearth_studio_label_inventory` | The annotations of a model's training selector, or a project's, counted |
| `olmoearth_compare_results` | How 2 to 8 results differ: pair, group, series or ensemble |
| `olmoearth_review_set` | The review set from scores, inline or a scores file |
| `olmoearth_review_set_from_result` | The review set of a Studio result on a grid (10 by default, 2 to 16) |
| `olmoearth_scores_from_file` | A direct run's files to a scores file |
| `olmoearth_scores_from_studio` | A Studio result's own files to one scores file per UTM zone |
| `olmoearth_compare_review` | Where two inferences of the same windows differ, not which is right |
| `olmoearth_review_budget_ceiling` | `min(1, budget / error_rate)`, and the share of it a quoted capture reached |
| `olmoearth_plan_label_sample` | Which windows to label, by design, with a labelling sheet |
| `olmoearth_estimate_map_error` | The error rate and per-class accuracy, or which of two maps is more accurate |
| `olmoearth_certify_zone` | The certified zone of a random design |
| `olmoearth_write_certificate` | The zone packed into a certificate |
| `olmoearth_verify_certificate` | A certificate checked against a copy of the map |
| `olmoearth_case_narrative` | The stakeholder report |
| `olmoearth_provenance_summary` | The run's calls, the verification calls' seeds and the replay skeleton |
| `olmoearth_load_skill` | An instruction skill's `SKILL.md` |

**Read-only mode.** Studio's keys can be read-only, and `/users/me` does not show it. With `OLMOEARTH_READ_ONLY=1` the
two tools that write to Studio (`olmoearth_create_project`, `olmoearth_submit_prediction`) are left out, and the
capability card says why. Drawing an area stays: the web UI saves it with `POST /areas` and answers a refused save
with HTTP 403. A request Studio refuses with 401 or 403 raises `StudioAccessError`, which says nothing was created or
changed and is never retried. Writes with a read-only key have not been tried live.

## Answer checks

The model reads the brief and calls a tool; the harness runs it (its arguments checked against the tool's schema),
feeds the result back and repeats until the model answers. The system prompt is the soul (`harness/soul.md`) plus a
capability card built from the tools' code: what each tool does, needs and cannot do (`harness/capabilities.py`). A
large result goes to a workspace file. At the turn cap the last tools run, then one call without tools asks for an
answer. When a tool fails the same way twice in a run, its result tells the model to stop retrying and report the limit.

Before an answer is shown, the harness checks (`harness/checks.py`) its numbers against the tool results and the
user's messages; its directions, places and magnitudes against the facts the tools computed; its claims of files saved
or items listed against what the run did; and the claims the tools forbid or require. One more call asks the model to
fix every problem found. A sentence that still fails is marked `[unverified: <check>]`; nothing is deleted. A required
statement left out is appended as the tool states it. `OLMOEARTH_CHECK_NUMBERS=0` turns off the number check,
`OLMOEARTH_CHECK_ANSWER=0` the rest. With `OLMOEARTH_CHECK_CLAIMS=1` the agent's own model also reads the answer
against the run's record and flags what no tool output supports, in the same rewrite (`harness/claim_check.py`; at
most two more model calls).

## Operational rules

The harness enforces these. They keep their numbers so code comments can cite them as rule N; the numbers not listed
are retired. One tool result and one tool schema still write rule 1 as "rule 3.1".

**Rule 1. No raw coordinates in chat.** Never output lat/lon, WKT or full GeoJSON in a conversational response; save it
to a file and give the path. Tools name windows and cells by `(row, col)` and index
(`tests/tools/test_compare_results.py::test_no_mode_returns_a_coordinate`).

**Rule 4. No code execution.** No tool runs code the model writes. The one child process, olmoearth-inferencex's
command line (`tools/inferencex.run_cli`: this interpreter with `-P`, no shell, a 600 s limit), runs with the Studio,
LLM and cloud keys removed from its environment (`security/child_env.py`): defence in depth, not a network sandbox.
Training outside Studio (a local rslearn pipeline) is outside the agent, as the soul says: it says so and hands over no
pipeline code. The studio-job-config package's `scripts/recommend.py` is the user's to run.

- **Egress guard.** Every outbound endpoint is checked against a per-capability host allowlist and an SSRF block
  (private, loopback and metadata ranges) before a request leaves the process (`security/egress.py`;
  `OLMOEARTH_EGRESS=audit` by default, `enforce` or `off`; `OLMOEARTH_EGRESS_ALLOW` to extend). Only the starting
  endpoint is checked: the Studio client and the local model server's health probe do not follow redirects, while
  the OpenAI and Anthropic SDK clients follow them unchecked (the Anthropic SDK sends its key header along). Each
  SDK client is handed the endpoint that was checked.
- **Workspace path guard.** The files the tools write stay under their root (`security/paths.py`, `safe_path`, under
  `tools/review_set.write_root`: `OLMOEARTH_SCORES_ROOT` when set, else the workspace root), so a `../../` or absolute
  name cannot write outside it. The model-chosen read paths (`scores_path`, `design_path`, `labels_path`,
  `certificate_path`, `map_path`) are confined by `tools/review_set.read_roots`: `OLMOEARTH_SCORES_ROOT` when set, else
  the working directory plus the workspace root.

**Rule 8. Studio quota awareness.** Studio caps API keys at 10 per account; never create or rotate keys from code.

**Rule 9. Async job pattern.** `olmoearth_submit_prediction` returns at once with the prediction's id; later tools take
the id and poll. Never block on a job.

**Rule 11. Refuse silent overwrites.** `olmoearth_write_certificate` never writes over a certificate already written for
its plan, and a `seed_from_map` plan's sheet never takes other labels than those first graded
(`tests/tools/test_certificate_tool.py::test_a_certificate_rests_on_the_labels_first_given`).

**Rule 13. Provenance manifest.** Every tool call writes a `ProvenanceManifest` entry through
`ProvenanceLog.record_tool_call` (`provenance/log.py`). The verification calls are also kept whole
(`ProvenanceLog.verification`), and skill #4's report states their numbers from them.

## LLM backends and serving

The model sits behind one seam, `chat(messages, tools=..., mode=...)` (`llm/types.py`). The default is a local model
through `OlmoEarthLLM` (`llm/client.py`); the web UI can use a hosted Claude, ChatGPT or Gemini; and a Claude Code
session can play the model ([below](#claude-code-as-the-model)).

**The local default.** [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) at quant
`UD-IQ4_XS` (`DEFAULT_MODEL` in `llm/config.py`): a 4-bit GGUF, about 17.7 GB, of
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (35B total, 3B active, hybrid Gated-DeltaNet and
MoE), used text-only with function calling. [llama.cpp](https://github.com/ggml-org/llama.cpp)
(`ghcr.io/ggml-org/llama.cpp:server-cuda`) serves it on `http://localhost:8000/v1`. `make serve` runs
[`scripts/serve-llm.sh`](../scripts/serve-llm.sh), which starts [`docker/llama.compose.yml`](../docker/llama.compose.yml)
(by hand: `docker compose -f docker/llama.compose.yml up`) and waits for `/health`. The server's flags:
`-hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS --host 0.0.0.0 --port 8000 --jinja -ngl 999 -c 16384 --parallel 1 --no-mmap`.

`--jinja` uses the GGUF's chat template, so the model emits structured `tool_calls` (without it the client recovers
calls from text). `--no-mmap` avoids slow mmap over the Docker Desktop or WSL file bridge. `-ngl 999` puts every layer
on the GPU. `-c 16384 --parallel 1` is one 16,384-token slot: llama.cpp splits `-c` across slots, and a `SKILL.md` with
several tool results needs the room. The compose file bind-mounts the host's Hugging Face cache, so the GGUF downloads
once; `scripts/serve-llm.sh` sets `HF_CACHE_DIR` (default `$HF_HOME`, else `~/.cache/huggingface`, in the
`C:/Users/<you>/...` form on Windows; from PowerShell or cmd pass it yourself). `HF_PROXY` (as
`http://host.docker.internal:<port>`) is for a first download behind a host proxy; `HF_TOKEN` only for a gated repo.

**Hardware.** An NVIDIA GPU, Blackwell recommended, with the NVIDIA container toolkit. The GGUF fits a 24 GB card with
about 5 GB for the KV cache; larger 4-bit quants and NVFP4 weights leave too little.

**Defaults.** Each chat sends `chat_template_kwargs.preserve_thinking=True`, which keeps thinking across turns, and the
`thinking_general` preset (temperature 1.0, top_p 0.95, top_k 20, presence penalty 1.5). `llm/presets.py` holds the
model card's presets (`thinking_general`, `thinking_coding`, `instruct_general`, `instruct_reason`) and the claim
check's `instruct_verify`; the rewrite uses `thinking_coding`. The timeout is 600 s, the output budget 32,768 tokens.

**Cloud API.** Web UI, Agent settings -> Model backend: Claude (the native Anthropic SDK; `uv sync --extra claude`),
ChatGPT or Gemini (the OpenAI-compatible client). The key goes to the provider with each request (`X-LLM-Backend`,
`X-LLM-Key`, optional `X-LLM-Model`); `GET /api/llm/models` lists the provider's models. If the local backend is
chosen and no local model answers, the UI says so.

**A GPU cluster with vLLM.** Any OpenAI-compatible server works through `LLM_ENDPOINT`, `LLM_MODEL` and `LLM_API_KEY`,
the only model settings the agent reads. `nvidia/Qwen3.8-27B-NVFP4` under vLLM has run the agent end to end:

```bash
vllm serve nvidia/Qwen3.8-27B-NVFP4 --port 8000 --max-model-len 65536 --kv-cache-dtype fp8_e4m3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
```

Where the nodes lack `nvcc`, use the vLLM OpenAI container under Apptainer. vLLM listens on all interfaces: set
`VLLM_API_KEY` to a random token, pass the same token as `LLM_API_KEY`, and keep it out of the repository, logs and
chat. Forward port 8000 through the login host and run `make bridge`.

## Claude Code as the model

Claude can play the model two ways. Natively (`AnthropicLLM`, `llm/anthropic_client.py`), Claude runs through the
Anthropic API in the web UI, with an Anthropic key and the `claude` extra, paid per token; the model sees exactly the
loop's request. Through the replay bridge (`olmoearth-agent-bridge`), a Claude Code session writes the model's
responses; it needs ssh to the cluster, a checkout and the Studio key there, and the model sees the loop's request as
text beside its own context. The bridge runs the agent's real loop, tools and checks one model call per command, and
holds nothing between commands. Each command builds a `LeadAgent` as `olmoearth-agent` does (but with the claim check
and the short-answer clause off), with `ScriptedLLM` returning the recorded responses in order. Tool handlers are
wrapped, so argument checks, envelopes, retry hints, spill and provenance run as live; a cached result
(`tool_cache.jsonl`) is served again, and only a call past the cache runs. The step stops at the first unanswered model
call (`pending.json`) or writes the final answer. Each response keeps the sha256 of the request it answered; a replay
that makes another request or tool call stops.

```bash
olmoearth-agent-bridge begin --run-dir D --brief-file brief.txt [--history-file H] [--max-turns 8]
olmoearth-agent-bridge step --run-dir D --response-file response.json --call N
olmoearth-agent-bridge show --run-dir D --text     # the pending call again; nothing runs
olmoearth-agent-bridge prompt                      # the system prompt
olmoearth-agent-bridge tools [--run-dir D]         # the offered tools' specs
```

`-` reads stdin; `--call N` names the call answered, so a response sent twice never answers the next call; `step`
with no response starts a run whose first step failed. A response is `{"tool_calls": [{"name", "arguments"}]}` or
`{"content": "..."}`. The view holds the unseen messages, the system prompt (or its sha256 when unchanged) and the
tools' specs when they change. Use only argument names the specs declare: an undeclared one is ignored.

From the Mac, [`scripts/oe-bridge.sh`](../scripts/oe-bridge.sh) runs each command as one `ssh <host> srun` step:
`begin RUN BRIEF.txt [MAX_TURNS] [--text]`, `step RUN [RESPONSE.json] [--call N] [--text]`, `show RUN [--text]`,
`prompt`, `tools [RUN]`. Briefs and responses go over ssh's stdin, through no shell. The Studio key is read on the
cluster from `~/.config/olmoearth-agent/studio.env`. It is read-only unless `OE_BRIDGE_ALLOW_WRITES=1`, and a step with
another setting than `begin`'s is refused. The script's header sets the host (`OE_BRIDGE_HOST`), the paths, the srun
resources and `OE_BRIDGE_QOS`. Each step replays from the start, so it slows as the run grows. The skills
[`olmoearth-agent-model`](../.claude/skills/olmoearth-agent-model) and
[`olmoearth-agent-bridge`](../.claude/skills/olmoearth-agent-bridge) say how to play the model and run the bridge.

The run directory is private (0700) and holds account names and whole tool results: `brief.json`, `responses.jsonl`,
`tool_cache.jsonl`, `events.jsonl`, `pending.json`, `provenance.json`, and `step.lock` while a step runs (a second step
is refused; a lock older than 25 minutes is taken over). After a timeout or dropped connection, the step may still
run: give each wrapper call the longest Bash timeout (600000 ms), then run `show` once (never a polling loop). If a
step is running, wait; if the call moved on, answer the new one; if not, send the same response again with
`--call N`. A failed step saved no response. Tool calls it ran stay cached for the same response; another response
drops them (`dropped_tool_calls`), though a Studio write among them may have happened.

Unlike a live run: the model also holds its own context; the mode sets no sampling; tool results pass through JSON
once; provenance times are renewed each replay; and the check switches, `OLMOEARTH_READ_ONLY`, the soul and the code
must stay as at `begin`. Not yet tried live: a key that can write.

## Web UI

`make web` serves `webui/` as a static demo: sample projects, nothing sent. Opened as a file, or served by anything but
the bridge, the page stays a demo. `make bridge` (`olmoearth-agent-serve`; `--host 127.0.0.1`, `--port 8088` by
default) serves it as the live agent: a multi-turn chat with saved history, your Studio projects as a tree (project,
model, predictions, results), and briefs streamed through `LeadAgent.run_stream` as Server-Sent Events.

- **Keys.** The Studio key ("Add API key"), each LLM provider's key and the chat history stay in the browser's
  localStorage. The Studio key goes with each request in `X-Olmoearth-Key` (or `Authorization: Bearer`) and a
  provider's key in `X-LLM-Key`; the bridge forwards them. It reads no Studio or provider key from the environment,
  only the local model's `LLM_*` settings.
- **Agent settings** (the status chip's menu, foot of the sidebar). Max steps per run: 4, 6, 8, 10 or 12 (8 by default;
  the server accepts 1 to 12). Model backend: Local, Claude, Gemini or ChatGPT, each with its key and a detected model.
  Auto-route: a short read-only lookup runs on the local model instead of the hosted one (`X-LLM-Route: auto`,
  `llm/router.py`); it only ever moves a run to the local model.
- **In the chat.** `/` opens a menu of the 5 skills and pins the run to one. Text files and PDFs can be attached (the
  local model cannot read images). The area tool shows a map to draw on. Saved results (comparisons, the report as
  Markdown) stay in the sidebar. The workflow rail shows once a run calls a pipeline tool, with the stages reached
  (`harness/workflow.py`, mirrored in `js/run.js`).

| Endpoint | What |
|---|---|
| `GET /api/health` | liveness, the LLM and Studio wired, whether the local LLM answers; flips the page to live |
| `GET /api/skills`, `GET /api/llm/models` | each skill and the pipeline stages its tools touch; a hosted backend's models |
| `GET /api/projects`, `GET /api/projects/{id}/predictions`, `GET /api/predictions/{id}/results` | the project tree |
| `GET /api/results/{id}/extent`, `GET /api/tile/{z}/{x}/{y}` | a result's bounding box and tiles (a cached proxy, Studio's host only) |
| `GET /api/pixel-value` | one pixel of a result (the difference scan) |
| `POST /api/areas`, `GET /api/projects/{id}/areas`, `GET /api/areas/{id}` | create, list and fetch Studio areas |
| `POST /api/run` | a brief as SSE; body `{"brief", "max_turns"?, "history"?, "forced_skill"?}` |

The page has no build step: `index.html`, `styles.css` (tokens at `:root`) and ES modules under `js/` (`main.js` the
entry, `api.js` every bridge call).

In demo mode, a brief asking which windows to check first or how wrong a map is replays what `make verify-demo`
recorded on the Dynamic World tile (its review set, and its error rate from 300 random labels), and says so; other
briefs show the account's sample projects or explain what a Studio key would add. `webui/demo/record_demo.py` records
the README's walkthrough from these replays in headless Chromium (Playwright and imageio-ffmpeg, not package
dependencies; its docstring gives the commands) and writes `olmoearth-agent-demo.mp4`, the animated
`olmoearth-agent-demo.webp` the README shows (GitHub plays no video stored in a repository) and the cover
`olmoearth-agent-demo.png`.

## Security

Rule 4 states the guards. The egress allowlist: `studio` is `olmoearth.allenai.org`; `llm-cloud` is
`api.anthropic.com`, `api.openai.com` and `generativelanguage.googleapis.com`; `llm-local` is loopback only. `audit`
logs an unknown host and `enforce` refuses it; the guard resolves no DNS, so the allowlist is the main control, and it
covers the agent's own calls, not the child process. Beyond your own laptop, set `OLMOEARTH_EGRESS=enforce`. The web
UI's Studio and LLM keys stay in the browser and go with each request; the bridge never stores or logs them. A run and the
model list each check a hosted backend's endpoint and hand the SDK client that endpoint with the key (`enforce` refuses
a host off the list with HTTP 403); Claude's is always `api.anthropic.com`, so `ANTHROPIC_BASE_URL` in the bridge's
environment does not redirect a browser user's key. The Anthropic SDK still reads `ANTHROPIC_AUTH_TOKEN` from that
environment and sends it alongside a browser user's key, so leave it unset where users bring their own. A result's
`download_token` is a credential, masked in every error. The workspace root is not the working directory, so a written file cannot overwrite served code.

## Studio API notes

- Base `https://olmoearth.allenai.org/api/v1`, `Authorization: Bearer <key>`
  ([docs](https://docs.olmoearth.allenai.org/)), responses `{records, meta, errors}`. The
  [OpenAPI spec](https://olmoearth.allenai.org/api/v1/openapi.json) is 0.1.0, pre-1.0: fields and enums may change.
  `studio/client.py` is written by hand against it and live probes; do not invent endpoints.
- Resources: Areas, Projects, Datasets, Labelsets, Labels, Annotations, Tasks, Predictions, PredictionResults, Users.
  No `/jobs`: async work is a Prediction (the request) and its PredictionResults (the outputs).
- `/models` is live but not in the spec: `POST /models/search`, `GET /models/{id}`; it gives `model_type` (fine-tuned
  or embeddings), `prediction_type`, the no-data value and training `data_selectors`. After a 405, or a 404 as for an
  unrouted path, the client asks it no more.
- No webhooks: polling `GET /predictions/{id}` is the only way to see completion, with no documented interval.
  Prediction statuses: pending, running, completed, failed, cancelled. Annotation tasks have their own (pending,
  in_progress, completed, cancelled, to_be_reviewed, reviewed).
- `PredictionWrite` requires `name`, `project_id` and `model_id`; the area goes in `target_area_ids`; optional
  `start_time`, `end_time`, `geojson` and `probability_output` (`top_k`, segmentation only). `PredictionUpdate`
  changes only `name`; `progress` is free-form.
- No rate limits are documented; the one quota is 10 keys per account. Search `limit` caps: 1000 for projects,
  prediction results and labelsets, 10000 for predictions, areas, annotations, labels and datasets. Error codes:
  `not_found_error`, `permission_error`, `server_error`, `unauthorized_error`, `validation_error`,
  `not_implemented_error`.
- Results have `/tiles/{z}/{x}/{y}.{png|mvt}` and `/pixel-value`, which reads the web-optimized GeoTIFF and can differ
  from the native raster at a UTM-zone edge; `GET /prediction-results/files?download_token=<token>` returns the native
  output as a zip. Access levels are `public`, `restricted` and `internal`, per result. Identity is Firebase.
- Unknown: whether `/models` will enter the spec, the real rate limits, and the values of `PredictionResultRead.source`.

## Development

```bash
uv sync --all-extras && uv run pre-commit install
uv run pre-commit run --all-files       # black, ruff, mypy, interrogate and file checks
uv run pytest                            # unit tests
uv run pytest -m integration             # needs OLMOEARTH_API_KEY, or LLM_ENDPOINT for the live LLM tests
uv run pytest --cov                      # the coverage gate, 90%
```

There is no CI: run pre-commit and the tests before a PR. Tests that reach the live Studio API or an LLM are marked
`integration`; unit tests must not. Style: Black (88 columns); Ruff (`E`, `F`, `S`, `I`, `PERF`); mypy with
`disallow_untyped_defs`; NumPy-style docstrings, interrogate at 81%; Python 3.11+, `X | None`. Every source file
carries `SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License`; the skill packages are MIT and skipped by the
hooks. Tests mirror `src/olmoearth_agent/`.

Branch as `2imi9/feature-<short-kebab-slug>`, one concern each; no direct commits to `main`. A commit has a short
imperative subject under 72 characters. A PR has a release-note title and a line in [CHANGELOG.md](../CHANGELOG.md)
(Added, Changed, Deprecated, Removed, Fixed or Security), needs one maintainer's approval and is squash-merged.
Contributors follow the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

**DCO sign-off.** Sign off every commit (`git commit -s`): `Signed-off-by: Your Name <your.email@example.com>`
certifies the [Developer Certificate of Origin 1.1](https://developercertificate.org/) under the OlmoEarth Artifact
License. The sign-off names a human; AI tools do not certify the DCO. No `Co-Authored-By:` AI trailers; say in the PR
if an AI tool helped substantially, and be able to explain and have tested what you submit.

**Pitfalls.**

- Every tool a catalog row lists is in the default registry; only `OLMOEARTH_READ_ONLY` leaves tools out
  (`tools/registry.opted_in`). A `download_token` is a credential: never in a result, manifest, log or fixture.
- A change to the tools or the rules updates this page in the same PR. A change that breaks a rule updates the rule
  and its test, beside the tool it guards; a new rule comes with its test in the same PR. A behaviour of the loop is a
  middleware (`harness/middleware.py`, LangChain 1.x's hooks), not code in the loop, which is a single agent shaped
  after [DeerFlow v2](https://github.com/bytedance/deer-flow)'s lead agent without subagents or LangGraph.
- The model name, server, quant and `LLM_*` defaults repeat in `.env.example`, `docker/llama.compose.yml`,
  `scripts/serve-llm.sh`, the `Makefile`, `llm/config.py` and this page: when one changes, search for the old value
  and fix each hit. The CHANGELOG keeps history as written.

**Adding a skill.** One PR: an instruction skill's folder under `src/olmoearth_agent/skills/packages/<name>/`
(`SKILL.md`, optional `scripts/`, `references/`, `assets/`; [agentskills.io](https://agentskills.io)), whose
frontmatter description says when to use it ("Use whenever..."), since the model picks skills by it; any new tool in
[Tools](#tools), with a NumPy-style docstring (Parameters, Returns, Examples); a `tests/skills/test_<name>.py` against
the live Studio API, marked `integration`; its entry in [Skills](#skills); a CHANGELOG line under Added. An external
claim carries a real URL; an unverified one says so.

**Vendoring.** The studio-job-config package lives in
[`src/olmoearth_agent/skills/packages/`](../src/olmoearth_agent/skills/packages/), moved from the frozen
[`2imi9/OlmoEarth-Skills`](https://github.com/2imi9/OlmoEarth-Skills), so edits land here. It keeps its MIT licence;
the folder's `README.md` records the changes since the move.
