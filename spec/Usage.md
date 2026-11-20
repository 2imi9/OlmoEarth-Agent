# Usage

This page is the reference for the OlmoEarth Agent. Changes are in [CHANGELOG.md](../CHANGELOG.md), in short; the detail of each version is in [changes/](changes/).

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
upgrade line. One that is installed but does not import (a numpy or rasterio build that does not match) is said so,
with `pip install --force-reinstall 'olmoearth-agent[inferencex]'`; with none installed on Python 3.14 or later,
where the extra installs nothing, the answer names that Python and says to run the agent on Python 3.12 (the
setup page's `uvx --python 3.12` entry, or `uv sync --python 3.12 --all-extras` in a clone), not a pip line; one
that imports while rasterio does not is said so ("olmoearth-inferencex imports here, but rasterio does not"), with
the reinstall line. The scores, plan, estimate and certify answers keep the import's own error as `missing`.
`olmoearth_review_set`, `olmoearth_review_set_from_result`, `olmoearth_review_budget_ceiling` and the rest of
`olmoearth_compare_review` need no extra. You also need a Studio API key (Studio UI -> profile ->
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
| `make setup` | `uv sync --all-extras`, on the Python 3.12 the checkout's `.python-version` names (`uv run` reads it too; on 3.14 or later the `inferencex` extra installs nothing) |
| `make serve` | `scripts/serve-llm.sh`: starts the llama.cpp container and waits up to 600 s for `/health` |
| `make down` | stops the LLM (`docker compose -f docker/llama.compose.yml down`) |
| `make up` | setup, serve, then bridge; Ctrl-C stops the UI and leaves the LLM running |
| `make agent` | `uv run olmoearth-agent "$(Q)"`; `Q` defaults to "How many OlmoEarth Studio projects do I have?" |
| `make bridge` | `uv run olmoearth-agent-serve --port 8088`: the live web UI |
| `make web` | the static demo web UI on port 8080 (`python -m http.server`): sample data, no backend |
| `make verify-demo` | the verification tools on olmoearth-inferencex's expert-annotated Dynamic World tile, with the `inferencex` extra and no Studio key or LLM; writes [VERIFICATION_DEMO.md](../docs/VERIFICATION_DEMO.md) (`ARGS="--work DIR --docs DIR"`) |

`make help` lists them. Without `make` (Windows with Git Bash, say), run what they run: `uv sync --all-extras`,
`./scripts/serve-llm.sh`, then `uv run olmoearth-agent-serve --port 8088` or `uv run olmoearth-agent "<brief>"`.
`olmoearth-agent BRIEF` runs one brief and prints the answer; `--max-turns N` caps the model round trips (default 8)
and `--show-trace` prints the tool calls to stderr. It always uses the model the `LLM_*` variables name and asks it for
short answers, as for the local model. A run that cannot start or finish (no `OLMOEARTH_API_KEY`, the model server not
running, an error of the model's API) prints one line, `olmoearth-agent: <why>`, and exits 2; a run with no answer
exits 1, with "(no answer: the run reached the turn cap)" or, when the model returned no text before the cap, "(no
answer: the model returned no text before the turn cap; run with --show-trace to see its tool calls)", which names the
flag only when it was not given (with it, under the trace, "(no answer: the model returned no text before the turn
cap)"); the Claude Code bridge, whose view holds the trace, gives that plain note. The command line shows no map, so
it does not offer `olmoearth_request_aoi`: its capability card says to ask the user for the id of an area saved in
Studio.

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
| `OLMOEARTH_WEBUI_DIR` | the repository's `webui/` | what the bridge serves; where it is missing (an install from a wheel), `olmoearth-agent-serve` says so and exits 2 |
| `OLMOEARTH_SERVE_ALLOWED_HOSTS` | none | more `Host` names the bridge answers, comma-separated (a LAN name with `--host 0.0.0.0`; a port is ignored, a bare IPv6 address may be written without brackets); it answers 127.0.0.1, localhost and [::1] |
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
styling, STAC or Earth Engine fetches, embedding fetches and label-free change maps. One exception: pictures of a label
sheet's window from Earth Search's public catalogue and bucket, off until the user switches them on in the panel
([Satellite pictures](#satellite-pictures) says what leaves the computer).

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

On olmoearth-inferencex's expert-annotated Dynamic World tile, `make verify-demo` writes
[VERIFICATION_DEMO.md](../docs/VERIFICATION_DEMO.md). The run's figures: the [error rate](../webui/demo/figures/error_rate.svg)
from 300 random labels, the [review set](../webui/demo/figures/review_set.svg) and
[which part to trust](../webui/demo/figures/trust_zone.svg) (no zone was certified). There, the 791 windows (5%) the
review set puts first, with no labels, are wrong 67% of the time (532 of 791); a random 5% would be wrong about 19% of
the time, the tile's own rate (19.3%).

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
the model's output, not validated truth; a key Studio refuses there (HTTP 401 or 403) is the tool's error with the
refusal's hint and sentence, never "no value at this point". A prediction needs an area: `olmoearth_request_aoi` shows
a map in the web UI (or the MCP App), the user draws a rectangle or polygon, it is saved as a Studio area, and its
`area_id` comes back next turn. The command line and the Claude Code bridge show no map and do not offer it
(`harness/capabilities.without_map`); their capability card says to ask the user for the id of an area saved in Studio.

**A prediction id where a result id is asked.** The six tools that take a result id (`get_prediction_result`,
`pixel_value`, `compare_results`, `review_set_from_result`, `scores_from_studio`, and `result_id`/`result_id_b` of
`plan_label_sample`) also take the id of the prediction it came from (`tools/result_ids.py`). A UUID is written the
API's way (lower case) first, and an id with a character no Studio id has (anything but letters, digits, `-` and `_`)
is refused before Studio is asked. When Studio answers the result with 404, each UUID among the ids is asked as a
prediction: one with exactly one result is read as that result, and the result says so in `resolved_ids`; one with
none or several is refused, with their ids and properties, so the caller picks one. An id that is neither keeps
Studio's error as it was. Provenance hashes the call as it was made; the result also names the result id it ran on
under the argument's name (`result_id`, `result_id_b`, where the tool's result does not set it), so the provenance
summary repeats a plan with the result id its design file names. A result with a card gets one "Result used" row per
id resolved, after the tool's own rows. What Studio answers for a prediction id at
`/prediction-results/{id}` has not been probed; if it is not a 404 or an error, nothing changes.

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
figure is refused. An estimate the MCP App withdrew (its sheet's answers changed after its first result; the run's
record marks it `withdrawn`) is named with its map and "Withdrawn: ... this report states no rate or range from it",
and measures nothing; so is a zone or certificate call on its design file, which graded the same sheet ("Withdrawn:
this call graded the same sheet as an estimate withdrawn in this request: ..."), with no verdict, rate or range. With
no other result, the section begins "No error rate of this map is stated in this request: its estimate was withdrawn
(below), ...".

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
  GeoTIFFs and README are read, never extracted by name, sizes capped (`max_mb`, 1024). A zone of up to 64 million
  values (channels x pixels) is read whole. A larger one, up to 256 million, is read, written and pooled
  in strips of about 2 million values: the same counts, refusals and windows, with memory that follows one
  strip; the zip and the zone's own file are still held whole, and a zone's file over 513 MB is still refused before
  it is read (a float32 `[0, 1]` score whose GeoTIFF is not compressed inside: about 128 million pixels). A zone read
  in strips takes no certificate plan: `seed_from_map` has the package read the raster whole, so the plan is refused
  in plain words, and the result's next step names the zone; a random plan with a seed still gives its error rate
  and, with `olmoearth_certify_zone`, a certified zone, with no certificate file. Assumed, not verified: the
  README's layout for k above 2, and the result's model as its prediction's `model_id`. The zip as downloaded is
  kept once it was read: `<scores root>/studio_downloads/<result id>/files.zip`, mode 600, beside
  `kept.json` (when it was downloaded, its size and sha256; never the token). A later read of the same result within
  24 hours (`OLMOEARTH_STUDIO_KEEP_HOURS`; 0 keeps nothing, and the next download, or opening Settings, then
  removes the copies kept before) reads that copy and downloads nothing; its result's
  `read_from_copy` says when the copy was downloaded. Such a read saves the download alone: it still asks Studio
  for the result's record, each part is still read from the kept zip, decoded (so a refusal is the same as on a
  download) and pooled, and a part already written from the same file is not written again. An older
  copy, or one that is not the zip its record names, is downloaded again; a download that differs from the kept zip
  removes it before the decode and is kept in its place once it read, so a changed zip the decode refuses leaves no
  copy (a refused download of the same zip keeps it); a zip the decode refuses is never kept; keeping a zip removes
  the other results' copies past the limit. `fresh: true` downloads whatever is kept. Two reads of one result in one
  server never download it twice at once: the second reads the copy the first kept. Within the limit a result
  changed in Studio is not seen until `fresh: true` (a drawn part's own check that its files changed still compares
  what was read). The zip is kept in the same worker call that decodes it, so a read stopped while it decodes (the
  Check card's preparation past its time) still keeps it, and the card opened again, even while that decode still
  runs, waits for it to end and then reads that copy. The part files (`.<name>.<hex>.part`) a killed write left an
  hour ago or more, beside a kept zip or in a zone's folder, are removed at the next keep or zone write, and a kept
  folder left with nothing else goes too. `studio_results/` is made mode 700 at every read, before a zone is written
  or reused; each scores file is written mode 600, whole beside its target under a staged `.<name>.<random>.part` and
  then moved over it, so a reader never sees a cut file (a part a killed write left goes at the next write of that
  name once an hour old). Each zone is pooled in a worker thread, so the server keeps answering meanwhile.
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
  So is a raster that is not a GeoTIFF by its first bytes ("scores.tif is not a plain GeoTIFF by its first bytes (a
  VRT or another format); write the scores as a GeoTIFF"): a VRT named `.tif` would have GDAL read the files or URLs
  it names. The tool's own opens use the GeoTIFF driver alone, and a `.tif` of the run's folder that is not one is
  never opened to say why the folder is refused.

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

A sheet passed as `labels_path` and saved by a spreadsheet is read, or refused with the label sheet's save-again step:
one with no `window_index` column ("NAME has no window_index column; a spreadsheet may have saved it with ';'
or a tab. Save it again as CSV UTF-8, comma-separated"), one not in UTF-8 ("NAME is not UTF-8; a spreadsheet saved it
in another encoding. Save it again as CSV UTF-8, comma-separated"), and a row whose `window_index` is not a whole
number, by its line ("row 7 of NAME: window_index is 'x', not a whole number; keep the plan's window_index column as
it was written"). A UTF-8 byte-order mark (what a spreadsheet's "CSV UTF-8" writes first) is read past, and a row a
spreadsheet left with every cell blank is skipped.

`olmoearth_certify_zone` gives the largest most-confident share with error rate at most `alpha` (`rule` `prefix` or
`bonferroni`; `delta` 0.1 by default), on random designs only. Its meaning ends with the package's `ISSUED_NOTE`: delta
counts every sample, those that certify nothing included, so a zone is not a 1 - delta chance of being right.

A Studio class map without top-k scores has nothing to rank: no review set, no zone, no ranking design. Labels still
measure it over its grid points with `design='random'` or `'class'` (Olofsson et al. 2014: strata by map class, at
least `min_per_class` each, 50 by default). The plan gives the package's budgets: `plan.plan_error_rate` for an
interval no wider than `width` (0.10), and `plan.plan_class` for a `class_share`. The reviewer writes the class seen,
blind under the class design. A `reference_class` filled on only some rows is refused.

Every interval and every certified zone assumes the labels are right.

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
| `olmoearth_request_aoi` | Asks the user to draw an area in the web UI or the MCP App; saved as a Studio area. Not offered on the command line or the Claude Code bridge |
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

**When a call fails.** A tool's failure is `"<exception>: <text>"` with a hint for the model. When Studio itself fails
(a timeout, a dropped connection, HTTP 429 or a 5xx, after the client's own retries of a read), the text of a timeout
is "Studio did not answer in time" and the hint says the ids are not the cause: wait a minute and make the same call
once more, and after a create or a submit look for the record before trying again (`tools/registry.studio_busy`); the
second failure alike still says to stop. A create whose answer was lost (`StudioOutcomeUnknown`) and a refused key
bring their own hints instead. The Claude Code bridge's cache records the same text and hint, so each of its steps
gives the live envelope; a record cached before keeps its old text and the generic hint, as the model first read it.
Arguments that do not decode as one JSON object (`__raw_arguments`) are
refused before the tool runs, with the schema, rather than run on its defaults.

## Answer checks

The model reads the brief and calls a tool; the harness runs it (its arguments checked against the tool's schema),
feeds the result back and repeats until the model answers. The system prompt is the soul (`harness/soul.md`) plus a
capability card built from the tools' code: what each tool does, needs and cannot do (`harness/capabilities.py`). A
large result goes to a workspace file. At the turn cap the last tools run, then one call without tools asks for an
answer. When a tool fails the same way twice in a run, its result tells the model to stop retrying and report the limit.

Before an answer is shown, the harness checks (`harness/checks.py`) its numbers against the tool results and the
user's messages; its directions, places and magnitudes against the facts the tools computed; its claims of files saved
or items listed against what the run did; and the claims the tools forbid or require. One more call asks the model to
fix every problem found. A sentence that still fails is marked `[unverified: <check>]`; nothing is deleted. On the web
UI an earlier answer of the chat is a source for the next one only in its sentences that passed: a figure or claim left
marked is checked again when it is restated (`RunEvidence.checked_earlier`). A required
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
chosen and no local model answers, the UI says so. The local presets and output budget stay with the local model:
Claude gets no sampling parameter (its own defaults; the locked `anthropic` 1.8.0 takes none) and its 8,192-token
budget; ChatGPT and Gemini get the preset's temperature, top_p and penalties and no output cap unless a call names one
(the claim check's 4,096), which OpenAI's own API takes as `max_completion_tokens`; an OpenAI reasoning model (`o1`,
`o3`, `o4`, `gpt-5`) gets no sampling parameter either.

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
loop's request. Through the replay bridge (`olmoearth-agent-bridge`, not `make bridge`, which serves the web UI), a
Claude Code session writes the model's responses; it needs ssh to the cluster, a checkout and
the Studio key there, and the model sees the loop's request as text beside its own context. The
bridge runs the agent's real loop, tools and checks one model call per command, and
holds nothing between commands. Each command builds a `LeadAgent` as `olmoearth-agent` does (with no draw tool, as
there, but with the claim check and the short-answer clause off), with `ScriptedLLM` returning the recorded responses in order. Tool handlers are
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

## In Claude, with views (MCP App)

How to use the panels, step by step: [mcp-app.md](../docs/mcp-app.md). The words they use (square, point, part, range, rule,
Cannot tell): [terms.md](../docs/terms.md). This section is the reference for developers: what each panel shows and does, and
what Claude is told. Why a behaviour is as it is: [design/mcp-app-notes.md](design/mcp-app-notes.md).

[Setup](#setup) · [Every panel](#every-panel) · [The start card](#the-start-card) · [Sign-in](#sign-in) ·
[The projects panel](#the-projects-panel) · [The Check card](#the-check-card) · [The label sheet](#the-label-sheet) ·
[Satellite pictures](#satellite-pictures) · [The result card](#the-result-card) · [The plan card](#the-plan-card) ·
[Look-first](#look-first) · [The area map](#the-area-map) · [What Claude is told](#what-claude-is-told) ·
[Files and privacy](#files-and-privacy) · [Limits](#limits)

### Setup

`olmoearth-agent-mcp` serves the agent's tools to Claude as an MCP App on stdio (`src/olmoearth_agent/mcp_app`, the
`mcp` extra). Claude is the model. Every tool of the default registry is listed with its own schema and runs through
the registry's dispatch, so Claude gets the result the agent's own model gets, `must_state` and `forbidden_claims`
included. The server's instructions are the agent's prompt (`harness/soul.md`) and its capability card, plus rules for
Claude ([What Claude is told](#what-claude-is-told)). What the agent's loop adds does not run here: no answer check
reads Claude's reply, and the instructions say so. In the MCP handshake the server names itself `olmoearth-agent` with
this package's version (`olmoearth_agent.__version__`, which moves with `pyproject.toml`'s), not the mcp SDK's, so a
host's server list or a bug report tells one build from another.

**Quick setup.** Step by step, for a person or a coding agent: [mcp-app-setup.md](../docs/mcp-app-setup.md).

**The install command.** `olmoearth-agent-mcp install` (`mcp_app/install.py`) writes the `olmoearth-agent` entry into
Claude Desktop's config, so nobody edits the file by hand; `uninstall` removes only that entry. With no subcommand the
program serves on stdio, as a host starts it. Its safety rules: it never prints the config, any `env` value or another
server's entry; every other entry and top-level key is kept, in the same order, and an entry already there in Claude
Desktop's config keeps its `env` block (only `command` and `args` are replaced); a backup is made beside the file
first (mode 600; a full copy, so an agent must not open it either) and the new content replaces the file by an atomic
rename, keeping its mode; a config that is not a valid JSON object, or that holds `NaN`, `Infinity` or a number too
large to keep, is left as it is (exit 1); a config that cannot be written is as it was, and the line says so; the
download runs first, and a failed download changes nothing; it never opens the saved Studio key; and it sends nothing
anywhere except uv's own download from GitHub and PyPI. `--claude-code` also runs `claude mcp add --scope user`, after
Claude Desktop's config is written (an entry already in Claude Code is removed and added again, without its `--env`
settings, which must be given again; the command says so, and says when the old entry is gone and the new one was not
added); `--ref` pins a release or commit (without it the entry is set back to main); `--config` names another file
(an empty value is refused, exit 2); `--dry-run` only shows the file and the entry; `--no-download` skips the
download. Exit codes: 0 done or nothing to do, 1 a failure, 2 no uv, a bad `--ref` or an empty `--config`.

**Claude Desktop.** Settings, Developer, Edit Config, then add the server. Use uv's absolute path (`which uv`), because
an app started from the Dock does not have the shell's PATH:

```json
{
  "mcpServers": {
    "olmoearth-agent": {
      "command": "/absolute/path/to/uv",
      "args": ["run", "--directory", "/path/to/OlmoEarth-Agent", "--extra", "mcp", "--extra", "inferencex",
               "olmoearth-agent-mcp"]
    }
  }
}
```

**Without a clone.** The wheel carries the panels' pages, scripts and brand images: a wheel built and installed into a
clean Python 3.11 environment passed the stdio smoke test, run from an empty folder. With uv installed, this entry
needs no copy of the repository; it installs the package from GitHub into uv's cache on first start (so it works once
the app is on the repository's default branch):

```json
{
  "mcpServers": {
    "olmoearth-agent": {
      "command": "/absolute/path/to/uvx",
      "args": ["--python", "3.12", "--from",
               "olmoearth-agent[mcp,inferencex] @ git+https://github.com/2imi9/OlmoEarth-Agent",
               "olmoearth-agent-mcp"]
    }
  }
}
```

Each user signs in with their own Studio API key in the panel ([Sign-in](#sign-in)); the key stays on their computer.
Where the server reads and writes: [Files and privacy](#files-and-privacy).

**Which tools show a panel.** Panels (MCP Apps views, `ui://olmoearth-agent/*.html`) show beside the chat for the
tools in the sections below only: the app's own start card (`olmoearth_start`), sign-in (`olmoearth_sign_in`), Check
card (`olmoearth_check_map`) and label sheet (`olmoearth_label_sheet`), and the agent tools in the allow-list
`TOOL_VIEW` in `mcp_app/server.py`: the projects panel (`olmoearth_load_context`), the area map
(`olmoearth_request_aoi`) and the result card (`olmoearth_plan_label_sample`, `olmoearth_estimate_map_error`,
`olmoearth_certify_zone`, `olmoearth_review_set`, `olmoearth_review_set_from_result`). Every other tool, an
intermediate step, shows no panel, and its `must_state` is in the text Claude reads. A tool without a panel that needs
a sign-in says so in that text and names `olmoearth_sign_in`. The panels call tools of their own (`olmoearth_app_*`),
marked `visibility: ["app"]` and listed only to a client that advertises the MCP Apps extension
(`io.modelcontextprotocol/ui`); where panels show, the agent's tools and the app's own are listed with
`visibility: ["model"]`, so a panel calls only the app's own tools.

**Without panels** (Claude Code, or another client that does not advertise the extension), the tools still work:
`olmoearth_sign_in` returns the sign-in link, `olmoearth_label_sheet` says which column of the CSV to fill,
`olmoearth_start` names the jobs in its text, a review set keeps its listed rows, and `olmoearth_check_map` gives each
zone's label counts with the planner's sentences, for Claude to ask in the chat and plan each zone itself. A plan's
text keeps its list of windows there (`windows`, `n_windows_listed`). Each section below says what else differs.

**Menu entries (MCP prompts).** The server lists the same jobs as MCP prompts (`check_map_accuracy`,
`which_map_is_more_accurate`, `run_model_on_area`, and `windows_to_look_at_first` only while `FIRST_IN_STUDIO` is
true), with no arguments, so a client that shows prompts in a menu (Claude Code lists them as slash commands) offers
them without the user knowing what to type. Each gives the request the start card's button sends, as one user
message.

**One server, every chat.** Claude Desktop runs one server for all its chats, and MCP names no chat. The run's state
(the provenance log that `olmoearth_provenance_summary` and `olmoearth_case_narrative` read, and the counts behind the
stop-retrying hint) starts afresh when the server leaves a Studio key (for another account's, or on sign-out) and on
`olmoearth_new_task`, which the instructions ask Claude to call at the start of each new conversation or task. A first
sign-in keeps the task's records.

### Every panel

**Words.** The panels say "square" for a window, "point" for the single pixel read at a grid point of a Studio result,
"part" for a UTM zone and "range" for an interval ([terms.md](../docs/terms.md)). UTM zone, interval, sample, design, seed and
stratum stay in the folds; tool names, fields and every sentence the tools write keep "window". No panel face says
"window": not the held figure's sentence, the finish message ("I have answered every square of this sheet. ..."), the
pictures' caption ("Context, square outlined") and their refusals, the check request ("how many squares to check"),
the which-map request ("how many points I can label"), the Check card's Details ("20 squares, design random, seed 0")
nor a review set's refusals (the start card's "Find the windows to look at first", off by default, still says it).
Each such sentence carries a `WORDING:` comment in the code, so a word can be swapped with one search.

**Squares and points.** A plan or review set drawn from a Studio result reads single pixels at the points of a grid,
so the panels call its windows "points": a review set's legend line in Details says each square stands for the one
pixel read in it, and a sheet of points (`read_sheet`'s `unit` "point", `pictures_possible` false) says "Point 1 of
20", shows no Pictures button and promises no pictures. So does a plan drawn from the file of sampled points
`olmoearth_review_set_from_result` saves (told by that small file's own top level, never a raster). The plan card, the
result card ("50 points checked", "about 1 point in 8", "The 50 points you checked", the class lines), the unfinished
line and Claude's first lines say points for such a sheet too. A sheet that shows no pictures has no line that no date
is marked clear, and its Cannot tell is "for a point you cannot judge", with no dates. A sheet of squares' pictures
start with the square's size ("A square is 11 x 11 map pixels, about 110 m a side."). The plan card's button reads
"Open the sheet here" for both.

**Lines and folds.** Under a card's heading stand at most two plain lines, in the body colour; the start card and the
projects card have one, what to do next. What more a card has to say is in its closed "Details" fold where it has one
(the label sheet, the Check card, a result card); the start card and the projects card have no such fold. The Check
card's face holds more ([Limits](#limits)).

**Look.** The panels look like OlmoEarth Studio's working pages (its MUI theme, values read from olmoearth.allenai.org):
one slim line on top, the AGENT tag (inlined as a data URL) and the sentence that the panel is a Claude panel of the
OlmoEarth Agent, and, on the start card and the projects card once signed in, the account and a Sign out text button
at the sentence's right (under it on a narrow panel; with the app's own key, "Signed in with the app's own key" and no
Sign out), and under them, once Sign out is pressed, its reply; the cream canvas; white cards with 4 px corners and no
shadow (project cards 12 px); one contained dark-teal button per view, the rest outlined or text; joined toggles;
chips for kinds and statuses, the Ai2 mint only as the ring of a completed status. Controls of one kind side by side
have one size: a square's answers and a part's counts are equal cells of one grid, every row as tall as the tallest,
and the buttons of a joined toggle, the label sheet's Previous and Next and the look-first pane's, are each as wide as
the widest (a list row's answers where the row has room: beside the open pictures pane they keep their words' widths,
and on a panel under 480 px they stand one under the other). The panels do not follow Claude's light or dark theme.
Manrope and Roboto Mono, Studio's fonts, come from Google Fonts, with the system font until they arrive. The start
card's overview figure's narrow strip uses the same colours, and its one example map is a picture of a finished Studio
map from the docs; its wide form is the docs' own figure.

**The line under a job button.** A job button's press shows under the button: "Sent to the chat. Claude answers
there." once Claude took the message; when the host does not take it, "Claude did not get this message. Copy it into
the chat:" in red, over the whole text in a box that one click selects whole, to copy. The same holds for the Check
card's and a result card's buttons and a sheet's finish button. A start card tile or a projects panel job button says
"Sent to the chat. Press Send in the message box below." instead, and turns the card's other job buttons off (pointing
at one says "One job at a time: press Choose another job first, or ask Claude."); "Choose another job", at the end of
the card's line, turns them back on, empties the job lines and puts the focus on the first job. That line
is the only place that says it: the status line (at the card's foot; on the Check card once it shows its press or a
sheet, right under that press or sheet) does not repeat it, so a screen reader hears it once. "Tell Claude: this
project" and the sign-in's "Continue in the chat" show no line under them; a message the host does not take says so
in the status line at the card's foot. Once a later message is taken, that failure line goes from the status line (any
other line there stays). A press that ends before it sends anything first empties the line under its button, so it
never says "Sent to the chat." for an earlier press, and leaves it empty: a map without class scores, or "Which is
more accurate?" with a ticked run of two outputs, says why in the status line, and a row's job on a run with two
outputs opens the list, whose first line says so, while the row's job buttons are away. Closing a list does not empty
that line. The line under a button is a status region in place before the first press, out of the layout while empty,
so a screen reader announces its first words; the message to copy is outside it and is not read out with it. The
start card's and the projects card's job buttons take one press at a time. So do a result card's and the Check
card's buttons ("Check how wrong this map is", "Check each class the map draws", "Check this map on points", "Run it
again"), "Tell Claude: this project" and "Continue in the chat": while the message is being sent the button is
greyed and keeps the keyboard focus, and a press does nothing; the second click of a fast double-click does nothing
either, even when the host answered the first (a press of a key always counts), so a double-click asks Claude once.

**Links.** A link opens in the user's browser through the host. The start card's docs link and "Request access" say
"The link was not opened. Open this address in your browser:" over the address to copy in a line of their own, beside
the link (a status in place before the first press, taking no room while empty), and nowhere else; a look-first
pane's Street map says it under its own row of buttons. For a link with no such line (the sign-in page; a street map
link on a sheet) the status line at the card's foot says "The link was not opened."; a later link the host opens
clears that line, and no other.

**When a card cannot go on.** A call the host stops before its result replaces the card's waiting line with "This step
was stopped before it finished. Ask Claude to run it again."; a stop after the result changes nothing. An error the
host bridge reports is one toast, "Something went wrong in this card.". An error in a card's status line is announced
to a screen reader at once (an alert), any other message politely. Each page runs a small watchdog before its script:
if the card has not connected after 10 s (its client or the map's Leaflet did not load), it says "This card could not
start here. Carry on in the chat.", on the panel's cream canvas in its text colour and a system sans-serif: each page's
head carries those few rules before the card's own scripts, and its style takes over once its script runs.

**Host report.** After it connects, and when Claude closes it, each panel sends the server one report of what its host
supports (`olmoearth_app_host_report`: the capability keys, display modes and context keys by name, the kind of height
and the host's name, never a value), and the server logs it in one INFO line (logger
`olmoearth_agent.mcp_app.server.host`); it also logs once the client's extension keys and the UI extension's MIME
types.

### The start card

`olmoearth_start` is a tool of this app's own (not one of the registry's tools). Its description asks Claude to call it
when the user first turns to the app without naming a job, or asks what it can do.

**Its line.** Under its title: "Sign in first, then press a job or ask Claude in the chat."; signed in, "Press a job, or
ask Claude in the chat." The title says what the card is; what a check cannot tell is said on the Check card and the
sheet, where labelling happens.

**The jobs.** Each job is a tile (a button holding its label and its one line, 85 characters at most) that sends one
complete request as the user's message. The main job comes first: "Check how wrong a map is" ("You check random
squares of the map; you get how often it is wrong, with a range."); signed in, its button is the card's one contained
button. Under a small heading, "Other jobs": "Say which of two maps is more accurate" and "Draw an area and run a
model" ("Makes a new map once you pick a model, an area and dates. Needs a key that can write."), and "Find the windows
to look at first" only while `FIRST_IN_STUDIO` is true. The tiles are one column, each full width and one size at every
width (at least 86 px tall), the line inside its tile, named by its label with the line as its description. Signed
out, a tile that is off keeps its label and its line in the readable secondary text colour, and "Sign in to start a
job." stands over the tiles, inside their card, at body size in the body colour. Each request asks Claude to show the
user's projects, where the projects card's job buttons pick the prediction or the model. A job's press says under its
row "Sent to the chat. Press Send in the message box below.", and the other tiles turn off until "Choose another job"
is pressed ([the line under a job button](#every-panel)). With Asta on in the
card's Settings box, one more tile, "Search papers with Asta", stands last under "Other jobs", on signed in or out
([Settings](#settings)).

**The figure.** The card shows one figure: how OlmoEarth makes a map, and where this app comes in. On a panel at least
680 pixels wide (780 px, for one) the jobs come first and the figure after them: the title, the line, the sign-in
while signed out (still first), the jobs (the dark main job, then "Other jobs"), then the figure with its caption and
docs link. On a narrower panel (380 px) it stands under the card's one line, before the sign-in and the jobs. The
figure's box moves in the page (not by a CSS order), so a screen reader and Tab follow what is seen, again when the
panel's width crosses 680 px, and a docs link that has the focus keeps it. On a panel at least 680 pixels wide it is
the overview figure of the OlmoEarth docs itself (used with Ai2's permission; a JPEG in `views/brand/`, inlined into
the start view, so nothing is loaded from elsewhere); on a narrower panel, where that figure's words would be too
small to read, a strip of three boxes drawn in code says it in plain words ("Studio makes a map", "You check random
squares", "You get how often it is wrong, with a range"); its first box shows a finished Studio map from the docs, cut
from inside its prediction layer (no place, base map, Studio panel or legend: a 160 x 160 JPEG in `views/brand/`,
inlined into the start view alone). It states no number. The figure, its caption and the docs link sit in one white
box, as wide as the figure and no wider (at most 466 px with the strip, 774 px with the docs figure), so no blank white
band stands beside the figure on a wider panel; the strip's two light boxes are the canvas colour inside it. Under it,
one caption, which says what is shown: beside the docs figure "How OlmoEarth makes maps (a figure from the OlmoEarth
docs). This app comes after: it checks how often a map is wrong.", beside the strip "How a check of a Studio map
works.", and a "What is OlmoEarth? (docs)" text button, underlined inside the caption's line, that opens
https://docs.olmoearth.allenai.org/ in the browser (when the host does not open it, the line under the caption says
"The link was not opened. Open this address in your browser:" over the address to copy).

**Signed out.** The jobs are off, and the card shows the sign-in (a white card; its button reads "Open the sign-in
page") with one sentence: the Studio API key is the user's own, a secret code Studio makes (profile, then API Keys);
"The key goes to the OlmoEarth Agent here, not through Claude." (the words of every sign-in text). Then a line of its
own: "No Studio account? Request access on olmoearth.allenai.org. Do not use someone else's key.", where "Request
access" is a text button, underlined as the docs link is, that opens https://olmoearth.allenai.org/ (OlmoEarth's site,
whose own "Request access" asks for an account) in the browser through the host; when the host does not open it, the
address is shown under the line to copy, under "The link was not opened. Open this address in your browser:". A
signed-in card has no such line.

**Who is signed in.** Who is signed in ("Ana (Lab)"), with Sign out, is in the slim line on top, not on the card; with
the key the app was installed with, the line reads "Signed in with the app's own key" and has no Sign out, and after a
Sign out that leaves that key in use the focus goes to the main job's button. The reply to Sign out ("Signed out; the
saved key was removed." or "Still signed in: this app was installed with its own key. Only whoever installed it can
remove it.") is said in the slim line, under where Sign out stood (its words end at Sign out's right edge, or start at
the line's left edge when Sign out stood nearer it than the words are wide; when the panel's width changes, under where
Sign out stands, or would stand), a polite status (an error an alert, in red), not in the status line at the card's
foot; the same holds on the projects card. It stays while the line shows what it is about (signed out, or the app's
own key) and goes once the line shows another key's or account's name, as after a later sign-in; an error goes when
the card is drawn again. A Sign out also empties the status line at the card's foot while it still says the sign-in
made on the card ("Signed in to OlmoEarth Studio as ..."), so that line never outlasts its sign-in. The account card's
Sign out (`olmoearth_sign_in`) keeps the status line. While a Sign out is answered it is greyed, keeps the keyboard
focus and takes no second press; a Sign out that fails leaves the focus on it. On the account card, one that goes
through draws the card again with the focus on its main button: the sign-in's button, or "Use another key" when the
app's own key is still in use. When the card opens it asks the server once whether the user has
signed in or out, or another key or account is now in effect, since, and redraws if so.

**Without panels.** The tool's text names the same jobs, the main one first with what it gives, then the others with
what each needs, and says to sign in first when the user is not signed in.

### Sign-in

The tools call Studio with the user's own key, taken in this order:

1. a sign-in on the sign-in page;
2. `OLMOEARTH_API_KEY` in the server's environment;
3. the key file, `~/.config/olmoearth-agent/studio.env` (or `OLMOEARTH_KEY_FILE`). A sign-in with "remember" writes its
   `OLMOEARTH_API_KEY=` line, mode 600 (set on the file's path, so saving works on Windows too), and keeps any other
   line. When the app was installed with its own key (2.), that key comes first again after a restart, so a key
   remembered beside it says so under "Signed in": "Saved on this computer, but this app was installed with its own
   key (OLMOEARTH_API_KEY): after a restart, that key is used again. Only whoever installed the app can remove it."

**The sign-in card.** A white card whose note is a few plain lines: "The button opens a sign-in page in your browser, on
this computer. Enter your Studio API key there (in Studio: your profile, then API Keys)." and "The key goes to the
OlmoEarth Agent here, not through Claude.", the words every sign-in text uses; a wrong key shows its own error. The
Check card and a result card say those two lines alone. Its button, "Open the sign-in page", opens the page in the
user's browser. In a client without panels, `olmoearth_sign_in` returns its link for Claude to give the user.

**The account card.** Once signed in, `olmoearth_sign_in` shows the account (user, organization, number of projects,
and a "Key" row: "Entered on the sign-in page", "The saved key file", or "The app's own key, set when it was installed
(OLMOEARTH_API_KEY)"), Sign out (none for the app's own key, which its press could not remove) and "Use another key".
Signed in with the app's own key, the tool's note tells Claude the panel has no Sign out and that only whoever
installed the app can remove that key.

**The sign-in page.** It is served by this server on `127.0.0.1`, on a free port, at a path with a random token, for
ten minutes and one sign-in; opened again while it is open (the tool called again, or the panel's button), the same
link has ten minutes from then. It opens with two short paragraphs: "Paste your Studio API key (in Studio: your profile,
then API Keys). It goes from this computer to the OlmoEarth Agent, not through Claude." and "A read-only key can check
maps; to save an area or run a model, use one that can write." (a key Studio refuses shows its own error there); its
one field has a label of its own, "Studio API key", which stays in view once the key is pasted (the placeholder "Paste
the key here" does not), and its card is white on the cream page. The key goes from that page to this server and never
travels in an MCP message, so it is in no tool call the Claude app relays or logs. The page answers only its own
`Host` (`127.0.0.1:<port>`, so a DNS-rebound name is refused) and a form post whose `Origin` is its own or `null` (what
a browser sends from a page that withholds its origin); a site elsewhere cannot know the path's token. Reading a
request has 60 s for its head, then 60 s for its form (`auth.READ_SECONDS`; a request that never ends is a bad
request). Studio's check of a typed key has its own bound, 100 s (`auth.CHECK_KEY_SECONDS`), after which the page says
"Studio did not answer in time; nothing was saved. Try again."; a Studio that cannot be reached gives "Studio could not
be reached (<kind>); nothing was saved."; an answer that is not JSON (a proxy's or maintenance page, or
`OLMOEARTH_BASE_URL` set to the site's root; `auth.StudioNotJson`) gives "Studio's address answered with a web page
instead of Studio, so the key could not be checked; if this lasts, whoever installed the app should check Studio's
address (OLMOEARTH_BASE_URL); nothing was saved." Never "Bad request", and never a refused key.

**Checking the key.** A key typed on the page is checked with Studio (`/users/me` and one page of projects) before it
is used or saved; a refused key is neither. A key from the environment or the key file is used as given, and a refusal
by Studio offers the sign-in again. A refusal is recorded for the key the refused call was sent with, only while that
key is in effect (a call still running on the old key after a sign-in with another one records nothing, a tool's or a
panel's: a panel's call is charged to the client it used, `server._CALL_STUDIO`), and a sign-in on the page clears it,
the same key too. The account a panel shows is the one read for the key in effect: when the key
file changes under the server (another server, Claude Code beside Claude Desktop, signed in with another key), the new
key's account is read. When Studio cannot be reached to check a key (offline, Studio down), the status carries
`studio_unreachable` ("Studio could not be reached (<kind>)"), not `key_problem`: the key is kept, the start card is not
signed out, `olmoearth_sign_in` opens no page, and both tell Claude "Studio cannot be reached right now (the network or
Studio is down). The key is kept, so no sign-in is needed. Try again shortly." When Studio's address answered with a
page that is not Studio's (an answer that is not JSON), `studio_unreachable` is the check's own words (above, without
"nothing was saved"), and Claude is told instead "Studio's address answered with a web page instead of Studio, so the
key could not be checked. The key is kept, so no sign-in is needed. Tell the user that Studio may be under maintenance,
or that the app's Studio address (OLMOEARTH_BASE_URL) may be set wrong; if it lasts, whoever installed the app should
check that setting and restart the app." (`server.STUDIO_NOT_JSON_NOTE`), never that the network is down or to try
again shortly. Either way the start card and the sign-in card (over its account card) show one error line, never the
status's own text: "Studio is not answering right now, so your key could not be checked. It is kept: there is no need
to sign in again. Try again in a little while; if this lasts, tell whoever installed the app." (`OE.STUDIO_UNREACHABLE`),
the start card's jobs still on. Without a key, a Studio tool sends nothing and its result says `needs_sign_in`. The
key never appears in a result, an error or a log. No tool takes the key.

**Sign out.** Sign-out forgets the key and removes the key file's key line (the file only when nothing else is left in
it); a key in the environment (the key the app was installed with) stays, and the panel says so: "Still signed in:
this app was installed with its own key. Only whoever installed it can remove it." An `OLMOEARTH_API_KEY` that is no
key, such as the placeholder, is no key in effect: the user is then signed out, and the panel says so. The start card,
the sign-in card and the projects card offer no Sign out for the app's own key. Signing out or into another account
starts the run's state afresh ([Setup](#setup)).

### The projects panel

`olmoearth_load_context` shows the page of projects the tool read, headed "Projects in this Studio account".

**Projects.** As cards (name, description, creation date), or as compact rows (name, created) past eight projects,
with a search by name or description (its grey hint, like every field's, is the text's colour at .7, 5.3:1 on white).
A date reads "1 Oct 2026" (the day, the month's three-letter name from a fixed table, "Sep" too, the year), which no
country reads as another day. A long project name shows on up to two lines on its card, its whole name in the tooltip.
The card's one line says what to press: "Press "Show its maps" on the project that holds your map, then "Check how
wrong it is" beside the map." The descriptions are read after the panel opens (the first 200 projects) and added to
the cards already shown; a project with none shows none. A closed card's bottom row sits at its foot, so cards in one
row end their buttons on one line. Every project has a "Show its maps" button ("Hide its maps" once open): it opens
the project, as a press on the project's name does.

**An open project.** A project opens into its predictions, then its areas, newest first (five, then "See all", with
"Recent" over them only then), its models (in a closed "Make a new map" fold), and a prediction into its outputs ("Show
outputs", each with its property names). A prediction's "Show outputs" and job buttons are a line of their own under
its row, that toggle at the row's left edge and 100 px wide, the wider of "Show outputs" and "Hide outputs", so the job
buttons start at one left edge whether a row's outputs are open or not; on a panel of 480 px or less the box, the ring
and the dates are the row's first line and the name, the model and the status the next, at full width. An area's name
sits in a column of its own and wraps inside it, so its button stays at the row's right edge, as on every other area
row.

**What each map shows.** Under each completed prediction but a class map's (its Check button says it can run) one line
says what the map shows and whether the check can run on it: "Shows a number at each pixel. Check works if that number
is a score from 0 to 1; its Check card tells you." for a model whose record says per pixel, and the same line without
"at each pixel" for a model whose record's grain is the window (said "per-tile" on its row) or one whose record does
not say; "Shows detections, not a map of classes. Check cannot run on it."; "What it shows is not recorded. Its Check
card says whether it can run." when the model's record names no kind, or the prediction names no model; "What it shows
was not read here. Its Check card says whether it can run." when the prediction names a model whose record was not
among those read; an embeddings row says it is not a map these jobs can check. A failed prediction's row says "This
run failed in Studio. There is no map to check."; a running one says nothing.

**Buttons.** A project and an area each have a "Tell Claude: this project" ("this area") button that sends its ids,
and the Studio name quoted as data, as the user's message; it starts no job. On a project it is a text button, and its
press also opens the project in the panel when it is closed. The others have job buttons instead, each sending one
complete request with a verb, the ids and the name quoted as data:

- a completed prediction: "Check how wrong it is", the main job, named as on the start card (Claude opens the result's
  Check card, where you pick how many squares to check). It is not offered for an embeddings or detection model, where
  the check can only fail; a regression keeps it, and the Check card says why when it refuses a band in other units.
  "Where to look first" is offered only while `FIRST_IN_STUDIO` in `mcp_app/server.py` is true (false by default), and
  then only for a class map with top-k scores: a model of classes or one not recorded, and on a result whose first
  property is the base of its top-k bands; a press on a prediction with no such result says so and sends nothing. The
  press reads the prediction's results once (`olmoearth_app_results`): one result is sent by its id; with none the
  panel says so; with several it opens the list, which starts "This run made more than one map; Studio names them only
  by the ids below. Not sure which is yours? Ask whoever made the run.", the one line that says so, and moves the
  focus to the first output's Check; each result row has its own buttons. While a run's list of several outputs is
  open (by a press or by "Show outputs"), the row's own job buttons are away, since their press only opens it; "Hide
  outputs" brings them back. A lone output has no buttons of its own: the row's use it. A failed or running prediction
  has no job;
- two completed predictions, ticked (the boxes show when the project has two or more, and only then the note under the
  Predictions heading adds "Tick two finished maps of the same area and dates to compare them.", above the first box):
  "Which is more accurate?" under the list, live only while exactly two are ticked, beside how it works ("You label
  only where the two maps differ. You see which is right more often there, or that your labels cannot tell, but not
  how accurate either map is."). It plans a labelled sample where the two maps differ (its request asks how many
  points the user can label: two Studio results are sampled at points of a grid), and each prediction must have
  exactly one result;
- a model: "Run on an area" (Claude opens the map to draw or choose an area, asks for the period and submits the
  prediction; this needs a Studio key that can write); its tooltip says it starts nothing yet. The Models card lists
  only models the project's predictions name, so a model the project has never run is not offered here.

Each press's reply is [the line under a job button](#every-panel).

**Signing out here.** Who is signed in, and Sign out, are in the slim line on top; after Sign out the keyboard focus
goes to the sign-in's button, and the reply ("Signed out; the saved key was removed.") is said under Sign out, in that
line, not at the card's foot, as on [the start card](#the-start-card). With the key the app was installed with
(`OLMOEARTH_API_KEY` in its settings) in use, the slim line reads "Signed in with the app's own key" with no Sign out,
from the card's first drawing (the tool's panel data names the key's source). After a Sign out that leaves that key in
use, its projects are shown and the focus goes to the search once they are drawn; a read of them that fails says "The
projects of the key still in effect could not be read: ..." and takes the focus. Claude is then told the user is
still signed in, not signed out.

**What Claude is told here.** Where panels show, the tool's result starts with a note for Claude: the user picks by
pressing; every finished map but a detections or embeddings one has a "Check how wrong it is" button; only a press
opens a project; asked what to press, Claude answers in one line with the press for the job at hand ("Show its maps"
then "Check how wrong it is" to check a map; two ticks then "Which is more accurate?" to compare two, in a project
with two or more finished predictions; "Make a new map" then "Run on an area" to make one, in a project whose
predictions name a model); and the rows already say what each map outputs, so Claude does not list the predictions
again unless asked. The note says that, where the panel shows, this comes before the first half of the rule kept after
it (say what each prediction outputs, which then holds only when Claude does list them); the rule's other half, never
to call one recommended, always holds.

**In progress.** Above the search, "In progress" lists the label sheets on this computer that are part labelled (some
windows labelled, not all) and have their plan's design file beside them, newest first: what the labels are for
("Checking how wrong a map is", "Certifying a zone", "Saying which map is more accurate"), the counts and the day the
sheet was last saved ("Last saved 10 Nov 2026."), the sheet's file name, and Resume, which sends Claude "Continue
labelling my label sheet NAME: open it with olmoearth_label_sheet, labels_csv_path PATH.", so the sheet opens again
in its own panel, read fresh from the file. The panel asks the panel-only `olmoearth_app_sheets_in_progress`
(`sheet.sheets_in_progress`), which reads only the top of the tools' write root and of the workspace root (never a
subfolder, a hidden file or a link), the newest 25 sheets at most, without the sheets' lock, and lists at most five
("Older sheets in progress are not listed." when more may exist); it reads a design only for a sheet it lists, and
counts as the plan card does. The list is counts only, never a window, a place, a class or a label. Sheets name no
Studio account, so every account's sheets on this computer are listed. Each row also carries the sheet's
`labels_csv_path` (the same as its `path`), which the panel tells Claude. Each row's words sit in one column and its
Resume in a second, the same for every row, so the Resume buttons line up; a long file name wraps in its column. The
card is white as every other card. It is hidden when nothing is in progress or the read fails, takes no focus and has
no contained button. A fully labelled sheet that was never graded is not listed. The list picks its sheets by their
filled label cells, before it reads a design, so on a sheet labelled with the class seen it can differ from the plan
card, rarely: a class code typed in a spreadsheet, with `wrong` not yet worked out, is not a label, so a sheet whose
every label cell is filled but not all worked out is not listed, though its plan card says it is part way.

**Glyphs.** A small glyph says what each thing is, never how good it is (no mint, no animation; the status chip stays
the only status sign). The server computes its fields (`mcp_app/glyph.py`) and sends only those, never a record:

- a prediction: a ring of twelve months, January from twelve to one o'clock with a tick at the top, dark where its
  requested range (`start_time` to `end_time`, in UTC; an end at 00:00 on the first of a month is exclusive) covers the
  month, with the year beside it (in a column of one width; "no dates" when the record has none, its ring then named
  "Dates not recorded"). The box, the ring and the year are tracks of a fixed width and the name a column of its own
  that wraps inside itself, its status chip after it or under it, so every row's name starts at one left edge, on a
  wide or a narrow panel and with a long name (at 480 px or less the name's column is a line of its own under them,
  and the year takes only its own width);
- a project card (eight projects or fewer): a month darker the more of its predictions cover it (none, 1, 2 to 4, 5 or
  more), all years and areas counted together, with the count beside it. What a ring shows is said once, in one line
  above the cards ("The ring on each project shows the months of the year that project's predictions cover, all years
  and areas counted together."); the line sits in one block with the cards and shows only while a card in view has its
  ring: not before the rings are read, when that read fails or gives none, or when the search leaves no card with a
  ring. Beside a ring only what differs: "No maps yet" for a project with no prediction, "dates not recorded", or how
  many predictions have no dates. One predictions read per card, in one call (`olmoearth_app_project_glyphs`, at most
  eight, three at a time); compact rows draw none;
- a model, in a row of the project's Models card: its output as a legend tile (4x4 per pixel, 2x2 when the record's
  grain is the window, which its caption and its predictions' rows call "per-tile"; classes as hollow cells while their
  colours are not read, values as grey bands, detection as a box, embeddings as stripes), its sources as `S2`, `S1`,
  `L8/9` badges and its size as a word, and a caption with its output and the months a period context needs (the tile
  a fixed track, the name and its badges a column that wraps inside itself, and the caption indented by the tile's
  track and the gap after it, so the name and its caption start at one left edge), from one `models/search` (one page
  of 200; a model not on a full page is said to be not read, not absent) when a project opens (`wizard_answers`:
  `prediction_type`, `encoder_variant`, `foundation_model_version`, `imagery_sources`, `temporal_context`,
  `period_duration`, `start_month_constraints`; the key names were read from live records, the value spellings not
  seen).

What a record does not hold, or holds in a spelling not known, is never guessed: a prediction or project ring with no
readable dates, and a tile with no output kind (or classes or values with no grain), are drawn as a dashed outline; a
model's months, sources and size are then said to be "not recorded" in its caption.

### The Check card

`olmoearth_check_map` is a tool of this server's own, not the registry's (which keeps its 25), for how wrong one Studio
map is. `result_id` is a Studio result, or its prediction when that prediction has one result. The server reads the
result's files with `olmoearth_scores_from_studio`, one map per UTM zone, passing `clip_out_of_range`: a `[0, 1]` score
with values outside 0 to 1 is clipped into it and counted, not refused, however many pixels lie outside, while its
values stay above -1 and below 2; a score whose values reach -1 or 2 is not read as a score from 0 to 1, and the card
says so with its lowest and highest value and offers no press; the flag changes nothing for a class map with top-k
scores. For each zone the card offers the label counts the package plans before any label (`plan.plan_error_rate` for
intervals no wider than 20%, 10% and 5% from their low end to their high end, at its 50% planning rate; the planner's
sentence says a width that way, never in "points", which name a Studio grid's points), none picked.

#### The card's face

Its title, the map's Studio name and period, four small step pictures (step 1 names the square's size when every
part's is the same; step 3 reads "Someone who can recognise what the map is meant to find looks at each one in a
satellite picture."), then "How many squares do you want to check?" and under it, while a part offers more than one
number, one line: "More squares: more work, and a narrower range."

**Counts.** Each part, headed by its name and size ("West part (1,234 squares in all)"), offers the package's counts,
each button its number alone ("20 squares") with no drawing, all of one size (one grid of equal columns, a row each
under 480 px, so the card never scrolls sideways). Nothing is picked for the user. A chosen number has the pale fill, a
dark line and a tick before it ("✓ 20 squares"), the look of a chosen answer on the sheet; the tick is drawn by the
style and left out of the button's name, and the button keeps its size; a count's line never breaks, so a chosen count
of four digits stays one line in the narrowest columns (a part's card from 481 px), its words running into the
button's right padding, inside its border. A picked count shows the planner's own sentence for it, whole, with its
`rate_note`; a second press on it clears it.

**Parts.** A map saved in several parts, each a strip of its own, says why under that line, beside a picture from the
OlmoEarth docs of the world cut into strips, one in pink, in one white card (`views/brand/world-strips.jpg`, recoloured
to the views' greys with the docs' words, zone numbers and state taken out): "Studio saves a map in one part for each
strip of the world it covers, like the pink strip here. This map has 2 parts; each gets its own check."; a map whose
parts are not one per strip (one strip on both sides of the equator, saved as a "North part" and a "South part"; a
file that is no strip; a strip saved twice) gets "This map is saved in 2 parts. Each part gets its own check." in a
white card with no picture, since the strips sentence is untrue there. Each part of a map in parts is one white card
that holds its name, its counts or its state, said once, a section named by its heading for a screen reader (two
parts' "Open the sheet" are told apart by their part; a map in one part has no heading and no name). A part is named
"West part"/"East part" or "North part"/"South part" only when that is certain (two UTM zones side by side, or one zone
in both hemispheres), else "Part 1", "Part 2" (never a zone's name on the face).

**"How this check works".** Once a part is picked (its squares drawn, by a press on the card or before it opened), the
four step pictures and the strips card (the picture with its line) go into one closed fold, "How this check works",
after the parts, the press or the sheet open in the card and the status line under it, and right before "Details", so
the parts and their sheets come first; opened, it shows the same panels and picture. Before any part is picked the
card is as described here; a map in one part folds its steps alone; the plain parts line holds no picture and stays
where it is; a card whose map is being prepared keeps its steps.

**The press.** One line over "Pick the squares": "Once you press, your rule and number are fixed. You get a result
once every square is answered." The line for a user who comes back to a card drawn before a part was picked, "Picked
before? Press again: the same squares open." (the server picks nothing twice), is in "Details", after how the pick is
fixed, only while the card has a press and shows no part as picked or as pressed with no reply (beside a picked part
the press picks another part's new squares). Under the press one line, in body type, says why the press is off or what
it leaves out: "Choose a number of squares above."; on the rule gate with neither, "Write your rule and choose a
number of squares first."; with the gate's rule too short, "Write the labelling rule first, in one sentence of at
least 10 characters."; with 1 to 9 characters in the optional box beside the app's rule, "Write at least 10 characters
in the box, or empty it to use the rule above."; and on a map in parts, for the parts that still offer numbers and have
none, "East part has no number chosen: this press leaves it out." (up to three parts by name, more by their number). A
press held by the rule or by the box only puts the focus in the box. While a part's sheet is being opened the press is
greyed and does nothing, and the line under it says "A sheet is opening. Wait for it.", so no pick starts under a sheet
about to open. The planner's sentences, files, zones, square size, seed and notes are in a closed "Details" fold; the
card's percent figures sit only in that fold.

**After a press.** A part not picked says in its own card "Not picked yet. To pick this part's squares, ask Claude to
open the Check card again."; a picked part's card says how far its sheet is answered, beside its open button. The
labelling rule, its scope, its example pictures and its box sit in one white card; a word too long for a line breaks
inside that card, inside a part's, and inside a part's lines in "Details" (the server's words on a sheet that cannot
be read, with its file's name, are there). Once no part offers a number the heading stops asking: "Your squares", or,
with every picked part answered whole and no sheet open, "Every square is answered. Open the sheet, then press
Estimate the error rate." on a map in one part and "Every square is answered. In each part's sheet, press Estimate the
error rate: one result per part." on a map in parts where more than one part has a sheet (a map in parts whose other
parts cannot be checked, or failed for good, says "Open the sheet"). Where a drawn sheet already gave its result, whose
finish button reads "Show the result again", the heading names that button, "Every square is answered. Open the sheet,
then press Show the result again." (one part) or "Every square is answered. In each part's sheet, press Show the result
again: one result per part." (every part's sheet gave one), and names none when only some did or one's answers changed
since its result: "Every square is answered. Open each part's sheet to get its result: one result per part." ("Every
square is answered. Open the sheet to get its result." on one part); the card reads it from the sheet's counts
(`graded`, `graded_changed`), never a grade.

**Part lines.** A press on a part picked before, in another window, says "Already picked on <day>: <n> squares. That
pick stays; the number chosen now is not used." When an open sheet gives way with answers it could not save, its part
says so in its own red line, by the part's name, with the next step ("1 answer for Part 1 was not saved: its sheet was
changed somewhere else. Open its sheet again and answer that square again."; the line goes when that sheet opens
again, and the toast in the same words with it). A part that cannot be picked says what to tell Claude ("This part
cannot be picked again here. Tell Claude that the pick failed for this part."; "No number is planned for this part. Ask
Claude to plan it with a number of your own."; "This part cannot be checked: the app could not cut its map into
squares.", its reason in its lines in "Details": "This part cannot be checked: <reason>."). A failed pick says "The
pick failed." once on its part's face ("The pick failed. Press Pick the squares to try again with 80 squares." when the
press retries it; "The pick failed. Not in this press: it waits for a later press of Pick the squares (80 squares)."
when more picks failed than one press takes and this one waits; the line under the press calls that press "this press"
too), the server's reason in the part's lines in "Details"; a press that did not pick every part says "Not every
part's squares were picked; each part says so." After a press that got no answer back the card picks nothing more: the
gate's one line is the one place that says to ask Claude to open the card again, and the focus, when the press had it,
goes to that line; the part pressed says "No answer came back: this part may or may not be picked.", every other part
"Not picked yet.", and no press stays; a screen reader hears "The press got no answer back." from the page's hidden
status.

#### Preparing a large map

All of a map's preparation (the read, the refusals, the offers, the rule, the squares' size, the cut and the clipped
counts) runs in the background on the server, one per result id as given. The call waits for it up to 20 s
(`OLMOEARTH_CHECK_WAIT_SECONDS` changes it), so a small map's card opens ready. A large map's first read takes about a
minute or more: such a call answers at once with a card that is preparing (view data `preparing`, the `result_id` as
given, the `stage`, and `map` once the map's name and period are read; no parts) and tells Claude that the card fills
in by itself, so not to call `olmoearth_check_map` again for that result while it reads.

**What the card shows while it waits.** Its title, the map's name and period when the view has them or once a status
answer brings them (the server reads them after the download; drawn once, under the title), the four step panels (step
1 without the square's size, not known yet) and one line, "Preparing this map: Downloading the map from Studio." (the
server's stage after the colon), beside a small turning mark with no number in it (it stands still where the system
asks for less motion; a screen reader hears the line change, not the mark, and the line is written only when its words
change; the mark sits by the line's first row, its turning edge in the text's colour), then, under that line and
outside it, so a new stage is heard without it, "This card fills in by itself when the map is ready; you can keep
chatting." (it goes when the card cannot ask about its map and when it stops waiting). There is no press, no count, no
rule and no box until the map is ready, so nothing typed can be lost. The card asks the panel-only
`olmoearth_app_check_status` 2 s after each answer (an ask that fails is asked again).

**Ready or failed.** Once it answers ready, the card is drawn by the same function, from the same view data, as a card
that opened ready, and tells Claude its parts as that card does, then, once, silently, the ready answer's
`model_text` (below); none of it is on the card, and a card drawn in its place drops it. Once it answers failed, the
card is the failed card ("The check could not start. ... Ask Claude what to do next: it has the reason."), and Claude
is told that the preparation ended without a check, with the words the card shows for it (a refusal's reason, else the
server's plain error, which the card holds in Details) quoted as data, or that it gives none. Either way a screen
reader hears it: "The map is ready to check." or "The map could not be made ready to check.", from one hidden status
outside the card; the focus does not move. After 10 minutes without an end the card stops asking and says "Preparing
this map took too long. Anything picked or saved before is kept. Ask Claude to open the Check card again." in the
failed card's red card, still the line a screen reader hears. While it waits, the card tells Claude that it fills in
by itself.

**Handing Claude the map's text.** Where the host takes the card's silent updates, the card, once it fills in, tells
Claude its parts and then hands it a part that starts "Its map is read. What olmoearth_check_map answers for it, to
take as that tool's answer:", then the text a call in time gives (the say-first lines first). It hands it on whole: a
text block of its own after the panel's facts, never cut, not under their head and outside their 800 characters, in
every later update of the card too; over 12,000 characters it says instead, under the same opening words, that the
text is too long to hand on and to call `olmoearth_check_map` once for that result before stating any result of it.
The note calls that part the tool's answer, the one exception to the rule that what panels tell Claude is facts, not
requests, and keeps one more call: once the card told Claude its parts without handing it the map's `must_state` and
`forbidden_claims`, or when no line of the card reached Claude by the user's next message (a host that takes no
context updates), Claude calls `olmoearth_check_map` again for that result before it states anything of the map,
answers a question about it or plans a part the card does not draw; otherwise not unless the user asks for its card
again.

**The preparation on the server.** The preparing answer carries the rules that hold for any map (no count picked and
no plan for a part the card draws, the blind rule, each part its own map, no grade before every window is labelled);
without panels it says to call the tool again in about a minute, which answers from the finished preparation, and not
to grade. A second call while it runs joins it (never two downloads of one result), with the same key in effect only:
after a sign-out or with another account a call stops that read and starts its own. A call after it ended answers once
from that outcome, ready or failed (unless it asks `fresh` and that read was not fresh); one answered from already is
started again, so a card opened again shows what was picked since. A call that answers a ready card from a read that
ended before it writes the offers and reads the rule again from that read, so a part drawn since shows as drawn;
nothing is read from Studio again. A preparation stops after 540 s (under the card's 10 minutes) and ends failed:
"Preparing this map took too long. Ask Claude to open the Check card again." (stopping it stops the download, not a
decode already running).

**The status tool.** The panel-only `olmoearth_app_check_status` (`{"result_id"}`) answers `{"state": "preparing",
"stage"}` (with `map` once known), `{"state": "ready", "view", "model_text"}` for a card of parts, with the view data a
call in time gets and, as `model_text`, the text that call gives Claude, word for word, one text block per line, and
`{"state": "ready", "view"}` for a result whose files gave nothing to read (the card tells Claude its reason; neither
is marked answered from, so a later call answers from it at once), or `{"state": "failed", "view"}` with the failed
card's (also for a preparation made with another key than the one in effect now). The stage is one of a few fixed
plain lines, set as the read goes: "Asking Studio about the map", "Downloading the map from Studio", "Reading this
computer's copy of the map", "Reading part 1 of 2" (or "Reading the map"), "Cutting part 1 of 2 into squares" (or
"Cutting the map into squares"), "Planning how many squares to offer".

**Kept copies.** The result's files read once are kept on this computer for a day ([Where the scores come
from](#where-the-scores-come-from)): a card opened again reads them from there with no download (each part is still
decoded and pooled, so a large map's card can still open preparing), and says "Read from this computer's copy of 18
Nov 2026, 10:04 UTC." once, in its Details (Claude's text says it too, as `read_from_copy`). `fresh: true` downloads
Studio's files again; the card has no button for it.

**Kept counts.** The counts planned for a part are kept on this computer too: `checks/planned/<sha256>.json` in the
output root, mode 600, written whole (a new file beside it, moved over it), one per number of squares, widths,
olmoearth-inferencex version and digest of the app's modules that word the counts' sentences (`check.py`,
`tools/estimation.py`, `tools/statistical_rules.py`), with the planner's answer as its JSON text whole. It is read
before planning, and a card opened again in a new server offers the same counts and sentences, word for word, without
planning; a file that does not parse, names other inputs or holds another shape is planned again and replaced, a link
there is never followed, and with no package version known nothing is kept. Each time one is written, the files there
named as kept answers and last written more than 30 days ago are removed, so an app update's older files go; a link, a
folder or another name is never removed, and a removal that fails is logged by its kind and leaves the file. The
package is still loaded first, so without it the counts are refused.

#### Picking the squares

One press ("Pick the squares", through the panel-only `olmoearth_app_check_draw`) draws each zone picked once, through
the plan tool, with the count, design `random` and seed 0 fixed and logged on the server (`mcp_app/check.py`: records
under `checks/` in the output root, a file lock another server process honours, one line per zone of each press in
`log.jsonl`). Each line holds the press's time, taken once for the whole press before the server's lock and every
claim (so a slow plan, an earlier zone of the same press or a press it waited behind does not move it; a zone's first
draw record holds the same time), the check's key, the result, the zone's file, the count pinned and the count sent,
the design, the seed, the outcome and the draw's state after the press. A press the server refuses whole (not a list,
an empty list, or more than 16 zones) draws nothing and writes no line, and a line that cannot be written does not stop
the draw: the draw record, not the log, holds the settings pinned at the press.

**At most 16 zones.** One press draws at most 16 zones (the card's data gives the page that limit as `draws_max`). The
failed draws it retries go first, at most 16 (a retry has no count to unpick), then the counts picked, up to the room
left. With more picked than that room, the press is off and a line under it says how many to unpick (and how many
failed draws this press retries), then to open the card again for the rest. When failed draws alone pass 16, the press
stays on and retries the 16 that have waited longest (by the time each zone's draw failed, earliest first, one with no
time first, ties in the card's order; a draw left drawing past 600 seconds counts from when it was claimed; a draw that
fails again waits behind the others, so each gets its turn), and the line says the rest wait for the next press, or for
the card opened again. The time only orders the retries: it is never shown or told to Claude.

**Which count a press may send.** A first press must send a count the card offers. A press on a zone drawn already gets
the stored draw ("repeat"), whatever count it sends; a draw under way, a final failure and a zone whose files changed
also take any count and draw nothing. A draw that failed for a reason that may pass is retried with the count its first
press pinned, never the count sent now; the retry is taken when it sends that pinned count or any count the newest
offer holds, and is refused ("This card is out of date. Ask Claude to open the Check card again.", the same refusal as
for a first press with a count the newest offer does not hold) while the newest offer holds no count (the package
missing, say), so the zone stays one to retry. A draw that will not pass (a refusal, a scores file gone) is not drawn
again and points to Claude; a zone whose Studio files changed after its draw is refused. A draw runs the plan tool as a
call Claude makes would, so a failed one also counts among the run's failures, which feed the hint that tells Claude to
stop retrying.

**A press that stops.** A press cancelled while its plan runs (the page's call timed out, or the host dropped it)
records its draw stopped ("The draw stopped before it finished.", which a later press retries with the pinned
settings; a draw already drawn by then is kept) and logs the zone `cancelled` at the press's time, both shielded from
the cancel, which then goes on; nothing is drawn twice. The page waits up to 600 seconds for a press (the server's
limit for a draw under way). A refusal of the whole press by the server says "No square was picked"; a press that gets
no answer back says it may have drawn, its zones are not pressed again from this card, and Claude opening the card
again shows each zone's draw.

**The sheet in the card.** The first zone drawn opens in the label editor in place of the press, unless a sheet was
opened here while the press ran. The press is not shown while an editor is open, and a zone not drawn then says to
open the card again. A zone drawn earlier keeps a button to its sheet, with its progress (counts only); with no zone
left to draw, that button is the card's contained one. The open buttons are off while a press runs. Opening another
zone's sheet first saves what the open editor holds; when that save fails, the open sheet stays, with its Save, and
nothing else opens, and the open sheet says so once, in its footer's line under Save labels, after the save's reason
and in place of its "Press Save labels to try again.": "The sheet open here has a change that is not saved, so it stays
open: press Save labels to try again, then open the other sheet."; that line goes once a save is taken. On a card of
parts the status line sits right under the press or the sheet open in the card, before "How this check works" and
"Details", and goes back to the page's foot before the panel draws another card, so that card has the page's one
status region in place before its first message.

**An editor that stopped saving.** An editor whose sheet changed since the panel opened it is not saved again: after
any save or read under way, the other sheet opens, and when the editor held answers not saved, its part says so in its
own red line, and one toast says the same words ("1 answer for Part 1 was not saved: its sheet was changed somewhere
else. Open its sheet again and answer that square again."). The line counts answers and never names which squares; it
goes when that part's sheet opens again, and so does the toast while it still says those words (the sheet open then
saves its answers; any other toast stays), and the file name stays in the sheet's "Details". When it held none, the old
editor's last toast about that change is cleared.

**A sheet that cannot be opened.** A sheet that cannot be opened after a good draw keeps that button, and its part says
in plain words what to do, naming the button by the label it shows, in double quotes ("The sheet could not be opened.
Press "Open the sheet here" to try again; if it fails again, tell Claude."). A part picked earlier whose sheet cannot
be read when the card opens says, before any press, "The sheet of West part could not be read when this card opened.
Press "Open the sheet" to try again; if it does not open, tell Claude." (no part's name on a map in one part) over its
button, until a press of that button opens the sheet, or its counts read when the panel is shown again; a press that
fails too puts in its place, red, "Its sheet cannot be opened now. Tell Claude that the sheet of West part cannot be
opened." and adds no second line, and Details say the reason once. The reason, in the server's words, is in the part's
lines in "Details": "NAME does not exist; plan the sample again", or, for a sheet that cannot be read as CSV, "NAME
cannot be read as CSV (...); save it again as CSV UTF-8, comma-separated".

#### The labelling rule

**No rule.** A map that is not one of presence and absence comes with no rule. Its Check card asks for the rule in one
sentence (10 to 300 characters) in a box labelled "Your rule" in body type, with an example in a small line under the
box ("For example: the class that fills most of the outlined square"); no red line. That the pictures are 10 m images
is in "Details". The press stays off until the rule is typed. The rule is pinned with the draw, saved beside the sheet
(`.<sheet>.rule`, refused once any answer is saved), shown on the sheet right above the answers and on the result, and
reused by the map's other parts. Without panels nothing is stored, and Claude asks for the rule in the chat.

**The app's rule and its box.** A map of presence and absence shows the app's general rule on the card's face with
"This rule came with the app, for any map of present and absent." While the rule is not settled, a box stands beside
it, labelled in body type "Did the map maker mean something else? Write their meaning of Present and Absent
(optional).", with a small line under the box, "Left empty, the rule above is used." (the box has no grey hint, and
neither line names a map). The app does not guess what the map shows. An empty box keeps the app's rule; 1 to 9
characters hold the press; 10 to 300 replace the app's rule for this map: pinned with the draw like the gate's rule,
and shown on the sheet, the pictures, the result and to Claude with "Typed by you before the squares were picked,
<day>" (a later part keeps the day it was first typed), never the app's rule's scope. The rule is settled once a part
is drawn or any sheet of the result holds an answer; a press that still sends another rule draws nothing and its reply
says so ("This map's labelling rule was set when its first part was picked."). Every press's reply carries the map's
rule as the server holds it, which the card shows from then on, so a second card on the same map shows the rule the
first one pinned; after a press with no reply the card shows no rule and no box, only "This card can pick no more
squares. Ask Claude to open the Check card again: it shows the rule and what was picked.", and picks nothing more (no
part keeps its numbers, so no press shows) until the card opened again gives the rule. Claude is told the card shows
the app's general rule with that box, where the user writes the map maker's meaning if it differs, and not to write it
for them unless they ask. Without panels, Claude is given the rule with its scope and told to give it to the user
before any plan and ask whether the map maker meant something else by Present and Absent, then for that meaning in one
sentence, which it repeats with the result. Maps that are not of presence and absence keep the gate above.

**An earlier wording of the app's rule.** A map a part of which was drawn with an earlier wording of the app's rule
shows that wording and its scope, as its draw records hold it, and so does a map with no part drawn whose sheet
planned in the chat holds answers under it, as that sheet's pictures' record or graded mark holds it
(`check.result_app_rule`; the wordings: [The rule on the sheet](#the-rule-on-the-sheet)). A press that failed, or was left
drawing, drew nothing and does not count, so an earlier rule shows only once the rule is settled; each first press,
and the retry of a press that drew nothing, pins the app's rule it is drawn with, the map's own when it has one.

**Example pictures.** Under the app's rule (and before its box), the card shows the same two example pictures as the
sheet when the app ships them, with the same captions in general words and, under them, the same small line "Example
pictures from a kelp canopy map.", and their place line in "Details"; never beside a typed rule, nor after a press
with no reply.

**Typed rules.** A rule pasted with no-break or zero-width spaces is kept with plain spaces; a rule the server refuses
says why ("one line of 10 to 300 characters"). A Check card drawn by an earlier version of the app refuses its press
and asks to be opened again.

#### Maps the card does not cut into squares

**When the check cannot start.** A zone not read as a map, or with no count planned, says so (a zone not read as a map
in plain words, "This part cannot be checked: the app could not cut its map into squares.", the server's reason in its
lines in "Details"). When the check cannot start (the output folder cannot be written, Studio's files could not be
read, or any other refusal that is not a class map's, a map of numbers' or the sign-in), the card offers no press:
under its title a red card says "The check could not start. Anything picked or saved before is kept. Ask Claude what
to do next: it has the reason.", and the tool's words are in a closed "Details". The line does not send the user to
open the card again (most refusals come back the same on every open, and Claude's text says not to retry with the same
arguments) and does not say nothing was saved (a card opened again keeps its earlier picks and answers). A refusal the
server wrote for the card's face, a map that is not a score from 0 to 1 (below), is marked `plain` in the view data:
its sentence is the red card's first line, over that line, with no fold.

**Error words.** A card's error is the tool's own words without the exception's name at their start ("ValueError: "; a
`KeyError`, whose own words are only the missing key, reads "the tool did not find an entry it needs ('<key>')") and
without an OSError's number at the start of the error, after the exception's name if there is one, when words follow
it ("[Errno 28] ", or on Windows "[WinError 32] ": "The pick failed: No space left on device." in the Check card's
"Details", its part's face saying "The pick failed." alone; a number with no words after it stays). No card shows the
hint written for Claude ("Check that referenced ids exist ..."): a card's view data holds a hint only when the card
offers the sign-in, and that card shows the sign-in with its own note, not the hint. Claude's text and a check's
stored draw record keep the whole error and the hint.

**A class map without top-k scores.** Its grid spans every zone, so the Check card cannot draw it. Under its title the
map's name and period, which the server reads at best for the refusal (the result's record once, its download token
dropped at once, only its prediction id and period read; the prediction's name as on the ok path; the view's `map`
`{name, period_text}`, never the model's text; no line when Studio does not answer in time). Its face says two lines:
"This map can be checked on points picked at random, instead of squares." and, before any count is fixed, "The app
shows no pictures of points: each needs your own imagery or records from the map's dates."; one button, "Check this
map on points", sends Claude that request with the result's id (the view's `points_request`); a closed "Details" holds
what the check gives ("You get how often the map is wrong over a grid of points spread across the map, with a range,
not for every pixel."), then the tool's sentence. Claude is sent to `olmoearth_plan_label_sample` and told to say first
that each point needs the user's own imagery or records from the map's dates and that the app shows no pictures of
points (the card's own line, for a client with no card), then to ask the user for the budget and for design `random`
or `class` in two plain questions (how many points; the whole map only, or each class too), never saying "design" or
"budget", and to pick neither for them. That case is known by the package's contract (the forbidden claim
`zone_for_a_class_map` and the refusal's fixed end), never by words in the error.

**A map that is one score from 0 to 1.** Its two classes are "<name>" and "not <name>", so its Check card asks for the
labelling rule as above. Right over the rule's box one read-only line says the cut: "This map gives each pixel a score
from 0 to 1, not a yes or no. This check counts a pixel as "<name>" when its score is above 0.5." (strictly above, as
the pooling counts it). The same line is among the notes in "Details" and first in what Claude must state. Where pixels
lay outside 0 to 1, "Details" say for each part how many were set to the nearest end, and Claude reads the count
(`clipped_pixels`). The cut is fixed: the card does not ask for another, and it changes no count the tools compute. A
score whose values reach -1 or 2 gets no such card: "This map's scores run from <lowest> to <highest>, so it is not
read as a score from 0 to 1.", with no press; that sentence is on the face, the red card's first line, over its "The
check could not start. Anything picked or saved before is kept. ...", with no fold (the server marks that view
`plain`).

**A map of numbers that is not a score from 0 to 1.** A result that declares one regression field on another range
(say from 0.2 to 1.2) is refused by `olmoearth_scores_from_studio` in the tool's own words, which name the tool Claude
needs (`olmoearth_plan_label_sample`, once a threshold is given) and stay in Claude's text. The Check card's face says
two plain lines in their place, with no press: "This map gives a number at each pixel ("<property>", from <min> to
<max>), not classes and not a score from 0 to 1. This check works on a map of classes." and "To check it, tell Claude
the value from which a pixel counts as a yes. The check then asks you about single points, without pictures." (the
view's `error` and `next`). The refusal is known by its fixed start and end with exactly one field between them; a
result with several regression fields, or none, keeps the tool's sentence on its card. The ends are the tool's own, in
plain digits ("1e+06" reads "1000000"). A field with no declared range (or an end that is no number, such as "inf")
reads "This map gives a number at each pixel ("<property>", no range declared), not classes. This check works on a map
of classes." (the tool found only that it is not declared [0, 1]). A range the tool's text rounds to "[0, 1]" (say 0 to
0.9999999) keeps the tool's sentence, since its true ends are not in that text.

#### What Claude reads from the card

Each zone is its own map, with its own sample and error rate; no rate combines zones. The card shows no rate, interval,
figure, window, place or class, and takes no focus when it opens. Claude reads each zone's name, file, windows, share,
state and scores path; for a drawn zone also its `labels_csv_path` and its plan's `facts` (among them
`prefix_is_random_sample`: every one of the sheet's rows labelled for the routes that read the design file),
`must_state` and `forbidden_claims`; `failed_for_good` on a failed zone, and `raster_changed` where Studio's files
changed after the press. The card's view data has no plan `facts` or `forbidden_claims`; the model's text, which the
host also hands to the page, has them: the plan's generic contract wording, with no window, class or place. It never
reads a check's key, and with panels never the counts.

The card's note tells Claude, for a zone the card draws, not to pick a count and not to plan it, and to wait; a zone
with a scores path that the card does not draw (no count planned, its draw failed for good, or its files changed) is
planned only if the user asks, with the user's count, that scores path, design `random` and seed 0. The note ends with
the blind rule ("Do not tell the user any window's map class, and do not rank or sort these zones' windows while their
sheets are being labelled.") and the context rule ([What Claude is told](#what-claude-is-told)). Each drawn zone has a
note that it is not drawn again and that its entry carries its plan's `facts`, `must_state` and `forbidden_claims`;
while its files are as they were at the press, a second note says the same settings open the same sheet with its labels
kept, and any other count or seed is another sample with a blank sheet, made only if the user asks. A zone whose files
changed or whose draw failed for good has its own note. The tool's description adds: "Its card tells you silently how
many windows of each zone's sheet are labelled; never grade before every window is." In a client without panels Claude
gets each zone's counts and sentences to ask in the chat, and plans each zone itself with the same settings; that note
ends with the no-grade rule ("Never grade a sheet or give a rate before every window is labelled and the user asks.")
but not the blind rule, since the user then fills a CSV that shows `map_class`.

### The label sheet

`olmoearth_label_sheet` opens a plan's `*_to_label.csv`, one row per window, labelled the way the estimate reads it
(the sheet's label rule, from the design file beside it). The same editor opens inside the Check card and the plan
card.

#### What each sheet asks for

- a scores map whose design names its classes and holds the map's class of every window (every plan on an
  `olmoearth_scores_from_file` scores file): the class seen, one of the design's names (Present, Absent, Cannot tell on
  a map of presence and absence; no Cannot tell on a certificate plan). The server writes the class code into
  `reference_class` and works out `wrong` from the design's `population.map_class` (never the sheet's `map_class` cell)
  at each save, on every row that holds an answer, not only the rows sent; a `wrong` sent by the panel is refused. A
  row counts as labelled, and the panel shows its answer, only once its `wrong` is worked out, as the estimate reads it:
  a code typed into `reference_class` in a spreadsheet with `wrong` blank is neither counted nor shown until the next
  save from the panel, and a `wrong` edited by hand is worked out again then;
- any other scores map: right or wrong (`wrong`) on every window, or Cannot tell (`?`). A scores file whose manifest
  names no class names its classes `class_0`, `class_1`...; names that are all of that form name no class, so such a
  map is labelled right or wrong;
- a class map: the class seen (`reference_class`) on every window, Cannot tell (`?`) where it cannot be told. The panel
  has no switch to right or wrong and no blind tick box: the sheet never shows the map's answer. A sheet already
  labelled right or wrong (in a spreadsheet) opens on that face, which shows the map's class; the server still refuses
  a save on the other face once one face holds answers. A mix made in a spreadsheet, which the estimate refuses, is said
  in plain words ("This sheet holds two sets of answers, so no result can come from it."), with a button that removes
  the set in the way and names both by what they hold, never by a column ("Remove the right-or-wrong marks, keep the
  classes picked"); the tool's own sentence is in the sheet's Details;
- two maps: the class seen, `?` where the user cannot tell.

Only the cells the user changed are written; every other cell stays as the plan wrote it. A class outside the list can
be typed in. There is no running error rate, no interval while labelling, no suggested label and no sort by
confidence; the counter of labelled windows stays.

#### What reaches the panel

The panel never gets a row. `olmoearth_app_sheet` sends a page copy (`sheet.read_sheet`): each window as a handle (`w`
and 16 base32 characters) keyed by a secret the server keeps in a hidden file beside the sheet (`.<sheet>.handles`,
mode 0600), with its answer in the one column the panel writes (the face), in the order of the handles, the same at
every read and after a server restart; the sheet's id names the windows drawn, so a sheet drawn again with other
windows refuses a panel opened before. No window index, row, column, place, order or stratum reaches the page, and the
map's class only on the right-or-wrong face. The answers and their keys come from the server: on a named scores map the
codes from the design, never from the rows; on a class map the design's classes, then those the sample shows, sorted by
name. The pictures, the saves and the map link name a window by its handle; a panel of an older server, which names it
by its index, is told to open the sheet again. The handle keeps the window from the page, not from the model, which
reads the plan. A named scores map's sheet already labelled right or wrong (`wrong` filled with `reference_class` blank
on some row) stays right or wrong, with the map's class shown and a note saying why; nothing is read anew.

Labelling the class seen is blind, as the plan's `how_to_label` asks: the map's class is not sent, no page is ever sent
a stratum, and a two-map sheet offers every window the same classes in the design's order. Right or wrong shows the
map's class (by name, where the design names it), so it is not blind, and a sentence says so. The panel reads the file
each time it opens: the tool's result carries only the sheet's counts and job (as the plan's button reads them), never
a row, so a chat opened again shows the sheet as it is now, and a sheet that cannot be opened or read then shows a card
that says so and what to do ("This label sheet could not be opened. Ask Claude to open it again."), with the reason in
its Details.

#### Saving

Each save names the sheet's id, the face and the label rule the panel drew; a sheet drawn again, or whose rule has
changed since (its design file moved, or `wrong` filled in a spreadsheet), refuses the save and is left as it was. A
retry would be refused the same way, so the panel stops saving, says so, and offers "Reopen the sheet", which reads the
sheet again and draws it as it is now (the changes not saved are not kept). Meanwhile its footer has no Save button:
beside the counter it says how many answers given since were not saved ("1 answer not saved", or "Not saving"), and
beside "Reopen the sheet", in body type, what happened and what to do ("This sheet was changed somewhere else, so 1
answer here was not saved. Reopen the sheet, then answer that square again."). A press of the finish button that found
the sheet changed somewhere else sends nothing; "Reopen the sheet" takes the finish button's place and its focus, and
that line is the description of "Reopen the sheet", so a screen reader hears it with the button. After the panel's own
removal of a set of answers (two sets on one sheet, above) whose sheet it then could not read again, that line is "This
panel could not read the sheet again. Reopen the sheet to carry on." (nothing else changed the file), the one place
that says what failed; the status line says only which answers were removed ("The right-or-wrong marks were
removed.", a plain message, not an error). That footer line is the description of "Reopen the sheet", which takes the
focus of the Remove button pressed, so a screen reader hears what failed with the button. In a client without panels,
the file route keeps the `wrong` column on every scores map.

**One live copy.** The page copy and the sheet's counts carry its version: a keyed digest of the file's bytes under the
sheet's own secret, so a server restart keeps it. Each save names the version the panel read. Once the sheet was saved
elsewhere since (another panel, a spreadsheet), the save is refused as stale and the file is left as it was; the panel
then reads the sheet again and takes it in. A window it did not change shows the label saved there; a window it changed
keeps its change where the label there is still the one it read, and takes the label saved there where both changed
it: the first save wins, and the panel names those windows by their place in its list ("Square 4 keeps the label saved
there"), never by an index. Then it sends what is left, with the new version, at most twice in a row; after that it
says the sheet keeps changing elsewhere and waits for a press of Save labels. Those words, like any save that failed
("Not saved: ... Press Save labels to try again.") and a press of Save labels with nothing waiting ("Every change is
saved to ..."), are a line in the sheet's footer under the row with the finish button and Save labels, at its right
end, while Save labels shows; a save that went through, "Saved to ...", is said beside the counter in the place of
"Saved automatically", which shows only while nothing waits (so it never stands under Save labels or beside "1 change
not saved yet"); and "This sheet was saved elsewhere; it shows those labels now." stands beside the counter whose count
it changed. None is in the status line at the page's foot: a failure is an alert in red, the rest a polite status; a
save that went through clears the counter's line, whose note on the squares kept then ends the "Saved to" words, and
the line under the row; an editor already off the page, another part's sheet opened in its place, still uses the
status line. The panel also asks for the version (counts only) when it is shown again or gets the focus, at most once
every 3 seconds, and before its finish button sends: a sheet saved elsewhere is taken in, and a finish press that found
a change sends nothing, so the user sees it and presses again; a sheet the counts say gave its result since the panel
drew it (Claude graded it after the finish press) is read and drawn again, its answers locked, once nothing waits to be
saved. The finish press first waits for a save or a read under way.

**When answers are saved.** A save from the panel that also worked out rows it did not send (on a sheet labelled with
the class seen, a class typed into the CSV gets its `wrong` at any save) answers `rederived: true`, which names no
window or row; the panel then reads the sheet again and takes it in, every change made in the panel winning, so the
counts it shows and tells and its version name the file as saved. Answers wait 1.5 seconds before they are saved (a run
of answers given closer together saves at the latest 5 seconds after its first), and go at once when the panel is
hidden or unloaded (best effort: a panel torn down may not get to send them). When Claude closes the panel, the answers
not yet saved are saved quietly (no toast, at most 3 s). A save that fails after another sheet was opened in the
editor's place names the windows not saved and says to open that sheet again. While the panel clears a column or
changes its face it takes no answer; a clear that worked but whose read after it failed offers "Reopen the sheet".

#### Cannot tell

Cannot tell (`?` in `wrong`, the package's own token there, or in a class map's `reference_class` with `wrong` blank or
`?` on the same windows) is counted right at each interval's lower end and wrong at its upper
(`estimate_error_rate(unjudged=)`): the point becomes a range, the confidence and proportional designs' second interval
becomes the union bound over the strata, there is no per-class table and no figure, and the estimate's `must_state`
says how many windows could not be judged. Only an estimate with the plan's `design_path` takes `?`; the certified
zone, a certificate, a certificate plan (`seed_from_map`, whose sheet and panel offer no Cannot tell) and the estimate
of windows labelled without a design take 1 or 0 on every window and refuse `?` by name. A `wrong` cell is read exactly
(`0`, `1`, `0.0`, `1.0` or `?`); `0.5` is refused. A class map's sheet offers Cannot tell after its classes, on the
face, in the list and on the `?` key.

#### The two-map switch

Only a two-map sheet keeps a blind switch, "Hide what the two maps say", locked until every square is answered and
saved and the sheet's result was asked for (the finish press whose message Claude took, or a sheet that already gave
its result). Locked, it is ticked and greyed as a busy button is (`aria-disabled`), with "Shown once every square is
answered and the result is sent." under its words as its description, and a press on it or its label, or Space,
changes nothing and asks the server for nothing. A press whose message the host did not take (shown to copy into the
chat) keeps it locked, the line then "Shown when you open this sheet again after Claude grades it.". Then, unticked, it
asks the server for each window's pair, which then leads its list, at that press; ticked again, it hides them, though
the user has seen them. Once it was unticked the sheet's answers are locked, as a graded sheet's are (below), ticked
again or not, since the user has seen what the two maps say, so no answer changes after the maps' answers were shown
and the estimate reads the sheet as it was answered blind.

The lead's first line is the locked answers' description and says they are locked either way: unticked, "Say what you
see in each square's satellite pictures. The two maps' answers are shown now, so your answers are locked."; ticked
again, or locked again by an answer taken in from elsewhere, "Say what you see in each square's satellite pictures. The
two maps' answers were shown, so your answers are locked." On a two-map sheet with no pictures and no places that
line's tail (", without looking at the maps. No records? Stop and tell Claude.") gives way the same way, so it says so
too. Nothing on the face asks for an answer then: the two-map sheet's own instruction and the line that no rule was set
go, as on a graded sheet. This lock is the panel's alone: the sheet opened again starts unlocked (the server does not
record that the maps' answers were shown). An answer changed after the press and before the switch is unticked, in the
panel, or taken in from elsewhere, undoes the press: the switch is locked again, ticked, and the pairs hide, until a
new press Claude takes. A class map's sheet and a named scores map's have no switch (an `unblind` on any other face is
ignored), and their Details say the seen sheet's blind line.

#### The finish button

The finish button sits in the footer from the first open, greyed until every window is labelled and saved. While a
window has no answer, "Answer every square first." ("point" on a sheet of points) stands beside it, so the grey button
has its reason in words, and the button names that line as its description (`aria-describedby`), so a screen reader
that reaches it hears why (not while the sheet stopped saving, which has its own footer words, not while only a save is
pending, and never on a sheet that already gave its result, whose answers are locked: there that place says "A square
has no answer; a new result needs a fresh sheet."). Once ready, "After you press, do not change an answer." shows in
that place, on the counter's row (on a sheet that already gave its result, "This sheet already gave its result.", as
its note says). Beside the counter, "Saved automatically" is plain text (after a save in the panel, "Saved to ...");
"Save labels" shows only while a change waits, a save runs or one failed. The finish button keeps to the footer's
right edge, the counter to its left, also when the line beside the button wraps.

The finish button is named for the sheet's job, and it sends Claude a message that starts in plain words ("I have
answered every square of this sheet. Estimate the error rate."; "every point" on a sheet of points; a two-map sheet's
and a certificate plan's say their own job in the same place), then names the tool to call, with the plan's
`design_path` and the sheet as `labels_path`:

- "Estimate the error rate": `olmoearth_estimate_map_error`;
- "Say which map is more accurate" (a two-map sheet): the same tool, which grades the two maps;
- "Certify the zone" (a certificate plan, `seed_from_map`): `olmoearth_certify_zone` at the alpha the plan fixed, the
  only alpha it takes, then an offer to write the certificate (`olmoearth_write_certificate`).

A sheet with no design file beside it keeps "Ask Claude to grade it". When Claude took the message, the footer says
"Sent to the chat. Claude answers there."; when the host does not take it, "Claude did not get this message. Copy it
into the chat:" with the whole text, as the job buttons do. Until the host answers that message the finish button stays
greyed, keeps the keyboard focus and takes no second press, and the second click of a fast double-click does nothing,
so Claude gets the message once. In a client without panels, `olmoearth_label_sheet` names the same call in its text.

**A sheet that already gave its result.** It says so in a note in body type, in two lines ("This sheet gave its
result; its answers are locked. A second opinion needs another person and a fresh sheet."; why, "A change made after
seeing where the map disagrees makes the result invalid.", is in its Details), and keeps the first of the lead's lines
only. Its answers are locked, and so are a sheet's that says it is not saving (its sheet changed somewhere else, or its
own removal of a set of answers could not read it again; the sheet reopened saves and takes answers again): every
answer button, "Other class...", a typed class's field (read only, its text kept) and OK, the list's menus and each
row's answers are greyed as a busy button is (`aria-disabled`; the answer given keeps its tick and bold words, in
grey), each described by the line that says why (the note, the lead's first line on a two-map sheet whose maps'
answers were shown, or the footer's line beside "Reopen the sheet"), and no press, key or menu change sets or clears an
answer (a change that gets through goes back); Previous, Next, the arrow keys, Pictures, Street map, the list and the
finish button still work. Once the sheet is ready, its finish button reads "Show the result again": it sends the same
message as before, and the same result card comes back. Where the sheet's answers changed since that first result (the
page copy's `graded.changed_since_first`) it reads "See why the result does not count" instead: the press then gives
the withdrawn card, "This result is out of date", not the result again. A graded sheet that is not ready (a blank
square, two sets of answers: nothing in the panel can fill or remove them) keeps the job's label on its grey button.
What the panel tells Claude names the button by the words it shows. So the panel cannot blank a square of a graded
sheet; a square still blank on one (blanked in a spreadsheet after the grading) leaves its finish button grey, with "A
square has no answer; a new result needs a fresh sheet." beside it as its description, never "Answer every square
first.". A graded sheet that holds two sets of answers keeps its notice of them but offers no button that removes one;
its notice says only "This sheet now holds two sets of answers." (the note and the footer say that a new result needs
a fresh sheet).

#### The sheet's face

"Label sheet" and the part it is for; then two short lines (the lead; one on right or wrong, which has no what-to-do
line): what to do, "Say what you see in each square's satellite pictures, without looking at the map." (on a two-map
sheet "..., without looking at the two maps.", and the locked line above once its maps' answers were shown; a sheet
with no pictures and no places has its own line, below); and one Cannot tell line, "Cannot judge a square? Press
Cannot tell; do not guess." (what a guess costs is in Details; or a certificate's own need; none on a two-map sheet).
There is no third line: the footer says "Answer every square first." beside the greyed finish button.

Then the face's own instruction where it has one, and never on a graded sheet, which asks for no answer, nor on a
two-map sheet whose maps' answers were shown, whose answers are locked (right or wrong; the class seen with the period,
when the sheet knows it, "Pick what was there during 1 Apr 2024 to 30 Jun 2024, using the map's class names."; a
two-map sheet's "For each square, pick the class you see there; press Cannot tell when you cannot judge it."). A class
sheet that does not know the map's period has no such line on its face: "For each square, pick what was there in the
period the map covers, using the map's class names; ask whoever made the map for that period." is the first line of
its Details fold after the file name.

**Details.** The fold says what Cannot tell does: you still get a result, with a wider range, but without the
per-class lines. Everything else (the file name, the rule's scope, on a sheet that already gave its result why its
answers must not change now, the keys, what Cannot tell does to the result, what never reaches the panel, the Street
map line, the clear-date note, that the answers are a reading of satellite pictures and not a visit on site, and with
the example pictures "A public place, not your map. Contains modified Copernicus Sentinel data <year>.") is in one
closed "Details" fold under the answers.

**No pictures.** A sheet with places and no pictures, of squares or of points, says first that it has no pictures and
what to judge from ("This sheet has no pictures. See where each square is, then say what was there from your own
imagery or records, without looking at the map."). A sheet with no pictures and no places says "This sheet has no
pictures and no places, so it cannot show where a point is. Answer only from your own records of each point, without
looking at the map. No records? Stop and tell Claude." (on a two-map sheet "... without looking at the maps. No
records? ...").

#### The rule on the sheet

Right above the answers and above the list stands the rule, in body type with "Rule:" in bold, in one box with a
hairline edge that also holds the cloud line, where the face has one, and its example pictures; who set the rule and
when (its scope) is in the Details fold, after the class line with no period. A sheet with no rule that can show
pictures says there, from the first open, "No rule was set for this map. Decide your rule before the first answer and
use it for every square." (a two-map sheet's says "No rule was set for these two maps. ..."; a sheet that already gave
its result does not say it, as nothing under its note invites an answer). Under the app's rule, general for any map of
presence and absence, its scope ("This rule came with the app, for any map of present and absent.", in the Details
fold) is all that follows it.

**Earlier wordings.** A sheet made with the app's earlier rule ("Present: on at least one clear date shown, canopy
fills more than half of the outlined square. Absent: on every clear date shown it fills half or less.") keeps that rule
and its old scope, "This rule came with the app (set 11 Nov 2026 for kelp canopy maps).", as its records hold them
(`imagery.app_rule_of`, in this order: while the sheet holds an answer, the rule its pictures' record printed last, as
beside a blank sheet the record can be an earlier sheet's of the same name; the rule its graded mark kept; the rule its
Check card draw pinned; a mark or a draw recorded before the general rule holds none and counts as made with the kelp
rule; else the general rule); nothing is written again to say so. The same holds for a sheet made with the general
rule's first words ("Present: on at least one clear date shown, what the map shows covers more than half of the
outlined square. Absent: on every clear date shown, it covers half or less."): it keeps them, with the general scope,
as its records hold them, and so does its map's card and a new part of that map, unless a sheet of that map planned in
the chat holds answers under the kelp rule: the map then keeps the kelp rule, the earlier one, wherever that sheet sits
among its sheets. `imagery.APP_RULES` holds the three wordings with their scopes: `imagery.READING_RULE` (the general
rule now), `imagery.KELP_READING_RULE` and `imagery.R5L_READING_RULE`.

**The cloud line.** On a face that takes Cannot tell (not a certificate plan's) and can show pictures, the rule's box
holds, under the rule, "If cloud covers the square on every date, press Cannot tell.", under any rule, so a square
clouded on every date is not pressed Absent. The pictures pane, when the page copy has no rule and the pictures answer
has one (both come from `imagery.rule_for`, so only when the page copy's rule could not be read), shows instead the
rule's Present half on its "Rule:" line, its Absent half word for word under it (from "Absent:" on, where the rule has
it), and, on a face that takes Cannot tell, that same line (the Check card and the result already show the scope).
Every face but right or wrong (which shows the map's class), a two-map sheet's (above) and a sheet with no pictures and
no places (above) has a first line ending ", without looking at the map.".

**Example pictures.** Beside the app's rule only, when the app ships two example pictures (`views/examples/present.png`,
`absent.png` and `examples.json` with the year; none ship yet), the rule block ends with them, captioned in general
words "Present: what the map is meant to find covers more than half of the square on a clear date." and "Absent: what
the map is meant to find covers half of the square or less on every clear date.", with one small grey line under them,
as on the Check card, "Example pictures from a kelp canopy map." (where they come from; the only words of the sheet's
own that name kelp), the Check card's words (`examples` in `olmoearth_app_sheet`'s answer).

#### One square at a time

The panel opens on one window (the face), in this order: "Square 3 of 20" with Previous and Next, then Pictures and
Street map (outlined buttons) and the window's pictures (until pictures are on, one line in body type says "Press
Pictures to see satellite pictures; you are asked first what is sent."), then what the window shows beside its answer
(the map's class on right or wrong, and what an unticked blind switch shows) and the answers the server sends for that
sheet as buttons, each with its key ("Present (P)"; past nine answers, the rest in a list, Cannot tell always a button;
"Other class..." where any class may be typed, the answers' last cell, as large as each answer, with a dashed edge; no
Cannot tell on a certificate plan's sheet). Street map asks the server for that window's OpenStreetMap link at the
press (`olmoearth_app_window_link`) and opens it (it never shows the map's answer); Pictures: [Satellite
pictures](#satellite-pictures). The two picture views are one joined toggle ("False colour" and "True colour"), and
each date is its own outlined block.

**Answers.** An answer's key sets the answer, never clears it, and the next window not yet labelled shows, in the
sheet's order, wrapping round; the left and right arrow keys move to the window before or after. A press on the answer
given clears it. The answer given is told by a pale fill, a dark outline, bold words and a tick before them (left out
of the button's name for a screen reader), not by the dark fill of the sheet's one action button; the pressed button of
a joined toggle (the two views, the pictures' colours, an answer in the list) is bold with a dark line inside its edge.
The answers are equal cells of one grid, every row as tall as the tallest (two a row on a narrow panel, four on a wide
one; three answers stand two and one under 516 px; the list of more classes takes two cells), so a tick does not widen
the answer given, and a class name of one long word breaks inside its cell, after an underscore where it has one (the
name's text is as written); Previous and Next are one width, and so are the buttons of each joined toggle. A list
row's answers are one width where the row has room: side by side on a wide panel, one under the other on a panel under
480 px; beside the open pictures pane they keep their words' widths, so the row stays one line.

**The box and the focus.** Everything about the window shown (its heading, Previous and Next, what it shows, its
answers, Pictures, Street map and its pictures) sits in one box with a hairline edge and 12 px inside it (6 px at its
sides on a panel of 420 px or less, so each date holds a 240 px picture; the rule's box above it has the same inset);
the focus ring is drawn 2 px outside that edge. The focus stays on the face after an answer (a click inside it also
gives it the focus), and goes to the finish button once every window is labelled and saved; the face never takes the
focus when the panel opens.

**Dates.** Inline, the face is a pager with the dates side by side, wrapped onto as many rows as they need at every
width (a look-first card's pane draws its dates the same way): each date keeps its own size, a date with pictures as
wide as its pictures and never wider than the face, so on a narrow panel a date's Detail picture goes under its Context
picture, and each starts at the top of its row, so a month without a picture is a short block; nothing scrolls
sideways, and the arrow keys page there as elsewhere on the face. The face is then taller than the view, so an answer
that moves on to another square, and a move by the arrow keys, brings the new square's heading (and the pictures under
it) into view, the least scroll, none while it is in view: a press of Next or Previous, which stand in the heading's
row, moves nothing, and the last answer stays where it is. In the host's full screen the dates are a grid. While the
pictures are open or on they follow the window shown at once: what the server gave for it, or "Loading", asked for 300
ms after a move, so one window's pictures never show under another's heading.

**Full screen.** Full screen is a button at the sheet's top right; in full screen it reads "Back to the chat" and takes
the card back. Each label shows only where the host offers the mode it asks for (else the button is hidden); the panel
never asks for a mode by itself, and the label follows the host's own controls too. The area map's button works the
same way.

#### The list

The list is one press away ("How to show the squares": One square or List, kept for the sheets opened after in the same
panel): "#" is the window's place, a button that shows it on the face; then the map's class where the face shows it
(and, once a blind switch is unticked, what it shows), the row's pictures button, which reads "Show" under the column's
head "Pictures" and "Hide" on the row whose pictures are open, where a press closes them (its name for a screen reader
or a voice command starts with the word it shows: "Show pictures: square 3", "Hide pictures: square 3"), and the answer
(on a class sheet a menu of the classes, which writes Cannot tell as the face's button does, "Cannot tell (?)"; a row
with no answer shows "(not labelled yet)" there, the filter's and the counter's word). Street map is said once, in the
head of the open pictures pane, for that square; a sheet with places and no pictures has Street map on each row, in a
column of that name. While pictures are off, the line under the list reads "Press Show under Pictures to see a
square's satellite pictures; you are asked first what is sent." (the face's line names its own Pictures button). The
filter (any, unlabelled, labelled) is the list's. On a panel at least 760 pixels wide the open pane sits beside the
table; on a narrower one it stays under it.

**Step by step in the list.** After an answer the focus goes to the next window not yet answered, in the order the
sheet shows, wrapping round: to that row's answers as a group, not to an answer button, so Enter or Space alone
answers nothing (for the class seen, to the row's list); once every window is answered it goes to the finish button
when that shows. A press that clears an answer stays where it is, and so does going through a class list with the
arrow keys. A held key answers once: its repeats set nothing and press no button, the finish button included. Keys on
the focused row set its answer and never clear it, and a line in the Details fold names them:

- a named scores map: the first letter of each answer (P present, A absent), `?` for Cannot tell, or 1 to 9 for the
  first nine answers in order;
- right or wrong: R, W and `?` (no digits, since 0 and 1 are the cells' own values);
- the class seen: 1 to 9 for the first nine classes of the row's list, and `?` where the sheet takes Cannot tell (a
  two-map sheet, a class map's).

**Pictures and scrolling.** The pictures follow the focus to the next window when the pane is open or pictures are on.
A closed pane with pictures off stays closed, and the server still asks before it fetches anything. When a press of
Pictures (Show in the list) draws that question, also one kept from an earlier press, the panel scrolls it into view:
the whole pane above the sheet's footer when it fits, else its top and at least the row of "Turn on pictures" above
the footer, so that button is not left under the footer of a panel of fixed height; the press of One square does the
same for a question kept for the square it shows. The focus stays where it is, and the pane scrolls nothing else: a
pane that follows an answer, Previous or Next leaves the scroll as it is (the face's heading, above, is brought into
view by the answer or the arrow key itself). Each control of the sheet outside its footer (an answer, a menu, a field,
a fold, a row's answers, the face) keeps the footer's own height and 8 pixels free under it whenever the browser brings
it into view as it takes the focus. That happens when the keyboard moves the focus, and when the sheet moves it after
an answer (to the face, or to the next row's answers), a mouse press included. Near the panel's foot the control then
ends at least that far above that edge, where the panel is tall enough for it, so the sheet's footer does not hide it
whatever it holds (about 60 pixels high on a wide panel with nothing said; on a narrow one about 210 with Save labels
and a line of two rows under its row): the sheet measures the footer each time it changes and when its width does. In
the list, the table's own frame (at most 420 pixels high) scrolls the same way, so the focused row stays as far above
the frame's foot. The margin takes no room: no control moves. The footer's own buttons get none: the footer cannot
hide them, and a margin there would only scroll the sheet each time one takes the focus.

### Satellite pictures

The server fetches small dated Sentinel-2 pictures of a window from Earth Search's public catalogue and bucket (no key,
no account), the one exception to "no STAC fetches" ([What it checks](#what-it-checks)), for the label sheet and for a
look-first list. It is off until the user switches it on in the panel, after a question that names the two hosts and
what each receives: the catalogue (`earth-search.aws.element84.com`) the map's extent, rounded outward to 0.1 degree,
and its period, once per sheet or review set (one search per month, paged; sent again only if the searches did not
complete); the bucket (`e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com`) byte-range requests for parts of
four band files for each scene tried (at most three per month) and of the scene's own cloud mask (at most six per
month); both see the computer's IP address. Nothing goes to
Studio or to Claude.

**The question.** Its face is three short lines: "Show satellite (Sentinel-2) pictures of each square? To fetch them,
this computer sends the box around the map, rounded out to 0.1 degree, and the map's months." (two rows of the panel at
780 px), "They go to Earth Search, a public catalogue, and its file store; both see your IP address." and "Nothing is
sent to Studio or to Claude." (one row each), then "Turn on pictures", a closed fold, "What is sent, in full", and "Try
the public test scene" (on the label sheet's pane and on a look-first list's pane alike). The fold, in the text's
colour, holds the two hosts by name, what each gets (the box and the months once for all the squares) and "No key and
no account are used.", then the test scene's question, which calls them the same catalogue and the same file store; so
the test scene's button comes after the fold that says what it sends. The answer is kept in the output root
(`.imagery_on.json`) with what it was given to (the question's words, its face and its fold, and whether the build
fetches), so a changed question asks again; a yes given to these words counts.

**The fetch.** It has its own HTTP client: no credentials, redirects never followed, no proxy or `.netrc` from the
environment, every URL checked to be https on exactly one of the two hosts; GDAL opens no socket (rasterio reads each
band through a Python opener, and the server fetches every byte); one call has a 40-second deadline for its search and
one for its reads, a byte budget and a request cap; two calls run at a time, and pictures switched off while a call
waits stop it before it sends anything. Errors are fixed sentences, with no coordinate, path, host or scene id. The
panel waits up to 200 seconds for a picture call (the server can take 160). A failure another try can get past (a slow
fetch, the catalogue, the image files: the answer's `again`) shows its sentence with a "Try again" button in the pane,
which repeats the call for that square; a call that gets no answer back, or fails with no sentence of the server's own,
shows "The pictures did not arrive. Press Try again." and never the error's own text. When no month of the map's period
has a scene with data on the square (the answer's `no_image`), the pane says "No satellite image was found for this
square in the map's period." in place of a list of empty months, with " Press Cannot tell." where the sheet takes
Cannot tell; a search the catalogue cut short is told by the record alone. It needs the `inferencex` extra (numpy,
rasterio). The switch and the pictures are app-only tools: in a host with panels, the consent rests on the host keeping
app-only tools from the model.

**Where the square is.** It comes from the files on disk, never from the page: the sheet, its design
(`population.source.scores_path`), the scores file (`grid`, `patch_px`, `raster.path`, `date_window`) and the map
raster's header; the window's centre is checked against the sheet's row before anything is fetched. A plan drawn from a
Studio result without a scores file has no raster to place the window, so it gets no pictures ("Street map" stays).

**The dates and the views.** For each calendar month of the map's period (at most 12; a period ending on the first day
of a month does not add that month), the scene with the least cloud and shadow on the outlined square by the scene's
own cloud mask (its Scene Classification Layer, classes 3, 8, 9 and 10), whose data covers the picture and at least
90% of the outlined square; up to six masks read and up to three scenes' bands tried, least scene cloud first; a date
without a mask is ranked by scene cloud; the same rule for every window, blind to its map class, stratum and order.
Dates the mask marks cloudy are folded away in the pane, not dropped. Per date: near-infrared false colour (B08, B04,
B03, reflectance 0 to 0.30) and true colour (B04, B03, B02,
0 to 0.20), one fixed stretch, nearest neighbour onto the map's own grid; a context view (7 squares a side) with the
square outlined over the picture ("Context, square outlined"), and a detail view (3 squares a side). The outline is 2
pixels of yellow with a 1-pixel black line outside and inside it, so it shows on cloud, sand or snow; the frame sits one
pixel further out, so no line covers the square. Magenta is no data. On the label sheet's pane each date's block is
headed by its day alone ("10 Apr 2024"); its scene cloud, which is the whole satellite image's and not the square's, is
in the pane's own closed fold "Scene cloud of each date", with each date's label whole (a review set's pane keeps the
label as its heading).

**The rule beside the pictures.** One the user typed on the Check card before the squares were picked comes first, on
any map, with "Typed by you before the squares were picked, <day>". Else, for a map of presence and absence, the app's
labelling rule ("Present: on at least one clear date shown, what the map is meant to find covers more than half of the
outlined square. Absent: on every clear date shown, it covers half or less."), with its scope, "This rule came with the
app, for any map of present and absent." (a sheet made with an earlier wording keeps that rule and its scope: [The rule
on the sheet](#the-rule-on-the-sheet)); another map gets no rule, and the sheet's face says so where the rule would
stand, with what to do ("No rule was set for this map. Decide your rule before the first answer and use it for every
square.", or on a two-map sheet "No rule was set for these two maps. ..."). The sheet's rule block shows the rule (its
scope is in the sheet's Details fold); the pane shows the rule's two halves (and, on a face that takes Cannot tell, the
cloud line) only when the page copy has no rule, as both come from `imagery.rule_for`. The pane also says what a label
then is: a visual reading of the map's own sensor, so agreement with that reading "at this square size", not ground
truth; an estimate, certified zone or certificate graded from a sheet whose pictures were shown says so in its
`must_state`.

**The record and the test scene.** A record beside the sheet (`<sheet>.imagery.jsonl`, JSON lines) keeps what was
shown: the rule, the stretch and the date rule; the search (once per sheet, reused for every window); and for each
window shown, the scene chosen for each month (id, date, scene cloud, links). The pictures themselves are not kept. The
question card also offers a public test scene (January to March 2024), unrelated to any project: it sends that scene's
rounded extent, nothing of the sheet, and leaves pictures off for the sheet. Its button is on the question's face,
right after the question's fold, which says what it sends. Beside a sheet its caption says it is not a square of the
sheet and "The outlined square is 11 pixels."

### The result card

A result with a figure or a sheet (`olmoearth_plan_label_sample`, `olmoearth_estimate_map_error`,
`olmoearth_certify_zone`, `olmoearth_review_set`, `olmoearth_review_set_from_result`) shows one card. A plan's card is
[the plan card](#the-plan-card), a review set's [look-first](#look-first).

**The tool's rows.** A card of at most six rows the tool writes from its own result (its `card`), then one "Result
used" row for each prediction id read as its one result ([a prediction id where a result id is
asked](#2-olmoearth-predict)): an estimate's error rate, its interval valid at every sample size and the one its design
earns, each named as the result's sentence names it, with its method as the detail ("Interval this design earns"
carries its level there: "nominal 95%, approximate" on the confidence and proportional designs, whose interval the
package calls nominal, unless a window is not judged), the labels and the design; a which-map grade's verdict, what
decided it and the difference in accuracy; a zone's coverage and its test, never the whole-map rate; a plan's windows
to label, design, population and seed; a review set's windows to look at first and what ranks them (a `boundary_first`
set: windows touching a class boundary first, then margin). On a plan's and a review set's card, and on an error
rate's card (below), the rows are in the closed "Details" fold; on another card (a which-map grade, a certified zone)
they stand on top while the card opens with no lines of its own. Its
numbers are written as the result's `must_state` writes them, and the panel computes none. A refusal the tool returns
(no zone on a stratified design, say) shows its reason instead. Then the determination figure and its caption, the
result's `must_state`, and for a plan, a button to label its sheet in the panel. The figure SVG and the card go to the
panel only (the result's `_meta`), not into the text Claude reads; the agent's own loop, outside Claude, still reads
the card with the result.

#### An estimate's card

**The headline.** For an estimate of a sheet: "Result: West part of Q2 map", the period and how many squares were
checked (", 2 Cannot tell" when some are); one headline that carries the rate in words, the exact rate and its 95%
range ("The West part is wrong on about 1 square in 8: 12.5%, 95% range 6.1% to 21.7%."; its subject the part when the
part is known, so the sentence cannot be passed on as the whole map's, else "The map", with the hedge under it), over a
line that draws the rate as a dot and its range as a pale band (the interval valid at every sample size), with the
band's two ends as numbers under it ("6.1%" and "21.7%", the server's `low_text` and `high_text`; one centred "6.1% to
21.7%" where the band is too narrow for two; the 0% or 100% label left out where an end's number would touch it).

**The rate in words.** Up to 0.5, the nearer of "about 1 in N" and "about N in 10" (41% reads "about 4 squares in 10";
a tie keeps "1 in N" (50%, 20%), and 25% stays "1 in 4", the nearer), then "about N in 10" below 0.95, "more than 9
squares in 10" from 0.95 up to 1, and "was wrong on every square you checked" at exactly 1 (as "right on every square"
at 0, where the headline writes the rate "0.0% wrong"). Below 0.95 the words are the nearer of these phrases that lies
inside the 95% range as printed; when none does, the headline gives the rate plainly ("Error rate: 44.0%, 95% range
41.9% to 46.1%."). Only the words change with the phrase: the exact rate and the range are the estimate's.

**Under the range.** At the size of the other text, the headline block's last line: "Checked against satellite pictures
(or your answers), not a site visit; no verdict on whether the map is good enough." Directly under that block, what
the check did not cover: one class too few squares had ('Only 3 of the 80 squares had "present"; the map agreed on 0
of them. Too few to say how well the map finds "present".'); two or more, or any with Cannot tell answers, by name only
in one line ('Too few squares to say how well the map finds: "A", "B".'), with each one's counted line first under
Details (none with Cannot tell: a Cannot tell square may hold the class); then the classes no square had ('You did not
see "A", "B" in any square, so this check says nothing about how well the map finds them. The rate above is for every
square of the map, all classes together.'; "You did not see 6 of the map's 8 classes in any square, ..." for more than
five, with their names under Details; Details keeps "Classes you saw in no square", and Claude's line its own words;
"every square of this part" on a map in parts, "every point of the map's grid" for a sheet of points, "every square of
the map they were picked from" beside the hedge when the part is not known). The line on the classes no square had
shows for a random design only, a simple random sample of the whole map or part, so its rate is that whole's.

**A second check.** Under those lines, when the server names the Studio result (and, on a map in several parts, the
part's file), one outlined button, "Check each class the map draws", sends Claude a request for a new check of that
result and part picked class by class (`olmoearth_plan_label_sample` on the `scores_path` that `olmoearth_check_map`
gives for that part, since the planner has no part argument and by result id samples the whole result; points by
result id), its own check and ranges, never pooled with this one, asking first how many squares (or points); Details
say what that check is.

**The tally.** Then the squares checked as a tally, one row per kind of square that occurs, at every number of
squares: each row is the kind's own square, marked by the class seen (the first class filled dark, the second a hollow
dark square, the third a mid teal with a dark edge, never mint, which the look keeps for a completed status) and
outlined in red where the map said otherwise, then "× N" (the squares of that kind, with thousands commas: "× 2,400"),
then its words ('You saw "absent"; the map agreed', 'You saw "absent"; the map said otherwise', "Cannot tell"), in the
order the squares come (each class seen where the map agreed, the same class where it said otherwise, Cannot tell
last); a kind with no square has no row, the counts add up to the squares checked, and the rows where the map said
otherwise are the only red of the figure. Every row is one height and the counts one column, in the figures' small type
(as the parts' labels); there is no legend, since each row says what it is, and the figure has one name for a screen
reader, each row's words with its count.

**Parts.** "One result per part, never averaged" with the parts figure (each part's words, "West part: this result",
inside its own box, every box in the row one height; a part's width follows its share from 561 px up, never narrower
than its label on one line, and the parts that do not fit one row go on to the next, never wider than their shares
there, so a short last row stays short, at the left; on a panel of 560 px or less the parts are stacked, each full
width), and while a part is unchecked: "Check the other part too. Ask Claude to open the Check card again." ("Check the
other 2 parts too. Ask Claude to open the Check card again."; Claude's own lines do not have that sentence).

**Details.** In this order: the rule used, printed with "(typed before the squares were picked)" or "(the rule that
came with the app)" (the earlier kelp rule, on a sheet made with it, keeps "(the app owner's rule, 11 Nov 2026)"), no
day (Claude gets the app's rule with its scope and nothing more, and no day for a typed rule); a count line per class
seen in 5 squares or more, its name in straight quotes as the sheet gives it ('Where you saw "Present" (74 squares),
the map also said "Present" on 70; agreement range: ...', from what the reader saw; an all-digit name 'class "3"'), so
the face holds one range; on a map of two parts or more, "Report the parts' results side by side; do not average or
add them: this check gives no range for a combined rate." (Claude's lines and the Check card's say the same reason);
the rare classes' counted lines, and on a map of more than two classes "This check says nothing about the area of any
one class." (Claude's line keeps "It says nothing about the area of any one class."); the unseen classes' names; what
the second check is; the tool's rows, with the range's method; the figure and where the sheet file is; the notes.

**With Cannot tell.** The rate is two figures, the rate with every Cannot tell counted right and then with every one
counted wrong (the package's own pair), said in the same everyday form with which is which: "The West part is wrong on
10.0% of squares, or 12.5% if the 2 Cannot tell were wrong; 95% range 6.1% to 21.7%." ("if the Cannot tell was wrong"
for one; in this sentence "to" is the range's alone; "between" stays for a rate of two figures with no Cannot tell),
drawn as a short dark bar over the pale band, the bar labelled with the headline's two figures and its own word, "10.0%
or 12.5%" ("between 10.0% and 12.5%" with no Cannot tell, a label long enough that near either end of the line it runs
inward from that end, so the figure's edge never cuts it), and the figure named "10.0% or 12.5%; 95% range 6.1% to
21.7%" for a screen reader (Claude's line and the Details keep their words; no number changes). The squares stay: a
Cannot tell square is an empty dashed square, its row ("Cannot tell") the last of the tally, and no class has a count
line; one line takes their place ("No split by class: 2 squares were Cannot tell, and each could be either."). Claude's
lines add a fourth with the classes seen too rarely or not at all (with Cannot tell, "You saw X in no square." for
those no square had); the third ends with the area sentence on a map of more than two classes, and on a map of two
parts or more whose parts are all graded the second still says to report the parts side by side. A graded part whose
sheet was cleared since counts as not checked yet. A part that was not read as a map (no scores path, or a scores file
that gives no count of its squares; its Check card says "This part cannot be checked") is drawn in the parts figure
as "East part: cannot be checked" and is no part left to check: the card says nothing about checking it, and Claude
is not told to check it.

**Which card.** The squares and the class lines show only for a random design (every Check card's), where the share of
red squares is the rate's own estimate; a stratified design picks squares with unequal chances, so its card leaves
them out. When it is not known which part of its map the sheet is (no Check card drew it), the card is titled "Result:
the squares checked" and Claude adds that it may be one part's rate; a plan drawn on a grid over a whole Studio result
is the whole map ("Result: this map", no such line). An estimate's labels are read as the estimate reads them: `wrong`
given is inline; `reference` alone is inline for a class map (with or without top-k scores) or two maps, while a scores
map's estimate still grades the sheet's `wrong` column, so that sheet is graded and marked. Inline labels get no
squares, rule, parts, sheet name or note that the labels came off pictures. A sheet with squares left unanswered gets
"Not finished yet: k of n squares on this sheet have no answer. Answer the rest on the label sheet, then press Estimate
the error rate there." instead of an error (the card has no button of its own). The button it names is the sheet's own
finish button, as the sheet draws it: "Certify the zone" on a certificate plan's sheet, "Say which map is more
accurate" on a two-map sheet, "Ask Claude to grade it" on a sheet with no design file beside it. From 1,000 the card
writes its counts with commas, as the headline does ("1,200 of 4,000 squares", "4,000 squares checked, 1,200 Cannot
tell").

#### Graded and withdrawn results

**Graded sheets.** An estimate marks its sheet graded (`.<sheet>.graded`, the version graded), only when the sheet did
not change while the estimate read it and only for an estimate of the sheet itself. The mark is tied to the design's
content (a digest of its `to_label`), so the plan tool writing a lost design file again keeps it, and a sheet that
holds no answer has no mark: a fresh sheet planned again under the same name starts unmarked, and its first save
removes the old mark. A save that empties a graded sheet (a column cleared, or every answer cleared) keeps the mark,
marked changed, so answers given again after the result was seen grade as changed. Whether the answers changed is told
by a digest of the answer cells, not the file's bytes, so a spreadsheet's re-save with no answer changed is no change.
Reopened, the sheet says changing an answer now makes the result invalid; an estimate after the answers changed is
withdrawn on the card and to Claude, also when a save landed while the estimate read the sheet (no new mark then). A
scores map's sheet graded beside an inline `reference` that differs from the sheet's own answers draws no squares and
no uncovered lines.

**A withdrawn result.** A result whose answers changed after its first result is withdrawn: its card's title says "This
result is out of date", under it what was checked (the part and the map, "West part of Q2 map · 1 Apr 2024 to 30 Jun
2024 · 80 squares checked"), then one line, "The answers on this sheet changed after its first result, so this result
does not count. A new result needs another person and a fresh sheet; ask Claude for one.", and nothing else on its
face: no rate, no range, no figure, tally, parts or button. The line names no button: the sheet's graded mark stays
changed, so pressing its finish button again only gives this card again (the sheet itself says its answers are
locked). Its closed "Details" hold the rate and range as "Out of date, do not use (computed ... from the changed
answers): " and the headline whole, with the time the grading that computed them ran (never the first result's date;
no time when that grading wrote no mark), then "Answers on this sheet changed after its first result (...). Answers
changed after seeing where the map disagrees make this result invalid." with the first result's date. Claude says
first that the result (the part's, on a map of two parts) is out of date and withdrawn and that a new result needs
another person and a fresh sheet, the same with panels or without, and is told to state no `must_state` item and no
rate or range from it, and, only if the user asks for a new result, to call `olmoearth_plan_label_sample` with the
same map, design and count as the plan that drew the sheet and another seed, for another person to label, for a zone
the Check card drew too, since the same seed and count open the locked sheet again; no number is recomputed or
changed. The run's record of the estimate is marked `withdrawn` too (`server._record_held`, on a copy: the result
Claude and the card read is unchanged), so the stakeholder report (`olmoearth_case_narrative`) states no rate or range
from it either, nor from a zone or certificate on its design file; an estimate's record is made only once its card
knows whether it is withdrawn (`server._estimate_recorded`), so a report made meanwhile finds no rate rather than this
one.

### The plan card

A plan's card (`olmoearth_plan_label_sample`) holds the rows, the figure and the notes of [the result
card](#the-result-card) in its closed "Details" fold, and a button to label its sheet in the panel.

**Its lines.** Two lines under its overline: "The squares to check were picked at random and are now fixed." and
"Answer every one on the sheet to get the result." ("points" for a sheet of points). Once the card reads that every
square (or point) is answered, while the sheet is not open in it, the second line reads 'Every one is answered. Press
"Open the sheet" to get the result.' (the button is named by its label in double quotes, never repeated bare), and a
later read below the whole sheet puts the first words back. A read of the counts that answers only after another card
was drawn in its place changes nothing on the new card. Once the sheet opens in the card, the second line goes with the
button: the sheet's footer then says "Answer every square first." beside its grey finish button, so the card says it
once.

**Its button.** The card's one contained button comes right under those lines and before the closed "Details" fold,
the card's last block; once the sheet opens in the card, Details sits under the sheet. The button remembers its sheet.
Beside it, the panel says how far the sheet is labelled ("2 of 5 labelled", with ", 1 cannot tell" when a window says
so), and the button reads "Open the sheet here" before any label, "Continue labelling" part way, and "Open the sheet"
once every window is labelled. The panel reads these counts from the file when it opens, each time it is shown again,
when it gets the focus and after each save of the sheet opened in it, through `olmoearth_app_sheet_progress`, a
panel-only tool that answers counts and the sheet's job, never a row, a place or a class; an older read that answers
after a newer one is dropped, and when a read fails, the button keeps its first words and no count shows. The panel
never takes the keyboard focus when it opens.

**The sheet as a file.** The face shows no file path: the closed "Details" fold ends with "The sheet is also a file on
this computer: <path>. It shows the map's class at each square, so answering there is not blind; use the sheet here
instead." ("each point" for a sheet of points: the plan's view data carries `unit`). A sheet labelled right or wrong
shows the map's class itself, so there the line ends "It shows each square's place, order and stratum, which the sheet
here hides; use the sheet here instead." (the view data's `right_or_wrong`, from the counts' `label_rule`). The path
wraps anywhere, so a long path never makes a narrow panel scroll sideways.

**The plan's figure.** It shades each window by the map's confidence, which on a confidence design is its stratum. So
while its sheet is labelled blind (the class seen: a named scores map, a class map, two maps; the counts' `blind`) and
is not complete, the server holds the figure: the plan's result gives the panel the caption but not the SVG, and a
sentence says so: "The plan's figure shades the map by its confidence, so it shows here once the whole sheet is
labelled." A sheet whose counts the server cannot read is held the same way. A sheet labelled right or wrong, which
shows the map's class anyway, gets the figure at once. Once a read of the counts allows it, the panel asks for the
figure through `olmoearth_app_plan_figure`, a panel-only tool that reads the figure beside the sheet only when the
sheet's counts allow it, and otherwise refuses with the same sentence; a refusal's words show in the figure's place,
and the next read asks again. A read that fails draws no figure. A two-map sheet is blind too, so its figure (where the
maps differ) waits as well.

**What Claude reads.** Where panels show, a plan's result tells Claude that its panel opens the sheet, so the sheet is
not opened twice; `olmoearth_label_sheet` reopens a sheet later. Where panels show, a plan with a sheet also comes to
Claude without its list of windows: its text has no `windows` and no `n_windows_listed`, so no window's index, row,
column, order, map class or stratum, because the host hands that text to the plan's panel too, which labels blind, by
handle. A client without panels keeps both, for the user who fills the CSV. The plan's `how_to_label` is the package's
sentence, passed on as written.

### Look-first

**When it is offered.** "Where is my map likely wrong" on a Studio result does not go straight to
`olmoearth_review_set_from_result`: no recorded experiment supports that order on a Studio result (top-k, a [0, 1]
score or a threshold), and it reads a grid of about 100 single pixels. The instructions rewrite the prompt's review-set
rule: a direct model run's scores file is ranked with `olmoearth_review_set`; on a Studio result Claude says so, offers
`olmoearth_check_map` (not for embeddings) and runs the grid only if the user still asks. The tool's listed description
says the same before any run. When it is run anyway, Claude's text starts with "No recorded experiment supports this
order on a Studio result, and it reads only N single pixels on a GxG grid", before any other note. The start card's
"Find the windows to look at first" and the projects card's "Where to look first" show only while `FIRST_IN_STUDIO` is
true (false by default).

**The card.** Its overline says "Look first". Its first line says how many it lists was chosen in advance and is not a
count of doubtful squares or points: "The 14 squares the model was least sure of, among the 280 read; the number 14 was
chosen in advance, not a count of doubtful squares. No labels were used, so the map may be right or wrong there." A
review set's grid picture, whose own title, subtitle, legend and caption say "windows", "review budget" and "review
set", is the first thing in the closed "Details" fold, with the tool's caption; its one legend line ("Shade is the
order only, most sure to least sure ..., not an amount of doubt") follows it there, nothing of it on the face. A plan's
and a review set's card table and notes are always in the closed fold, and their first rows read "Squares to label",
"Points to label" (a Studio result's grid) and "Squares to look at first". A Studio result's review set tells Claude it
is not a random sample, no error rate follows from it and no point on it is known to be wrong.

**Squares with pictures.** A review set ranked from a scores file that records its map (the map raster's path, the
window size, the period and each window's centre, as `olmoearth_scores_from_file` and `olmoearth_scores_from_studio`
write it) shows each listed window as a card, in rank order: twelve, then "Show 12 more". A card says "Square k of N",
N the review set's whole count (`n_review`), and what the map says there ("Map says: water"). Every card is one size at
every panel width: one height for two caption lines, the rank in a column two digits wide, the cards in columns of at
least 160 px (two on a phone); a caption longer than two lines is cut, its whole words in the card's name and the pane.
At most 50 are listed; when fewer than N are, a line above the cards says "The first 50 of N squares are shown here;
the rest are in the full list." The cards scroll inside a box at most 399 px tall (six rows and half of the seventh, so
a longer list shows half a card, not a row with its edge shaved), and a card's press brings its pane into view.
Pressing it opens the window's pictures in a pane under the cards, as the label sheet shows them (the dates, the colour
switch, the square outlined: "Context, square outlined"), with Previous, Next (the two one width, as on the label
sheet's face), the square's one outlined "Street map" and Close. The pane asks the panel-only
`olmoearth_app_review_image` by an opaque key and the window's rank: the page never gets a path or a place. The server
keeps the review's record (`review_sets/<key>.json` in the output folder: the scores file's real path, size and
modification time, and the listed windows); a press on a scores file changed since, or moved outside the read roots,
is refused. The first press asks the pictures question ([Satellite pictures](#satellite-pictures)), with the public
test scene, which the pane asks as a review set's (`review: true`), so its caption and a failed fetch name no sheet or
label; a failure shows its sentence, with "Try again" where another try can get past it (a call with no answer back:
"The pictures did not arrive. Press Try again."; no image in any month: one line, as on the label sheet). The record of
what was shown is `review_sets/<key>.imagery.jsonl`. Where the panel pictures the windows, Claude's text holds no
listed row and starts with a note to name a window by its place in the panel.

**No pictures.** Inline scores, a file that records no map (the grid of single pixels a Studio result gives), or no
panel: no cards; the rows stay in Claude's text, and the panel says why there are no pictures, by cause: the period is
not recorded, a Studio result's sampled points are not pictured, the file cannot be read here, or the map is not
recorded. The rank, window index, row, column, margin and score sit in a closed fold ("The numbers for these squares"),
each number to three places, the last part of the card's one "Details" fold.

**A Studio result's review set.** `olmoearth_review_set_from_result`, asked in the chat, reads single pixels, so its
block's `unit` is "point": the fold is "The numbers for these points". Under its list come "You label a random sample.
This list is not one." and a "Check how wrong this map is" button, which sends the projects card's request for the same
result, the card's one contained button, before "Details"; its lead then has no second line, since that card says it
with its button (a review set without that card keeps its second line).

**Street map.** The open square's pane has one outlined "Street map" button beside Previous and Next, named for its
square. A Studio result's points whose scores file records each point's centre get a row of Street map buttons, one per
point by its rank, each an outlined button, 1 to 9 as wide as 10 to 50, in place of the numbers' table (and still no
pictures). A press asks the server for that one link by the review's key and the rank (`olmoearth_app_review_link`) and
opens OpenStreetMap in the user's browser; the place never shows in the panel, a toast or what Claude is told. A press
that fails ("No map link for square 2.", "The map was not opened: ...", or "The link was not opened." when the host
does not open it) is said right under the row of buttons it came from, in the open square's pane or under the points'
row, in red and announced at once; the line tells the latest press only (a late reply of an earlier press says
nothing), belongs to its square (closed and opened again, the square shows none), stays in place while the pane
redraws (so its alert is not read out again), and takes no room while empty.

**The list with places.** One outlined button, "Save the list with places (CSV file)", gives the list in rank order
with `rank`, `square` (or `point`), `map_says`, `lon`, `lat` and a `note` on every row ("look-first list, not a random
sample; no error rate comes from it"), never a rate column, as `look_first_not_a_sample_<key>.csv`
(`olmoearth_app_review_csv`): through the host's download where it offers one, else written under `review_sets/` (mode
600) and named under the button. That one line under the button tells the latest press only: a press empties it, and a
press that fails says there, in red, "The list was not saved: ...", announced at once to a screen reader (role alert,
as the status line's errors are), the file's name politely; a download the host does not allow is still said in the
status line.

### The area map

`olmoearth_request_aoi` shows a map (OpenStreetMap tiles, Leaflet.draw). The user draws a rectangle or polygon, or
picks one of the project's saved areas from the list under the map, and presses the button. The card's one
instruction, "Draw a rectangle or polygon with the tools on the map, or pick a saved area." (then what was drawn or
saved), is the first line under the map, at body size, before the Project and name fields and the saved areas; the
sign-in, when it opens, stands above it. A drawn area is created in Studio. Its `area_id`, `project_id` and name go to
the chat as the user's next message; no coordinates do, and the panel shows none. The tool's description, as this
server lists it, says so (the agent's own web UI keeps its words). A Save whose answer from Studio was lost may have
been saved: the server's failure carries `outcome_unknown: true` and "Studio's answer was lost, so the area may be
saved after all. The list of saved areas is read again: look for it there before you save it again."
(`StudioOutcomeUnknown`, [Studio API notes](#studio-api-notes)), which the panel shows alone, never after "Not saved";
a Save that got no answer at all (the host's timeout) says "Studio's answer did not come back, so the area may be saved
after all. The list of saved areas is read again: look for it there before you save it again." Either way the panel
reads the project's saved areas again, so an area that was saved is listed (not chosen), and Claude is told nothing.
Its full screen button works as the label sheet's.
The map tells Claude the project and the area chosen, by id, or that a shape is drawn and not saved; its result carries
no context rule.

### What Claude is told

**The instructions.** The server's instructions are the agent's prompt (`harness/soul.md`) and its capability card,
plus rules for Claude. Claude Code keeps about the first 4,100 characters of them, so the review-set rule is rewritten
inside that part ([Look-first](#look-first); `harness/soul.md` itself is unchanged), and they add: when the user picks
among their own predictions, say what each outputs and whether the job can run on it, never "recommended".

**What Claude says first.** The Check card's and an estimate's text start with three lines Claude says first (keeping
their words for what was checked, squares or points, the part and the range; a withdrawn estimate has one line
instead, that it is out of date and withdrawn and that a new result needs another person and a fresh sheet, naming no
button, and no rate or range), in a client without panels too, where Claude then asks how many squares to check in each
part. The share of each part is `share_of_pixels_with_data`, a size, not a score. It is written from the parts'
pixel counts, unrounded: a part holding all but a sliver is ">99.9%", never "100.0%" beside the sliver's own share,
and a part under 0.0001% is "<0.0001%", never "0.0%" (`statistical_rules.share_percent`); the result's
`studio_zones` sentence writes them the same way.

**The notes.** Each panel's result carries a note for Claude; the sections above say what each says. They are
constants in `mcp_app/server.py`, so a developer finds each by its name:

| Note | Claude reads it with |
|---|---|
| `START_CARD_NOTE`, `START_TEXT_NOTE` | the start card; its text without panels |
| `PROJECTS_PANEL_NOTE` (it starts with `PROJECTS_PRESS_NOTE`) | the projects panel, where panels show |
| `CHECK_CARD_NOTE`, `CHECK_TEXT_NOTE` | the Check card; its text without panels |
| `CHECK_NO_RULE_NOTE`, `CHECK_OWNER_RULE_NOTE` | a map with no rule; a map of presence and absence, with the app's rule and its box |
| `CHECK_PREPARING_NOTE`, `CHECK_PREPARING_TEXT_NOTE` | a Check card still preparing; the same without panels |
| `CHECK_DRAWN_NOTE`, `CHECK_SAME_SHEET_NOTE` | a drawn zone; the same settings open the same sheet |
| `CHECK_CHANGED_NOTE`, `CHECK_FINAL_NOTE` | a zone whose files changed; a zone whose draw failed for good |
| `CHECK_CLASS_MAP_NOTE` | a class map without top-k scores |
| `SHARE_SIZE_NOTE` | a part's share of the map |
| `PLAN_SHEET_NOTE` | a plan with a sheet, where panels show |
| `REVIEW_PANEL_NOTE` | a review set whose windows the panel pictures |
| `SAY_FIRST_NOTE`, `SAY_WITHDRAWN_NOTE`, `GRADED_AGAIN_NOTE` | the lines said first; a withdrawn estimate |
| `CONTEXT_RULE` (its second half is `NO_GRADE_RULE`) | every panel that tells how far a sheet is labelled |

**What the panels tell Claude.** Unless the host says it takes no context updates (the MCP Apps
`ui/update-model-context`; the stable spec defines the method but no capability flag, so only `updateModelContext:
false` stops them), a panel tells Claude silently what it shows, so that "this one" or "how far am I" needs no copying:
the label editor its sheet's name, `labels_csv_path` and `design_path`, how many windows are labelled (and how many say
Cannot tell), whether every change is saved, and once every window is labelled the words of its finish button; the
plan card its sheet's counts and button; the Check card the result's id and, one clause per zone, a drawn zone's counts
first, then its count, design and seed, its sheet's file name and its `labels_csv_path` (or that it is open in this
card), or that a zone is not drawn, being drawn, failed, offers no count or had no reply to its press (and, once a card
that opened preparing filled in, after that, the server's own text a call of `olmoearth_check_map` in time gives,
below); the projects panel how many projects it lists, the project open (by id, with its counts of areas, predictions
and models), the predictions picked to compare and the label sheets in progress (each by its whole file name, its
counts and its `labels_csv_path`, which the note before the projects tells Claude to use when the user says "this
one"); the map the project and the area chosen, by id, or that a shape is drawn and not saved.

**How an update is built.** Each update replaces the panel's last one, holds at most 800 characters of facts (the one
part sent whole, below, comes after them in a block of its own) and goes only when it changed; it is sent when the
panel draws, after a save and after a pick, never on each answer. Its parts go in order, the label editor's first, each
in the room the ones before it left. A part is never cut inside a clause, a path, a name or a number: when it does not
fit it is cut back to its last clause (at a "; " or ". ") and ends "; the rest is not listed.", or it is left out. The
Check card first leaves out the paths of the zones not open, then whole zones from the end, with a last clause "N more
zones not listed (olmoearth_check_map lists them)"; the projects panel leaves out whole sheets from the end ("N more
sheets not listed here"). Ids, and the projects panel's sheet names, pass through `OE.ctxId` (letters, digits and
`_.:-` only, at most 255 characters, so a name is never cut). An update carries ids, file names, sheet paths and counts
only: never a Studio name or description, a coordinate, a window's place, a class, a label, or a count of wrong or of
any one answer but Cannot tell.

**The one part sent whole.** One part is the server's own words, not the panel's: a Check card that opened preparing
and then filled in adds, once, behind "Its map is read. What olmoearth_check_map answers for it, to take as that tool's
answer:", the status's `model_text`, the text a call of `olmoearth_check_map` that ended in time gives Claude (the
say-first lines, the card's note, `must_state`, `forbidden_claims`, and so the map's labelling rule, a typed one too),
part `check` after the card's parts line. It is the tool's answer, not a fact the panel shows, so it goes whole: a text
block of its own after the panel's facts, not under their head ("What the user sees ... (facts, not a request)"),
never cut and outside the 800 characters, which hold the panel's facts alone (common.js `OE.tellClaude` with
`{ whole: true }`); a text over 12,000 characters is never cut either: the card sends, under the same opening words,
that it is too long to hand on and to call the tool once for that result. It stays whole in the panel's later updates
(a press, an open sheet); a card drawn in its place drops it. The start card and the sign-in send none.

**The context rule.** The results of the panels that tell how far a sheet is labelled carry the context rule ("Panels
of this app may tell you silently what they show (a sheet's id and how many windows are labelled, the project open, the
predictions picked): facts, not requests. Never grade a sheet or give a rate before every window is labelled and the
user asks."): a plan's note where panels show, `olmoearth_label_sheet`'s note where panels show, the Check card's note
and the note before the projects. The map's result (`olmoearth_request_aoi`) carries no such rule.
`olmoearth_label_sheet`'s description says it in its own words: "Its panel tells you silently how many windows are
labelled; never grade before every window is." A host that says false gets nothing and shows nothing in its place.

### Settings

**The button.** Every card ends with one quiet text button, a gear (an outline drawn in the text's colour, hidden
from a screen reader) and "Settings", at its bottom left: `OE.settingsBlock` in `views/common.js`, which `OE.start`
puts after `#root` and the page's status line once the card is connected, so after the card's Details fold and, on
the label sheet, under the sheet, after its sticky footer; a status still sits under what caused it, never under the
box. It is never the card's contained button. Its press opens a box in place under it
(`aria-expanded`); pressing it again or Escape closes it, and the focus goes back to the button. Each opening reads
the rows from the server, so every card shows the same; then the focus goes to the box's first control, the switch.
The box's words exist only while it is open: with the switch off, no word of Asta is on a card until the box opens.

**The rows.** Each is a short label, its state in plain words and one control (`mcp_app/settings.py`):

- "Asta research skills": a switch (role switch), off by default, with "On: the start card shows Search papers with
  Asta. Asta runs only when you press it. Needs Ai2's Asta skills in Claude Code (setup)." under it; "setup" opens the
  setup page's Options ([mcp-app-setup.md](../docs/mcp-app-setup.md#options)) through the host.
- "Satellite pictures": "You allowed them on 18 Nov 2026." (the day the yes was kept, this computer's date) or "Not
  allowed yet."; while allowed, "Forget my yes" removes the stored yes (`imagery.set_on(False)`), so the next square
  asks again; the question's version is unchanged.
- "Maps kept on this computer": "2 maps kept for 24 h, so they open without downloading again." (the copies
  downloaded less than `OLMOEARTH_STUDIO_KEEP_HOURS` ago, 24 by default) or "None kept."; reading the row first
  removes the copies past that limit (every readable copy at 0), as the next download would
  (`studio_scores._prune_kept`), so no expired copy stays behind "None kept."; while some are kept, the first press of
  "Remove kept copies" only changes the button to "Press again to remove" for six seconds, and a second press in that
  time removes every kept copy whose record names its folder, an expired one too (`studio_scores.forget_kept` for
  each); Escape on the button, the focus moved away or the six seconds passing puts the words back and removes
  nothing.

**The tools.** Two app-only tools, listed to a client with panels with `visibility: ["app"]` and quiet (the model
reads `{"ok": true}` from them on every path, a failure too): `olmoearth_app_settings` (no argument) answers the
three rows, `{"asta": {"on": bool}, "pictures": {"allowed": bool, "said": text}, "kept": {"count": n, "said":
text}}`; `olmoearth_app_setting` takes one change, `{"name": "asta", "on": bool}`, `{"name": "forget_pictures"}` or
`{"name": "remove_kept"}`, and answers the rows after it; another name, or a switch given no true or false, is
refused in plain words ("Not changed: ..." in the box) and changes nothing.

**The file.** The switch is kept in `settings.json` in the output root, mode 600, written by an atomic rename:
`{"format": 1, "asta": bool}`, no key, no id and no place. A missing, unreadable or broken file reads as the
defaults (the switch off).

**The Asta tile.** With the switch on, `olmoearth_start`'s view data carries `asta: true` (the key is left out while
it is off), and the start card shows one more tile, last under "Other jobs": "Search papers with Asta" ("Ai2's
research skills find and summarise papers. Claude asks what to search first."), on signed in or signed out (it needs
no Studio). Its press is the only thing in the app that asks for Asta: it sends Claude one message, to use Ai2's Asta
skills (asta-tools) to search the scientific literature, to ask first what to search for, to use the quick
literature search, to say what a long or multi-step run will do and roughly how long it takes and wait for a yes, and,
when the skills are not available, to say so and point to the setup page. Every start card follows the switch: it
reads `olmoearth_app_settings` when it opens (a card reopened from the history drew its tool's old view data) and
again when its panel is shown or takes the focus, and the tile's press reads it first (off, the tile goes and nothing
is sent; a read that fails sends as before). Only the tile, and "Other jobs" when no other job is under it, is added
or removed in place: the other tiles, a press under way and a message to copy stay. Signed out with the tile shown,
the card's line says "Sign in first for the Studio jobs. Press a job or ask Claude in the chat." and the line over
the tiles "Sign in to start a Studio job.", since the tile needs no sign-in. No server instruction, tool description
or text for the model mentions Asta, and the tile is not one of the server's MCP prompts.

### Files and privacy

**Where it works.** The server works in, and the tools write to, `~/olmoearth-agent-outputs` (made mode 700) unless
`OLMOEARTH_OUTPUT_ROOT` is set. The tools read model-chosen files only from there, or from `OLMOEARTH_SCORES_ROOT`
when it is set; put a model run's folder under one of them. Results over 60,000 bytes are spilled to a file there
unless `OLMOEARTH_TOOL_RESULT_SPILL_BYTES` is set.

**What the app keeps on this computer.** Each is described in its section above:

| File | What it holds |
|---|---|
| `~/.config/olmoearth-agent/studio.env` (or `OLMOEARTH_KEY_FILE`) | the Studio key, after a sign-in with "remember" (mode 600) |
| `<scores root>/studio_downloads/<result id>/files.zip`, `kept.json` | a Studio result's zip, kept 24 hours (`OLMOEARTH_STUDIO_KEEP_HOURS`) |
| `<scores root>/studio_results/` (mode 700); the scores and design `.json` files the tools write (mode 600) | each zone's decoded map; the windows, with their places |
| `checks/` in the output root, with its `log.jsonl` | each Check card's draws, a file lock, one line per zone of each press |
| `checks/planned/<sha256>.json` | the counts planned for a part (mode 600); one older than 30 days is removed when another is written |
| `.<sheet>.handles` beside a sheet | the secret behind the sheet's handles (mode 0600) |
| `.<sheet>.rule` | a rule typed on the Check card |
| `.<sheet>.graded` | the version graded, and whether the answers changed since |
| `<sheet>.imagery.jsonl` | what the sheet's pictures showed, never the pictures |
| `.imagery_on.json` in the output root | the answer to the pictures question, with the words it answered |
| `settings.json` in the output root | the Settings box's switch for Ai2's Asta skills (mode 600) |
| `review_sets/<key>.json`, `review_sets/<key>.imagery.jsonl` | a look-first list's record and what its pictures showed |
| `review_sets/look_first_not_a_sample_<key>.csv` | the list with places, when the host offers no download (mode 600) |

**What never leaves.** The Studio key goes from the sign-in page to this server and never travels in an MCP message;
it is in no result, error or log. No page shows a window's place. The label sheet's page gets no window index, row,
column, place, order or stratum: it names windows by handle. A look-first list's page gets each window's rank, index,
row and column, and asks the server for pictures and map links by an opaque key and the rank. Its list with places is
either handed by the page straight to the host's download, without being shown, or written by the server to a file.
The server reads every place from the files on disk. The panels show no coordinate, and the area map sends none to the
chat; a Street map press opens OpenStreetMap from the server's answer and shows the place nowhere. What a panel tells
Claude silently is ids, file names, sheet paths, counts and a few fixed words (a design, a finish button's words),
never a coordinate, a class or a label. Once, a Check card that filled in also hands on the server's own text for it
([the one part sent whole](#what-claude-is-told)). A button's press sends ids, and Studio names quoted as data, as
your message, never a coordinate. The satellite pictures send the map's rounded extent and months, and byte-range
requests for the scenes, to Earth Search, and nothing to Studio or Claude.

**Where the panels load from.** The pages load their scripts only from pinned versions, each checked by its SRI hash:

- the MCP Apps client, `@modelcontextprotocol/ext-apps` 2.0.4, from unpkg;
- the map's Leaflet 1.7.1 and Leaflet.draw 1.0.4, from jsDelivr.

Fonts come from Google Fonts. Each view declares just those origins (and OpenStreetMap's tile server for the map) in
its CSP; the host blocks the rest. A page never calls Studio; it calls the server's tools.

### Limits

- **What a local server cannot do.** Claude's "app tools" (tools a view registers for Claude to call), entry points
  and embedded websites need a remote server, as the Claude docs say; this server uses none of them.
- **Tried live in Claude Desktop:** the server started from the config above, the sign-in on the real Studio, and the
  project list. Not yet tried live there: the other panels (start card, Check card, label sheet, result card, map) and
  the projects card's "In progress". Whether Claude Desktop honours the context updates, and whether it shows the MCP
  prompts, has not been tried. The repository's tests run the server's handlers, not a host; the panels were also run
  under a stand-in host in headless Chromium, by a script kept outside the repository.
- **The projects note's order.** Whether a model follows the note's press-first answer or the rule kept after it (say
  what each prediction outputs) has not been measured.
- **More than two lines on the Check card.** Beside its figure, a score map's face holds its cut line, the rule's
  question with its example under the box, the count's question with its one line on what more squares give, the line
  over the press and, while the press is off or leaves a part out, the line that says why. A class map with no rule
  that came with the app holds the same less the cut line, and, when it is in parts, the strips picture with its line
  on the map's parts (or "This map is saved in N parts", with no picture, where the parts are not one per strip). A map
  of presence and absence holds the rule with its scope, the box's label with the line under it, the count's question
  with its line, the line over the press and the line that says why. Which of these goes is not decided.
- **The pictures' consent.** In a host with panels it rests on the host keeping app-only tools from the model.

## Web UI

`make web` serves `webui/` as a static demo: sample projects, nothing sent. Opened as a file, or served by anything but
the bridge, the page stays a demo. `make bridge` (`olmoearth-agent-serve`; `--host 127.0.0.1`, `--port 8088` by
default) serves it as the live agent: a multi-turn chat with saved history, your Studio projects as a tree (project,
model, predictions, results), and briefs streamed through `LeadAgent.run_stream` as Server-Sent Events. It answers
only requests addressed to 127.0.0.1, localhost or [::1] (any other `Host` gets HTTP 403, so a page that rebinds its
own name to this machine cannot drive it; `OLMOEARTH_SERVE_ALLOWED_HOSTS` adds names, each read as a `Host` header
carries it: lower case, without a port, a bare IPv6 address in brackets). Where the web UI is not found
(an install from a wheel ships no `webui/`), `olmoearth-agent-serve` says where it looked and exits 2.

- **Keys.** The Studio key ("Add API key"), each LLM provider's key and the chat history stay in the browser's
  localStorage. The Studio key goes with each request in `X-Olmoearth-Key` (or `Authorization: Bearer`) and a
  provider's key in `X-LLM-Key`; the bridge forwards them. It reads no Studio or provider key from the environment,
  only the local model's `LLM_*` settings.
- **Agent settings** (the status chip's menu, foot of the sidebar). Max steps per run: 4, 6, 8, 10 or 12 (8 by default;
  the server accepts 1 to 12). Model backend: Local, Claude, Gemini or ChatGPT, each with its key and a detected model.
  Auto-route: a short read-only lookup runs on the local model instead of the hosted one (`X-LLM-Route: auto`,
  `llm/router.py`); it only ever moves a run to the local model, and only when the local model answers (else the run
  stays on the hosted one).
- **In the chat.** `/` opens a menu of the 5 skills and pins the run to one. Text files and PDFs can be attached (the
  local model cannot read images). The area tool shows a map to draw on. Saved results (comparisons, the report as
  Markdown) stay in the sidebar. The workflow rail shows once a run calls a pipeline tool, with the stages reached
  (`harness/workflow.py`, mirrored in `js/run.js`). A project the agent creates, or a prediction it submits, drops the
  tree's cached reads, the bridge's and the page's, so the tree shows it when opened next.
- **The difference scan** (two results compared on a grid of pixels): a cell Studio did not answer for is counted
  apart from no-data, said in the status line ("N cells could not be read from Studio") and kept as `n_failed` in the
  saved comparison; when no cell could be read it says "Studio did not answer for N of M cells; try again later". A
  bridge error (HTTP 5xx) is tried once more only, since the bridge's Studio client has already retried it. A key
  Studio refused is no outage: the bridge passes on its 401 or 403, the cell is not asked again and is counted apart
  (`n_refused`, "Studio refused the key for N cells"), and with no cell read the line says "Studio refused the key for
  N of M cells; check the Studio key."

| Endpoint | What |
|---|---|
| `GET /api/health` | liveness, the LLM and Studio wired, whether the local LLM answers; flips the page to live |
| `GET /api/skills`, `GET /api/llm/models` | each skill and the pipeline stages its tools touch; a hosted backend's models |
| `GET /api/projects`, `GET /api/projects/{id}/predictions`, `GET /api/predictions/{id}/results` | the project tree |
| `GET /api/results/{id}/extent`, `GET /api/tile/{z}/{x}/{y}` | a result's bounding box and tiles (a cached proxy, Studio's host only) |
| `GET /api/pixel-value` | one pixel of a result (the difference scan); a refused key keeps its 401 or 403, other Studio failures are a 502 |
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
- The creates (`POST /areas`, `/predictions`, `/projects`) are sent once, never retried. One whose answer is lost after
  it went out (a read or write timeout, a dropped connection, a gateway's 502 or 504) raises `StudioOutcomeUnknown`:
  Studio may have made it, so its hint says to search the project's areas, predictions or projects by name before any
  new write, and its sentence says the same. A failure to connect sent nothing and stays a plain error. The registry
  gives that hint on every surface (`registry.own_hint`; the Claude Code bridge's cache keeps it), the MCP App with
  `outcome_unknown: true` on the tool's answer; the web UI's area save answers HTTP 502 with the error's words and drops
  that project's cached area list, so a reload shows an area that was saved.
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
imperative subject under 72 characters. A PR has a release-note title, its detail in `spec/changes/<version>.md`
(Added, Changed, Deprecated, Removed, Fixed or Security) and, when a user would notice it, a short line in
[CHANGELOG.md](../CHANGELOG.md), needs one maintainer's approval and is squash-merged.
Contributors follow the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

**DCO sign-off.** Sign off every commit (`git commit -s`): `Signed-off-by: Your Name <your.email@example.com>`
certifies the [Developer Certificate of Origin 1.1](https://developercertificate.org/) under the OlmoEarth Artifact
License. The sign-off names a human; AI tools do not certify the DCO. No `Co-Authored-By:` AI trailers; say in the PR
if an AI tool helped substantially, and be able to explain and have tested what you submit.

**Pitfalls.**

- Every tool a catalog row lists is in the default registry; only `OLMOEARTH_READ_ONLY` leaves tools out
  (`tools/registry.opted_in`), and a surface with no map leaves out the draw tool on top
  (`harness/capabilities.without_map`: the command line and the Claude Code bridge). A `download_token` is a
  credential: never in a result, manifest, log or fixture.
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
the live Studio API, marked `integration`; its entry in [Skills](#skills); a line under Added in `spec/changes/<version>.md`. An external
claim carries a real URL; an unverified one says so.

**Vendoring.** The studio-job-config package lives in
[`src/olmoearth_agent/skills/packages/`](../src/olmoearth_agent/skills/packages/), moved from the frozen
[`2imi9/OlmoEarth-Skills`](https://github.com/2imi9/OlmoEarth-Skills), so edits land here. It keeps its MIT licence;
the folder's `README.md` records the changes since the move.
