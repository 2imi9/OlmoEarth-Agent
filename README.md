OlmoEarth Agent (beta)
======================

A beta version developed by a collaborator. It is not the final OlmoEarth Agent.

Drives [OlmoEarth Studio](https://allenai.org/blog/olmoearth) from a plain-language brief, and checks the maps it gets
back.

- It finds a model in Studio, runs predictions and compares their results.
- It checks a map: which windows to review first, how wrong the map is, which part you can trust, and which of two
  maps of one time is more accurate. The statistics come from
  [olmoearth-inferencex](https://github.com/2imi9/olmoearth_inferenceX).
- It checks each answer against the tool results before showing it.
- It runs on a local LLM (llama.cpp) or on your own cloud API key (Claude, ChatGPT or Gemini).


Install
-------

Needs [uv](https://docs.astral.sh/uv/) and an OlmoEarth Studio API key (Studio, your profile, **API Keys**). The local
LLM also needs Docker and an NVIDIA GPU; a cloud API key needs neither.

```bash
git clone https://github.com/2imi9/OlmoEarth-Agent.git && cd OlmoEarth-Agent
make setup      # uv sync --all-extras
```

- On a cloud API key: `make bridge` serves the live web UI on http://localhost:8088. Pick the provider in
  **Settings -> LLM backend** and paste its key.
- On the local LLM: `make up` starts it (the first run downloads the model), then the same web UI.

Open the page, paste your Studio key and send a brief. Both keys go with each request; the server does not store them.

From the terminal, on the local LLM:

```bash
export OLMOEARTH_API_KEY=...
make serve && make agent Q="How many OlmoEarth Studio projects do I have?"
```

- `make help` lists the targets; `make down` stops the local LLM; `make web` serves a demo UI with sample data on
  port 8080.
- A read-only Studio key: set `OLMOEARTH_READ_ONLY=1`. The two tools that write to Studio are then left out.
- Without `make`, run what the targets run ([quick start](docs/Usage.md#quick-start)).


What it answers
---------------

Ask in plain words; the agent picks the tools. The main questions:

- **Which windows do I check first?** `olmoearth_review_set`, or `olmoearth_review_set_from_result` for a Studio
  result, with no labels.
- **How wrong is the map?** `olmoearth_plan_label_sample`, label the windows it draws, then
  `olmoearth_estimate_map_error`.
- **Which part can I trust?** `olmoearth_certify_zone`, on the labels of a random sample.
- **Which of two maps of one time is more accurate?** `olmoearth_plan_label_sample` with `design='disagreement'`,
  label where they differ, then `olmoearth_estimate_map_error`.
- **How do my Studio results differ?** `olmoearth_compare_results`: a pair, a group, a series over dates, or an
  ensemble's disagreement, with no ground truth.

<details>
<summary>Every route, with what it needs and what it gives</summary>

| Route | Needs | Gives |
|---|---|---|
| **Which windows to check first**<br>`olmoearth_review_set`, `olmoearth_review_set_from_result` | A map with scores: a Studio `[0, 1]` score band (another regression band needs a `threshold`), a Studio class map with top-k scores, a direct run's scores, or your own. No labels. | The windows to open first at your review budget, least confident first. Boundary-first is optional. |
| **How wrong one map is**<br>`olmoearth_plan_label_sample`, then `olmoearth_estimate_map_error` / `olmoearth_certify_zone` | A map with scores and labels on a planned sample. A Studio result is sampled on at most 16 x 16 points (256 windows), so its figures cover those points, not every pixel; reading its own files (below) covers every valid window of the files instead. | The error rate: the interval valid at every sample size first, then the design's (approximate under the default confidence design, exact under a random one). Per-class accuracy where labels name the class, each class's valid interval first, its producer's accuracy beside its user's, and whether the map draws it too little or too much. A random design can certify a zone. |
| **A Studio class map without top-k scores**<br>`olmoearth_plan_label_sample` (`design='random'` or `'class'`), then `olmoearth_estimate_map_error` | A classification result (classes, no scores) and labels on a random or class-stratified sample. The plan gives olmoearth-inferencex's label budget for a random sample. | The error rate and per-class accuracy, the interval valid at every sample size first. They cover the sampled grid points (at most 16 x 16, one pixel each), not every pixel. No review set, no certified zone. |
| **A Studio result's own files**<br>`olmoearth_scores_from_studio` | A class map with top-k scores (its prediction submitted with `probability_output`) or a `[0, 1]` score band. | One scores file per UTM zone, pooled from the native-resolution files to windows of `patch` pixels (a window counts when at least half its pixels are valid; pixels past the last whole window are left out and counted as `pixels_outside_window_grid`), for the review set, plan, estimate and certified zone. With no `patch` given, each zone takes the smallest side from 4 up whose windows a ranking takes (at most 200,000). Each zone is its own map, and a certificate on one binds the raster the agent decoded, which Studio's own file is not. |
| **A direct OlmoEarth run and its certificate**<br>`olmoearth_scores_from_file`, `olmoearth_write_certificate`, `olmoearth_verify_certificate` | `oe-inferencex from-olmoearth`'s `scores_<EPSG>.tif` and `olmoearth_output.json`, or `score_area.py`'s raster and `manifest.json`, with probabilities or logits (class ids are refused). A certificate needs a `seed_from_map` random plan; its labels are fixed once first graded. | Every window of the run, for the review set, estimate and two-map route. Its certified zone in a certificate anyone with the map can recheck; its labels, the map's date and single draw are the issuer's word. It adds no guarantee. |
| **Which of two maps is more accurate**<br>`olmoearth_plan_label_sample` (`design='disagreement'`; `date_a` / `date_b` date a scores file), then `olmoearth_estimate_map_error` | Two maps of one area and time: two Studio results, or two scores files of one window grid (not class-map rasters). Labels where they differ; the plan advises how many. Different, overlapping or one-sided dates are refused when given; missing dates are noted. Maps of different times are measured each on its own sample. | Which map is more accurate, or neither (exact sign test; the interval past a fifth of the differing windows), and the accuracy difference with a 95% interval. Studio results cover at most 12 x 12 sampled points, not every pixel. |
| **What labels Studio holds**<br>`olmoearth_studio_label_inventory` | A prediction, model or project id. | The annotations a model's training selector selects now, counted by status, split and class. No error rate: Studio records no random design for them. |

</details>

- The agent registers 25 tools by default, in 5 skills: configure a Studio job, run and compare predictions, record
  provenance, write a stakeholder report and check a map. All 25 are sent on every turn. See
  [Skills](docs/Usage.md#skills) and [Tools](docs/Usage.md#tools).
- The verification tools need the `inferencex` extra (olmoearth-inferencex 1.20.3 or later, Python 3.11 to 3.13);
  `make setup` installs it.


Use from Claude Code
--------------------

A Claude Code session can play the agent's model through the replay bridge, `olmoearth-agent-bridge` (not
`make bridge`, which serves the web UI). Each command is one model call: `begin` starts a run from a brief and `step`
answers the pending call, while the agent's own loop, tools and checks run unchanged. Each response is kept with the
sha256 of the request it answered, so a replay that diverges stops. The project skills `olmoearth-agent-model` and
`olmoearth-agent-bridge` ([.claude/skills](.claude/skills)) say how; details in
[Claude Code as the model](docs/Usage.md#claude-code-as-the-model).


What it checks
--------------

Every tool call goes into a provenance manifest, and tool arguments are checked against each tool's schema. Before an
answer is shown, the harness checks:

- its numbers, against the tool results and your messages;
- its directions, places and magnitudes, against the facts the tools computed;
- its claims of files saved or items listed, against what the run did;
- the claims the tools forbid or require.

One request asks the model to fix every problem found; a sentence that still fails is marked `[unverified: <check>]`.
`OLMOEARTH_CHECK_NUMBERS=0` turns off the number check and `OLMOEARTH_CHECK_ANSWER=0` the rest. With
`OLMOEARTH_CHECK_CLAIMS=1`, the model also reads the answer against the run's record. The stakeholder report states
only the accuracy the verification tools measured in the same request, or says none was. Details:
[answer checks](docs/Usage.md#answer-checks).


Example
-------

`make verify-demo` runs the verification tools on the expert-annotated Dynamic World tile that olmoearth-inferencex
ships (southern Peru, 9 June 2019), with no Studio key and no LLM. It writes
[docs/VERIFICATION_DEMO.md](docs/VERIFICATION_DEMO.md).

- **The map.** 15,813 expert-annotated 40 m windows. Their true error rate is 19.3%, from every window's expert label,
  which a user does not have. It is not Dynamic World's 10 m accuracy.
- **The estimates.** 300 random labels give 19.0%, interval 14.8% to 23.9%. 360 class-design labels give 17.4%, valid
  interval 11.8% to 25.2%. Both cover 19.3%; that is one draw, not evidence of how often such intervals do.
- **The review set.** At a 5% budget it lists 791 windows to check first. Against every window's expert label (the
  script's check), they hold 17.5% of the errors; the attainable ceiling is 26.0%.
- **No certificate.** No zone was certified at alpha 0.1 with 300 labels, so none was written. The certificate round
  trip and a tampered copy are tested on a synthetic tile.


Limits
------

- Without labels it tells you where to look, not how wrong the map is. A review set is not a sample.
- A Studio result is read on a grid of sampled points (at most 16 x 16), so its figures cover those points, not every
  pixel. For a map with scores, `olmoearth_scores_from_studio` reads its own files and covers every valid window.
- Comparing results uses no ground truth: a series estimates movement, not verified change on the ground.
- A certificate adds no guarantee: its labels, the map's date and the single draw are the issuer's word.
- Intervals and certified zones assume the labels are right.
- The egress allowlist only logs by default; beyond your own machine, set `OLMOEARTH_EGRESS=enforce`. It does not
  cover the olmoearth-inferencex command line the agent starts ([security](docs/Usage.md#security)).


More
----

- Every skill, tool, rule and setting: [docs/Usage.md](docs/Usage.md). Changes: [CHANGELOG.md](CHANGELOG.md).
- Development: `uv sync --all-extras`, `uv run pre-commit install` and `uv run pytest`. Sign off each commit
  (`git commit -s`, the DCO); see [development](docs/Usage.md#development).
- [OlmoEarth Artifact License](LICENSE) (Ai2): free use with restrictions: no military, defense, surveillance or
  extractive uses; cite Ai2 and pass the terms on downstream. It matches
  [allenai/olmoearth_pretrain](https://github.com/allenai/olmoearth_pretrain/blob/main/LICENSE).
- Built in collaboration with Ai2, on OlmoEarth Studio. A research demo, not an official product.
