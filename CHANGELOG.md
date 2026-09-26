# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog v1.1.0](https://keepachangelog.com/en/1.1.0/); this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

PR titles for each merged change appear under the appropriate section below.
See [`CONTRIBUTING.md`](CONTRIBUTING.md#7-documentation) for the convention.

## [Unreleased]

### Added
- **The claim check: the agent's own model reads the answer against the run
  (`harness/claim_check.py`).** exp86 round 9's audit found 16 material
  claims in 10 of 30 answers, most where the tools said nothing and the model
  filled the gap: offers of a step no tool takes or whose preconditions do
  not hold (7 of 16), claims about the world no tool checked, a margin read
  as a probability of error, a tool's summary read against another set. The
  rules catch 14 of round 8's 17 material claims on replayed calls but about
  half of new wordings (23 of 46). The check sends the draft, the user's
  messages (and on the web the earlier answers), the capability card and
  every tool call with its result as the model read it (whole, or a spilled
  result's summary and file, `spill.llm_view`) to the agent's own client
  (`OlmoEarthLLM.chat`, no tools, the new `instruct_verify` preset:
  temperature 0.1, no presence penalty, `enable_thinking` false, at most
  4,096 tokens), and asks for `[{"sentence", "kind", "why"}]`: a sentence
  that states what no tool output or user message supports (`unsupported`),
  contradicts a tool output (`contradicts`), or offers a step no tool on the
  card can take, or whose inputs the run lacks (`offer`); not a restatement,
  a hedged reading the outputs support, or an offer the card lists with what
  it needs. The reply is read leniently (prose, a fence, a thinking block, a
  Python literal, a reply cut at the budget keeps its complete items); each
  quote is matched to the answer's own sentence (Markdown, bullets, case and
  punctuation aside, a quote over two sentences names both, a near copy of a
  long one names it) and a quote found nowhere is reported, never marked. A
  malformed reply, a failed call (named by its exception type only) or a
  record over 200,000 characters fails open. `AnswerChecksMiddleware` runs it
  as the check `claims` after the rules: its flags join theirs in the one
  rewrite, with a section that asks for a possible step and what it needs in
  place of an impossible offer, and it reads the rewrite again, marking what
  it still flags `[unverified: claims]`; at most two model calls per run.
  Each call emits one `check` event (`revise`, `marked`, `passed` or
  `failed_open`) with the verifier's raw `reply`, its `version`
  (`claim-check-1`, with the prompt's digest pinned in the tests),
  `unmatched`, `reply_complete`, `finish_reason` and any `error`. On by
  default; off with `LeadAgent(check_claims=False)` or
  `OLMOEARTH_CHECK_CLAIMS=0`. `harness/capabilities.py` is a stub of the
  capability card (each tool's name and description) with the signature the
  card's own branch keeps. A preset's `chat_template_kwargs` are merged with
  `preserve_thinking`, no longer replaced by it; the web UI shows a check
  event that passed, failed open or appended as such, not as a rewrite.
  Measured offline, without a model: the verifier prompts of round 9's 30
  runs are 7,078 to 12,762 tokens (chars / 4; median 9,798, of which the
  card is 6,172).
- **`olmoearth_scores_from_file`: a direct model run's map as review-set
  input (the cluster scores provider).** Studio's API gives map tiles and a
  point lookup, never per-class scores. A run outside Studio (the companion
  repository's `scripts/score_area.py`, Ai2's fine-tuned AWF model on a GPU
  cluster) writes a georeferenced `(C, H, W)` float32 `scores.tif` of logits
  or probabilities with NaN no-data, and a `manifest.json`. The tool reads
  such a directory under `OLMOEARTH_SCORES_ROOT` (the scores files' path
  rule; a symlink or a manifest naming a file outside the directory is
  refused, and the raster's sha256 must be the manifest's), pools it with
  `oe_inferencex.assess.assess_prediction` as `oe-inferencex assess` does
  (`is_logit` from the manifest, `patch` 4 by default), and writes one scores
  file: each valid window's row holds the package's window confidence at its
  majority class and 0 elsewhere (exp64's mapping), with its pooled top-1
  probability and class beside it. `olmoearth_review_set` on it is the
  package's review set, `olmoearth_plan_label_sample` draws what
  `oe-inferencex sample` draws, and two runs go to `olmoearth_compare_review`.
  A channel the manifest marks as the label fill value (`nodata_value_<c>`;
  AWF's channel 9, never a training target) is left out of the margin and the
  class vote unless `include_fill_class` is set. For probabilities the
  confidence is the package's, the top-1 probability. The result gives the
  scores file, the window grid, valid and no-data windows, the classes, the
  model and revision, the date window and the provenance (full model output,
  not Studio point samples); never the area. Core, not deferred: it opens a
  path whose other tools are all core (a default turn now carries about 7,700
  spec tokens). On the first real run (AWF, 512 x 512 px, 16,384 windows of
  4 px) the 5% review set is `oe-inferencex assess --logits`'s, the same 819
  windows in the same order; leaving channel 9 out changes none of them,
  since it is the lowest channel at every pixel. The soul routes a map from a
  direct model run to it.
- `olmoearth_review_set` reads a scores file's own `score_type`, `signal`,
  per-window class and class names; the estimation tools read its `p1` and
  `map_class`.
- **`olmoearth_review_set_from_result`: which windows of a Studio prediction to
  check first.** A live trial (Qwen3.8-27B-NVFP4, 24 September 2026) asked
  which windows a reviewer should open first; with no tool turning a Studio
  result into review-set input, the model ranked hand-sampled pixels and put
  the most confident cells (0.97, 0.99) first. The new tool samples the
  result's band on a grid (one pixel-value per window, 2-16 per side, bounded
  concurrency, no-data dropped and counted), reads a `[0, 1]` score `s` as
  `[1 - s, s]` and ranks with `analysis.review_set.review_set`, least decided
  first, stating the assumption (the score is P(positive) for ranking only;
  not a probability of error). Other regression ranges need a `threshold`;
  classification bands are refused. Windows are `(row, col)` and index; the
  scores and window locations go to a file under `OLMOEARTH_SCORES_ROOT` that
  `olmoearth_review_set` and the estimation tools read.
- **How wrong is the map: `olmoearth_plan_label_sample`,
  `olmoearth_estimate_map_error`, `olmoearth_certify_zone`.** Wrap
  `oe_inferencex.estimate` from `olmoearth-inferencex` >= 1.3.0, a new
  optional extra (`pip install 'olmoearth-agent[inferencex]'`; the core stays
  numpy-free, and without the extra the tools answer `available: false` with
  the install line). The trial's model had proposed a pooled stratified and
  near-threshold sample with a simple-random-sample interval; the tools draw
  the package's design (confidence strata by default, random for a certified
  zone), save it with a labelling sheet, return the error rate with the
  interval that design earns and its `method`, per-class accuracy, and the
  certified zone, and refuse a review set offered as a sample.
- `olmoearth_compare_review` takes `date_a`, `date_b` and `labels_date`; with
  the extra it attaches `oe_inferencex.compare.dates_reading` and, across
  different dates, declines to grade which side is right; without it, it says
  the dates were not assessed.
- `olmoearth_search_predictions` returns a `models` map (name, `model_type`,
  `prediction_type` from the model record's `wizard_answers`) and says the
  listing is this account's models only; `olmoearth_fetch_results` and
  `olmoearth_get_prediction_result` return each output's declared regression
  range or classes from `result_metadata`.
- **Skill #18 `olmoearth-review-set` — label-free error ranking.** Answers the
  question nothing in the catalog answered: *which windows should a human open
  first, and how much of the error do they catch at that budget?*
  `olmoearth_review_set` ranks windows by the model's own top-1-minus-top-2
  margin (boundary-first optional, using a nine-level indicator over the 8
  neighbours); `olmoearth_grade_review_rule` scores **any** candidate suspicion
  signal against that margin and a no-model control, with the scene as the unit
  of replication and a one-sided exact sign test; `olmoearth_review_budget_ceiling`
  reports `min(1, budget/error_rate)` so a quoted capture is read against what
  was reachable rather than against 1.0. New `analysis/review_set.py` — pure
  Python, no new dependencies — carrying ports of `oe_inferencex.metrics`
  (`aurc_expected`, `oracle_aurc`, `excess_aurc`, tie-aware `capture_at_budget`,
  `auroc`), verified against the numpy originals to < 1e-9 across 400 randomised
  cases including heavy ties and degenerate inputs. Tie handling is by
  expectation under random tie-breaking, so no result depends on raster order.
  Every result ships the measured evidence and the honest caveats: on all 24
  tasks of Ai2's own published embedding suite (14 sources, 6,435,473 graded
  units) the margin beat the best no-model control, 24/24, p = 6e-08, and
  ensemble disagreement, cross-model disagreement, two-view disagreement and
  feature-space typicality were each measured against it on expert labels and
  none ranked errors better — but the margin only **ties** ensemble predictive
  entropy, one flood event has a no-model NDWI control that matches it, and
  these numbers order a review well while being badly miscalibrated as
  probabilities. Evidence from
  [`2imi9/olmoearth_inferenceX`](https://github.com/2imi9/olmoearth_inferenceX).
- **A capability card in the system prompt, built from the tools' code, and
  two rules in the soul** (exp86 round 9's audit: 7 of 16 material findings
  were offers of actions no tool can do or whose preconditions did not hold,
  such as a review set of a regression band with no threshold, "I'll set up a
  direct model run", labels on flagged windows turned into "a proper
  error-rate estimate" and a comparison "rerun with labels_date", which takes
  no labels; others were claims no tool checked, such as "no ground-truth
  labels exist"). Every `RegisteredTool` now carries a `Capability`
  (`tools/registry.py`), declared beside its handler: what it `does` in one
  line, what it `needs` (preconditions it enforces, e.g. a threshold for a
  regression band not in [0, 1], a `design='random'` plan to certify) and
  what it `cannot` do (e.g. say which map is right, even with `labels_date`,
  for `olmoearth_compare_review`; take a review set as its sample, for
  `olmoearth_estimate_map_error`). `harness/capabilities.capability_card(registry,
  loaded_groups=())` assembles the card: each core tool and each tool of a
  loaded group with its required arguments from the spec, the groups not yet
  loaded by name, and what no tool of the registry can do (run a model
  outside Studio; look up or fetch ground-truth labels, or say whether any
  exist; read what a Studio model was trained on; read files outside the
  workspace and the scores root), each dropped once a registered tool
  `covers` it (the opt-in `olmoearth_run_python` covers files and model
  runs). `LeadAgent` puts the card right after the soul
  (`soul.with_capability_card`), before the skill index and the other
  clauses, and rebuilds the system message on the turn after a group loads.
  The card is about 1,390 tokens (characters / 4) for the default registry,
  about 1,700 with every group loaded. The soul's new rules: state as fact
  only what a tool of this run returned or the user said, and say what no
  tool looked up is not known from this run; propose next steps freely, but
  only actions the card lists, with their preconditions, and say plainly
  when no tool can do what the user may want. Its routing no longer presumes
  that no labels exist ("when the user gives no ground-truth labels"). A test
  fails for a tool of `build_default_registry()` with no declaration, and
  each `needs` and `cannot` entry has a probe that calls the tool in that
  condition and gets the refusal or the absence it states
  (`tests/tools/test_tool_capabilities.py`), so the card cannot drift from
  the code.

### Changed
- **The lead-agent loop is a middleware chain with LangChain 1.x's hook
  interface** (`harness/middleware.py`; the owner chose on 25 September 2026
  to build the layer rather than migrate, so that a later migration is a
  mechanical replacement). `AgentMiddleware` has LangChain's async hooks
  (`abefore_agent`, `abefore_model`, `awrap_model_call`, `aafter_model`,
  `awrap_tool_call`, `aafter_agent`), its order (`before_*` in list order,
  `after_*` in reverse, wraps nested with the first outermost), node hooks
  that return state updates or `{"jump_to": "model" | "tools" | "end"}`, and
  `@hook_config(can_jump_to=[...])`; an undeclared jump raises.
  `ModelRequest` (messages, system message, tools, tool choice,
  `model_settings` = `OlmoEarthLLM.chat`'s keyword arguments, state,
  runtime) and `ToolCallRequest` have `.override()`; `ModelResponse` wraps our
  `ChatResponse`. The runtime's `emit(event)` publishes an event, and
  `run_stream` yields it in order as it is emitted, even from inside a model
  or tool call. The loop's behaviours are ported onto three middlewares:
  `TurnCapMiddleware` (the forced answer without tools at the cap, and
  `max_turns`), `RetryHintMiddleware` (the stop-retrying hint, moved from
  `ToolRegistry.dispatch`, whose `retry_hint=True` default still applies it to
  direct calls; a subclass that overrides `dispatch(call, ctx)` must now accept
  the `retry_hint` keyword, since the agent passes it) and
  `AnswerChecksMiddleware` (the checks, the one rewrite in
  `REVISION_MODE`, the marking and the appended required statements, with
  their `check` and `grounding_check` events). `LeadAgent(middleware=[...])`
  replaces them; the default chain runs as the loop did: the existing tests
  pass unchanged, and 3,000 random scripted runs give the same events, model
  calls and state as the loop before it. When a wrap hook replaces a tool
  call, the provenance log and the answer checks record the call that ran. A chain that would start a turn past
  `max_turns + 1`, or enter the model step more than 10 times in one turn,
  raises `MiddlewareError`.
- The `inferencex` extra is `olmoearth-inferencex[geo]` (>= 1.3.0): reading a
  scores GeoTIFF needs rasterio, and the package's `geo` extra brings it, with
  geopandas, shapely, pystac-client, planetary-computer, matplotlib and scipy.
  The core install is unchanged and numpy-free.
- **One comparison tool.** `olmoearth_compare_results` takes 2-8
  `result_ids` and a `mode`: `pair`, `group`, `series` (one model, ordered by
  date) or `ensemble`; the default `auto` picks pair, series or group from the
  results' models and dates. It replaces `olmoearth_compare_group`,
  `olmoearth_trace_shifts` and `olmoearth_ensemble_uncertainty` and keeps their
  statistics. Breaking: `result_id_a`/`result_id_b` inputs are now
  `result_ids`; hotspots are grid `(row, col)` and index, and
  `shared_extent_bbox` is gone (rule 3.1).
- **Deferred tool groups.** A turn sends the core tools only (about 7,400
  spec tokens instead of 13,700). The self-run training tools, the
  caller-array tools and the negative sampler are sent once their skill is
  loaded with `olmoearth_load_skill`, or forced from the web UI.
- `olmoearth_load_context` pages projects (`limit`, `offset`, `total`) and
  replaces `olmoearth_search_projects`; `olmoearth_litsearch` takes an
  `identifier` and replaces `olmoearth_litsearch_resolve`;
  `olmoearth_list_skills` is removed (the index is in the system prompt).
- **Skill #9 `olmoearth-uncertainty` now routes the error-ranking question to
  #18** instead of silently answering it. Both its tool descriptions and its
  `SKILLS.md` section state the split: #9 owns self-consistency and the
  out-of-distribution regime, #18 owns which windows are wrong. This is a
  scoping fix, not a correction — `StudioClient.pixel_value` returns only
  `raw_value`/`classification`, never probabilities or logits, so with Studio
  results alone the margin is not computable and ensemble disagreement is the
  only uncertainty signal reachable. #18 is therefore bring-your-own-scores,
  mirroring #8's bring-your-own-embeddings.
- **Catalog corrections.** Skill #17 `olmoearth-rslearn` was missing from both
  `SKILLS.md` tables (Catalog and Example briefs) although its section and its
  webui entries existed; both rows added alongside #18. Skill counts updated
  from 17 to 18 across `README.md`, `SKILLS.md`, `PLAN.md`, `docs/CANON.md` and
  `webui/js/skills.js`.
- `tests/skills/test_skill_registry.py` gained a catalog-integrity test
  asserting every tool a catalog row advertises is actually built by
  `build_default_registry()`.
- Ruff `per-file-ignores` for `tests/**` now also allows `S311`: seeded PRNG
  fixtures must be reproducible, not unguessable.
- **`olmoearth_compare_results` refuses two results of different properties**
  (`comparable: false`, both property names and declared ranges) unless
  `allow_different_properties=true`, which returns only the correlation as
  meaningful, with a warning (the web UI's compare card then shows the warning
  instead of a difference map); the trial had narrated a binary score and a
  count as one quantity. `olmoearth_ensemble_uncertainty` refuses the same.
- The harness soul routes "which windows to check" to the review-set tools and
  "how wrong is the map" to the estimation tools, and describes a model by its
  `prediction_type`. `olmoearth_compare_group` routes dated series to
  `olmoearth_trace_shifts`. Skill #18's catalog row lists all eight of its
  tools. The review-set and ensemble descriptions were shortened (the
  evidence stays in the results).
- `olmoearth_get_prediction_result` returns `result_metadata` without its
  `geometry` (rule §3.1).
- Shared grid sampling (`tools/sampling.py`) replaces four copies of the
  pixel-value loop and two band readers; a Studio class (`{label, color}`)
  reads as its label, so categorical agreement and vote counts no longer
  depend on an unhashable object.
- mypy skips numpy's stubs (3.12 syntax) and the untyped companion package.
- **Repository reorganised.** The four instruction skills (#1-#3, #17) ship
  inside the package under `src/olmoearth_agent/skills/packages/` (moved from
  the frozen `2imi9/OlmoEarth-Skills`) and the `vendor/olmoearth-skills`
  submodule is removed; five research notes and the web UI design notes move
  to `docs/archive/`; nine unused dataclasses are removed from `types.py`;
  `docs/serving.md` documents the vLLM cluster setup.
- **The turn cap no longer ends a run without an answer.** When the last
  allowed turn still asks for tools, `LeadAgent.run_stream` yields
  `max_turns` (`final_answer_forced`), tells the model the cap is reached,
  and makes one more call with no tools; its text is the `final` event,
  marked `forced_by_turn_cap` (a model that writes nothing gets a harness
  answer naming the cap). The CLI prints it with a note on stderr and the
  web UI shows the same note above it; the Claude backend writes tool turns
  as text in a request without tools. exp86 round 1's brief 4 on a Studio
  result ended all three runs at the cap with no answer.
- The soul asks for numbers exactly as the tools returned them, no ratio,
  difference or percentage of the model's own, and no figure quoted from a
  tool's description as a finding (exp86 round 1).
- **The answer's numbers are checked before it is shown** (exp86 round 3's
  fix, 25 September 2026). In each of three rounds the only genuine fault was
  a number no tool had returned (round 3: "5,565+ windows" between the budget
  cut and the median; the scores give 7,373), and each tool fix only moved
  the model to a new derived quantity. `LeadAgent.run_stream` now reads every
  number of the final answer (`harness/grounding.py`: thousands separators,
  a U+2212 minus, percents, `k`/`M` suffixes, digit runs inside ids and
  dates, integers up to 8 exempt) and looks for it in the run's full tool
  results, the brief, the history and the saved preferences, never in the
  system prompt or a tool description. When one is found in none of them it
  yields `grounding_check` (`action: "revise"`, the numbers as written),
  sends the draft back with a harness note naming them and makes one more
  call with no tools; the rewrite is the `final` (`grounding_revised`), or
  the draft if the rewrite is empty. A rewrite that still states such a
  number is shown as written, with a second `grounding_check`
  (`action: "shown"`); nothing is asked twice. The answer forced at the turn
  cap is checked too; the harness's own fallback text is not. `AgentResult`
  carries `grounding_revised` and `grounding_checks`. The CLI names the
  numbers still unsupported on stderr (the rewrite request under
  `--show-trace`); the web UI notes them above the answer and puts the
  request in the steps. Off with `LeadAgent(check_numbers=False)` or
  `OLMOEARTH_CHECK_NUMBERS=0`. Over the 87 answers of rounds 1-3 the reader
  reports seven numbers in six answers, each a fault the trial's diagnoses
  found ("51-70%" from a description, "47 dropped", "> 90%", "6.5%",
  "delta/18", "5,565+"), and none of the other 3,053. The soul says the
  harness checks.
- **A comparison of different properties returns only what its warning allows**
  (exp86 round 4's fix, 25 September 2026). With `allow_different_properties`,
  `olmoearth_compare_results` used to return the mean difference, the mean and
  maximum absolute difference, the RMSE and the agreement fraction between the
  two quantities, beside a warning not to read them; two of round 4's three
  answers put them in a table as findings. A pair of different properties now
  keeps `n_samples`, each map's own mean and the correlation, and names the
  rest in `statistics_left_out`. In a group, a pair of one property keeps its
  statistics, a pair of two keeps only those, and the ensemble and the
  most-divergent pair are not computed. The number check's `revise` event
  carries the answer it replaces (`draft`), so a trace shows what was removed.
- **The number check reads "1,000" as a thousand** (exp86 round 5's fix). A
  thousands-separated number was also tried as its parts, so that a window
  written "(24,108)" is supported by its row and column; "1,000" became 1
  and 0, both exempt, and "e.g. 1,000+ labels", in no tool output, passed.
  Only a number standing alone in brackets is split now. Over the 147
  answers of exp86's five rounds the reader reports the same seven answers
  as the trial's scorer, and no other.
- **A comparison of two inferences counts every class change, not only the
  listed ones** (the exp86 round 6 audit). `olmoearth_compare_review` listed
  the first differing windows in window order (the top rows of the grid) and
  gave no breakdown, and answers read a direction and a place off that
  listing: "most differing windows flip class 1 -> 0" when 1,247 of 1,570
  flipped 0 -> 1, "a long strip along the north edge" for 1.3% of the
  differences, a "dominant contrast" of 11.7% where another pair held 37%.
  The output now carries `class_changes` (each (class A, class B) pair with
  its count and share of the differing windows, largest first, up to ten),
  `n_class_changes`, `a_more_confident_share_of_differing`, and
  `listing_order`, which says the list is the first windows in order, not a
  sample.
- **The review-set tools give the aggregates an answer needs, scope their
  evidence to the case, and carry the output contract** (the exp86 round 7
  diagnosis and the blind audit of rounds 6 and 7). `olmoearth_compare_review`
  adds a `spatial` breakdown (each of 4 row and 4 column bands' share of the
  differing windows and of all windows), `class_pairs` (each unordered class
  pair with both directions' counts, beside the directed `class_changes`),
  `b_more_confident_share_of_differing`, and class names when both scores
  files give them; it lists only the first 10 differing windows inline and
  saves every one to `differing_path` (`n_differing_total`, `listing_order`).
  A result now carries the shared output contract: `facts` (one-sentence
  statements computed by code), `must_state` (at most 3 sentences of at most
  25 words, each a limit the answer must convey) and `forbidden_claims`
  (`winner_without_labels`, `another_date_settles_it` across dates,
  `error_rate_without_labels`, `combined_statistic_across_properties` in
  `olmoearth_compare_results`; each id once per result). The facts:
  `dominant_change` (`from_class` -> `to_class` from map A to map B, with
  `reverse_n`, `reverse_share` and `tied_with_reverse`; a tie with any other
  direction is named in the sentence), `more_confident_side` (`side` A, B or
  `neither`, decided on counts; `share_a`, `share_b`, `share_equal`),
  `concentration` (`grid`, `n_differing`, `top_band_share` = the northmost
  row band's share of the differing windows, and `max_band`, the band of
  either axis holding the most, with its `axis`, `band` from 0, `of_grid`
  and `share`), and `margin_ratio` (the median margin over the listed
  windows' margins and over all windows in the review set:
  `listed_low`/`listed_high`, `review_set_low`/`review_set_high`; the margin
  summary adds the review set's range). The inline `evidence` and `caveats`
  blocks are replaced by one `evidence_scope` sentence: Ai2's 24-task
  embedding suite, window level, OlmoEarth family, and whether that covers
  this case; when it does not, `must_state` gets a short sentence stating the
  limit ("No recorded experiment grades a regression score read as a
  probability." for a Studio band; "This ranking uses the logit margin; on
  Ai2's suite one minus the top probability ranked errors better on 14 of 16
  multi-class tasks." for multi-class logits, exp76), and the full text is
  saved to `evidence_detail_path`. The old caveat that the measured evidence
  is for the logit margin is dropped: on the suite the probability margin
  beat the logit margin on all 16 multi-class tasks. A ranking of a scores
  file carries the file's warnings whole in `scores_file_warnings` (the
  provider now writes its `package_warnings` into the file) and the limit a
  known one states in `must_state`, never the provider's "pass form='top1'",
  an argument no agent tool takes. A scores file's rows are placed on the
  file's own grid, never on a grid the model passed (`grid_note` says when
  one was set aside; two files naming different grids are refused).
  `olmoearth_scores_from_file` adds `model_and_raster` ("model ... at
  revision a347b15; raster sha256 a7c40be9 ...": a hash was given as a
  revision), a ranking with a grid adds `boundary_neighbours_means`
  (neighbours of a different PREDICTED class, not true boundaries, not
  error), and a comparison adds `boundary_means`. A spilled tool result
  previews its summary fields and copies the contract whole (a result that
  is not a dict keeps the raw JSON prefix), and its note no longer asks for
  the saved path in the answer and says not to describe the file beyond the
  preview. No number, ranking or parity-checked field changes.
- **The estimation tools state the statistical rules as data: next steps,
  facts and forbidden claims** (the exp86 rounds 6 and 7 blind audit). The
  answers broke rules the tools had just applied: a looser alpha or a
  Bonferroni re-run offered after the prefix rule certified nothing (every
  p_value was at least 0.554), a zone offered from a confidence design, an
  error rate offered for a regression declared 0.2 to 1.2 with no threshold,
  "a third dated map" offered to settle which of two is right, and "~23%"
  (69/300) and "127 unused labels" (300 - 173) worked out by hand. Every
  output of `olmoearth_plan_label_sample`, `olmoearth_estimate_map_error` and
  `olmoearth_certify_zone` now carries `next_steps`, written by code from the
  design and the outcome (nothing certified under a random design: label
  more windows under a new random design fixed in advance, or report the
  whole-map estimate; never a looser alpha or another rule), and the output
  contract's `forbidden_claims`, each id one of its fixed ids and listed once
  per result: `post_hoc_alpha` and `rule_switch_after_failure` on every
  certification (the latter says prefix accepts levels from the smallest
  zone upward while p_value <= delta and stops at the first failure, and
  bonferroni any level at p_value <= delta/J, with the p_values each rule
  reads); `certify_from_nonrandom_design`; `error_rate_without_labels` on a
  plan; `subset_labelling_sufficient` on a stratified plan only, whose sheet
  lists the strata in turn, least confident first, so its first rows are one
  stratum; and `simple_random_interval_for_stratified_design` on a stratified
  estimate. A random plan's sheet is in the package's random draw order, so
  it carries the fact `prefix_is_random_sample` instead (its first k rows,
  k fixed before labelling, are a smaller random sample: unbiased, with a
  wider interval; the sentence names the route that estimates them). An
  estimate and a certification carry the fact `whole_map_estimate`
  (estimate, low, high, design; the sentence adds the package's warning when
  it gives one, such as starved strata or no labelled window wrong). A rate
  over a Studio result's grid (plan, estimate and certification alike) and a
  certified zone carry a `must_state` scope. At the largest Studio grid a
  budget above the valid windows is planned at all of them with the fact
  `unused_labels` (requested, planned, n) instead of refused; below it the
  refusal says to keep the budget at grid 16, and inline or file scores
  still refuse, stating the labels left over in the error and as an
  `unused_labels` fact on the failed envelope (the registry carries a
  refusal's `facts`, `must_state` and `forbidden_claims` beside its
  unchanged error). `olmoearth_compare_results` names
  `error_rate_for_unthresholded_regression` for a regression band that is not
  a [0, 1] score, and `combined_statistic_across_properties` for results of
  different properties (refused or allowed); `olmoearth_compare_review` names
  `another_date_settles_it` for maps of different or overlapping periods;
  both name `winner_without_labels` on every comparison, since neither takes
  labels. The builders live in `olmoearth_agent.tools.statistical_rules`. No
  number, draw or ranking changes: the six fixed-input parity calls of exp86
  round 7, replayed, keep every key and value and pass against the package.
- **The answer is checked against the run, not only its numbers** (the exp86
  round 6 and 7 audits, for exp87). The blind audit found a false or
  unsupported statement in 23 of 30 answers of each round; the number check
  reads numbers only, and passed "~23%" (69/300, derived) against an
  unrelated count of 23 in six rounds. The post-answer checks are now a list
  in `harness/checks.py`, each taking the answer and the run's evidence and
  returning violations (`{check, text, detail}`, `text` the sentence):
  `numbers`; `direction`, which reads the tools' `facts` under the contract
  the tools and the harness share (`dominant_change` with `reverse_n`,
  `reverse_share` and `tied_with_reverse`; `more_confident_side` with `side`
  A, B or neither and `share_a`, `share_b`, `share_equal`; `concentration`
  with `grid`, `n_differing`, `top_band_share` of the northmost row band and
  `max_band`; `margin_ratio` with the listed windows' and the whole review
  set's range; `unused_labels`; `whole_map_estimate` and unknown ids are
  shown, never checked): a dominant change stated backwards, or another
  pair named the dominant change, read only in a sentence about change
  between the maps (an arrow, "from X to Y", a change verb, a transition or
  contrast; never what the maps are made of) and only for classes written
  by name or as `class N` (a bare digit is never a class, so "2 to 4 rows"
  is none), and no contradiction when the reverse ties the top count; the
  other side named the more confident, unless the clause places it in a
  region ("in the north"); a clause that puts the bulk of the differences
  (mostly, most, concentrated, clustered, dominated, a strip, along) at the
  north edge or the top rows when the northmost band holds under 10% of
  them, at another edge when `max_band` is another band on that axis, or
  in rows too few to hold half of them, never a window's own coordinates
  ("the highest-ranked window is at row 12, col 40") or a count of a minor
  part; a magnitude that compares the review windows' confidence with the
  typical window's ("20x less confident", "orders of magnitude more
  uncertain than typical windows") outside `margin_ratio`, not any "N
  times" near a confidence word; a wrong count of unused labels. `actions`:
  a file said to be saved that no tool of the run reports writing (a path
  under the same key as the argument it echoes is an input; a key that
  names an output, such as `out_path`, or another key, reports a write
  though the caller chose the path; the harness's own spill file of a
  result too large for the context is written too), a list said to be
  saved in a file that is not one, items said to be listed above or below,
  or the first N said to be listed, that the answer does not hold ("as
  shown above" or "see above" points at prose and is none; on the web,
  where the tool results are shown above the answer, only "above" is not
  checked). `forbidden_claims`, a detector per id a tool names
  (`post_hoc_alpha`, which allows the alpha the tool ran at, the alpha the
  user asked for, and either split over k levels in a sentence about
  levels; `rule_switch_after_failure`; `certify_from_nonrandom_design`,
  which takes advice to draw a random or probability sample as correct;
  the error rate without labels or for an unthresholded regression; a
  subset said to suffice; a winner without labels; a statistic across
  properties; another date said to settle it, where "the earlier map" or
  "the later image" is map A or B, not a new date);
  `simple_random_interval_for_stratified_design` has no detector (the
  package's own interval is a Wilson interval on the design's effective
  sample size, which no wording tells from a naive one) and is ignored like
  any unknown id. And `must_state`, the statements a reported result
  requires: the first three per result, of at most 25 words each. The
  detectors hold back on a negated clause, a hedge or an example, and read
  each sentence once into its brackets, clauses and word spans, so a 12 KB
  listing written as one sentence is checked in milliseconds (about 2 s
  before). When any check fires, one rewrite call (no tools; the
  `thinking_coding` sampling, `REVISION_MODE`, with no presence penalty)
  lists every violation; the checks run again, and each sentence still
  flagged is shown with `[unverified: <check>]` after it, never deleted (a
  required statement still missing is added at the end). Each check that
  fires yields a `check` event (`action` `revise`, with the `draft`, then
  `marked`); the number check keeps its `grounding_check`. The number
  check's sources are the tool results, the paths the harness spilled
  results to, the brief and the user's turns, never the saved preferences;
  an earlier assistant turn is a source on the web only, where it was
  checked when it was shown. A percent is supported only by a share: any
  value in [0, 1] (so 0% and 100% pass against 0 and 1, as before), a
  percent written in a string, or a value under a key naming a percent,
  share or rate; a count above 1 under another key never. `LeadAgent` takes
  `surface` (`"cli"`, the default, or `"web"`, which the bridge passes) and
  `check_answer` (`OLMOEARTH_CHECK_ANSWER=0` switches every check but the
  numbers off); `AgentResult` carries `revised`, `checks` and `marked`, the
  CLI and the web UI name the checks. The LLM client reads a server's
  reasoning from `reasoning`, then `reasoning_content` (round 7 recorded no
  thinking), for the `thinking` event only.
  `scripts/validate_answer_checks.py` runs the checks over recorded
  answers; with the tools' keys simulated from exp86's recorded results
  under the contract, rounds 6 and 7 get 27 flags in 21 of 60 answers: 22
  on sentences the blind audit confirmed false (every confirmed E3 action
  claim, 2 of 3 E2 directions, 4 of 10 E1 listing readings, 1 of 20 E5, 2
  of 16 E6, 2 of 5 E4, 8 of 45 others), 2 on sentences its verifiers added,
  2 on the same claims the audit confirmed in sibling answers, and 1
  required statement the answer omits (the same flags as before these
  fixes). On the unaudited rounds 1 to 5 the
  fixed detectors add 4 flags, each the row 0 strip read off the head of an
  index-ordered listing that the audit confirmed false in rounds 6 and 7
  (two of the four scope it to the listing, "visible" or "the top of the
  list"). The percent rule adds 13 flags over the 207 answers of rounds 1
  to 7, all derived (11 of them "~23%"), and removes none. The `inferencex`
  extra requires olmoearth-inferencex 1.3.1.

### Fixed
- **The tools state what round 9's model filled in from general knowledge**
  (the audit of exp86 round 9, on agent c62538f). Most of round 9's material
  errors sat where the tools said nothing. Offers are not restricted; the
  tools now say what holds.
  - Labels. A Studio model summary (`olmoearth_search_predictions`'
    `models`, `olmoearth_review_set_from_result`'s `model`) carries
    `trained_on_labels`, the `label_field_id` and the train/val/test `split`
    when the fine-tuning wizard names a label field; the requester is never
    read. `olmoearth_search_predictions`, `olmoearth_compare_results` and
    `olmoearth_review_set_from_result` state the fact `labels_in_studio`:
    which models were fine-tuned on a label field of their project, so labels
    for it may exist in Studio, and that no tool of the run looked them up
    for this area. The comparisons' `winner_without_labels` reason, and
    `olmoearth_compare_results`' framings and `method`, say no labels (no
    ground truth) were given to this comparison, never that none exist.
    Round 9 (B3/studio run 1) answered "no ground-truth labels exist" of two
    models each fine-tuned on a label field of the user's project.
  - Review sets. `margin_summary`'s `listed`, `not_listed` and `review_set`
    give the `ranks` they cover, `not_listed` names what it counts after
    (`after`: the tool's own listing), and the `reading` says a table of fewer
    rows leaves out more windows, with margins no higher. Round 9 (B2/studio
    run 1) showed 5 of 8 listed windows and called the tool's "83 not listed"
    "the other 83 windows", margins 0.516 to 0.991, where three unshown
    windows had margins of 0.26 to 0.47. The fact `review_set_classes` counts
    every window of the review set by predicted class, listed or not (named
    by the scores file's class names, a Studio score's side of its threshold,
    or the class number; ten by name, the rest together). Round 9
    (B8/cluster run 2) said montane_forest and woodland_forest pairs
    "dominate" from the 10 listed windows; the 819 hold 256 grassland_barren
    and 246 shrubland_savanna against 125 woodland_forest and 84
    montane_forest.
  - Lists in files. The fact `list_file` names, by its full path, the file a
    comparison's differing windows or a review set's full list was written
    to, and what it holds ("All 3,807 differing windows are listed, in window
    order, in <path>, each with both maps' class and margin."), and
    `listing_note` and `listing_order` name the file, not the key that holds
    its path. Round 9 (B7/files run 1): "All 1,570 differing windows are
    listed in `differing_path` above".
  - A margin is not a probability of error. The contract gains the fixed id
    `margin_as_error_probability`, emitted with every ranking
    (`olmoearth_review_set`, from rows or a scores file, and
    `olmoearth_review_set_from_result`). Its reason forbids calling a window
    "most likely wrong", "probably an error" or "likely mislabeled", and,
    where the ranking evidence does not or may not cover the case, an order
    by likelihood ("the likeliest spots for a wrong call"). The answer
    checks' detector reads the same: a window called probably wrong ("they're
    most likely mislabeled", "the argmax label is most likely wrong there",
    "probably errors", "more likely wrong than right", a margin called a
    probability of error) is flagged; an order ("the most likely places for
    a wrong label", "the windows most likely to be wrong", "where it is most
    likely wrong") only where the emitting result's
    `evidence_covers_this_case` is "no" or "not known": the blind audit of
    rounds 7 and 8 refuted two such orders of an OlmoEarth model's logits,
    and the audit of round 9 confirmed one of a Studio score. A negated,
    quoted or example claim, an error rate, "least decided", "most
    uncertain", "checked first" and the tools' own sentences pass. Over the
    1,824 sentences of rounds 6 to 9's 120 answers, with the id emitted in
    the 24 runs that ranked a review set, it flags 4: round 9's three
    audited claims (two material), and round 6's "where it is most likely
    wrong" (no finding), an order that passes under the current tools but
    is read there because round 6's result predates
    `evidence_covers_this_case`. Emitted in all 120 runs, it adds round 8's
    "where each model is likely wrong" (B3/studio run 2), a confirmed
    finding.
  - Boundary shares. `boundary_means` says the shares measure no contiguity
    and do not show whether whole regions flip or only their edges; `where`
    says the differing windows lie on map A's predicted-class boundaries more
    (or no more) often than windows overall, "not whether whole regions
    flip", in place of "mostly on class boundaries" and "spread across the
    scene"; and the fact `boundary_share` states both shares with that
    limit. Round 9 (B3/cluster run 3): "mostly boundary reclassification
    rather than wholesale area flips", where the round's audit found blocks
    of 696, 543 and 467 differing windows.
  - The integrated check (outside the repo) replays the tool calls of
    rounds 8 and 9 (30 runs each) on these tools. Answers built from the
    tools' own sentences raise no violation, with or without the extended
    notes. Round 8's recorded answers keep 14 of 17 material findings
    flagged, with the same 8 other flags; round 9's are flagged on 2 of its
    16 material findings (the two "most likely" claims) and on one
    immaterial confirmed finding, and nowhere else.
- **The comparison and review tools state what their outputs cannot support**
  (the blind audit of exp86 rounds 7 and 8, round 8 on agent 6d25307).
  - `olmoearth_compare_results`: every correlation it returns (a pair's, each
    pair's of a group, each step's of a series, with
    `allow_different_properties` too) is a `correlation` fact with `r`, `n`
    and its 95% interval by Fisher's z (none below 4 cells), whose sentence
    says the sample cannot say whether the maps co-vary when the interval
    holds 0, the sign otherwise, and that a correlation says nothing about
    where. `must_state` says so; `spatial_pattern_from_one_correlation` is
    forbidden whenever a correlation is returned, and
    `agreement_from_uncertain_correlation` when its interval holds 0 and
    reaches 0.3 on a side; below 50 cells `next_steps` names a denser `grid`
    (or says the mode's cap is reached). Round 8 read r = -0.0172 over 25
    cells (interval -0.41 to 0.38) as "do not agree spatially at all" and
    "one is high where the other is indifferent".
  - A regression band with no threshold forbids a review set
    (`review_set_for_unthresholded_regression`, beside the error-rate claim)
    in `olmoearth_compare_results` (a pair or a series; not a group or an
    ensemble, whose own ranking of where the results disagree needs no
    threshold) and in `olmoearth_review_set_from_result`'s refusal, its reason
    scoped to a margin-based review set of the band it names; both
    comparisons forbid labelling only the low-confidence windows
    (`subset_labelling_sufficient`: not a sample of the map; the plan tool's
    designs draw from every window).
  - Two dated maps (`olmoearth_compare_review` with dates, a temporal pair or
    a series in `olmoearth_compare_results`) forbid
    `one_reference_settles_two_dates`, and their dates sentence says labels
    for one date grade only that date's map and each map needs its own
    date's reference. A temporal pair also forbids another date as what
    settles which map is right (`another_date_settles_it`); a series does not,
    since the trend over more dates is its own question.
  - Where the evidence does not, or may not, cover the case, no field
    carries the experiment's figures or its dataset: `evidence_scope` names
    the source (exp58 or exp70 of 2imi9/olmoearth_inferenceX) and says it does
    not cover the case, the `more_confident_side` fact drops "51 to 70
    percent", the Studio review set's caveat drops exp78's "1.8 to 5.8
    times", a provider's multi-class limit reaches `must_state` only where
    the suite covers the case, and `evidence_outside_its_scope` is forbidden.
    Round 8 wrote "51-70% of the time in comparable cases" and called a
    GEOID-Flood pair "Sen1Floods11 flood maps". Where it covers the case only
    in part (an OlmoEarth model's multi-class logits; the suite measured the
    probability margin), `evidence_scope` carries none of the suite's 24-task
    figures, only exp76's about the logit margin itself (worse than one minus
    the top probability on 14 of 16 multi-class tasks), and
    `evidence_outside_its_scope` names the logit margin as what the suite did
    not measure. The file keeps the full text.
  - A review set listed short (`olmoearth_review_set`, and the largest
    budget's in `olmoearth_review_set_from_result`) saves every window of it
    to a CSV beside the evidence file (rank, window, row, col, margin, class
    or score; no coordinates): `review_list_path`, `review_list_rows`, and a
    `listing_note` that says `review_set_evidence.json` holds evidence text
    only (as `olmoearth_compare_review`'s `listing_order` now does), where
    round 8 said the full list was saved in it.
  - `margin_ratio` gives `listed_n` and, past ten listed, the first ten's own
    ratio (`first_n`, `first_low`, `first_high`); round 8 gave ten shown
    windows the 50's "13 to 41 times" (theirs: 20.68 to 41.31).
  - `olmoearth_compare_review` calls row band 0 the northmost (or southmost)
    only for scores with window centres or a transform; otherwise it is the
    grid's first rows and the result says the rows need not run north to
    south (round 8 called row band 0 of F4's 400 chips stacked in dataset
    order "the northmost band"). `top_band_share` and the other fields keep
    their names; `spatial` and the fact add `row_order`.
- **The estimation tools rank the weakest classes in code and never promise
  a certified zone** (the blind audit of exp86 round 8, B5/files/2). One
  answer named class 1 (user's accuracy 22.5%) and class 6 (45.1%) the
  weakest while class 5 had none of its 10 map-labelled windows correct
  (interval 0 to 27.8%), and another offered "a guaranteed-certifiable
  region" from a random plan. `olmoearth_estimate_map_error` with reference
  classes now carries the fact `weakest_classes`: the classes with at least
  5 map-labelled windows ranked by user's accuracy (of the windows the map
  puts in a class, the share the labels agree with; `correct` and
  `labelled` from the confusion matrix, the estimate and its interval),
  a sentence naming every class with none correct first and then the next
  lowest up to three, but never every ranked class (of two, one is the
  lowest; with one, no class is ranked against it and none is named the
  lowest), the classes on fewer windows named as too few to rank, and a
  class with enough windows but no estimate named as unranked; a next step
  points to it when it names a lowest class. Every next step of
  `olmoearth_plan_label_sample`, `olmoearth_estimate_map_error` and
  `olmoearth_certify_zone` that names a random design for a certified zone
  says it makes one possible, not certain (`olmoearth_certify_zone` may
  certify nothing; on the same map's 300-label random design every level
  failed), as do `design_requirement`, the confidence plan's `design_note`
  and the certify tool's description, and each of those results emits the
  forbidden claim `certification_guaranteed`. No number the package
  returns changes.
- **The answer checks catch exp86 round 8's audited claims** (the blind
  audit of rounds 7 and 8, and the review of branch 2imi9/fix-r8).
  `forbidden_claims` gains a detector for each of the contract's six new
  ids, each run only when a tool of the run emits it:
  `spatial_pattern_from_one_correlation` (where, or in what pattern, the
  maps agree, read from one pooled correlation: "anywhere", "nowhere", "high
  where the other is low", "large parts ... while the rest differs", its
  label read with the clause; not a breakdown a tool computed by rows or
  bands, a share such as "a large part of the disagreement", "agree
  spatially", which says how much, not where, or one map's level in a place,
  "KarstBinary is low almost everywhere": the clause must relate the two
  maps); `agreement_from_uncertain_correlation` (that the maps do or do not
  co-vary: "do not rise and fall together", "unrelated", "strongly agree"; a
  negation does not exempt it, a statement of what the correlation can say
  does: "only whether they rise and fall together is meaningful", and so do
  the tools' own sentences, which say the sample cannot say whether the maps
  co-vary, what an interval holds, or the sign the tool found);
  `review_set_for_unthresholded_regression` (a review set, margins or the
  least decided windows *offered*, in the review's own clause, unless a
  threshold is stated as what the review needs, "once a threshold is named",
  while "without a threshold, I can still build a review set" is read; and
  only for the bands the reason names, so an offer that names only another
  band of the run, by its map's or its property's name or as a [0, 1] score,
  is not it); `one_reference_settles_two_dates` ("either date", "one or both
  dates", "at least one, plus a date-matched second inference", and one
  map's year named alone, read against the comparison's own `dates`; a
  clause that restricts one date's labels to that date's map, as the tools'
  must_state does, is the rule); `evidence_outside_its_scope` ("in
  comparable cases", "of such windows", "behind this rule", "why these: on
  Ai2's suite ...", and a name only the tool's `evidence_scope` holds, such
  as round 8's "the two Sen1Floods11 flood maps", unless the sentence states
  the scope; a bare "measurement" is the run's); `certification_guaranteed`
  (a design, sample or plan in the promise's own clause:
  "guaranteed-certifiable", "enough labels to certify", "a random design
  will certify"; not a requirement, a hedged offer, or the test's own
  guarantee, "the certified zone's error is at most alpha", "the guarantee
  covers only that alpha"). `subset_labelling_sufficient` also catches a
  sample of the least confident windows promised a sound rate ("a targeted
  labeling sample from the lower-confidence windows ... defensible error
  rates", round 8 B3/studio run 3, which the rewrite had kept unmarked),
  unless the sentence names a stratified or weighted design; "not enough" is
  no offer. A negation governs a claim only before it in its clause or
  within six words after it (round 7's "... or treat this as a
  change-detection layer rather than a contest" had exempted a one-date
  offer). `actions`: a list, a ranking or the windows said to be saved in,
  or listed in, a named file must be one a tool names as holding a list by
  its key (`differing_path`, `labels_csv_path`); `review_set_evidence.json`,
  which `evidence_detail_path` names and which holds evidence text only, is
  no longer read as a list by the "review" in its name (round 8, B8/cluster
  runs 2 and 3), and only a string that is a path counts as a file a tool
  wrote or names (no whitespace, or a rooted path): the review tools'
  `listing_note`, a note that names `review_set_evidence.json` in prose
  under a key that names a listing, had made that file a list's. A list or
  the windows right before a file ("the full ranked list is in <file>", "all
  3,807 differing windows are in <file>") is read as the claim too.
  `direction`: a `concentration` violation calls row band 0 the northmost
  only when the fact's `row_order` runs north to south (a fact without the
  key is read as north-up), and with row 0 south a compass place is not read
  against row band 0. The detectors read a sentence in time linear in its
  length: quotes, negations, clauses and offers are found once per sentence
  and looked up by bisection, where the review-set and spatial detectors had
  searched the whole clause once per match (14.7 s for one 130 KB listing
  line with the id emitted; now well under a second with every id emitted).
  On the recorded answers of rounds 6 to 8, with each id emitted wherever it
  could apply, the detectors flag 43 sentences: 37 hold a confirmed audit
  finding (in two, the finding is another claim of the same sentence), 3 a
  finding the verifiers added, and 3 none, which read as true catches the
  audits did not record; none is a false alarm. `actions` flags 5 sentences,
  all confirmed findings (3 at c3d2e28). With round 8's 30 runs replayed on
  these tools (every recorded call, Studio served from the run's recorded
  responses; 5,508 of 5,508 numbers at shared paths reproduced), an answer
  made of the results' own facts, must_state and notes raises no violation
  in any run, and the checks flag 14 of round 8's 17 material findings in
  the recorded answers; the other three are a margin ratio of the 50 listed
  windows given for the 10 shown, a dataset name the tools no longer return,
  and weakest classes named by eye, which `weakest_classes` now ranks.
  Rounds 6 and 7, checked against their recorded results, gain and lose no
  flag against c3d2e28.
- **exp86 round 1's tool faults** (the trial's diagnosis, 24 September 2026):
  - `olmoearth_plan_label_sample` held a Studio result's grid to 16 in
    silence (20, 30 and 40 all gave the same 173 valid windows) and crashed
    on the `[rows, cols]` form its schema advertised. The grid is N or
    `[N, N]`, 2-16, stated in the schema, and the result's `sampling` gives
    the grid used, the grid asked for and whether it was capped
    (`olmoearth_review_set_from_result` states its cap the same way).
  - Its budget error said "a finer grid" at the cap. It now states the
    ceiling ("a Studio result sampled at 16x16 = 256 points has 173 valid
    windows here, so at most 173 labels can be planned from it") and offers
    only what works: a smaller budget, a finer grid only below 16 and only
    when 256 points can reach the budget, or a direct model run through
    `olmoearth_scores_from_file`.
  - The second failure of a tool with the same error in a run (numbers
    aside) replaces the generic hint with: stop retrying, tell the user the
    limit, answer with what you have (`same_error_count`).
  - Measured figures left the tool descriptions, which reach every model
    call: `olmoearth_compare_review`'s "51-70%" (quoted as a finding in a run
    that never called it; its `caveats` keep it), `olmoearth_review_set`'s
    "all 24 tasks", `olmoearth_review_budget_ceiling`'s worked example and
    `olmoearth_area_of_applicability`'s "13/27 scenes, sign p=1.00" (now in
    its output's `error_ranking`). A test fails any description that states
    one.
  - `olmoearth_compare_results` reports `n_cells` beside `samples_requested`,
    a pair's `n_cells_compared` and `n_cells_dropped`, and a `sampling_note`
    (the answer had read 72 samples of a 6x6 grid as 72 cells).
  - `olmoearth_certify_zone` gives a `verdict` and says that a level whose
    `upper_bound` is below alpha is not thereby certified: certification is
    the package's exact test under the rule (every brief 6 answer claimed
    zones at alpha 0.10-0.15 where the package certifies none).
- **exp86 round 2's tool faults** (the trial's diagnosis, 25 September 2026):
  - `olmoearth_certify_zone` gives `levels_tested`: the number of levels and
    their coverages, the per-level delta of each rule over them (bonferroni's
    written out, e.g. "delta/18 = 0.1/18"), how many were accepted, whether
    any certifies, and why these levels (`min_labels_to_certify`, `c_min`).
    Every output, refusals included, says certification needs
    `design='random'`. Round 2's only model number was a "delta/18" the model
    counted itself, and one answer recommended a confidence design for
    certification. The smallest alpha that would certify is not given: the
    levels tested, and bonferroni's divisor, change with alpha.
  - `olmoearth_compare_review`'s `date_a`, `date_b` and `labels_date` state
    "YYYY-MM-DD, or a YYYY-MM-DD/YYYY-MM-DD period; not a bare year or
    month", as `oe_inferencex.compare.dates_reading` reads a string; a bare
    year or month is refused with the interval that means it. Every brief 3
    cluster run had passed `'2023'` first.
  - `margin_summary` (`olmoearth_review_set`,
    `olmoearth_review_set_from_result`) labels its fields (`lowest_margin`,
    `median_margin`, `highest_margin`, `margin_at_budget_cut`), gives the
    listed and unlisted windows' margin ranges, and says the median is not a
    lower end; an answer had read the median as the unlisted windows' floor.
    Breaking: `min`, `median`, `max` and `cut_at_budget` are gone.
  - A test drives the round-1 harness fixes end to end (the stop-retrying
    hint and the forced answer at the turn cap) through the loop, the CLI and
    the web bridge; round 2 never triggered them.
- `pyyaml` is a declared dependency: without it `olmoearth_rslearn_compose`
  returned no YAML.
- **No-data entered the statistics of every grid sampler.** Studio's
  pixel-value returns no-data as a value (`raw_value: -1.0` beside
  `regression: {min_value: 0.0, max_value: 1.0}`), and only `None` was
  dropped. In the trial, 11 of 36 points were `-1` in both maps and
  `olmoearth_compare_results` reported correlation 0.946 ("karst in largely the
  same places") where the valid points give -0.017 (agreement 0.306 -> 0.0).
  One helper, `analysis.raster_compare.band_is_nodata`, now decides no-data (a
  missing or NaN value, the model's `wizard_answers.nodata_value`, a value
  outside the band's declared range, a classification band with no class) in
  `olmoearth_compare_results`, `olmoearth_compare_group`,
  `olmoearth_ensemble_uncertainty`, `olmoearth_trace_shifts`,
  `olmoearth_review_set_from_result`, `olmoearth_pixel_value` and the web UI's
  difference scan (`/api/pixel-value` returns `value: null, nodata: true`);
  every sampling result reports `n_nodata_dropped`. The trial's 36 value pairs
  are a regression test that keeps both numbers reproducible.

## [1.4.0] - 2026-08-14

### Added
- **`olmoearth_trace_shifts` — timeseries shift tracing** (skill #5 family).
  Traces how ONE model's estimates shifted across 3-8 dated prediction results
  over the same area: discovers each result's date itself (its prediction's
  `start_time` — result records carry no date), orders the series
  chronologically (loud given-order fallback when a date is missing, since a
  silently wrong order flips every shift sign), samples every result on one
  shared grid, and returns per-step stats (Pearson correlation / mean delta /
  agreement — the pixel-correlation trace), per-point trajectories (net
  change, OLS slope, largest step + its dates, top shifted points), and
  legend-calibrated shift sizes (`value_range` from the project's annotation
  form when known, observed range otherwise, provenance-labeled). Categorical
  results get class-transition counts instead. New `analysis/trace_shifts.py`
  (pure Python, index-based) + `tools/trace_shifts.py`; fills the deliberately
  unclaimed seam between `olmoearth_compare_results` (2 results),
  `olmoearth_compare_group` (N models), and `olmoearth_change_detect`
  (ready-made series). Honest framing: estimate movement, not verified ground
  change, not accuracy. Also extracted the verbatim-duplicated
  `_result_bbox` helper from `tools/predict.py` + `tools/uncertainty.py` into
  `analysis/raster_compare.result_bbox` (the new tool would have been its
  third copy).
- **Whole-agent eval scenario seeds** (`evals/agent/`): six rubric-weighted
  scenarios (run-order, two soul-guardrail probes, compare-without-truth,
  memory-preference, simple-lookup-stops) + the scenario/rubric format README.
  Data-first prep for the planned rubric-eval runner + LLM judge; no runner yet.
- **Production-posture docs:** README now documents `OLMOEARTH_EGRESS=enforce`
  as the recommended setting for any deployment beyond a personal laptop
  (default stays `audit` so a first run never breaks) — closing the open
  follow-up from the 2026-06-10 security review.

- **Soul as a versioned artifact.** The system prompt moved out of a Python
  string literal in `harness/agent.py` into `harness/soul.md`, a markdown file
  with an explicit `## Guardrails` section separated from workflow rules —
  editing behavioral boundaries is now a reviewable markdown change, not a code
  change (Shippy-style soul/skills/config anatomy, after Ai2's "What building
  Shippy taught us about building agents"). `harness/soul.py` loads it;
  `OLMOEARTH_SOUL_PATH` points at an alternative soul file (a broken path falls
  back to the packaged soul rather than yielding a boundary-less agent).
- **Tool-argument validation at dispatch.** `ToolRegistry.dispatch` now
  validates model-emitted arguments against the tool's declared JSON Schema
  (`tools/validate.py`: `required`, `type`, `enum`, nested objects/arrays;
  deliberately lenient on undeclared extras and str-coercible scalars) *before*
  the handler runs. A malformed call comes back as a self-documenting rejection
  naming the bad argument plus an `expected_arguments` sketch and a recovery
  hint, instead of a bare `KeyError` from handler internals; handler exceptions
  now also carry the tool name and a recovery hint.
- **Oversized tool results spill to disk instead of the context window.**
  `harness/spill.py`: a tool result whose JSON exceeds
  `OLMOEARTH_TOOL_RESULT_SPILL_BYTES` (default 20 000; `0` disables) is written
  whole to `<workspace>/tool_results/` (confined under the `security/paths.py`
  workspace root) and the LLM receives a compact envelope — preview, structural
  sketch, saved path, guidance — so one big `fetch_results` payload can't eat a
  16k-token local-model context. The webui event stream and the provenance log
  keep the full result.
- **Eval regression gate.** `evals/skillopt/scripts/regression_gate.py` diffs a
  new `eval_skill.py` `summary.json` against the committed baseline
  (`evals/skillopt/baselines/*.json`, seeded from `RESULTS.md`) and exits
  non-zero on any hard/soft drop, printing a score-change report — a
  skill/model change that regresses the suite doesn't ship. `--update-baseline`
  promotes an accepted improvement (all four shipped via #147).
- **Cross-thread preference memory** (Shippy roadmap: "carry persistent facts
  ... and apply them automatically"). New `harness/memory.py` stores durable
  user preferences (default project/area, preferred sources, reporting style)
  in `<workspace>/memory/preferences.json`; the bridge and CLI inject the
  stored facts into every run's system prompt, so "run it over my usual area"
  works in a fresh conversation. New core tools `olmoearth_remember` /
  `olmoearth_forget` let the model save a preference the moment the user
  states one (soul rule added). Injection-hardened: slug-only keys, values
  one-line and capped at 240 chars, 50-fact cap, and the prompt block labels
  the contents as data-not-instructions.
- **Opt-in model routing** (Shippy roadmap: "not every question needs a
  frontier model"). New `llm/router.py` classifies a brief as a simple
  read-only lookup vs a complex investigation (deterministic word-list
  heuristic; ties break complex). With a hosted backend selected and the new
  webui "Auto-route simple briefs to the local model" toggle on
  (`X-LLM-Route: auto`), the bridge demotes simple lookups to the free local
  model — demotion only, so a wrong "simple" verdict costs quality on one
  lookup, never money, and a user-forced skill is never demoted.
- **Optional Asta backend for litsearch.** `olmoearth_litsearch` accepts
  `source="asta"`: when Ai2's [Asta CLI](https://github.com/allenai/asta-plugins)
  is installed and authenticated, the search upgrades from metadata-only
  arXiv/OpenAlex to Asta's full-text ranked retrieval with AI relevance
  judgements and supporting snippets (new `analysis/asta.py`, Shippy-style:
  the agent invokes the deterministic CLI — argv, no shell — which writes its
  JSON artifact to the confined workspace). Fully detection-gated: without the
  CLI the tool answers with install guidance and the key-free sources keep
  working; the subprocess env is credential-scrubbed (agent Studio/LLM keys
  dropped, `ASTA_*` passed through). `OLMOEARTH_ASTA_BIN` /
  `OLMOEARTH_ASTA_TIMEOUT` configure the binary and wall-clock cap.

### Changed
- README/SKILLS.md aligned with the #147 features (soul artifact, dispatch
  validation, result spill, preference memory, litsearch `source="asta"`).
- `interrogate` `fail-under` set to the measured floor (81, was an
  unenforceable 90) so the pre-commit docstring gate actually gates; ratchet
  upward as coverage grows. `reviews/` (local session audit notes) is now
  gitignored — durable findings live in issues, not loose untracked files.

## [1.3.0] - 2026-06-10

### Security
- **Symmetric egress guard on the Claude backend.** `AnthropicLLM.__init__` now
  validates its endpoint (`egress.validate_endpoint`, capability auto-picked like
  `OlmoEarthLLM`) before binding the BYO key, instead of relying solely on the
  `serve.py` call site — so a malicious `base_url` can't exfiltrate the key/chat,
  consistent with the local-model client.
- **Path-traversal guard: model-controlled tool file I/O is confined to a
  workspace root.** `olmoearth_export_data` (`out_dir`), `olmoearth_nndm_cv`
  (`output_path`), and `olmoearth_negative_sampler` (`positives_path` /
  `candidates_path` / `out_path`) took file paths straight from the model and
  read/wrote them with no confinement, so a prompt-injected `../../` or absolute
  path could read or write arbitrary files under the operator's account (clobber a
  served `webui/js` module, plant a startup script, reflect a JSON file back into
  chat). A new `security/paths.py` (`safe_path`) resolves every such path under a
  single workspace root and refuses anything that escapes it. The root is
  `OLMOEARTH_OUTPUT_ROOT` if set, else `<cwd>/olmoearth_outputs` (deliberately not
  the CWD/repo, so a relative path lands harmlessly inside the workspace).
- **Egress redirect re-validation on the litsearch / HuggingFace fetchers.** Both
  `analysis/litsearch.py` and `analysis/automate.py` validated only the *initial*
  URL, then followed redirects automatically — an allowlisted host (arXiv /
  OpenAlex / HF datasets-server) could 3xx-redirect the request to an internal /
  cloud-metadata address with no second allowlist check. The clients now use
  `follow_redirects=False` and re-run `egress.validate_endpoint` on every `Location`
  (bounded hops), so a redirect to a non-allowlisted host is blocked in `enforce`
  mode. `ARXIV_API` is also switched to `https://export.arxiv.org` so the hop can't
  be tampered with in transit.
- **The LLM client now routes its endpoint through the egress guard.** Studio,
  litsearch, and HF calls were already validated via `security/egress.py`, but
  `OlmoEarthLLM` built its OpenAI-compatible client straight from `LLM_ENDPOINT`
  with no check — so a malicious/misconfigured endpoint could exfiltrate the
  conversation (and a hosted-provider key). `OlmoEarthLLM.__init__` now validates
  the endpoint: loopback → the `llm-local` capability, otherwise the `llm-cloud`
  allowlist (`api.anthropic.com` / `api.openai.com` / `generativelanguage.googleapis.com`).
  Consistent with the rest of the guard — audit-logs by default, blocks under
  `OLMOEARTH_EGRESS=enforce`. SSRF to the cloud metadata IP (`169.254.169.254`) is
  blocked in enforce mode.

### Added
- **Multi-group compare for skill #4 `olmoearth-predict`:** new
  `olmoearth_compare_group` tool — quantitatively compare 2-6 prediction
  results (different models) over the extent shared by all of them, with no
  ground truth. Samples every raster on the same grid (pointwise pixel-value)
  and returns a **pairwise matrix** (the two-way agreement/difference stats
  for every pair, via the existing `compare_numeric`/`compare_categorical`)
  plus an **ensemble consensus**: the fraction of cells where ALL models
  agree (within tolerance, or same class), mean spread / ensemble std
  (regression) or unanimity + mean majority share (classification), the most
  divergent pair, and the most-contested grid points as lon/lat hotspots
  (capped at 5 so the tool result stays context-light). Cross-model only by
  design: exactly TWO results route to `olmoearth_compare_results` (richer
  two-way stats + temporal mode), and one model across MULTIPLE DATES routes
  to `olmoearth_change_detect` (trajectory) — the tool description encodes
  both boundaries. Group logic is pure
  (`analysis/raster_compare.py: intersect_bboxes / compare_group_numeric /
  compare_group_categorical / compare_group_narration`); the default grid is
  3x3 because each result adds grid^2 slow Studio pixel-value calls.
- **Standalone `olmoearth_pixel_value` for skill #4 `olmoearth-predict`** (#92).
  Exposes the existing `StudioClient.pixel_value` as a first-class tool: reads one
  prediction result's model output at a single user-supplied lon/lat (`raw_value`
  for a regression layer, the class for a categorical one), surfaces every band's
  value, and returns `available: false` with a reason off-raster instead of
  raising. Each value is paired with the exact point queried — not an interpolated
  raster read and not validated truth (use skill #7 for accuracy). Closes the
  "pixel-value follows" gap; feature-search remains honestly unbuilt.
- **Ensemble-disagreement confidence for skill #9 `olmoearth-uncertainty`** (#92),
  delivering the catalog's previously-aspirational "repeated pixel-value"
  confidence — honestly. `prediction_confidence` (pure, in `analysis/uncertainty.py`)
  turns per-point repeated/ensemble draws into dispersion stats (std /
  coefficient-of-variation / range for regression; majority class / vote fractions /
  normalized Shannon entropy / margin for categorical) folded into a confidence in
  [0,1]; `olmoearth_ensemble_uncertainty` (live) samples **two or more distinct
  prediction results** on a shared-extent grid (Studio pixel-value) and treats the
  >=2 values at each point as ensemble members, so the variance is real — never a
  fake re-read of one deterministic result. Explicitly labelled epistemic
  uncertainty (model self-consistency), NOT a calibrated correctness probability;
  pair with the AOA OOD flag.
- **QGIS provider files + an honest COG recipe for skill #11 `olmoearth-qgis-bridge`**
  (#92). `olmoearth_qgis_bridge` now also returns a ready-to-drop QGIS `.qlr`
  layer-definition (`build_qlr`) and a GDAL_WMS/XYZ descriptor (`build_gdal_wms_xml`)
  pointing at the authenticated tile endpoint, with the Bearer key kept out of the
  file (an unresolved sentinel + an `authcfg` note). Because the agent is GDAL-free,
  a true Cloud-Optimized GeoTIFF is returned as a user-run `gdal_translate -of COG`
  recipe + rationale (`cog_recipe`), not a fabricated file. The credential-bearing
  files are gated on the Studio egress allowlist (`egress.check_endpoint`), so a
  model-controlled (e.g. prompt-injected) non-Studio `tile_urls`/`base_url` host is
  refused rather than baked into a `.qlr` that would exfiltrate the user's key.
- **Skill #17 `olmoearth-rslearn` — rslearn for non-experts.** Four torch-free
  in-repo tools that let a domain scientist who doesn't know rslearn still set up a
  correct OlmoEarth/rslearn experiment: `olmoearth_rslearn_recommend` maps a
  plain-language research goal to a complete, *explained* setup (task, data layout,
  `encoder -> decoder -> head`, task knobs, fine-tune schedule), and
  `olmoearth_rslearn_validate` catches the config errors rslearn only surfaces hours
  into a run (encoder-dim vs decoder `in_channels`, `out_channels` vs `num_classes`,
  task vs label-type, missing bands, the Faster R-CNN background-class +1); `olmoearth_rslearn_compose` emits a full valid finetune `model.yaml` (MultiTaskModel + the right decoder/head, mirroring the proven data-prep template); and `olmoearth_rslearn_diagnose` turns a failing `prepare`/`ingest`/`materialize` run into plain-English fixes. Logic in
  `analysis/rslearn_advisor.py` mirrors rslearn's tasks/models/config schemas as data
  (no torch import; verified against the rslearn repo with a version pointer). This
  also promotes the previously-uncatalogued vendored `olmoearth-rslearn` SKILL.md to
  catalog row #17, so the catalog is now **17 skills** (was 16); the slash menu, cards,
  and `forced_skill` allow-list pick it up automatically.
- **Multi-source fusion for skill #17 `olmoearth-rslearn`** (a refinement of `compose()`,
  not a new skill). `olmoearth_rslearn_compose` and `olmoearth_rslearn_recommend` now take
  `modalities` (2+, e.g. `['sentinel2','sentinel1','dem']`); a new `recommend_fusion()`
  picks a **pre / mid / post** strategy with a plain-English why, and `compose()` emits the
  config. `mid` (the OlmoEarth-native default) emits a **valid multi-input `model.yaml`** —
  one OlmoEarth encoder fed every modality, fused internally via cross-modal attention (the
  decoder's channel contract is unchanged); `cross_attention` returns a grounded
  `rslearn.models.xatt_fusion.CrossAttentionFusionExtractor` skeleton (used automatically when
  a modality such as DEM isn't in `OlmoEarth.MODALITY_NAMES`); `pre` and `post` return input-
  concat caveats and a train-per-modality-and-ensemble recipe. Per-modality bands/data-sources
  (`FUSION_MODALITIES`) are mirrored as torch-free data, verified against the rslearn clone
  (Sentinel-1 `vv,vh`; WorldCover `B1`; the five `MODALITY_NAMES` + their native flag). The
  emitted mid config round-trips through `olmoearth_rslearn_validate` with no errors. Tool count
  unchanged (still four). Closes #120.
- **Comparison `kind` (`cross_model` | `temporal`) for `olmoearth_compare_results`.**
  The two-raster compare (skill #4) now distinguishes *two different models over the
  same area* (`cross_model`, the default — "how much do they agree?") from *one
  model's output at an earlier (A) vs a later (B) date* (`temporal` — "net change over
  time, later minus earlier"). The numbers are identical; the *narration* differs. A
  new `compare_narration()` helper (`analysis/raster_compare.py`) emits kind-aware A/B
  labels, a one-line headline, and a framing caveat, and the tool returns them plus
  the comparison `kind` (the old `kind` data-type field is renamed `value_type`). The
  in-chat compare card, difference-scan caption, A/B legend, saved-comparison view, and
  overlay slider in `webui/js/viz.js` all branch on the kind (e.g. temporal shows
  "Change map (later − earlier)" with a *decreased / increased* legend instead of
  "Difference map (B − A) / A higher / B higher"); records without a `kind` fall back
  to `cross_model`, so saved comparisons stay readable. Closes #124.
- **Server-side forced-skill routing** (`forced_skill`): `POST /api/run` accepts an
  optional allow-list-validated `forced_skill` slug; `LeadAgent` appends a
  `FORCED SKILL:` directive to the system prompt, and the web UI sends a clean
  brief + the structured field instead of the client-side rewrite (`parseSkillSlash`
  replaces `applySkillSlash`). Prompt-steering is the one mechanism that works across
  all 17 skills and the local Qwen backend (which has no `tool_choice`); a Claude-only
  `tool_choice` hard-force tier was considered and deferred.
- **Result-raster legend data** (`reporting.qgis.build_legend`): a structured
  value→color legend (continuous ramp stops, or categorical class entries) so a
  viewer can actually *read* a score/class map. It shares the same color ramp as the
  SLD, so a QGIS export and an in-chat legend agree; `olmoearth_qgis_bridge` now
  returns a `legend` alongside the `sld` (and takes optional `classes` for a
  categorical layer). The **web UI** renders it under each result raster as a
  value→color legend — a hover-readable gradient bar for a continuous score, or
  labelled swatches for a class map (`webui/js/viz.js` + `styles.css`), built
  against the Claude-design component spec.
- **Per-skill workflow stages** (`harness.workflow` + `GET /api/skills`): the source
  of truth for which run-pipeline stages (`load context → AOI → find model → submit →
  poll → fetch → report`) a skill's tools actually touch. A **run** skill (predict)
  gets the full pipeline; an **advisory** skill (e.g. `negative-sampler`, which never
  submits) gets just `context → report`. The **web UI** workflow rail is now
  **horizontal and relevance-filtered** — it renders only the stages a task will
  reach (so an advisory question shows 3 steps, not 7 with submit/poll/fetch stuck
  "pending"), from the forced skill's stages or inferred from observed tool calls
  (`webui/js/run.js`, mirroring `harness/workflow.py`).

- **Sidebar "switch bar" (Claude-desktop-style nav reorg).** The three stacked
  collapsible sections (Chats / Projects / Comparisons) are replaced by one
  always-visible **Chats** list plus a 3-way **segmented switcher**
  (Projects · Comparisons · **Areas** — folder / compare / map-pin glyphs) feeding a
  single secondary pane. Claude-style, the **active** segment shows its icon + label
  while the inactive ones collapse to icon-only (names also in `aria-label` + tooltip).
  Only one list shows at a time, the active segment is persisted, and Chats + the pane
  share the flexible middle so each scrolls independently. The switcher is a proper **WAI-ARIA tablist**:
  ←/→/Home/End keyboard nav with roving `tabindex`, and `aria-controls`/
  `aria-labelledby` linking each tab to its pane. *New chat* stays pinned
  at the top and the Studio-key chip at the bottom. **Areas is new**: every AOI
  flattened across *all* projects, each labelled with its project (e.g. `Drawn AOI ·
  PA Karst Final`) and draggable into the chat like a tree area; since Studio has no
  single "all areas" endpoint it fans out one fetch per project and **fills in
  progressively** (cached) with a "loading N more projects…" footer rather than
  blocking. Projects/Areas show a *Connect your Studio key* nudge with no key;
  Comparisons shows a *save a pair* nudge. Honors `prefers-reduced-motion`; the
  240px rail collapses to the existing hamburger drawer on mobile. Built from the
  Claude-design sidebar spec (`webui/index.html`, new `webui/js/switcher.js` +
  `webui/js/areas.js`, `webui/js/{projects,compares,main}.js`, `styles.css`). The
  Projects list keeps its shallow drill-down tree (the designer's named alternative
  to flatten-into-the-main-panel) so drag-to-attach of results and areas still works.

### Changed
- **Skill #2 `olmoearth-studio-job-config` rewritten to Studio's real "new model"
  wizard** (submodule bump to `2imi9/OlmoEarth-Skills@6839f1a`). The skill now
  walks the full wizard in order (model name -> model type -> foundation model ->
  label field -> training data -> data split -> temporal context -> image
  sources -> surrounding area -> review): the patch ("surrounding area") default
  is corrected 320 -> **640 m** (the wizard's recommendation for most tasks;
  160/320 m for fine-grained targets or <1,000-label datasets, 1280 m for
  large-region labels or detection); the missing **Model name**, **Training data**
  (full vs a proportion) and **Data split** (Spatial recommended vs a metadata
  field, + a new `references/data_split.md`) steps are added; model type carries
  the wizard's fine-tuned-vs-embeddings framing; and the skill content now states
  the rule the #142 routing fix enforces — **Studio trains on Ai2's compute**:
  never ask about the user's local hardware, never reroute Studio model creation
  to local training. `scripts/recommend.py` emits + validates the new fields with
  the 640 m default and a <1,000-label step-down to 320 m (all 14 presets
  round-trip its validator). The SkillOpt jobconfig benchmark was regenerated
  from the updated oracle (`evals/skillopt/`, incl. a neutral patch placeholder
  in the rollout prompt): the previous skill scores 0.357 hard on the faithful
  targets, the rewrite **0.714 hard / 0.906 soft** (local Qwen3.6 target,
  temperature 0) — recovering the optimized level. SKILLS.md #2 and the registry
  row updated to match.
- **Re-recorded the README walkthrough demo** against the current polished UI --
  the lead GIF/MP4 had been frozen at the pre-polish v1.0.0 layout. Corrected the
  caption accordingly (recorded-on v1.0.0 -> **v1.2.0**). Both the basic
  walkthrough (`webui/demo/record_demo.py`) and the feature showcase
  (`webui/demo/record_showcase.py`) now reflect the shipped UI.
- **The feature-showcase demo now leads with the in-chat skill slash-command
  flow** -- the short highlight GIF opens on `/` -> filter to a skill -> Enter ->
  the routed brief; the workflow rail + difference-scan radar move into the full
  MP4 (`webui/demo/record_showcase.py`).
- **Lighter difference scans + pooled Studio connections.** The diff scan's default
  grid drops 7->5 (49->25 sampled cells, ~half the pointwise pixel-value calls), and
  the bridge now reuses one connection-pooled `StudioClient` per key for
  `/api/pixel-value` and `/api/tile` (`_pooled_studio`, closed on shutdown;
  `StudioClient.get_bytes` for tile bytes) instead of constructing + closing a client
  per request — connection hygiene that most helps the map's tile burst. **Note:** this
  does *not* speed up the pixel-value scan itself; a fresh pixel-value call still takes
  ~28-70 s, because the cost is Studio's per-point compute (+ proxy round-trip), not the
  TLS handshake — so the real lever is the smaller grid. (`webui/js/viz.js`, `serve.py`,
  `studio/client.py`)

### Fixed
- **litsearch no longer overflows the local model's context on multi-call
  sessions.** Live coverage-run finding: five `olmoearth_litsearch` calls (all
  ok) accumulated 18,278 tokens, exceeding the 16,384-token window, so the
  final synthesis failed with a 400 and no answer. Abstracts are now capped at
  500 chars (was 1,200 — the `url` is the citation surface, the first
  sentences carry the claim), both litsearch tool descriptions budget the
  model explicitly (2-3 searches per task; results accumulate in context;
  reuse retrieved records), and the served-context default in
  `docker/llama.compose.yml` is bumped to the verified one-big-slot
  configuration (`-c 16384 --parallel 1`, was `-c 8192` split across ~4 slots
  of ~2k tokens each — too small for a single SKILL.md). `docs/serving.md`
  and `PLAN.md` aligned.
- **"Configure a model" now routes to the Studio wizard, not local training.** A
  plain "set up / configure / build a model to map X" was landing on the rslearn
  tools (`olmoearth_rslearn_recommend`/`compose` — a local `model.yaml` + freeze
  schedule) or the embeddings `olmoearth_automate` flow, which asked the user
  about their **local compute** (CPU / Colab / GPU) — irrelevant on Studio, where
  Ai2 runs the training. Narrowed those tool descriptions to *self-run* training
  (defer to `olmoearth-studio-job-config` for Studio model creation) and added a
  system-prompt routing rule: a create/configure-model request loads the Studio
  job-config skill and walks the wizard, and never asks about local compute.
  Verified live: "map crop types in Iowa" / "detect solar farms in California"
  now route to the Studio wizard with no compute / `model.yaml` questions.
- **Tool-input failures no longer report as successful calls.** `olmoearth_run_python`
  (empty code) and `olmoearth_load_skill` (unknown skill) returned an
  `{"ok"/"error": ...}` dict, which the dispatch envelope wraps as
  `{"ok": True, "result": ...}` — masking the failure as a success in provenance
  and the agent loop. They now raise, so dispatch reports `ok=False` (the
  unknown-skill error carries the available names for retry).
- **Skill #11 honesty: removed the unbuilt "COG / WMTS / full-precision GeoTIFF /
  sidecar uncertainty raster" claims** (#92). The agent is GDAL-free and only has
  tile URLs, so it cannot itself export a COG. `SKILLS.md` (catalog row #11 + §11
  In/Out/What/Tools) and `registry.py` (#11 summary) now describe what actually
  ships — a `.qlr` + GDAL_WMS descriptor + SLD + a user-run `gdal_translate -of COG`
  recipe (noting a COG built from public tiles is a *visual* raster, not
  full-precision scores) — and the fabricated skill-local tool names
  (`cog_export`, `qgis_provider_xml`, `write_sld_style`) are corrected to the real
  `olmoearth_qgis_bridge` / `reporting/qgis.py` functions.
- **Skill #9 honesty: replaced the fictional `repeated_sampling` tool name and the
  "Repeated pixel-value" / bare "confidence map" claims** (#92) with the shipped
  `prediction_confidence` + `olmoearth_ensemble_uncertainty`, framed as ensemble
  disagreement across distinct results (real variance), not a deterministic re-read.
- **`forced_skill` is existence-checked, not just shape-checked.** `POST /api/run`
  validated the slug's *shape* (a `[a-z0-9-]` regex) but not that it names a real
  skill, so a well-formed but unknown slug (e.g. `totally-fake-skill`) was injected
  into the `FORCED SKILL: …the '<slug>' skill` prompt directive. The bridge now also
  checks membership against the actual catalog (`SKILLS` slugs, the same 17 the webui
  "/" menu offers) and ignores anything that isn't a real skill (`serve.py`).
- **Skill #17 doc accuracy: "two tools" → four.** `olmoearth-rslearn` has shipped
  four in-repo tools since #119 (`recommend` / `validate` / `compose` / `diagnose`,
  the last two gaining fusion via #120), but several docstrings/descriptions still
  said "two torch-free tools (recommend/validate)". Corrected the count and added the
  missing tools across `tools/rslearn.py`, `analysis/rslearn_advisor.py`,
  `skills/registry.py` (catalog #17), and `SKILLS.md`. Docs only, no behavior change.
- **Slash menu lists all 17 skills** (was capped at the first 8) — the `/` command
  palette is scrollable, so the cap only hid skills 9–17 (`webui/js/slash.js`).
- **In-chat result-block download buttons are right-aligned**, matching the
  comparison-modal footer: the leading "Open overlay" action stays left and the
  download buttons sit flush-right (`webui/styles.css`, scoped to `.result-viz`).
- **Generalized over-specific skill descriptions.** `olmoearth-baseline-compare` (#6)
  read as "OlmoEarth vs AlphaEarth" and `olmoearth-similarity` (#8) as "over OlmoEarth
  Base embeddings", but both tools are general — compare vs *any* baseline foundation
  model / kNN over *supplied* embeddings. Reworded the capability descriptions to
  "vs a baseline foundation model (e.g. AlphaEarth)" and "supplied embeddings (e.g.
  OlmoEarth Base)" across `registry.py`, `SKILLS.md`, `README.md`, the tool
  description, and the web UI card; AlphaEarth stays as the worked example + the
  science citation. Doc/wording only, no code change (closes #121).

## [1.2.0] - 2026-06-08

### Changed
- **Tool-calling order guidance in the system prompt.** `DEFAULT_SYSTEM_PROMPT`
  now includes an explicit "typical run order" recipe with dependency guards:
  load context -> draw/resolve an AOI -> discover a reusable `model_id` ->
  submit (only with `project_id` AND `area_id` AND `model_id` AND a time range)
  -> poll to completed -> fetch -> analyze/report last; plus "search before
  create" and "do not repeat a succeeded call." Helps the model (especially the
  small local one) avoid out-of-order calls (submitting before discovering a
  model, fetching before completion) and redundant loops. No behaviour change to
  the harness; prompt-only.
- **Skill catalog consolidated 19 -> 16.** Three pairs of catalog entries were
  each one capability split across two rows; merged and renumbered:
  - **#1 `olmoearth-data-prep`** <- the former `olmoearth-studio-upload` (#1) +
    `olmoearth-rslearn-config` (#2): they were two rows for the single vendored
    `olmoearth-data-prep` SKILL.md package (now the vendored set = the 3 actual
    `vendor/olmoearth-skills` dirs).
  - **#3 `olmoearth-embeddings`** <- the former `olmoearth-embeddings` (#4) +
    `olmoearth-automate` (#17): one embeddings-vs-fine-tune decision with two
    facets (vendored guidance + notebook, and the in-repo `olmoearth_automate`
    one-call tool).
  - **#5 `olmoearth-change-detection`** <- the former `olmoearth-change-detect`
    (#6) + `olmoearth-latent-change` (#19): change detection with two engines
    (in-process Studio multi-date trajectory + out-of-process JEPA latent
    residual).
  No tool was removed and the vendored skill index is unchanged, so the agent's
  routing surface (tool specs + skill index) is identical -- the merge is a
  catalog/documentation consolidation, not a behaviour change. Propagated across
  `registry.py`, `SKILLS.md`, `README.md`, the web UI (`skills.js` cards +
  `index.html` counts), `docs/CANON.md`, `PLAN.md`, every tool/analysis
  docstring, and a live-regenerated `docs/SHOWCASE.md`.

### Added
- **Web UI visual-polish pass** -- a cohesive loading & busy-states kit
  (spinners, skeletons, progress pill, button-busy state), a **difference-map
  radar-scan animation** (sweep beam + cells igniting as they resolve, replacing
  the plain pop-in), a polished **in-conversation result block** (stat chips +
  map/chart + caption + download bar as one card), and a **consecutive-workflow
  indicator** rail above "Reasoning & tools" that tracks a prediction pipeline's
  stages. All plain CSS against the existing `:root` tokens (no build step, no new
  deps), honoring `prefers-reduced-motion`. Plus a feature-showcase demo
  (`webui/demo/record_showcase.py` -> `olmoearth-agent-showcase.{gif,mp4}`).
- **In-chat skill slash-commands**: type `/` in the composer to call a specific
  skill directly (like Claude Code's `/commands`) -- a filterable menu of the 17
  skills, arrow-nav + Enter/click to select. The brief is then routed to that
  skill (the displayed message stays as typed). Client-side brief-rewrite in
  `webui/js/slash.js`; a deeper server-side forced-skill mode is future work.
- **Result visuals in the conversation + downloadable artifacts**: a tool
  result's visual (map / chart / compare / difference map) now renders in the
  conversation flow instead of inside the collapsed "Reasoning & tools" (the raw
  chips + JSON stay there). Chat artifacts are downloadable: a difference
  comparison offers **Download JSON / CSV** (the real stats + sampled grid), a
  qgis-bridge result offers **Download .sld**, and a case-narrative offers
  **Download report (.md)**. Client-side Blob downloads (`js/download.js`); no
  server round-trip.
- **Comparisons panel**: a sidebar section (parallel to Chats and Projects)
  that **stores** two-raster comparisons. A completed difference scan offers
  "Save comparison"; the saved record (real stats + the sampled diff grid,
  captured straight from the live scan -- never fabricated) is persisted to
  `localStorage` and re-openable in a modal that redraws the stored difference
  map and its stats with no re-fetch. `js/compares.js`; decoupled from the
  scan via an `oe:save-comparison` DOM event.
- **More in-chat charts + auto difference map + a result cache**: (1) skill #4
  `olmoearth_compare_results` now auto-renders a stat card (correlation,
  agreement, mean |diff|, RMSE) **and** kicks off the difference-map scan inline
  -- no button needed. (2) skill #7 `olmoearth-baseline-compare` renders grouped
  metric bars (model A vs B + overall winner); skill #8
  `olmoearth_classification_metrics` renders overall + per-class F1 bars. (3) the
  bridge now caches proxied tiles and pixel values in-memory (LRU, key-namespaced
  + SSRF-guarded), so panning a result map or re-running a difference scan over
  overlapping cells is instant instead of re-fetching through the proxy.
- **Difference-map scan (in chat)**: comparing two result rasters now produces a
  visible **difference output**, not just numbers. A "Scan difference map" button
  on a two-raster view samples both rasters on a grid over their shared extent
  (via a new `GET /api/pixel-value` proxy) and paints each cell by `B - A`
  **progressively as it scans** -- blue where A is higher, pink where B is higher,
  opacity by magnitude -- then reports mean |diff|, correlation, and agreement.
  Pure client-side (`js/viz.js renderDiffScan`); pointwise, so it's a grid
  estimate (default 7x7), shown live.
- **Quantitative two-result comparison** (`olmoearth_compare_results`, skill #4):
  compare two prediction results numerically with **no ground truth**. It
  samples both rasters on a grid over their shared extent (pointwise
  `pixel-value`, new `StudioClient.pixel_value`) and returns model-vs-model
  agreement -- mean / mean-absolute difference, RMSE-between-models, Pearson
  correlation, and a threshold agreement fraction (regression) or class
  agreement (classification). This fills the gap behind the side-by-side raster
  preview: previously the agent could only describe two rasters or recommend a
  manual GIS overlay, because the accuracy tools (`olmoearth_classification_metrics`)
  need labels. Pure-Python stats in `analysis/raster_compare.py` (no GDAL/numpy);
  a system-prompt rule routes "compare these two results" to it. Verified live
  on two karst runs (correlation 0.997, 96% agreement within 0.1).
- **In-chat result visuals**: tool results now render an inline visual, not just
  text. Raster/tile results (predict / fetch-results / qgis-bridge) render on a
  **Leaflet map fit to the raster's extent** (so the picture actually shows,
  rather than being an invisible speck at world zoom) over an OpenStreetMap
  basemap; two-or-more rasters render **side by side** to compare. The extent
  comes from a new `GET /api/results/{id}/extent` (the result's
  `result_metadata.geometry` bbox); maps appear immediately and snap to extent
  as it resolves. **Dragging prediction results** from the sidebar into the
  composer previews them as rasters above the input -- pull two in to compare
  them side by side before sending (the original "compare two rasters" case).
  Studio tiles are auth-gated
  and browser `<img>` requests can't carry a header, so a new bridge tile-proxy
  (`GET /api/tile/{z}/{x}/{y}?src=...`) adds the Bearer key server-side and is
  hard-restricted to the Studio host (no open relay / SSRF). Change-detection
  results render an **SVG trajectory chart**. New `webui/js/viz.js` (the result
  visualizer) + `webui/js/leaflet.js` (a Leaflet loader now shared with the AOI
  draw widget); pure-SVG charts, no new deps. Resolves the gap where comparing
  rasters showed nothing visual in the chat.
- **AOI draw-in-chat**: select an area of interest by **drawing** it on a map
  in the chat instead of typing a bbox. A composer "Draw AOI" button opens a
  Leaflet map (OSM basemap, rectangle/polygon tools; loaded from CDN, no build
  step); the drawn polygon is stored as an **OlmoEarth Studio area**
  (`POST /api/areas` -> new `StudioClient.create_area`, verified against the
  live `POST /api/v1/areas`) and attached to the next brief carrying its
  `area_id` (for `olmoearth_submit_prediction`) and bounding box (for
  bbox-based skills such as #19 `olmoearth-latent-change`). The agent surfaces
  the widget itself: a new foundational tool `olmoearth_request_aoi` lets it
  ask the user to draw an area when a task needs one and none was given,
  rendering an inline "Draw the area" button that seeds a follow-up turn.
  The modal can also **reuse a saved area**: pick one of the project's
  existing areas (`GET /api/areas/{id}`) to render it on the map and attach
  its `area_id` without creating a duplicate. Saved areas also appear as an
  **"Areas" branch under each project in the sidebar tree** and can be
  **dragged into the chat** to attach (like prediction results). The dev
  bridge now serves the static web UI with `Cache-Control: no-cache` so
  edited (no-build) ES modules are never served stale.
  Pure-Python geometry helpers in `analysis/aoi.py` (GeoJSON Polygon/
  MultiPolygon validation + bbox; no GDAL/numpy, agent stays torch-free).
  Resolves issue #59 (map/draw AOI) and the concrete sub-piece of #90.
- **Skill #19 `olmoearth-latent-change`** (catalog 18 -> 19): a JEPA latent-prediction
  change detector on **frozen** OlmoEarth embeddings, shipped as a thin
  **out-of-process** link to the new standalone heavy-ML repo
  [`2imi9/olmoearth-jepa-change`](https://github.com/2imi9/olmoearth-jepa-change)
  (PyTorch + CUDA, kept *out* of this torch-free agent). A lightweight head predicts the
  time-2 patch embedding from time-1; the prediction residual is the change score
  (I-JEPA, Assran et al. CVPR 2023). On the OSCD test split (frozen OlmoEarth-v1-Base)
  it beats the cosine baseline by **+0.22 F1** (0.25 -> 0.47) and ~3x AP, reaching
  unsupervised-SOTA-level **F1 0.54 label-free** (robust threshold), integrity-verified
  (9x chance, permutation control). A Phase-2 gate study found current general VLMs
  cannot deliver calibrated, localized raster change comparison, so the agent needs this
  calibrated pixel-level tool. **Catalog / skill-contract only here** (`SKILLS.md` #19);
  no heavy dependencies added. Complementary to #6 `olmoearth-change-detect` (Studio-API
  trajectory diff). Out = georeferenced heatmap GeoTIFF + %-area-changed + top-k GeoJSON.
- **Skill #18 `olmoearth-negative-sampler`** (catalog 17 -> 18): generates the
  missing negative / background class for a *presence-only* label set as
  buffered, spatially-thinned (optionally embedding-dissimilar) pseudo-absences,
  and writes a combined GeoJSON that round-trips back through the vendored
  `olmoearth-data-prep` audit -- clearing its negative-class check, which
  hard-FAILs a presence-only set. Resolves the documented dead-end where the
  audit detected the missing class but `--negative-class auto` was deferred
  upstream. In-process tool `olmoearth_negative_sampler`; pure-Python logic in
  `analysis/negative_sampler.py` (no GDAL), inverting skill #9's similarity
  ranking for the environmentally-dissimilar pseudo-absence selection (#75).
- **NNDM-LOO cross-validation for skill #8 `olmoearth-evaluate`** (tool
  `olmoearth_nndm_cv`): Nearest Neighbour Distance Matching Leave-One-Out CV
  (Mila et al. 2022) for an unbiased map-accuracy estimate over the actual
  prediction area, where the random-vs-spatial inflation check only flags the
  risk. Pure-Python `evaluation/nndm.py` (no GDAL; reuses `spatial_cv`'s
  haversine), a faithful port of R CAST's `nndm` verified against the reference
  algorithm (an exact hand-traced execution + the `predpoints==trainpoints`
  invariant from CAST's own tests). Reports the optimistic-bias before (LOO) vs
  after (NNDM) and writes the per-fold train/test/exclude indices inline or to a
  file. Closes the catalog's previously-aspirational NNDM-LOO claim (#92).

### Changed
- **Skill #18 `olmoearth-negative-sampler` -- accuracy refinement**: addresses
  the pseudo-absence accuracy concern (#92) with (1) an optional embedding-based
  **contamination guard** (`contamination_threshold`) that drops candidates
  resembling any positive too closely -- likely *unmapped positives* -- before
  ranking (guarding environmental contamination on top of the spatial buffer),
  and (2) a **quality report** on every result (nearest-positive-distance +
  similarity-to-positive stats + an honest caveat) so the negatives' credibility
  is inspectable and the thresholds tunable, rather than trusted blindly.

### Fixed
- **Web UI polish fixes** from post-review + a multi-agent doc<->code<->
  architecture audit (26 raw findings -> 7 adversarially confirmed): the
  consecutive-workflow rail no longer marks every stage done when a turn pauses to
  draw an AOI, and no longer wipes a failed stage at end-of-turn; the trajectory
  chart's axis labels sit in the gutters (no overlap or clipping); the compare
  card no longer renders the A/B legend twice; the **difference scan is
  reproducible** -- it retries flaky pixel fetches so the same raster pair yields
  the same numbers; the comparison-modal download buttons are right-aligned;
  README "18-skill catalog" -> **16**, with the omitted `js/` modules + bridge
  endpoints documented; `prefers-reduced-motion` now covers the remaining
  animations (modal/detail entry, typing dots, caret, tool-call pulse, flash);
  and a `--mint-hi` design token replaces a hardcoded hover color.
- **Skill #9 `olmoearth-similarity` catalog wording** corrected to match the
  implementation: it described "FAISS" but the tool runs an **exact brute-force
  top-K kNN** in-process (FAISS is the scale-up follow-up, per the analysis
  docstring). Aligned `README.md`, `SKILLS.md` (catalog row + #9 section, incl.
  removing a fictional `faiss_index_build` tool), `skills/registry.py`, and the
  `system:python` sandbox wording. Honest-results doc fix from the #92 audit.

### Security
- **In-process egress guard** (`security/egress.py`), informed by NVIDIA
  NemoClaw's network-policy model and SSRF guard
  ([`docs/nemoclaw-assessment.md`](docs/nemoclaw-assessment.md)). A
  per-capability host allowlist (`studio`, `llm-cloud`, `llm-local`, `litsearch`,
  `hf`) plus an SSRF block of private / loopback / link-local / cloud-metadata
  address ranges, validating every outbound endpoint before a request -- and any
  credential -- leaves the process. Wired into `StudioClient` (guards the
  `OLMOEARTH_BASE_URL`-overridable base URL before the Bearer key is bound), the
  hosted-LLM path in `serve._llm_for_request` (before the BYO key is handed over),
  and the litsearch / HF fetchers. Modes: `OLMOEARTH_EGRESS=audit` (default; log
  only, never breaks a deployment) / `enforce` (block, HTTP 403 on the LLM path) /
  `off`; `OLMOEARTH_EGRESS_ALLOW` allowlists a self-hosted Studio or LLM host.
  Decisions are recorded host-only in the provenance manifest
  (`ProvenanceLog.record_egress`).
- **Credential-scrubbed subprocess** for the opt-in `olmoearth_run_python` tool:
  the agent's Studio / LLM / cloud keys are removed from the child environment so
  executed snippets cannot read and exfiltrate them. Defence-in-depth, not a
  network sandbox; OS-essential env is preserved so it still launches. Advances
  the sandbox spec (#54).

## [1.1.0] - 2026-05-31

Post-1.0 checkpoint: two new skills (catalog 15 -> 17), the `make serve` cache
fix, and the sandbox / GDM / quick-start documentation pass.

### Added
- **Skill #16 `olmoearth-litsearch`**: arXiv + OpenAlex literature search and
  DOI / arXiv-id resolution, so the agent can ground EO/geospatial citations in
  real papers instead of world-knowledge or hallucinated links. In-process tool
  bundle (`olmoearth_litsearch`, `olmoearth_litsearch_resolve`), key-free
  (OpenAlex polite-pool `mailto` via `OLMOEARTH_OPENALEX_MAILTO`), deduped across
  sources; informed by a read of Google DeepMind's Science Skills (#62).
- **Skill #17 `olmoearth-automate`**: auto-decides embeddings vs fine-tuning for
  an EO task and proposes a config (model size, classifier head, embeddings
  notebook command, fine-tune schedule, Studio job-config hand-off). In-process
  tool `olmoearth_automate`; reuses the vendored `olmoearth-embeddings` decision
  table and can introspect a Hugging Face dataset (rows + classes) via the public
  datasets-server (#58).
- **`docs/science-skills-assessment.md`**: a multi-agent assessment of Google
  DeepMind's `science-skills` against this repo's skill/harness architecture. It
  re-confirms the bundle is off-domain (genomics / proteomics / chemistry) and
  that arXiv / OpenAlex literature search is the one transferable capability (now
  skill #16), and captures the report's 3-tier-test + LLM-autorater methodology
  as a target for the SkillOpt harness (#69).

### Changed
- **Quick start reworked around the live web UI** (`README.md`): the walkthrough
  leads with `make bridge` and the browser flow, and the LLM is positioned as a
  "local or cloud API" choice ("provider" wording became "cloud API") (#66).
- **README skill count corrected to 17** as `olmoearth-litsearch` (#16) and
  `olmoearth-automate` (#17) landed (badge, table, and prose) (#71).
- **`olmoearth-embeddings` (#4) and `olmoearth-automate` (#17) disambiguated**
  in `README.md` and `SKILLS.md` so the two Configure skills no longer read
  identically: #4 is the embeddings-vs-fine-tune *guidance* plus a runnable-
  notebook generator (you run the notebook); #17 is the *one-call* auto-decide
  + proposed-config tool (with optional Hugging Face dataset introspection) that
  reuses #4's decision logic (#73).
- **Corrected the `system:python` sandbox spec across docs** (`PLAN.md`,
  `AGENTS.md`, `SKILLS.md`): it is the opt-in subprocess (`OLMOEARTH_RUN_PYTHON=1`,
  `python -I`, no persisted state, geospatial stack not guaranteed), not a
  preloaded persistent interpreter (which is now a labelled design target) (#68).

### Fixed
- **`make serve` reuses the cached model** (`docker/llama.compose.yml`,
  `scripts/serve-llm.sh`): the compose now bind-mounts the host Hugging Face
  cache (`cygpath -m`-normalized on Windows) and offers opt-in `HF_PROXY` via
  `host.docker.internal`, so a second `make serve` loads the already-downloaded
  GGUF instead of re-pulling ~17.7 GB (#67).

## [1.0.0] - 2026-05-31

First tagged release: the agent runs all 15 skills live against OlmoEarth
Studio and a local or hosted LLM.

### Added
- **Multi-provider LLM backends**, selectable from the web UI: local Qwen3.6
  (default) plus bring-your-own-key **Claude** (native Anthropic SDK),
  **ChatGPT**, and **Gemini** (OpenAI-compatible). `GET /api/llm/models`
  autodetects each provider's current models; keys are forwarded per request
  and never stored server-side (#52).
- **Model-vs-embeddings labelling in the project tree**: each synthetic model
  node shows the model's real name and a type badge (Embeddings / Fine-tuned),
  resolved via the live (openapi-undocumented) `/models` endpoint (#51).
- **Per-skill example briefs** in `SKILLS.md` (a realistic prompt per skill).

### Fixed
- V1.0 readiness review: release the per-request hosted LLM client after each
  run (connection-pool leak); `/api/run` + `/api/llm/models` return clean
  4xx/5xx for bad input; web-UI subtab stale-closure, silent no-key->local,
  concurrent-send, and tree-retry edges; Studio client surfaces `errors`
  envelopes and paginates client-side filters.

### Changed
- **Relicensed from Apache-2.0 to the OlmoEarth Artifact License** (Ai2), matching
  `allenai/olmoearth_pretrain`: free use with restrictions (no military/defense/
  surveillance or extractive uses; cite Ai2 and propagate the terms downstream).
  Updated `LICENSE`, every source SPDX header (`LicenseRef-OlmoEarth-Artifact-License`),
  `pyproject.toml`, the README badge, `AGENTS.md`, and `CONTRIBUTING.md` (#64).
- **README: dropped the Google "Earth Agent" / AlphaEarth comparisons** and
  tightened wording (`README.md`): the tagline and intro no longer position the
  tool against another product, and skill #7 reads "vs. a baseline foundation
  model".
- **README made concise with collapsible sections**: the 15-skill catalog and
  the Stack table are now `<details>` blocks, so the page reads short at a
  glance and expands on demand.
- **Demo GIF now leads with connecting a Studio key** before the first question
  (`webui/demo/record_demo.py` reordered to key -> tree -> brief -> answer),
  matching the real flow; the GIF and MP4 were re-recorded.
- **ASCII-only typography across docs and code**: replaced every em dash, en
  dash, and horizontal bar (`U+2014` / `U+2013` / `U+2015`) with natural ASCII
  punctuation (commas, colons, periods, parentheses, or ` - `) across all
  in-scope Markdown docs, `src/` strings and docstrings, `webui/`, `scripts/`,
  and the `Makefile` (52 files). Number ranges and compound names became
  hyphens (`1-4`, `Meyer-Pebesma`); arrows (`->`) and code were left alone.
  Line counts are unchanged and 196 tests + ruff + mypy stay green. The
  vendored `olmoearth-skills` submodule and `evals/` fixtures are intentionally
  untouched.
- **README `AGENT` wordmark recolored to white** (`webui/assets/agent-tag.png`):
  the lettering goes from dark teal to white on the same OlmoEarth-pink pill,
  so it reads correctly on both light and dark GitHub themes. The pill shape,
  rounded corners, and transparent background are unchanged.
- **Agent-settings menu labels formalized** (`webui/index.html`): a consistent
  `Title (kind)` style for the Papers & resources links (the
  `Embeddings -> fine-tuning` arrow is gone), `Max steps per run` instead of
  `Max steps / run`, and a header on every section.
- **Landing example briefs follow the selected mode** (`webui/index.html`,
  `webui/app.js`): the "Run a prediction" / "Analyze results" / "Prep &
  configure" tabs now swap the suggested example briefs (not just the input
  placeholder), and selecting a tab reveals them.
- **Example briefs show by default** (`webui/app.js`): the landing reveals the
  suggested briefs on load instead of hiding them behind the toggle.

### Removed
- **Skill #16 `roger-annotation-bridge` dropped**: the planned Roger
  Studio → Studio labelset bridge is no longer part of the project, so the
  catalog is now **15 skills** (was 16). Removed its `SkillSpec` from
  `registry.py` and its section from `SKILLS.md`, and corrected the
  skill-count in `PLAN.md` / `README.md` / `docs/CANON.md` (C9). (Historic
  CHANGELOG entries that mention #16 are left as-is.)

### Fixed
- **Web UI project tree no longer 502s on expand** (`studio/client.py`,
  `serve.py`): `search_predictions` sent a `project_id` field that Studio's
  `/predictions/search` rejects with HTTP 422; it now drops that field and
  filters client-side, like the sibling `search_prediction_results`. The bridge
  also surfaces the real upstream status in the 502 detail instead of a generic
  "Studio call failed".
- **Studio reads retry transient failures** (`studio/client.py`): `load_context`
  (the projects panel) makes two Studio calls, so one transient 502/503/504 or
  timeout failed the whole connect (the panel showed "Couldn't load - HTTP 502"
  while the key was valid). Idempotent reads (GET and the search POSTs) now
  retry up to 3 times with backoff; creates are never retried.
- **"Show example briefs" button now works** (`webui/styles.css`,
  `webui/app.js`, `webui/index.html`): a `.examples { display: flex }` rule beat
  the `[hidden]` attribute, so the chips were always visible and the toggle did
  nothing. Added `.examples[hidden] { display: none }` and a Show/Hide label.
- **Text-emitted tool calls are now recovered (`llm/client.py`)**: when the
  llama.cpp server returns a tool call as plain text (Hermes-XML
  `<function=..><parameter=..>` or `<tool_call>{json}</tool_call>`) instead of
  via the structured `tool_calls` field (which it does without a matching
  `--tool-call-parser`, and intermittently even with one) the client now
  parses it back into a `ToolCall` and sets `finish_reason="tool_calls"`.
  Previously the agent loop mistook the markup for a final answer and silently
  skipped the call. Only triggers when the structured channel is empty, so a
  well-behaved server is untouched. 3 new tests.
- **Underspecified label-array tool schemas (`tools/baseline_compare.py`,
  `tools/evaluate.py`)**: `y_true` / `y_pred` / `*_pred` used
  `{"type": "array", "items": {}}` (items = any type). The grammar
  llama.cpp derives from that lets the model emit the integer arrays as `[]`
  or `[{"value": 1}, …]`, corrupting the call. Typed the items as
  `["integer", "string"]` (class codes or names) so grammar-constrained
  decoding produces clean arrays on the first try.
- **Circular import in `tools/skill_tools.py`**: the module-level
  `from olmoearth_agent.skills.loader import SkillLoader` formed a cycle
  (`skills.loader` → `skills/__init__` → `skills.registry` → `tools.skill_tools`)
  that raised `ImportError` whenever `tools.skill_tools` was imported before
  `skills.registry`. Moved the import inside `build_skill_tools` (lazy) so
  import order no longer matters.

### Added
- **Chat file upload** (`webui/index.html`, `webui/app.js`, `webui/styles.css`):
  attach files in the composer (button or drag-drop). Text files (GeoJSON,
  JSON, CSV, code, txt, md) and PDFs (text extracted client-side via lazy
  pdf.js) are appended to the brief as context; images are accepted but
  labelled "not readable" since the local model is text-only. Per-file and
  total size caps keep the prompt bounded. Frontend-only: the bridge and agent
  are unchanged.
- **Drag a Studio result into chat** (`webui/app.js`, `serve.py`): prediction-
  result nodes in the project tree are draggable; dropping one on the composer
  attaches the result as context (id, properties, format, tile URL). The bridge
  now forwards the result tile URL.
- **`docs/SHOWCASE.md` + `scripts/generate_showcase.py`**: a skills-in-action
  page where **all 15 skills, in catalog order, are driven by the live LLM**:
  each transcript is a captured run of the agent loop against the served
  Qwen3.6 backbone (brief → reasoning → function call → real result →
  answer), nothing fabricated. #5 (read-only) and #13 run against the live
  Studio API; #1-#4 load the vendored `SKILL.md` bodies via
  `olmoearth_load_skill`; #6-#15 are real computation. Falls back to a short
  note for #5/#13 when `OLMOEARTH_API_KEY` is absent. Linked from the README.
  Regenerate with `set -a; . ./.env; set +a;
  uv run python scripts/generate_showcase.py > docs/SHOWCASE.md`.
- **Skill #7 `olmoearth-baseline-compare` (`src/olmoearth_agent/analysis/baseline.py`)**:
  `compare_metrics` runs OlmoEarth vs AlphaEarth head-to-head on shared
  ground truth (reusing skill #8's `classification_metrics`): a per-metric
  table (accuracy / macro-F1 / mean-IoU) with deltas, a per-metric winner,
  and an `overall_winner`. `difference_raster` gives the cell-by-cell gap
  between two score layers (mean / mean-abs / max-abs difference + which
  layer is higher where). Tool `olmoearth_baseline_compare` (metrics
  always; difference raster when score layers are supplied). Substantiates
  an "outperforms AlphaEarth on transfer regions" claim (Ma et al.
  arXiv:2601.00857). **No live GEE connection / Earth Engine MCP**: the
  AlphaEarth side is data the user exported from the now-public GEE
  "Satellite Embedding" dataset, kept the repo's pure/no-deps design.
  13 new tests.
- **Skill #9 `olmoearth-similarity` (`src/olmoearth_agent/analysis/similarity.py`)**:
  `similarity_search` returns the top-K embedding vectors most similar to
  a query (exact brute-force kNN, cosine or Euclidean; FAISS-at-scale is
  the deferred follow-up). `geographic_prior_check` is the honesty guard:
  it **warns when the top matches cluster geographically near the query**,
  because then the "similarity" may reflect *location* (same region /
  biome) rather than genuine feature resemblance: the classic
  similarity-search failure mode (cf. NASA Earthdata Similarity Search;
  OlmoEarth Base wins 15/24 kNN tasks, arXiv:2511.13655). Tool
  `olmoearth_similarity_search` (optional `ids`, `metric`, and
  `query_coord` + `coords` to enable the geographic-prior warning);
  returns matches + a summary, no raw coordinates (rule §3.1). Pure
  Python; reuses `haversine_km` from the evaluate skill. 21 new tests.
- **Skill #10 `olmoearth-uncertainty` (`src/olmoearth_agent/analysis/uncertainty.py`)**:
  `area_of_applicability` implements the Meyer & Pebesma (2021, MEE
  12:1620) Area of Applicability: standardize (optionally
  importance-weight) the training features, compute each point's
  dissimilarity index (nearest-training distance / mean pairwise training
  distance), and flag points whose DI exceeds the training data's own
  outlier-adjusted threshold (`Q75 + 1.5·IQR` of leave-one-out DIs, as in
  the R `CAST` package) as **out-of-distribution**. `ood_flag` returns
  per-point flags + OOD fraction + verdict (within-AOA / partially-OOD /
  mostly-OOD). Tool `olmoearth_area_of_applicability`. The point:
  **softmax confidence is not OOD detection**: a model can be
  confidently wrong on data unlike its training set (AlphaEarth's
  documented transfer failure is exactly what AOA flags). Algorithm-
  agnostic, pure Python (no numpy); the repeated-sampling confidence map
  is the documented follow-up. 18 new tests.
- **Skill #11 `olmoearth-cloud-mask-audit` (`src/olmoearth_agent/analysis/cloud_mask.py`)**:
  `ensemble_disagree` summarizes where several aligned cloud masks
  (CFMask / s2cloudless / Sen2Cor / MAJA, or any others) agree vs
  disagree: agreement/disagreement rates, per-algorithm cloud fraction
  (which algorithm runs aggressive vs conservative), pairwise
  disagreement, and a vote histogram: it surfaces **disagreement, not a
  single ground-truth mask**, because algorithms diverge on thin /
  semi-transparent cloud (Skakun et al. CMIX, RSE 274:112990, 2022).
  `verdict_classifier` takes a model-error mask and returns a
  **bad-mask-vs-bad-model verdict** (cloud-mask-limited / model-limited /
  inconclusive). Tool `olmoearth_cloud_mask_audit` returns summary stats
  only, no per-pixel geometry (rule §3.1). Algorithm-agnostic, pure
  Python; the STAC + s2cloudless `fetch_cloud_masks` step (needs an AOI
  bbox + date, plus heavier deps) is the gated live-smoke follow-up.
  18 new tests.
- **Skill #6 `olmoearth-change-detect` (`src/olmoearth_agent/analysis/change_detect.py`)**:
  `enforce_min_3_dates` + `diff_layers` turn a dated series of per-date
  layer summaries (one `value` per prediction date, e.g. positive-class
  fraction or mean score over the AOI, from a skill #5 result) into
  trajectory metrics: per-step deltas, net change, the largest-change
  interval, a **reversal count**, and a trend label
  (increasing/decreasing/stable/oscillating). **Refuses fewer than 3
  distinct dates**: a two-date diff reports net change but cannot tell a
  steady trend from a reversal (a flood that peaked then receded reads as
  "no change"), so the skill enforces a 3+-date trajectory (`SKILLS.md`
  #6; Ma et al. arXiv:2601.00857). Trend/reversal logic runs on the raw
  deltas, so output rounding can never flip a sign. Tool
  `olmoearth_change_detect` composes skill #5 (`olmoearth-predict`); new
  `analysis/` package (home for the coming Analyze skills). 16 new tests.
- **Skill #12 `olmoearth-qgis-bridge` (`src/olmoearth_agent/reporting/qgis.py`)**:
  `resolve_xyz_url` (relative tile template → absolute QGIS XYZ URL,
  preserving `{z}/{x}/{y}`) + `build_raster_sld` (well-formed OGC SLD 1.0
  color-ramp, default 5-stop YlOrRd over a 0..1 score). Tool
  `olmoearth_qgis_bridge` turns a result's `tile_urls` into XYZ URLs +
  SLD + load instructions (Bearer-auth header note; key never embedded).
  Verified on real PA Karst tiles (resolved URL + valid 5-stop SLD).
  Loading in QGIS desktop is the user's confirmation step; COG export
  follows. 5 new tests.
- **Skill #13 reframed → `olmoearth-data-export`.** The original
  "external-data" (wire third-party GEE/OSM/USGS/NOAA MCPs) needed
  external MCPs and wasn't core; reframed to the self-contained "export
  our own Studio data, grouped." `olmoearth_export_data` writes the
  user's projects + predictions to JSON grouped by `project` (default)
  or `status`, curated to ids/names/statuses/times (no raw geometry).
  `src/olmoearth_agent/reporting/export.py` (pure helpers) +
  `tools/export.py`. Verified live 2026-05-28: 5 project files (12
  predictions linked) + 2 status files from the real account. 7 new
  tests. (`exports/` gitignored.)
- **CLI entrypoint: the agent is now runnable.** `olmoearth-agent
  "<brief>"` (and `python -m olmoearth_agent`) wire the LLM client,
  Studio client, default tool registry, and vendored-skill index into a
  `LeadAgent` and run a natural-language brief. `--show-trace` prints the
  tool-call trace + provenance count to stderr. `src/olmoearth_agent/
  cli.py` + `__main__.py`; `[project.scripts]` console entry. Verified
  live 2026-05-28: `olmoearth-agent "how many projects do I have?"` →
  load_context + search_projects → "You have 5 projects: …". 5 new tests.
- **Project write path verified + `get_project` / `delete_project`.**
  `StudioClient` gains `get_project` (`GET /projects/{id}`) and
  `delete_project` (`DELETE /projects/{id}` → 202; unwraps the deleted
  record nested under `record`). **Verified live 2026-05-28**: a
  throwaway project create → get → delete → 404 (cleaned up, no
  residue). New **double-gated** live test
  `tests/studio/test_write_live.py` (requires `OLMOEARTH_WRITE_TESTS=1`
  *and* `OLMOEARTH_API_KEY`, so it never writes by accident). This
  closes the last unverified core capability: the `POST /projects`
  write path. 3 new tests (2 unit, 1 gated live).
- **Skill #15 `olmoearth-case-narrative` (`src/olmoearth_agent/reporting/`)**:
  `build_narrative` (pure) assembles a stakeholder Markdown report from
  prediction results (tile URLs + properties) + the run's provenance,
  with a **freshness gate** that withholds and strikes through tiles
  older than a configurable window (so a disaster-response brief never
  shows stale imagery). Tool `olmoearth_case_narrative` reads provenance
  from `ThreadState`. Verified live 2026-05-28: agent produced a
  "PA Karst Demo" report. 7 new tests.

### Changed
- **Dropped NVFP4; serving consolidated on the 4-bit GGUF via llama.cpp.**
  NVFP4 (~20 GB) doesn't leave KV-cache headroom on a 24 GB card; the
  GGUF `unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS` (~17.7 GB) is the
  verified path. Swept every doc + code default: `README.md`, `PLAN.md`
  (scope / §4 / §6 roadmap / §7.1 / §8), `AGENTS.md`, `docs/serving.md`
  (rewritten around llama.cpp), `.env.example`, `llm/config.py` default
  + module docstrings, test mocks. `docker/vllm.compose.yml` →
  `docker/llama.compose.yml` (llama.cpp service).
- **Renamed LLM env vars**: `VLLM_ENDPOINT` / `VLLM_MODEL` /
  `VLLM_API_KEY` → `LLM_ENDPOINT` / `LLM_MODEL` / `LLM_API_KEY` (the
  endpoint is backend-neutral, no longer vLLM-specific). Verified live.

### Added
- **`docs/CANON.md`**: single source of truth for cross-document facts
  (model, serving stack, quantization, env vars, Studio API, skill
  count) plus a grep-based alignment protocol. Update a fact there
  first, then fix every reference. Prevents the doc drift that motivated
  this pass.
- **Skill #5 predict, result output path**: `olmoearth_fetch_results`
  (tile URLs / property names / file format for a prediction) and
  `olmoearth_get_prediction_result` (by result id). `StudioClient` gains
  `get_prediction_result` + `search_prediction_results`. The API's
  `PredictionResultSearchRequest` has no `prediction_id` filter
  (openapi v0.1.0), so `fetch_results` scans and filters client-side.
  Verified live 2026-05-28 against PA Karst results (tile URLs like
  `/api/v1/prediction-results/{id}/tiles/{z}/{x}/{y}.png?property_name=sample_karst_score`).
  pixel-value / features-search remain the last predict follow-up. 3 new tests.
- **Skill #8 `olmoearth-evaluate` (`src/olmoearth_agent/evaluation/`)**:
  honest map-accuracy tools, pure Python (no heavy deps). `spatial_cv.py`:
  haversine, `spatial_block_folds` (Roberts 2017), `random_folds`, and
  **`cv_inflation_diagnostic`**, the headline: compares mean
  test-to-train nearest-neighbour distance under random vs spatial-block
  CV and reports the inflation ratio + risk band (operationalizes Ploton
  2020 / Meyer-Pebesma 2021). `metrics.py`: per-class
  precision/recall/F1/IoU + accuracy/macro-F1/mean-IoU. Tools
  `olmoearth_cv_inflation_check` + `olmoearth_classification_metrics`.
  Verified live 2026-05-28: agent flagged clustered data with a 353×
  inflation ratio and explained why random CV would overstate accuracy.
  13 new tests. NNDM-LOO (Milà 2022) is the remaining follow-up.
- **Skills #1-#4 vendored** from [`2imi9/OlmoEarth-Skills`](https://github.com/2imi9/OlmoEarth-Skills)
  as a git submodule (`vendor/olmoearth-skills`, pinned `a96427e`). The
  three upstream `SKILL.md` packages (`olmoearth-data-prep` [unifies #1+#2],
  `olmoearth-studio-job-config` [#3], `olmoearth-embeddings` [#4]) are
  consumed via a new `SkillLoader` (`skills/loader.py`) that gives the
  harness agentskills.io-style progressive disclosure: `olmoearth_list_skills`
  (name+description index) and `olmoearth_load_skill` (full `SKILL.md` body).
  `LeadAgent` accepts a brief `skill_index` for its system prompt. Graceful
  when the submodule is not initialized. Verified live 2026-05-28: the agent
  loaded `olmoearth-data-prep` and reported its real steps. 9 new tests.
  (Clone with `git submodule update --init` to populate the vendored skills.)
- **Skill #5 `olmoearth-predict` (core run loop)**: `tools/predict.py`:
  `olmoearth_search_predictions` (discover reusable `model_id`s) and
  `olmoearth_submit_prediction`; poll via the foundational
  `olmoearth_get_prediction`. `StudioClient` gains `search_predictions`
  and `submit_prediction` (the six `PredictionWrite` required fields:
  name, project_id, area_id, model_id, start_time, end_time).
  **Resolves the PLAN.md §4 `model_id` gap for the reuse case**:
  `predictions/search` returns each prediction's `model_id`, so a client
  discovers a reusable id by searching. Read paths verified live
  2026-05-28 (12 real predictions, all with model_ids); an agent run
  found PA Karst model_ids and recovered from a failed tool call mid-run.
  `submit` implemented + unit-tested (not live-created, to avoid side
  effects). Result sub-tools (pixel-value/features/files) are a follow-up
  within this skill. 3 new tests.
- **Skill #14 `olmoearth-provenance` (`src/olmoearth_agent/provenance/`)**:
  implements operational rule §3.13. `ProvenanceLog` lives on
  `ThreadState`; the lead agent records one `ProvenanceManifest` entry
  per dispatched tool call (tool name, sha256 of args, id-only result
  summary, never raw geometry). `to_json()` + `replay_script()` emit
  an auditable manifest and a replay skeleton. Tool bundle
  `olmoearth_provenance_summary` lets the agent report what it did.
  Added `ProvenanceManifest` to `types.py` (was spec-only in PLAN §2).
  Verified live 2026-05-28: agent run recorded `load_context` +
  `provenance_summary` with hashes and result summaries. 7 new tests.
- **Harness core (`src/olmoearth_agent/{types,studio,tools,harness,skills}`)**,
  the structure all 16 skills plug into:
  - `types.py`: harness dataclasses + `ApiEnvelope[T]` (the live
    `{records, meta, errors}` Studio response wrapper found 2026-05-28).
  - `studio/client.py`: async `StudioClient` (httpx, Bearer auth,
    envelope unwrap) with `users_me`, `search_projects`, `create_project`,
    `get_prediction`, `load_context`. Endpoints verified against the live
    API.
  - `tools/registry.py`: `ToolRegistry` + `ToolContext`; dispatch never
    raises (errors return to the model). `tools/studio.py`: the
    foundational `olmoearth_*` tool bundle (load_context / search_projects
    / create_project / get_prediction).
  - `harness/agent.py`: `LeadAgent` ReAct loop (DeerFlow v2 lead-agent
    shape): brief → LLM → tool dispatch → result, with a turn cap and the
    operational rules in the system prompt. `harness/state.py`:
    `ThreadState`.
  - `skills/registry.py`: manifest slotting all 16 skills (number,
    category, status, tools) + `build_default_registry()`.
- **Verified live end-to-end 2026-05-28**: `LeadAgent` →
  `OlmoEarthLLM` (Qwen3.6 4-bit GGUF via llama.cpp) → `StudioClient`
  (live Studio API) → answer. The agent called `olmoearth_load_context`
  and correctly filtered the user's real projects by topic.
- `tests/{studio,tools,harness,skills}`: 16 unit tests (mock HTTP +
  fake LLM) + 1 live integration test (`tests/harness/test_live.py`,
  needs `VLLM_ENDPOINT` + `OLMOEARTH_API_KEY`).
- `pyproject.toml` runtime dep: `httpx>=0.27`.
- **LLM serving client (`src/olmoearth_agent/llm/`)**: async OpenAI-
  compatible wrapper around the vLLM-served Qwen3.6-35B-A3B-NVFP4
  backbone. `OlmoEarthLLM.chat(messages, tools=..., mode=...)` returns
  a parsed `ChatResponse` with content, extracted `<think>` trace,
  tool calls, finish reason, and usage. Four sampling presets from
  the model card; default `thinking_general` with
  `chat_template_kwargs.preserve_thinking=True` for multi-turn agent
  runs. Synchronous `Tracer` protocol exposes request/response hooks
  for the provenance middleware (lands in PR #7).
- `docs/serving.md`: vLLM serve command, hardware requirements,
  **function-calling serve flags** (`--enable-auto-tool-choice
  --tool-call-parser`, required or tool calls come back as text;
  parser name flagged UNVERIFIED pending live confirmation), YaRN
  long-context recipe, agent-mode defaults.
- Client robustness: `_parse_completion` reads server-split
  `reasoning_content` (when served with `--reasoning-parser qwen3`)
  and otherwise extracts the inline `<think>` block, works either way.
- `docs/serving.md`: "Local development on ≤24 GB VRAM" section: 4-bit
  GGUF (`UD-IQ4_XS`) via llama.cpp `server-cuda` with `--jinja` for tool
  calling. **Function-call path verified end-to-end 2026-05-28** on an
  RTX 5090 Laptop (24 GB): NVFP4+vLLM stalls at memory profiling on
  24 GB (residual KV headroom too small), but the 4-bit GGUF loads to
  ~18.6 GB and the agent's `create_project(...)` tool call round-trips.
  Production stack stays vLLM+NVFP4 on datacenter Blackwell; this is a
  local-dev accommodation (same OpenAI protocol, client code unchanged).
- `docker/vllm.compose.yml`: pinned `vllm/vllm-openai:v0.19.0` for
  local dev (still requires Blackwell host).
- `tests/llm/`: mock-endpoint smoke tests via `pytest-httpx`: simple
  chat, `<think>` extraction, tool-call round-trip, preserve_thinking
  forwarding, top_k routing through `extra_body`, tracer hooks. Plus
  one live integration test (`@pytest.mark.integration`) that hits a
  real `vllm serve` instance when `VLLM_ENDPOINT` is set.
- `pyproject.toml` runtime dep: `openai>=1.50`. Dev deps:
  `pytest-asyncio>=0.24`, `pytest-httpx>=0.30`. Pytest config now
  pins `asyncio_mode = "strict"`.
- `.env.example`: `VLLM_ENDPOINT`, `VLLM_MODEL` (defaults pointing at
  local `vllm serve` on `http://localhost:8000/v1`).
- `PLAN.md` §4: explicit Studio API spec-version pin (`openapi.json` v0.1.0,
  pre-1.0) and verified findings on Firebase auth, `PredictionResultAccessLevel`,
  `PredictionUpdate` rename-only, free-form `PredictionRead.progress`,
  `*-management/` doc stubs, and the `TaskStatus`-vs-`PredictionStatus` enum split.
- `PLAN.md` §1: new `olmoearth.create_label` tool row (the Studio API treats
  labelset metadata and individual label classes as separate POSTs).
- `PLAN.md` §2: new `LabelsetSpec`, `LabelDef`, `DataPrepLabelSchema` dataclasses
  (clean separation between Studio-API schemas and the OlmoEarth dataset-prep
  layer's field names).
- **`SKILLS.md`**: detailed 16-skill catalog (Prep / Configure / Run /
  Analyze / Integrate / Report). Each skill has what / why / tools-composed
  with academic citations (Ploton 2020 spatial CV, Meyer-Pebesma 2021 AOA,
  Skakun CMIX 2022 cloud masks, WorldCereal 2025 lessons, IAMAP, NASA
  Similarity Search, etc.). Skill #16 (`roger-annotation-bridge`) added
  alongside the 15 from Ziming's source spec.
- `SKILLS.md`: "Existing implementations (upstream source)" section
  pinning skills #1-#4 to [`2imi9/OlmoEarth-Skills`](https://github.com/2imi9/OlmoEarth-Skills),
  upstream unifies skills #1 + #2 as `olmoearth-data-prep`
  (split-vs-unify decision deferred to first end-to-end skill PR).
  Skill #16 target pinned to [`2imi9/Roger-Studio`](https://github.com/2imi9/Roger-Studio).
- `SKILLS.md`: "Vendoring policy" section, submodule vs copy-with-
  provenance choice deferred to first vendoring PR.
- `PLAN.md` §4 Skills row: references upstream
  [`2imi9/OlmoEarth-Skills`](https://github.com/2imi9/OlmoEarth-Skills)
  as canonical home for skills #1-#4.
- `PLAN.md` §1: three new global tools: `olmoearth.pixel_value`,
  `olmoearth.features_search`, `olmoearth.fetch_embedding` (used by skills
  #4, #5, #9, #10).
- `PLAN.md` §2: four new dataclasses: `PixelValueResult`, `FeatureMatch`,
  `EmbeddingVector`, `ProvenanceManifest`.
- `PLAN.md` §3: new operational rule **13: Provenance manifest** (every
  `olmoearth.*` API call writes a `ProvenanceManifest` entry via
  `provenance_middleware` from skill #14).
- `PLAN.md` §7: new "Future work (parked)" section documenting the
  multimodal stack and train-time self-improvement tracks that are
  explicitly deferred.

### Changed
- `PLAN.md` bumped to **v0.4**. Scope narrowed to text-only LLM
  (`unsloth/Qwen3.6-35B-A3B-NVFP4`) with function calling. Multimodal
  stack (Prismatic / Q-Former / OlmoEarth embedding stream / NVFP4
  fine-tuning) moved to §7.1 Future work; train-time self-improvement
  moved to §7.2.
- `PLAN.md` §4 "Underlying stack" table collapsed from 7 rows to 5:
  dropped "Vision-language model" + "Geospatial encoder stream" +
  "Self-improvement"; added explicit "LLM" row pinning Qwen3.6-35B-A3B-NVFP4
  served via **vLLM ≥0.19.0** (upstream-recommended path for this NVFP4
  checkpoint; full `vllm serve` command in §4). Agent sessions use
  `chat_template_kwargs.preserve_thinking=True` per the model card.
- `PLAN.md` §7 "Future work" trimmed from per-bullet ref lists to two
  prose paragraphs. Noted that Qwen3.6 ships a native vision encoder,
  so re-opening §7.1 means using/replacing that tower, not training
  one from scratch.
- `SKILLS.md` "Existing implementations" + "Vendoring policy" sections
  tightened (split-vs-unify and submodule-vs-copy decisions left as
  one-line options rather than verbose A/B writeups).
- `PLAN.md` §6 roadmap rewritten from 7 generic phases (P0-P6) to
  skill-first: P0-P2 done (scaffold / gap closure / this rewrite),
  P3 = LLM serving + harness MVP, then one PR per skill ordered by
  case-study demand. Skills 14 (provenance) and 8 (evaluate) flagged
  for early landing because they're cross-cutting.
- `PLAN.md` §8 (was §7) references trimmed: in-scope LLM refs only;
  parked refs live in §7.1/§7.2.
- `PLAN.md` §3 rule list grew from 12 → 13 (provenance manifest).

### Changed (continued from PR #3)
- `PLAN.md` bumped to v0.3 (PR #3 increment, superseded by v0.4 here).
- `PLAN.md` §4: rewritten "Studio gaps" subsection from v0.1/v0.2's three
  UNVERIFIED items to verified findings: webhook absence CLOSED, fine-tune
  `model_id` field CONFIRMED but provenance still UNVERIFIED, rate limits
  CLOSED-as-undocumented.
- `PLAN.md` §5 example: uses `LabelsetSpec` + `create_label` flow and notes
  that every Studio Prediction requires a `model_id`.

### Changed
- `PLAN.md` bumped to v0.3.
- `PLAN.md` §4: rewritten "Studio gaps" subsection from v0.1/v0.2's three
  UNVERIFIED items to verified findings: webhook absence CLOSED, fine-tune
  `model_id` field CONFIRMED but provenance still UNVERIFIED, rate limits
  CLOSED-as-undocumented.
- `PLAN.md` §5 example: uses `LabelsetSpec` + `create_label` flow and notes
  that every Studio Prediction requires a `model_id`.

### Fixed
- `PLAN.md` §2 `PredictionStatus.state` enum corrected against the live
  `components.schemas.PredictionStatus`: `queued`→`pending`, `succeeded`→
  `completed`, added `cancelled` as a fifth terminal state.
- `PLAN.md` §2 `LabelSchema` retired: the v0.2 shape (`sample_category`,
  `es_label`, `oe_labels`) is the OlmoEarth dataset-prep / rslearn layer's
  schema, NOT the Studio API. Renamed to `DataPrepLabelSchema` and
  documented as a different layer; `LabelsetSpec` + `LabelDef` replace it
  for the API surface.
- `PLAN.md` §2 `PredictionRef.kind` docstring clarifies it's a client-side
  dispatch key, not a Studio API field (the API has no `kind` / `task_type`).

## [0.0.2] - 2026-05-27

### Added
- `CONTRIBUTING.md` modeled on NVIDIA earth2studio's developer workflow
  (DCO sign-off required, pre-commit mandatory, PR title into CHANGELOG,
  90% coverage gate, Black / Ruff / mypy / interrogate / NumPy docstrings /
  Apache-2.0 + SPDX headers).
- `AGENTS.md` following the [agents.md](https://agents.md/) spec for
  coding-agent onboarding.
- AI-assisted contribution policy: disclosure in PR description; no AI
  co-author trailers in commits (per OpenInfra Foundation convention).
- Branch naming convention: `2imi9/feature-<slug>`; no direct commits to `main`.

### Changed
- LICENSE copyright line updated to "OlmoEarth Agent contributors".

## [0.0.1] - 2026-05-26

### Added
- Initial `PLAN.md` and `README.md` defining the tool catalog, harness data
  classes, operational rules, and underlying-stack references, modeled on
  Google's Google Earth Agent shape (catalog + dataclasses + numbered rules).
- Apache-2.0 LICENSE.
- `.gitignore` covering Python build artifacts, virtual environments, test
  caches, type-checker caches, secrets (`.env` family, key files), model and
  data artifacts (safetensors, checkpoints, GeoTIFF, Zarr, Parquet), Hugging
  Face caches, and Claude Code agent state.
