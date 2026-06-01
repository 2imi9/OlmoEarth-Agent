# Google DeepMind "Science Skills" — assessment for OlmoEarth Agent

An architectural read of [`google-deepmind/science-skills`](https://github.com/google-deepmind/science-skills)
and its technical report, mapped onto how the OlmoEarth Agent **places** skills
and how the **harness manages** them — with a fit verdict and the patterns worth
adopting.

> **Sources.** Repo `google-deepmind/science-skills` (Apache-2.0, ~37 skill
> directories, tag v1.0.3) and the report *"Science Skills for Antigravity:
> Towards Efficient and Reliable Scientific Workflows"* (Google DeepMind,
> 2026-05-19,
> [PDF](https://storage.googleapis.com/deepmind-media/papers/google_deepmind_science_skills_for_antigravity_towards_efficient_and_reliable_scientific_workflows.pdf)).

## TL;DR

- **As a skill *bundle*, GDM Science Skills is off-domain** for OlmoEarth: the
  ~37 skills are genomics / proteomics / cheminformatics / clinical (ChEMBL,
  AlphaFold, ClinVar, …). Nothing there ingests satellite imagery, talks to
  OlmoEarth Studio, or touches geospatial workflows. The earlier "off-domain"
  call **stands**.
- **One capability *is* transferable and in-domain-adjacent: literature search.**
  GDM ships `literature_search_{arxiv,openalex,biorxiv,europepmc}` +
  `pubmed_database`. An arXiv/OpenAlex lit-search skill is already on our backlog
  (issue #62) — GDM's arXiv + OpenAlex skills are a concrete reference for it.
- **The report's *methodology* is the most valuable takeaway**, independent of
  domain: a three-tier test suite (unit → workflow → capability) with an
  LLM-autorater and falsifiable checkpoints, grounding-over-web-search, honest
  "report absence" behavior, and trigger-heavy skill descriptions. Quantified
  result worth remembering: with skills, **Gemini 3 Flash reliability went
  49% → 93%** and **tokens dropped ~1.5–2×**, letting Flash match Pro — strong
  external evidence that well-authored skills let a *weaker* model (our local
  Qwen3.6) punch above its weight.

## 1. What GDM Science Skills is

A curated, quality-tested bundle of agent skills for **Google Antigravity** (a
Gemini-based agentic dev platform), built on the cross-vendor
[agentskills.io](https://agentskills.io) standard. Core thesis (verbatim): the
bundle "improves the reliability and efficiency of scientific workflow tasks:
more tasks succeed and the number of tokens and model calls necessary to
complete tasks is reduced. With skills, smaller models of the Flash category
attain Pro level reliability." Skills close three LLM gaps — missing domain
knowledge, limited reliability, and weak grounding in primary data — by pointing
the agent at **authoritative APIs/data via bundled scripts** instead of
world-knowledge or generic web search.

## 2. How GDM places & runs skills

- **Skill = a folder.** `SKILL.md` (YAML frontmatter: exactly `name` +
  `description`) + `scripts/*.py` CLIs + optional `references/*.md`. The whole
  repo is one installable plugin (`plugin.json`, name `science`).
- **Discovery + progressive disclosure.** An external host (skills.sh / `npx
  skills add`, or the Antigravity plugin) installs the bundle and indexes each
  `SKILL.md`. The agent **selects** by matching the request against the
  `description`, then **opens `SKILL.md` first** (top layer) and reads
  scripts/references on demand (deeper layers). There is *no loader code in the
  repo* — it's a passive bundle.
- **Execution is out-of-process.** The agent emits
  `run_shell_command "uv run scripts/x.py <subcommand> --output file.json"`, the
  script hits the API, writes JSON to a **file**, prints only a short status; the
  agent then greps/`jq`s the file. Dependencies are declared with **PEP-723
  inline headers** and isolated by `uv`. The only shared library is
  `scienceskillscommon` (an `HttpClient` with cross-process `fcntl` file-lock
  rate-limiting, retry/backoff, `Retry-After`, gzip).
- **Auth.** Keys live in `~/.env`, loaded via `python-dotenv`, appended to URLs
  out-of-band, and **never** `cat`/`echo`/`printenv`'d into model context.
  arXiv/EuropePMC need no key; OpenAlex's is optional (premium).
- **Reliability is engineered and measured** (report §3–4): a three-tier test
  suite (**unit** → **workflow/multi-turn** → **capability**, 67 expert-validated
  tasks), an **LLM-autorater with explicit per-turn checkpoints and exact
  expected outputs**, grounding in reference data, documented **negative
  results** (without skills the agent hallucinates links, spams redundant web
  searches, fabricates tables, and gives a confidently-wrong answer), and a
  standardized `uv` environment as a reproducibility guardrail. **No provenance**
  layer — the report lists reproducibility as an open limitation.

## 3. How OlmoEarth Agent places & runs skills

Two distinct kinds of "skill":

- **(a) Vendored `SKILL.md` guidance** (#1–#4) in the
  `vendor/olmoearth-skills` submodule. Loaded by
  [`SkillLoader`](../src/olmoearth_agent/skills/loader.py): frontmatter
  (`name`, `description`) → `index()` injects a name + first-sentence list into
  the `LeadAgent` system prompt; the full body is pulled on demand via
  `olmoearth_list_skills` / `olmoearth_load_skill`
  ([`skill_tools.py`](../src/olmoearth_agent/tools/skill_tools.py)).
  **Guidance only — no bundled scripts.** This is structurally identical to a
  GDM skill (same agentskills.io anatomy, same progressive disclosure).
- **(b) Implemented Python tool bundles** (#5–#15) in
  [`src/olmoearth_agent/tools/`](../src/olmoearth_agent/tools/): each a
  `build_*_tools()` factory returning `RegisteredTool(spec=ToolSpec, handler=async fn)`,
  with heavy logic in a parallel `analysis/` / `reporting/` module.

The harness ([`LeadAgent`](../src/olmoearth_agent/harness/agent.py)) is a
single-agent ReAct loop: system prompt + injected skill index → the LLM emits a
`tool_call` → [`ToolRegistry.dispatch`](../src/olmoearth_agent/tools/registry.py)
awaits the async handler **in-process** → the JSON result is fed straight back,
until a plain-text answer or `max_turns`. Two differentiators over GDM:
`ToolRegistry.dispatch` **never raises** (returns `{ok: false, error}` so the
model self-recovers), and **every call is logged to a provenance manifest**
([`provenance/log.py`](../src/olmoearth_agent/provenance/log.py): sha256 of args
+ an id-only summary, never raw geometry) with a runnable replay skeleton.

## 4. Side-by-side

| Dimension | GDM Science Skills | OlmoEarth Agent | Takeaway |
|---|---|---|---|
| **Skill unit** | A folder: `SKILL.md` + `scripts/` + `references/` | (a) vendored `SKILL.md` guidance; (b) in-process `build_*_tools()` bundles | Our (a) ≡ a GDM skill; our (b) has no GDM analog (GDM never registers function-call tools — it always shells out). |
| **Discovery / disclosure** | Host indexes `SKILL.md`; agent matches `description`; opens `SKILL.md` then scripts on demand | `SkillLoader` index → `load_skill` for (a); for (b) the full `ToolSpec` is always in the tool list | For implemented tools there is **no `SKILL.md` gate** — the `ToolSpec.description` is the only routing signal, so it must be trigger-heavy. |
| **Execution** | Out-of-process `uv run` CLI scripts → write JSON to file → agent `jq`s it | In-process async handler; result fed straight back; only `olmoearth_run_python` shells out (opt-in) | **Don't** adopt uv-run scripts — it's a second, alien execution model. A new skill is a normal async tool bundle (like `StudioClient`). |
| **Deps / auth** | Stdlib + PEP-723/`uv`; shared `HttpClient` (POSIX `fcntl` lock); keys in `~/.env` | Real deps in `pyproject` (`httpx`); Studio key from env; no per-script isolation | **Don't** port `scienceskillscommon` — its `fcntl`/`/tmp` lock is POSIX-only (breaks on our Windows host) and an in-process loop has no cross-process contention. Reuse `httpx`. |
| **Distribution** | One installable plugin (marketplace) | Single Python package; skills = code + `SkillSpec` + `SKILLS.md` | No marketplace needed. A skill ships as code + a `SkillSpec` row + a `SKILLS.md` entry + tests, in one PR. |
| **Reliability** | 3-tier tests + LLM-autorater + negative results; Flash 49→93%, 1.5–2× fewer tokens | Structural + unit-tested; dispatch-never-raises; no autorater/benchmark in-tree (SkillOpt is separate) | Adopt the **testing methodology** and bake **no-fabrication / cite-source-URLs** rules into tool descriptions. |
| **Provenance** | None (open limitation) | First-class manifest + replay per call | We're **ahead**; a new tool gets provenance for free. |

## 5. Fit verdict

**Confirmed off-domain as a bundle.** The skills cover molecular/biomedical
databases and models that have no bearing on Earth observation. Porting any of
them wholesale would add off-domain surface area for zero in-domain value — the
same conclusion the project reached pre-1.0, now re-validated against the actual
repo + report.

**The one exception is literature search.** GDM ships first-class arXiv +
OpenAlex skills; our `SKILLS.md` already *cites* a body of EO literature
(Ploton 2020, Meyer–Pebesma 2021, WorldCereal, AlphaEarth transfer work, …) but
the agent has no way to *fetch* or *resolve* those citations — it can only rely
on world-knowledge or hallucinate links (exactly GDM's documented failure mode).
That gap is issue #62, and it is worth closing.

## 6. Adoptable patterns

| Pattern | Effort | Recommendation |
|---|---|---|
| Resilient shared async HTTP helper (retry 429/5xx + backoff + `Retry-After`) | S | Factor `StudioClient`'s retry into a tiny `httpx` helper reused by lit-search. **Not** GDM's `fcntl` lock — a per-host `asyncio.Semaphore` suffices in-process. |
| No-fabrication / cite-source-URLs / report-absence rules | S | Bake into the `ToolSpec.description` (the only behavioral surface for implemented tools); return a real source URL on every record so the model cites instead of inventing. |
| Result-size discipline (cap counts, curated fields, no full text) | S | Small default `max_results` (5) + hard cap (~25); return id/title/authors/year/venue/doi/url/abstract only; abstracts opt-in. GDM's `--output`-to-file discipline, translated in-process. |
| Three-tier test methodology (unit → workflow → capability) | M | Keep per-tool unit tests; add a small EO-flavored **capability** tier to the SkillOpt harness (e.g. "find the WorldCereal lessons paper, give its arXiv id" with the exact id as the autorater check). |
| Trigger-heavy description authoring | S | Stack task verbs + an explicit "Use when…" clause; generalize (no baked-in example) — matches our existing `SKILLS.md` convention. |
| Cross-source dedup delegated to a deterministic key | S | Merge arXiv + OpenAlex on lowercased DOI → normalized arXiv id → title-hash; keep single-source API relevance ordering (no semantic re-rank). |

## 7. Decision — issue #62 as an in-process tool bundle

Build a new **skill #16 `olmoearth-litsearch`** as kind (b) (an in-process
function-call tool bundle), **informed by** GDM's arXiv/OpenAlex skills but not
vendored as uv-run scripts and not importing `scienceskillscommon`:

- **Tools.** `olmoearth_litsearch` (unified arXiv + OpenAlex search; dedups across
  sources) and `olmoearth_litsearch_resolve` (DOI / arXiv-id → one record; the
  resolve-before-cite primitive).
- **APIs, key-free.** arXiv Atom API (`export.arxiv.org/api/query`) and OpenAlex
  `/works` via the **documented `mailto` polite pool** (preferred over GDM's
  `Referer`-header hack). Optional `OPENALEX_API_KEY` only for premium — not
  required.
- **Curated record:** `id` (normalized), `title`, `authors`, `year`, `venue`,
  `doi`, `arxiv_id`, `url` (citation link), `abstract` (truncated, opt-in),
  `cited_by_count` (OpenAlex), `source`.
- **Reliability:** shared `httpx` retry helper; dedup by DOI/arXiv-id/title-hash;
  result caps; no-fabrication/cite-URL rules in the description; emit one arXiv
  envelope **after** the parse loop (avoiding the real per-entry reprint bug in
  GDM's `search_arxiv.py`); never touch raw geometry (provenance-safe).
- **File plan:** `tools/litsearch.py` (bundle) + `analysis/litsearch.py`
  (testable query-build/parse/dedup) + register in
  `skills/registry.py` (`build_default_registry` + a `SkillSpec(16, …)` row) +
  unit tests (`pytest-httpx` mocked) + `SKILLS.md` + `CHANGELOG.md`.

## 8. References

- Repo: <https://github.com/google-deepmind/science-skills>
- Report PDF: *Science Skills for Antigravity* (GDM, 2026-05-19).
- Agent Skills standard: <https://agentskills.io>
- Our skill placement: [`skills/loader.py`](../src/olmoearth_agent/skills/loader.py),
  [`tools/skill_tools.py`](../src/olmoearth_agent/tools/skill_tools.py),
  [`skills/registry.py`](../src/olmoearth_agent/skills/registry.py).
- Our harness: [`harness/agent.py`](../src/olmoearth_agent/harness/agent.py),
  [`tools/registry.py`](../src/olmoearth_agent/tools/registry.py),
  [`provenance/log.py`](../src/olmoearth_agent/provenance/log.py).
- Related: `PLAN.md` §7 (parked multimodal/self-improvement tracks),
  `evals/skillopt/` (skill benchmarking vs local Qwen3.6).
