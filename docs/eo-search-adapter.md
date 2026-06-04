# Planetary embedding search -- an adapter, not a clone

A scoping note on the one capability the OlmoEarth Agent lacks relative to a
managed geo-embedding product like [LGND](https://lgnd.ai): **semantic search
over a planet-scale embedding corpus** ("find everything that looks like this").
The conclusion is that we should *not* build a planetary vector index in-process;
the in-pattern move is a thin **adapter** over an *open* embedding corpus (or an
external search service), reusing skill #9's kNN core.

> **Method.** Compared the agent's `olmoearth-similarity` (skill #9) against the
> capability LGND's MCP server exposes, using the LGND/Clay public material
> (their team built [Clay](https://clay-foundation.github.io/model/); the
> embeddings are released openly as **LGND Clay v1.5 Sentinel-2** on the AWS
> Registry of Open Data). Checked the gap against the actual code
> (`analysis/similarity.py`) and the project's design rules (no heavy geospatial
> deps in the sandbox; pixel-heavy work is surfaced to Studio or the user;
> "generalize skills, not examples").

## TL;DR

- **The gap is real but narrow.** Skill #9 is an *exact, bring-your-own-embeddings*
  top-K kNN with a geographic-prior honesty warning. It has no corpus, no
  approximate-nearest-neighbour (ANN) index, and no text->embedding alignment --
  so it cannot do LGND's headline "find 5 swimming pools in LA" planetary search.
- **Do not clone it.** A planetary index is a *platform* capability (a corpus +
  an ANN index + a serving layer, possibly + a text-alignment model). Building
  that in our sandbox violates the no-heavy-deps / surface-to-Studio design and
  duplicates what Studio (or a managed service) should own.
- **Build an adapter instead.** Both LGND and OlmoEarth wrap an *open* EO
  foundation model, and LGND publishes its Clay/Sentinel-2 embeddings openly. So
  a thin `olmoearth-embedding-search` skill can query a managed vector backend
  (bring-your-own endpoint) or the open embeddings directly, reusing the cosine
  kNN + geographic-prior machinery skill #9 already has -- no planetary index of
  our own.
- **Lowest-effort variant:** consume an external embedding-search **MCP** (e.g.
  LGND's) as a tool. This is exactly the "complementary, not competitive" framing
  -- their search-as-a-service plugs into our agent loop -- and it ties into the
  Studio-native vision (issue #90).
- **Honest caveat:** *text-prompt* search is the hard part. Clay (and OlmoEarth)
  are self-supervised, image-only encoders, so text queries need a CLIP-style
  text<->image alignment they do not ship. Image-chip and coordinate search are
  feasible dep-light; free-text search should be deferred or routed to a
  backend that owns the alignment. Do not over-promise it.

## 1. What skill #9 is, and what it is not (verified)

`analysis/similarity.py` provides:

- `similarity_search(query, corpus, ...)` -- exact brute-force top-K over
  embedding vectors the caller already holds (cosine or euclidean). Correct and
  dependency-free at the hundreds-to-thousands sizes the agent passes in; FAISS
  was noted as the scale-up follow-up.
- `geographic_prior_check(...)` -- the honesty guard that warns when the matches
  cluster near the query (so "similarity" is not just reflecting location).

What it deliberately is *not*: a managed index over a pre-computed planetary
corpus. There is no embedding store, no ANN structure, and no way to query by
text. That is the whole of the gap versus LGND.

## 2. Why an adapter, not a planetary index

| Concern | Build a planetary index in-process | Adapter over an open corpus / external service |
|---|---|---|
| Corpus | We would have to generate + store embeddings for a large AOI (a heavy batch job) | LGND Clay v1.5 Sentinel-2 is *already* public on AWS Open Data; OlmoEarth's own exported embeddings (skill #7's bring-your-own path) also fit |
| Index | An ANN library (FAISS/HNSW) + a vector store -- a new heavy dep + ops | The backend owns the index; we send a query, read top-K |
| Sandbox rules | Violates "no heavy geospatial/index deps; surface pixel work to Studio" | Honors them: we stay metadata/vector-thin, the heavy lift is elsewhere |
| Maintenance | We own a planetary pipeline | We own a thin client + the same kNN/geographic-prior guard we already have |

The design rule the repo already follows (pixel-heavy / index-heavy work is
surfaced, not done in the sandbox) points squarely at the adapter.

## 3. Two concrete shapes

1. **`olmoearth-embedding-search` (adapter skill).** A `build_*_tools ->
   RegisteredTool` bundle that takes a query (an image chip's embedding, or a
   coordinate that resolves to a cell's embedding) + a backend handle, returns
   top-K matches, and reuses `geographic_prior_check` for the honesty warning.
   The backend is bring-your-own: a managed vector endpoint, or the open
   LGND-Clay / OlmoEarth embeddings loaded behind a small service. Text queries
   are out of scope unless the backend supplies a text-aligned encoder.
2. **External search MCP as a tool (lowest effort).** Since LGND ships an MCP
   server and our agent can consume MCP tools, the cheapest integration is to let
   the agent call an external embedding-search MCP directly. No corpus or index
   of our own; the agent orchestrates the search and folds results into the rest
   of the workflow. This is the literal payoff of the "complementary" framing.

Either shape keeps our differentiator (the honest, full-lifecycle EO workflow)
and borrows the one capability we lack, without standing up a planetary index.

## 4. Scope, caveats, and relationship to other work

- **Text search is the genuinely hard part.** Image-only encoders have no text
  space; "find swimming pools" needs an alignment layer. Be explicit: ship
  image-chip + coordinate search first; treat free-text as backend-provided or
  deferred. Over-claiming text search would violate the honest-results ethos.
- **Generalize, not examples.** Any adapter must be backend-agnostic and
  corpus-agnostic (LGND-Clay, OlmoEarth, or a user's own store) -- never wired to
  one provider's specifics as if they were the design.
- **Ties to the Studio-native vision (issue #90).** Embedded in Studio, "select
  an area -> find similar across the AOI" becomes a natural map gesture; the
  adapter is the engine behind it. Issue #59 (draw/select an AOI) is the input
  half.
- **Not scheduled here.** This is a scoping note, like
  [`eo-skills-shortlist.md`](eo-skills-shortlist.md); it argues the shape so the
  build (if pursued) starts from the right premise.

## 5. References

- LGND (Clay-based geo-embeddings; open Sentinel-2 embeddings): <https://lgnd.ai/resources/geo-embeddings-101> · AWS Open Data "LGND Clay v1.5 Sentinel-2": <https://registry.opendata.aws/lgnd-clay-v1-5-sentinel2/>
- Clay Foundation Model: <https://clay-foundation.github.io/model/>
- In-repo: `src/olmoearth_agent/analysis/similarity.py` (skill #9), [`SKILLS.md`](../SKILLS.md) (#7 bring-your-own exported embeddings, #9 similarity), [`eo-skills-shortlist.md`](eo-skills-shortlist.md), issues #59 (draw AOI) and #90 (Studio-native skill + area + chat flow).
