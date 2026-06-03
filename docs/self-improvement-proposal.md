# Inference-time self-improvement -- adopt-first proposal

Which self-improving techniques (Stanford [CS329A](https://cs329a.stanford.edu/))
to adopt first in the OlmoEarth Agent, mapped onto the actual harness. Scope is
**inference-time only** (no weight updates); train-time RL stays parked per
[`PLAN.md`](../PLAN.md) section 7.2.

> **Sources.** CS329A (self-improving / self-evolving agents) plus the underlying
> papers cited inline; the GDM methodology in
> [`docs/science-skills-assessment.md`](science-skills-assessment.md); and a read
> of the harness (`harness/agent.py`, `tools/registry.py`, `harness/state.py`,
> `provenance/log.py`, `evals/skillopt/`). Code facts below were verified against
> the source.

## TL;DR

- **Adopt per-skill self-check (grounded verify-and-retry) FIRST.** It is the only
  candidate that is simultaneously cheap (at most one extra call, only on a
  verify-fail), high-payoff (Reflexion / Self-Refine show +20-22% absolute *when
  the feedback is a reliable external signal*), and a near-perfect ethos fit. The
  harness is already half a verify loop: `ToolRegistry.dispatch` returns
  `{ok, error}` instead of raising, and `evals/skillopt/` ships deterministic
  oracles that are exactly the grounded verifiers the evidence says you need.
- **The concrete first PR is small (S):** an optional `verify` predicate on
  `RegisteredTool`, a `reflections` list on `ThreadState`, and a one-retry verify
  gate in `LeadAgent.run_stream`, with a deterministic verifier for
  `olmoearth_automate` that recomputes `analysis.automate.decide()` from the
  echoed inputs and asserts the returned decision matches. Merge **only** if it
  raises the SkillOpt score at temperature 0.
- **Then, in order:** (2) a 3-tier test + LLM-autorater for SkillOpt (the honest
  *measurement substrate*), (3) repeated-sampling + verifier (higher ceiling,
  K-fold cost, layered on the same verify infra), (4) MemGPT-style tiered
  `ThreadState` memory (deferred; cheap deterministic Tier-1 only).
- **Train-time RL (STaR / SWiRL / GRPO / DAPO) stays parked.** No trace volume yet
  to justify it, and every technique here is weight-free. The self-check loop is
  itself the cheapest way to *start* producing the verifier-graded traces a future
  RL bootstrap would consume -- so inference-time first is the correct ordering
  even if RL is eventually wanted.

## 0. Scope and ethos

The agent is a text-only ReAct loop driven by a **weak local model** (Qwen3.6,
~3B active) or a hosted Claude/ChatGPT/Gemini backend. Three constraints shape
every recommendation:

1. **Honest results.** A verifier must read a *real external signal* -- a tool
   error, a JSON-Schema check, or a deterministic oracle -- never the ~3B model
   judging its own reasoning in a vacuum. This is not a stylistic preference: the
   evidence is decisive (see 1.1).
2. **Provenance per call.** Every tool call is logged to an append-only manifest
   (`provenance/log.py`); any new loop must keep that trail truthful.
3. **`dispatch` never raises.** `ToolRegistry.dispatch` returns
   `{"ok": False, "error": ...}` on any failure (verified, `tools/registry.py`).
   New machinery should *extend* this contract, not fight it.

## 1. The four techniques

### 1.1 Per-skill self-check (grounded verify-and-retry) -- ADOPT FIRST

**What.** After a tool returns, run a cheap deterministic check on the result; on
failure, append a one-line natural-language reflection (Reflexion-style) and
re-dispatch **once** (Self-Refine: gains are front-loaded in the first
iteration). No weight updates.

**Evidence.** Strong, but strictly conditional on the feedback being external and
reliable -- which is our case. Pro: Reflexion ([arXiv:2303.11366](https://arxiv.org/abs/2303.11366))
HumanEval 80->91% pass@1, +22% abs on ALFWorld; Self-Refine
([arXiv:2303.17651](https://arxiv.org/abs/2303.17651)) ~20% abs avg over 7 tasks
with no training. Boundary (decisive): Huang et al. ICLR'24
([arXiv:2310.01798](https://arxiv.org/abs/2310.01798)) -- *intrinsic* self-correction
without external feedback consistently **degrades** accuracy; Kamoi et al. TACL'24
([arXiv:2406.01297](https://arxiv.org/abs/2406.01297)) -- self-correction works
only with reliable external feedback. On a ~3B model an ungrounded "are you sure?"
flips right->wrong more than wrong->right.

**The gap it closes (verified in code).** `dispatch` wraps *any* successful
handler return as `{"ok": True, "result": ...}` and only sets `ok: False` on an
exception or unknown tool. So a handler that returns a **semantically wrong but
non-throwing** payload -- e.g. `olmoearth_automate` returning a `decision` that
violates `decide()`'s own table, or a job-config missing a required field -- still
comes back `ok: True`. The verifier must therefore run on `result["result"]`
*after* dispatch, not trust the `ok` flag alone.

**Maps to.** `harness/agent.py`, the per-call block in `LeadAgent.run_stream`
(between `dispatch` and the `role="tool"` message append). The grounded verifiers
already exist: the `{ok, error}` contract and the three SkillOpt oracles
(`recommend.decide()` / `PRESETS` / the pitfalls table), scoreable at temperature
0 with no LLM-autorater.

**Effort.** **S** beachhead; **M** full rollout (verifiers for the other
oracle-backed tools, a per-run retry budget, the proving ablation). The split
adversarial-LLM critic (the generator-verifier-gap upgrade, Weaver
[arXiv:2506.18203](https://arxiv.org/abs/2506.18203), ~+20% over plain
self-refine) is an **L deferred** layer -- only after the deterministic loop shows
a measured win, and never as an ungrounded critic on the local model.

**Risks.** (1) Ungrounded self-check on a weak model degrades accuracy -- mitigate
by *requiring* an external signal for every `verify`. (2) Token/latency blow-up --
cap at one retry per call and a per-run retry budget; never re-verify the retry's
own output. (3) A buggy `verify` predicate becomes a new failure source -- keep
verifiers pure, deterministic, unit-tested, and import the *same* `decide()` the
handler uses (no forked oracle copy).

### 1.2 3-tier test + LLM-autorater for SkillOpt -- ADOPT SECOND (measurement substrate)

**What.** Lift GDM's UNIT -> WORKFLOW -> CAPABILITY test taxonomy and a
rubric-anchored LLM-autorater into `evals/skillopt/`, on top of today's three
deterministic-oracle envs. (1) *Relabel* the existing oracles as the UNIT tier
(no new scoring code); (2) *add* a WORKFLOW tier of 2-3 multi-turn EO tasks scored
by per-turn checkpoints, driven through the real `run_stream`; (3) *add* a small
CAPABILITY tier for end-to-end utility + token counting.

**Why second.** It does not improve the agent's answers directly -- it is how you
*honestly prove* that 1.1 (and everything after) helps on the **multi-turn**
agent, beyond single-shot oracles. The trust comes from three stacked
inference-time tricks: rubric-anchored per-checkpoint booleans (the LLM judges
only the free-text residue; ids/numbers/tool-names stay code-side), repeated-
sample-and-vote on the judge, and a confidence gate that emits `needs_human`
rather than guess. A two-rater honesty guard (local Qwen + a hosted judge) with a
hand-labeled ~20-30 item slice and a reported Cohen's kappa gates whether the
autorater is authoritative or advisory -- closing the validity gap GDM itself left
open (they published no human-vs-autorater agreement).

**Maps to.** All under `evals/skillopt/` (no `src/` change for v1; the rater calls
the agent as a black box). `run_stream` already yields JSON-serializable per-turn
events and `AgentResult` carries `tool_calls` + `state`, so the WORKFLOW tier is
plumbing over an existing event stream, not new agent code.

**Effort.** **L** overall, but value lands early: UNIT relabel = S; one
rubric-anchored judge + one 3-turn workflow task = M (the first PR); the kappa
calibration slice is the long pole (human labeling, not code).

**Risks.** LLM-judge validity (keep it advisory until kappa is measured); the weak
local judge (hence the hosted second judge + majority vote, which reintroduces an
API-key/cost dependency the harness avoids); rubric-authorship bias (rubrics must
assert general decision-correctness, not memorized strings -- per the project's
"generalize skills, not examples" rule).

### 1.3 Repeated sampling + verifier -- ADOPT THIRD

**What.** Best-of-K at inference time, gated by a **real** verifier (never
verifier-free majority vote in chat). Draw K candidates for one checkable step,
keep the first that passes a deterministic check.

**Evidence.** Large Language Monkeys
([arXiv:2407.21787](https://arxiv.org/abs/2407.21787)): coverage (pass@K) scales
log-linearly (SWE-bench Lite 15.9%->56% over 250 samples) and cheap-model-x-many
beats one strong call -- **but** verifier-*free* selection plateaus (~39% while
oracle coverage exceeds 95%) and the gap widens with K. So it pays off *only*
where we already own an oracle, which we do: the `{ok, error}` envelope (resample
a failing call, keep the first `ok: True`) and the SkillOpt oracles.

**Why third.** Same honest, oracle-gated profile as 1.1 and an even higher
ceiling, but it spends K-fold tokens up front, against an 8-turn budget. Correct
sequencing: build the verify infra once in 1.1, then turn on K-sampling behind an
opt-in budget flag on the few oracle-backed steps, after 1.2 can measure whether
the spend pays off. **First step needs zero harness change:** a measurement-only
SkillOpt ablation (best-of-K vs majority-vote vs single-sample on the existing
test split).

**Effort.** **M**, most of it shared with 1.1. **Risks.** The no-verifier trap
(never apply to free-form answers); cost/latency (small K, gate on a prior
failure); determinism (K-sampling needs temperature > 0, which collides with the
temp-0 eval reproducibility -- keep it strictly opt-in, default K=1, and log every
attempt to provenance so a resampled run stays auditable).

### 1.4 MemGPT-style tiered ThreadState memory -- DEFER

**What.** Page between an in-context WORKING tier (system prompt + compact facts +
recent tail with recursive summaries), an out-of-context RECALL tier (the
provenance log, searched on demand), and an ARCHIVAL tier (embedding search).
MemGPT ([arXiv:2310.08560](https://arxiv.org/abs/2310.08560)) roughly tripled
fixed-context QA accuracy -- *but that assumes a capable controller*, and our
default driver is the weak local Qwen.

**Why deferred.** Three findings from reading the code: (1) the only unbounded
growth is **intra-run** -- cross-request history is already bounded in `serve.py`
(`_MAX_HISTORY_TURNS`, fresh `ThreadState` per request); (2) there is **no token
budget** anywhere, so "respect the context window" means *introducing* a budget;
(3) the ARCHIVAL tier the technique assumes is effectively a dead field
(`ThreadState.artifacts` is declared but rarely written). So the honest entry is a
**deterministic, read-only Tier-1 context assembler** (keep WORKING small,
summarize the older tail using the *same* provenance reducer that already strips
geometry to ids, never touch the manifest) with the model **out of the loop** --
not model-driven self-editing on a ~3B controller. Provenance integrity is
structurally safe: paging rewrites only the local `messages` list; the append-only
`ProvenanceLog` is separate.

**Effort.** **M.** **Risks.** Weak controller (PR 1 keeps the model out of the
paging decision); lossy summaries dropping a live id (pin all ids verbatim,
summarize only prose/large payloads); over-claiming the paper's QA numbers as our
EO gain (report only measured SkillOpt deltas + a token-budget number).

## 2. Ranking

| Rank | Technique | Adopt first | Payoff | Cheapness | Ethos fit | Effort |
|---|---|---|---|---|---|---|
| 1 | **Per-skill self-check** (grounded verify-and-retry) | **Yes** | High (+20-22% where feedback is reliable) | High (1 extra call on fail) | Excellent | **S** beachhead |
| 2 | 3-tier test + LLM-autorater (SkillOpt) | No | Measurement, not answers | Med (UNIT relabel free) | Excellent (closes GDM's validity gap) | L (S first slice) |
| 3 | Repeated sampling + verifier | No | Highest ceiling | Low (K-fold tokens) | Excellent (oracle-gated) | M (shares 1.1 infra) |
| 4 | MemGPT tiered ThreadState memory | No | Uncertain on weak controller | Med (Tier-1 deterministic) | Good (Tier-1 only) | M |

Two explicit non-recommendations, both surfaced by the research as negative
baselines: **ungrounded intrinsic self-correction** (Huang ICLR'24) and
**verifier-free majority vote** (the negative arm of 1.3) -- neither has an
external signal, both can make a weak model confidently wrong, and both violate
honest-results. Keep verifier-free voting only as a *control arm* in the SkillOpt
ablation to demonstrate the gap.

## 3. Adopt-first: the concrete first PR

A single small PR scoped to one tool with an objective post-condition --
`olmoearth_automate`, whose result must be consistent with
`analysis.automate.decide()`:

1. Add `verify: Callable[[dict[str, Any]], tuple[bool, str]] | None = None` to
   `RegisteredTool` in `tools/registry.py` (default `None` -> existing ~16 tools
   unchanged, zero risk).
2. Add `reflections: list[str] = field(default_factory=list)` to `ThreadState`
   (`harness/state.py`), next to the existing append-only `provenance`.
3. In `LeadAgent.run_stream`, after `result = await self.registry.dispatch(...)`:
   if the registered tool has a `verify` and `result["ok"]`, call
   `verify(result["result"])`; on `(False, reason)` append a one-line reflection,
   record the failed attempt to provenance, re-dispatch **once** with the
   reflection as a tool/system note, and use the second result. Cap at one retry.
4. Supply `automate`'s verifier as a pure function that recomputes `decide()` from
   the echoed inputs and asserts the returned `decision` / `model` match (a no-LLM
   oracle check, importing the *canonical* `decide()`, not a fork).
5. Add a SkillOpt ablation (single-shot vs verify+retry) on the
   `olmoearth_embeddings` / `jobconfig` test split at temperature 0.

**Ship only if the oracle score rises.** That proves the grounded loop before any
adversarial/LLM critic is considered. Auditability is free: a verify-fail + retry
shows up as two manifest entries sharing a request hash.

### 3.1 Outcome (measured)

The beachhead was built and the gate measured, with an honest split between the
*infrastructure* (always safe) and the *behaviour* (gated on a lift):

- **Infrastructure landed.** An optional `verify` predicate on `RegisteredTool`,
  a `reflections` list on `ThreadState`, and a capped verify-and-retry gate in
  `LeadAgent.run_stream` (`max_verify_retries`, default 1). It is a **no-op for
  every existing tool** (all have `verify=None`), and a unit test drives a
  synthetic failing-then-passing verifier through the loop to prove the
  reflection-and-retry path works (`tests/harness/test_verify_gate.py`).
- **`automate` verifier built** (`verify_automate_result`): recomputes `decide()`
  from the echoed inputs (a consistency invariant) and flags an under-specified
  fallback (`ask_for` non-empty). Pure, deterministic, unit-tested.
- **The gate does NOT clear the ship bar on the held-out test split.** A
  deterministic ablation (`evals/skillopt/scripts/ablate_verify_automate.py`),
  using the real verifier + `automate()` against the `olmoearth_embeddings`
  split, measures the gate's upper-bound lift (baseline = brief-only parse;
  verify+retry = re-supply the brief's full inputs on a verify-fail):

  | split | n | baseline | verify+retry | gate fired | lift |
  |---|---|---|---|---|---|
  | **test** | 12 | 12/12 | 12/12 | 2 | **+0** |
  | val | 9 | 7/9 | 7/9 | 1 | +0 |
  | train | 14 | 8/14 | 12/14 | 6 | **+4** |

  On the **test** split the parser already yields oracle-correct decisions (the
  `embeddings`/`tiny` answer is robust to the inputs it drops), so the gate
  cannot lift it. On **val** the two misses are *confident-but-wrong* (no
  `ask_for`), so the verifier never fires on them — a real limit: this verifier
  catches under-extraction-into-fallback, not silent miscalls. The mechanism is
  nonetheless real (**train +4**), where briefs state inputs the parser drops.

- **Decision (honest, per the gate):** the infra ships dormant; the `automate`
  verifier is **not wired on by default** because the held-out lift is 0. It
  remains importable for opt-in, and the gate is ready for the first verifier
  that *does* clear a held-out bar (a job-config cross-field validator, or a
  negative-sampler placement-shortfall retry, are better candidates than a tool
  that already calls its own oracle). The temperature-0 *agent-loop* ablation is
  superseded by this deterministic upper-bound, which is tighter and needs no LLM.

## 4. Why train-time RL stays parked

Per [`PLAN.md`](../PLAN.md) section 7.2, train-time RL (STaR / SWiRL / GRPO /
DAPO) re-activates *only* when "trace volume justifies." Two grounded reasons it
does not yet:

1. **No trace corpus.** v1.0 only just shipped; there is nothing to bootstrap
   from. The irony worth stating: the grounded self-check loop ranked #1 is itself
   the cheapest way to *start* generating the labeled (verify-pass / verify-fail +
   reflection) traces a future STaR/SWiRL run would consume. Inference-time first
   is the correct ordering even if RL is eventually wanted.
2. **Weight-free is the whole point here.** All four ranked techniques touch no
   weights and keep the `dispatch`-never-raises + per-call-manifest guarantees
   that make the agent auditable. RL would add a training stack and forfeit those.

RL re-activates only when (a) the self-check / repeated-sampling loops have logged
enough verifier-graded traces to bootstrap from, and (b) a specific skill is
empirically capped by what prompting + verification can reach. Neither holds today.

## 5. References

- Reflexion: <https://arxiv.org/abs/2303.11366> · Self-Refine: <https://arxiv.org/abs/2303.17651>
- Huang et al., "LLMs Cannot Self-Correct Reasoning Yet" (ICLR'24): <https://arxiv.org/abs/2310.01798>
- Kamoi et al., self-correction survey (TACL'24): <https://arxiv.org/abs/2406.01297>
- Large Language Monkeys (repeated sampling + verifier): <https://arxiv.org/abs/2407.21787> · Archon: <https://arxiv.org/abs/2409.15254>
- Weaver / generator-verifier gap: <https://arxiv.org/abs/2506.18203>
- "Let's Verify Step by Step" (process/checkpoint supervision): <https://arxiv.org/abs/2305.20050>
- MemGPT: <https://arxiv.org/abs/2310.08560>
- Course: Stanford CS329A <https://cs329a.stanford.edu/>
- In-repo: [`docs/science-skills-assessment.md`](science-skills-assessment.md) (GDM 3-tier + autorater), [`evals/skillopt/README.md`](../evals/skillopt/README.md), [`PLAN.md`](../PLAN.md) section 7.2.
