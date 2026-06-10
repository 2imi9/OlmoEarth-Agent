# Optimizing `olmoearth-studio-job-config` with SkillOpt on local Qwen3.6

Goal: use [microsoft/SkillOpt](https://github.com/microsoft/SkillOpt) to reduce the
known pitfalls in the OlmoEarth vendored skills, running entirely on the
project's local **Qwen3.6-35B-A3B** (4-bit GGUF via llama.cpp) — both the
*optimizer* (proposes skill edits) and the *target* (the frozen agent that runs
the task).

> **2026-06-10 note:** the job-config skill was later rewritten to match
> Studio's real "new model" wizard (patch default 640 m, training-data +
> data-split steps, Ai2-compute framing) and the benchmark dataset was
> regenerated from the updated oracle, so the numbers below are the
> historical (pre-rewrite) run. On the regenerated test split the
> pre-rewrite skill scores 0.357 hard / 0.806 soft (it teaches the old
> 320 m default; patch correct on 5/14), the rewritten `SKILL.md`
> **0.714 hard / 0.906 soft** (patch 13/14, output_type 14/14) — recovering
> the optimized level on the faithful targets.

## Setup

- **Benchmark env** (new): `skillopt/envs/olmoearth_jobconfig/` — a SkillOpt
  `EnvAdapter` that gives the target model the skill + a plain-English EO task
  and scores the wizard config it returns against the skill's own
  `recommend.py` oracle (`PRESETS` + `adjust_for_signals`), so ground truth can
  never drift from the skill's documented behavior.
  - `hard` (0/1) = all five core wizard fields correct (output_type,
    foundation_model, time_frame.mode, patch_size_m, imagery_sources-as-set).
  - `soft` (0-1) = fraction of all checked fields incl. mode-specific time
    sub-fields (a smoother gradient).
- **Dataset**: 37 natural-language tasks (paraphrases of the 14 verified
  presets + model-size signal-flip variants), split 14 train / 9 val / 14 test,
  stratified so the held-out test set carries genuine difficulty.
- **Models**: optimizer + target both `unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS`
  on `http://localhost:8000/v1`. Local adaptations needed:
  - Relaunched llama.cpp with a single large slot (`-c 16384 --parallel 1`) —
    the default `-c 8192` split across 4 slots gave only ~2048 tokens/slot,
    too small for the ~3.7k-token skill.
  - SkillOpt's optimizer path is OpenAI-only; pointed its `openai_compatible`
    client at the same llama.cpp server, and added an opt-in
    `SKILLOPT_OPENAI_COMPAT_EXTRA_BODY` hook so thinking can be disabled (a tiny
    token budget was otherwise consumed entirely by Qwen's reasoning trace).

## Baseline (original skill, frozen Qwen3.6 target, temp 0, deterministic)

| Split | exact-config (hard) | field-average (soft) |
|---|---|---|
| test (n=14) | 0.643 | 0.845 |
| train (n=14) | 0.643 | 0.816 |

Systematic, reproducible failure modes (consistent across splits):

- **`foundation_model`** — the data-volume adjustment (">20K samples → Tiny")
  was never applied; 60K/35K/30K-sample tasks all kept the preset's Base.
- **`start_months`** — model defaulted to all-twelve; oracle wants `[1]`, `[6]`,
  or `[3,4]`.
- **`imagery_sources`**, **`patch_size_m`** (flood→640 missed), **context
  months** — inconsistent.

## Refinement (the validated change)

A short **"Quick decision rules"** section near the top of the skill that makes
the easy-to-miss steps explicit — the data-volume model-size adjustment, the
`start_months` default, context-month defaults, the S1 families, and patch
sizes. (An earlier, longer draft that *restated* a model-size taxonomy
regressed `foundation_model` by conflicting with the oracle; the tightened
version that only adds the missing steps is what shipped.)

| Split | hard: base → refined | soft: base → refined |
|---|---|---|
| **test (held-out)** | 0.643 → **0.714** (+0.07) | 0.845 → **0.888** (+0.043) |
| train | 0.643 → **0.786** (+0.14) | 0.816 → **0.888** (+0.072) |

On train, `foundation_model` went 10/14 → **14/14** and `start_months` 1/9 → 5/9;
on the held-out test every targeted field improved or held. Shipped to
`vendor/olmoearth-skills` on branch `2imi9/jobconfig-quick-rules`.

## SkillOpt-on-Qwen (tool-alone, optimizer = Qwen3.6)

A fully-autonomous SkillOpt run with Qwen3.6 as *both* optimizer and target
(4 steps, patch mode, edit_budget 3): it accepted a new best at step 1
(selection/val hard 0.556 → 0.667), rejected steps 2–4, and emitted a changed
`best_skill.md` (+2,027 chars; 130 calls, 744K tokens, ~15 min).

Its self-reported test improvement was **hard 0.643 → 0.714 (temp 0.3)** — but
re-scored **deterministically (temp 0)** that gain disappears:

| skill (test, temp 0) | hard | soft |
|---|---|---|
| baseline | 0.643 | 0.845 |
| **Claude refinement (shipped)** | **0.714** | **0.888** |
| SkillOpt-on-Qwen `best_skill.md` | 0.643 | 0.804 |

So the weak 3B-active optimizer **found the right levers** — its edits target
`start_months` ("never all 12 by default") and the per-pixel-vs-window
distinction — but couldn't land a *robust* improvement (its temp-0.3 gain was
sampling noise, and deterministically it slightly regressed soft). This is the
expected limitation of a small local optimizer, and is exactly why the
shipped change is the hand-tightened, general-purpose version (Claude as the
optimizer, validated on the same frozen Qwen3.6 target).

## Conclusion

- **Shipped:** `2imi9/OlmoEarth-Skills` PR #1 — a general "Quick decision
  rules" block; robust +0.07 hard / +0.043 soft on held-out test.
- **Stopping point:** the residual misses (e.g. biomass→`[6]`, solar→`[1]`)
  are preset-specific values; encoding them would bake benchmark fixtures into
  the skill, against the general-purpose principle — so job-config tuning stops
  here. The next improvement is to apply the same harness to
  `olmoearth-embeddings` and `olmoearth-data-prep`.

---

# Skill 2: olmoearth-embeddings

Same method, second skill: a new `olmoearth_embeddings` benchmark env (35
plain-English tasks, 14/9/12 split) scored against the skill's own
`recommend.py:decide()` oracle; target = local Qwen3.6.

The baseline model jumped to fine-tuning regardless of compute and over-picked
the Base model. A general **"Quick decision rules"** precedence block fixes the
ordering (goal → <100→kNN → compute gates fine-tune → prod+strong → >2000+strong
→ 100–2000 → default; Base only when classes>5 OR samples<2000).

Held-out **test** (n=12, temperature 0, deterministic):

| skill | hard | soft | decision | model |
|---|---|---|---|---|
| baseline | 0.417 | 0.597 | 8/12 | 6/12 |
| **refined (shipped, PR #2)** | **0.833 (+0.42)** | **0.806 (+0.21)** | 10/12 | 11/12 |

Shipped: `2imi9/OlmoEarth-Skills` **PR #2**. Every fixed case maps to a general
rule, not a benchmark-specific value.

---

# Skill 3: olmoearth-data-prep — already optimal (no change)

A pitfall-diagnosis benchmark env (`olmoearth_dataprep`, 25 generic EO symptom
scenarios, 10/7/8 split): given a data-prep situation/error, name which of the
8 documented pitfalls it is (`pitfall_id`) + the fix action.

Held-out **test** (n=8, temperature 0, deterministic):

| skill | hard | soft | pitfall_id |
|---|---|---|---|
| **no skill (control)** | 0.375 | 0.688 | 3/8 |
| current skill (baseline) | **1.000** | **1.000** | 8/8 |

The no-skill control (0.375) confirms the benchmark is discriminating — the
model cannot assign the right pitfall number without the skill — yet the
**current skill already saturates it (1.000)**. So no change is warranted:
the pitfalls table is already effective, and editing it would be busywork with
only regression risk. "Improve *if necessary*" → not necessary here.

---

# Summary across the three vendored skills

| skill | baseline (test hard) | after | shipped |
|---|---|---|---|
| studio-job-config | 0.643 | **0.714** | PR #1 |
| embeddings | 0.417 | **0.833** | PR #2 |
| data-prep | 1.000 | 1.000 | — (already optimal) |

Method throughout: a SkillOpt benchmark env per skill, scored against the
skill's own oracle, **target served locally on Qwen3.6-35B-A3B (4-bit GGUF,
llama.cpp)**; refinements kept general (no benchmark fixtures), validated
deterministically on a held-out test split, vendored upstream via PR.

## Reproduce

```bash
# 1. big-context local server (single slot)
docker run -d --name oe-llama-so --gpus all -p 8000:8000 \
  -v C:/Users/Frank/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS --host 0.0.0.0 --port 8000 \
  --jinja -ngl 999 -c 16384 --parallel 1 --no-mmap

# 2. baseline vs refined (deterministic)
python olmoearth_local/eval_skill.py --skill skillopt/envs/olmoearth_jobconfig/skills/initial.md --split test --temperature 0
python olmoearth_local/eval_skill.py --skill skillopt/envs/olmoearth_jobconfig/skills/refined_claude.md --split test --temperature 0

# 3. full SkillOpt training (optimizer + target = local Qwen)
PYTHONUTF8=1 SKILLOPT_OPENAI_COMPAT_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}' \
  python scripts/train.py --config configs/olmoearth_jobconfig/default.yaml --out_root outputs/run_qwen2
```
