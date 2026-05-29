# SkillOpt harness for the vendored OlmoEarth skills

This directory captures the [microsoft/SkillOpt](https://github.com/microsoft/SkillOpt)
setup used to **measure and improve the three vendored guidance skills**
(`olmoearth-data-prep`, `olmoearth-embeddings`, `olmoearth-studio-job-config`)
against the **local Qwen3.6 backbone** the agent actually runs on.

SkillOpt itself is a separate tool; we keep only our *overlay* here (benchmark
envs, datasets, scripts, configs, results, and a small core patch) so the work
is reproducible from this repo. The skill *content* edits land upstream in
[`2imi9/OlmoEarth-Skills`](https://github.com/2imi9/OlmoEarth-Skills) (the
submodule at `vendor/olmoearth-skills`), and this repo picks them up by bumping
the submodule pointer once those PRs merge.

## What each benchmark measures

Each env gives the **target model** a skill's `SKILL.md` as its system prompt
plus a plain-English task, and scores the JSON it returns against the skill's
**own deterministic oracle** (`recommend.py` / the pitfalls table) — so the
benchmark rewards exactly the behaviour the skill documents. Both the optimizer
and the target are the local **Qwen3.6-35B-A3B (4-bit GGUF, llama.cpp)** server;
no API keys are used.

| env | skill | task → scored output | oracle |
|---|---|---|---|
| `olmoearth_jobconfig` | studio-job-config | task → wizard config (output type, model, time frame, sources, patch) | `recommend.py` PRESETS |
| `olmoearth_embeddings` | embeddings | task → {decision, model, classifier} | `recommend.py:decide()` |
| `olmoearth_dataprep` | data-prep | symptom → {pitfall_id, action} | pitfalls table (hand-keyed) |

## Results (held-out test, temperature 0, deterministic)

| skill | baseline hard | after | shipped |
|---|---|---|---|
| studio-job-config | 0.643 | **0.714** | [OlmoEarth-Skills PR #1](https://github.com/2imi9/OlmoEarth-Skills/pull/1) |
| embeddings | 0.417 | **0.833** | [OlmoEarth-Skills PR #2](https://github.com/2imi9/OlmoEarth-Skills/pull/2) |
| data-prep | 1.000 | 1.000 | — already optimal (no-skill control = 0.375, so the benchmark is real) |

Full breakdown + the SkillOpt-on-Qwen vs Claude-as-optimizer comparison is in
[`RESULTS.md`](RESULTS.md).

## Layout

```
evals/skillopt/
├── README.md            # this file
├── RESULTS.md           # full results + analysis
├── skillopt-core.patch  # 2-file patch to SkillOpt (env registry + thinking-off)
├── overlay/skillopt/envs/olmoearth_*/   # the 3 benchmark envs (copy into a SkillOpt checkout)
├── data/olmoearth_*_split/              # train/val/test items.json per skill
├── configs/olmoearth_jobconfig/         # SkillOpt training config (job-config)
└── scripts/             # eval_skill.py, gen_dataset*.py, make_refined_skill.py, probe_local.py
```

## Reproduce

Prereqs: the local LLM server from [`docs/serving.md`](../../docs/serving.md)
running at `http://localhost:8000/v1` (Qwen3.6 GGUF via llama.cpp). For the
optimizer/analyst path the context must hold a full `SKILL.md`, so launch the
server with a single large slot, e.g. `-c 16384 --parallel 1`.

```bash
# 1. Get SkillOpt at the pinned commit and install it
git clone https://github.com/microsoft/SkillOpt.git && cd SkillOpt
git checkout 75b5c7f31c040b4e8845877f1f2dd664bf366b11
python -m venv .venv && . .venv/Scripts/activate   # or bin/activate
pip install -e .

# 2. Apply our overlay (envs + the core patch + data + scripts + configs)
git apply /path/to/OlmoEarth-Agent/evals/skillopt/skillopt-core.patch
cp -r /path/to/OlmoEarth-Agent/evals/skillopt/overlay/skillopt/envs/olmoearth_* skillopt/envs/
cp -r /path/to/OlmoEarth-Agent/evals/skillopt/data/*            data/
cp -r /path/to/OlmoEarth-Agent/evals/skillopt/configs/*         configs/
cp    /path/to/OlmoEarth-Agent/evals/skillopt/scripts/*.py      olmoearth_local/

# 3. (Re)generate the benchmark datasets from each skill's oracle
python olmoearth_local/gen_dataset.py             # job-config
python olmoearth_local/gen_dataset_embeddings.py
python olmoearth_local/gen_dataset_dataprep.py

# 4. Baseline a skill on the held-out test split (deterministic)
python olmoearth_local/eval_skill.py --env olmoearth_embeddings \
  --skill skillopt/envs/olmoearth_embeddings/skills/initial.md \
  --split test --temperature 0

# 5. Compare the refined skill
python olmoearth_local/eval_skill.py --env olmoearth_embeddings \
  --skill skillopt/envs/olmoearth_embeddings/skills/refined_claude.md \
  --split test --temperature 0

# 6. (optional) Run the full SkillOpt optimizer with Qwen as both roles
python scripts/train.py --config configs/olmoearth_jobconfig/default.yaml \
  --backend qwen_chat --num_epochs 2
```

`gen_dataset*.py` import the vendored skills' `recommend.py` from
`../OlmoEarth Agent/vendor/olmoearth-skills/...`; adjust the path in those
scripts if your checkout layout differs.

## Notes

- The refinements are **general** (decision-precedence and rule clarifications),
  not benchmark fixtures — see the project memory *"Generalize skills, not
  examples."*
- The small local optimizer (Qwen3.6, ~3B active) reliably *finds* the right
  levers but does not always land a robust edit; the shipped changes are
  Claude-authored and validated on the same frozen Qwen target. See `RESULTS.md`.
