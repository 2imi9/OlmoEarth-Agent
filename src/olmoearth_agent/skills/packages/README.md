# Instruction skills

The [agentskills.io](https://agentskills.io) package the agent loads with `olmoearth_load_skill`
(catalog #1 in [`SKILLS.md`](../../../../SKILLS.md)). Its folder holds a `SKILL.md`, `references/`
and `scripts/`. The folder ships inside the `olmoearth_agent` wheel, and
`olmoearth_agent.skills.loader.SkillLoader` reads it by default (`OLMOEARTH_SKILLS_DIR` overrides it).

| Folder | Catalog |
|---|---|
| `olmoearth-studio-job-config` | #1 |

## Origin and licence

Moved from [`2imi9/OlmoEarth-Skills`](https://github.com/2imi9/OlmoEarth-Skills) at commit `6839f1a`.
That repository is frozen and points here; this copy is now the maintained one. The package is
under the MIT License, copyright (c) 2026 Ziming Qi: see [`LICENSE`](LICENSE) in this folder. The
rest of the agent is under the OlmoEarth Artifact License (the repository's root `LICENSE`).

Changes since the move:

- `scripts/*.py`: an SPDX header (`MIT`) added to each script.
- `olmoearth-data-prep`, `olmoearth-embeddings` and `olmoearth-rslearn` were removed on 31 October
  2026 as training-side (label preparation, an embeddings-vs-fine-tune notebook, running rslearn
  training), outside what the agent verifies; git history keeps them.
- `olmoearth-studio-job-config/SKILL.md` and `references/data_split.md` no longer route to the
  removed `olmoearth-data-prep` skill or to a self-run training skill: they say the agent does not
  prepare labels or configure training outside Studio.

The scripts are standalone command-line helpers for a user's own machine; the agent does not run
them. They are excluded from the repository's black, ruff and mypy hooks.
