# Instruction skills

The four [agentskills.io](https://agentskills.io) packages the agent loads with `olmoearth_load_skill`
(catalog #1-#3 and #17 in [`SKILLS.md`](../../../../SKILLS.md)). Each folder holds a `SKILL.md`, and
most hold `references/` and `scripts/`. The folder ships inside the `olmoearth_agent` wheel, and
`olmoearth_agent.skills.loader.SkillLoader` reads it by default (`OLMOEARTH_SKILLS_DIR` overrides it).

| Folder | Catalog |
|---|---|
| `olmoearth-data-prep` | #1 |
| `olmoearth-studio-job-config` | #2 |
| `olmoearth-embeddings` | #3 |
| `olmoearth-rslearn` | #17 |

## Origin and licence

Moved from [`2imi9/OlmoEarth-Skills`](https://github.com/2imi9/OlmoEarth-Skills) at commit `6839f1a`.
That repository is frozen and points here; these copies are now the maintained ones. The packages
are under the MIT License, copyright (c) 2026 Ziming Qi: see [`LICENSE`](LICENSE) in this folder. The
rest of the agent is under the OlmoEarth Artifact License (the repository's root `LICENSE`).

Changes since the move:

- `scripts/*.py`: an SPDX header (`MIT`) added to each script.
- `olmoearth-rslearn/SKILL.md`: the section on running Python described a sandbox with `rslearn`
  preloaded and `import` banned. It now describes `olmoearth_run_python` as it works
  (`src/olmoearth_agent/tools/system.py`): opt-in, a fresh isolated subprocess per call, normal
  imports, nothing preloaded.

The scripts are standalone command-line helpers for a user's own machine; the agent does not run
them. They are excluded from the repository's black, ruff and mypy hooks.
