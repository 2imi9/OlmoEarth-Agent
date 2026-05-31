# Contributing to OlmoEarth Agent

Thanks for the interest. This guide is modeled on [NVIDIA earth2studio's contributor workflow](https://nvidia.github.io/earth2studio/userguide/developer/index.html) and adapted for an agent-first codebase.

The agent's contract (the tool catalog, harness dataclasses, and operational rules) lives in [`PLAN.md`](PLAN.md). When in doubt about behavior, that file is the source of truth.

---

## 1. Code of conduct

Contributor behavior follows the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). Be patient, be specific, address ideas not people. Maintainers may close PRs from contributors who repeatedly disregard these norms.

---

## 2. Getting set up

```bash
git clone https://github.com/2imi9/OlmoEarth-Agent.git
cd OlmoEarth-Agent
# Preferred package manager: uv (https://docs.astral.sh/uv/)
uv sync
uv run pre-commit install
```

`pre-commit` is **required**. Contributions that have not run pre-commit will not be reviewed. This rule is borrowed verbatim from earth2studio ([Developer Overview](https://nvidia.github.io/earth2studio/userguide/developer/overview.html)) because it works.

---

## 3. Branch and PR workflow

- **Branch naming:** `2imi9/feature-<short-kebab-slug>` (e.g. `2imi9/feature-studio-mcp-skeleton`, `2imi9/feature-aoi-resolver`). One branch per concern; do not stack unrelated work.
- **No direct commits to `main`.** All changes land via PR.
- **PR target:** `main`.
- **PR title:** release-note quality. The title appears in [`CHANGELOG.md`](CHANGELOG.md) on release (same convention as [earth2studio's PR template](https://github.com/NVIDIA/earth2studio/blob/main/.github/PULL_REQUEST_TEMPLATE.md)).
- **PR body:** describe what changed, why, and how it was tested. Link any related issue with `closes #N`.
- **Merge style:** squash-merge.
- **Force-push policy:** allowed on your own feature branch before review; not allowed on `main`.

A typical session:

```bash
git checkout main && git pull
git checkout -b 2imi9/feature-<slug>
# ... edit, test, commit (signed-off, see §4) ...
git push -u origin 2imi9/feature-<slug>
gh pr create --base main --title "..." --body "..."
```

---

## 4. Commit conventions

- **DCO sign-off is required.** Use `git commit -s -m "..."` so each commit ends with `Signed-off-by: Your Name <your.email@example.com>`. By signing off you certify the [Developer Certificate of Origin 1.1](https://developercertificate.org/): that you wrote the contribution or have the right to submit it under the project's OlmoEarth Artifact License.
- **No AI co-author trailers in commits.** Do NOT add `Co-Authored-By: Claude`, `Co-Authored-By: Codex`, etc. AI involvement, if substantial, belongs in the PR description (see §8), not the commit trailer.
- **Commit message format:** free-form, but readable. A short imperative subject line under 72 characters, blank line, then optional body.
  ```
  Add spatial CV split utility for class-imbalanced AOIs

  Uses k-fold contiguous polygon assignment so train/val splits
  respect spatial autocorrelation. Refuses random splits per
  PLAN.md operational rule §3.6.

  Signed-off-by: Your Name <your.email@example.com>
  ```
- **Conventional Commits** ([spec](https://www.conventionalcommits.org/en/v1.0.0/)) prefixes (`feat:`, `fix:`, `docs:`, etc.) are welcome but not required. Match the surrounding history.

---

## 5. Code style

Configured in `pyproject.toml`:

- **Formatter:** [Black](https://black.readthedocs.io/) (default settings, 88-char line length).
- **Linter / import sorter:** [Ruff](https://docs.astral.sh/ruff/) with `["E", "F", "S", "I", "PERF"]` selected; `E501` (line length) deferred to Black.
- **Type checker:** [mypy](https://mypy-lang.org/): `disallow_untyped_defs = true`; type hints required on all public functions.
- **Docstrings:** [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html): enforced by [interrogate](https://interrogate.readthedocs.io/) at `fail-under = 90`.
- **Syntax level:** Python 3.11+, PEP 604 `|` over `Union` / `Optional`, enforced by pyupgrade.
- **License headers:** every source file carries the OlmoEarth Artifact License SPDX identifier (`LicenseRef-OlmoEarth-Artifact-License`).

These exact tools mirror [earth2studio's `.pre-commit-config.yaml`](https://raw.githubusercontent.com/NVIDIA/earth2studio/main/.pre-commit-config.yaml), pinned versions in our own `.pre-commit-config.yaml` (added in a future PR).

---

## 6. Tests

- **Framework:** [pytest](https://docs.pytest.org/).
- **Coverage gate:** 90%, same as [earth2studio's testing guide](https://nvidia.github.io/earth2studio/userguide/developer/testing.html). CI fails below this.
- **Integration vs unit:** integration tests that hit the live OlmoEarth Studio API are tagged `@pytest.mark.integration` and require `OLMOEARTH_API_KEY` in the environment; unit tests must not.
- **Operational-rule tests:** every rule in `PLAN.md` §3 has a corresponding test in `tests/test_operational_rules.py`. Adding a new rule? Add the test in the same PR.

---

## 7. Documentation

- **Public API:** every function exposed in the tool catalog (`PLAN.md` §1) needs a NumPy-style docstring with `Parameters`, `Returns`, and `Examples`.
- **Spec doc:** if a change alters the tool catalog, harness dataclasses, or operational rules, update `PLAN.md` in the same PR.
- **CHANGELOG:** every PR appends one line to `CHANGELOG.md` under the appropriate [Keep a Changelog v1.1.0](https://keepachangelog.com/en/1.1.0/) section (`Added` / `Changed` / `Deprecated` / `Removed` / `Fixed` / `Security`).

---

## 8. AI-assisted contributions

Coding agents (Claude Code, Cursor, Codex, Aider, …) are welcome to assist contributions. The rules:

1. **You are the author.** The DCO sign-off (`Signed-off-by:`) names a human. AI tools do not certify the DCO. This matches the [OpenInfra Foundation's AI policy](https://openinfra.org/legal/ai-policy/): *"The 'Signed-Off-By' label is a statement that you take responsibility for the entire contents of the commit, including any parts that were generated or assisted by AI tools."*
2. **Understand and verify what you submit.** If an agent wrote it, you must be able to explain it and have tested it. Reviewers may ask you to do so.
3. **Disclose AI involvement in the PR description**, not in commit trailers. A one-liner is enough: *"This PR was written with assistance from Claude Code; all logic reviewed and tested by the contributor."* For minor brainstorming use, disclosure is optional.
4. **No `Co-Authored-By:` AI trailers in commits.** If you need a machine-readable disclosure trailer, use `Assisted-by: <tool name>` in the commit body, per the [Apache / OpenInfra convention](https://openinfra.org/legal/ai-policy/). Default is no trailer.
5. **License compatibility.** Don't paste AI-generated code that the agent retrieved verbatim from an incompatible-licensed source. Most modern coding agents handle this; reviewers will still ask if a passage looks suspect.

Repeated disregard for these rules is grounds for closing the PR. Adapted from [Gradle's AI_POLICY](https://github.com/gradle/gradle/blob/master/AI_POLICY.md) and [Quipucords/Red Hat's AI_POLICY](https://github.com/quipucords/quipucords/blob/main/AI_POLICY.md).

---

## 9. Review process

- One maintainer approval is required to merge.
- Reviewers may post advisory feedback from an automated review bot (e.g. Greptile). Addressing every comment is not required: use judgment.
- Once approved and CI is green, squash-merge with the PR title as the merge commit subject.

---

## 10. References

- [NVIDIA earth2studio CONTRIBUTING + Developer Guide](https://nvidia.github.io/earth2studio/userguide/developer/index.html)
- [Conventional Commits v1.0.0](https://www.conventionalcommits.org/en/v1.0.0/)
- [Keep a Changelog v1.1.0](https://keepachangelog.com/en/1.1.0/)
- [Developer Certificate of Origin 1.1](https://developercertificate.org/)
- [GitHub: creating commits with multiple authors](https://docs.github.com/en/pull-requests/committing-changes-to-your-project/creating-and-editing-commits/creating-a-commit-with-multiple-authors)
- [agents.md spec](https://agents.md/), see this repo's [`AGENTS.md`](AGENTS.md) for agent-onboarding context
- [OpenInfra Foundation AI policy](https://openinfra.org/legal/ai-policy/)
- [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/)
