"""Produce a Claude-refined version of the job-config skill.

Inserts a crisp, ordered "Quick decision rules" section near the top of the
skill — distilling the skill's own scattered guidance (and presets) into an
algorithm the frozen target model can follow. Grounded in the systematic
train-split failures: the un-applied ">20K samples -> Tiny" downgrade, the
all-twelve start_months default, inconsistent S1 use, flood patch size, and
context-month defaults. Fully general (no benchmark/test fixtures).
"""
from __future__ import annotations

import pathlib

ENV = pathlib.Path("skillopt/envs/olmoearth_jobconfig/skills")
src = (ENV / "initial.md").read_text(encoding="utf-8")

BLOCK = """## Quick decision rules (easy-to-miss steps)

The per-field sections below remain the source of truth. These short rules just \
make explicit the steps the wizard most often gets wrong — apply them, then \
justify in the `rationale`.

- **Model size — pick the section default, then ADJUST for data volume \
(the most-missed step).** After choosing the default model for the task, look \
at the stated dataset size: if there are **more than ~20,000 labeled samples, \
step the choice DOWN to `tiny`** (abundant data lets a smaller, faster model \
match Base); if there are **fewer than ~2,000 samples, or more than 5 classes, \
step UP to `base`** (representation quality dominates when data is small or \
complex). If no count is given, keep the default. Do not skip this adjustment.

- **`start_months` — default to `[1]`, never all twelve by reflex.** Use the \
growing-season start for crops (`[3,4]` Northern hemisphere, `[9,10]` \
Southern), mid-year `[6]` for peak-canopy properties (tree height, biomass), \
and the full `[1,2,3,4,5,6,7,8,9,10,11,12]` only for targets that are stable \
year-round (e.g. mangrove).

- **Context months (`single_moment_with_context`).** before → soil moisture \
`3`, drought `6`, flood `1`, burn scar `1`; after → flood `2`, burn scar `1`, \
post-event cause `1–2`, otherwise `0`. At least one of before/after must be \
greater than 0. For `single_moment`, set `observation_window_hours` (default \
`12`).

- **Imagery — `["sentinel2"]` only by default.** Add `"sentinel1"` just when \
the signal is texture/structure or needs cloud penetration: soil moisture, \
biomass, flood, oil slick, vessel detection. Do not add S1 for crop type, land \
cover, mangrove, tree height, ecosystem type, or embeddings.

- **Patch size.** 320 m for per-pixel and window tasks; 640 m for flood or \
landscape-scale context; 1280 m for detection (Studio's recommendation); 160 m \
only for narrow features or sparse points.

"""

anchor = "plus the rslearn-side knobs the wizard implies but doesn't show.\n"
idx = src.find(anchor)
if idx == -1:
    raise SystemExit("anchor not found")
cut = idx + len(anchor)
refined = src[:cut] + "\n" + BLOCK + src[cut:]

out = ENV / "refined_claude.md"
out.write_text(refined, encoding="utf-8")
print("wrote", out, "chars=", len(refined), "(+%d)" % (len(refined) - len(src)),
      "approx_tokens=", len(refined) // 4)
