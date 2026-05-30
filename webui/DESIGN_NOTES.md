# Design notes

How the look was derived, so it can be revised deliberately.

## Source of truth

The palette and type are sampled from **asta.allen.ai's live computed styles**
(rendered DOM, not a screenshot guess). Asta's actual tokens:

| Token | Asta value | Used here as |
|---|---|---|
| canvas background | `#032629` | `--bg` (same) |
| body text | `#faf2e9` (warm cream, *not* white) | `--text` (same) |
| panels / elevations | `#0a3235`, `#105257`, `#153a3c` | `--bg-2`, `--panel-2`, `--line` |
| accent (links/buttons) | emerald `#0fcb8c`, link `#37cd8f` | `--mint` — kept as **secondary** |
| font | **Manrope** | `--font` (same, via Google Fonts) |
| radii | 4–10 px (crisp, not pillowy) | `--radius` 10 / `--radius-sm` 6 |

## The one deliberate divergence

Asta's primary accent is emerald green. OlmoEarth's brand (its logo, the Ai2
mark) is **pink `#F0529C`**. So the **primary accent here is pink** (logo,
"New run", the AGENT tag, active tab, send button, primary CTAs), with Asta's
emerald kept as a **secondary** (links, card icon chips, the live-status dot,
the reasoning rule). Dark teal is common ground — both Asta *and* the OlmoEarth
PRISM deck use it — so the swap feels native rather than grafted.

## Content mapping (Asta → OlmoEarth)

- Tagline structure mirrors Asta's ("A *<role>* that *<does X>*. It uses *<scale>*…").
- Segmented tabs: Asta's *Find papers / Generate a report / Analyze data* →
  **Run a prediction / Analyze results / Prep & configure** (the skill stages).
- Example queries are **real** agent briefs; the capability grid is the **15
  skills** from `SKILLS.md`; the transcript is the **real** `olmoearth_change_detect`
  showcase example.

## How it was checked

Built, then iterated against headless-Chromium screenshots (desktop hero, the
card grid, the transcript, and 390 px mobile) until the layout, type, and color
matched the reference and there were no console errors. The app uses an inner
scroll container (`.scroll`), so full-page captures need to scroll that element,
not the window.

## Not done yet

- Live data — the prompt/transcript are static; wire to `LeadAgent`.
- Light theme, keyboard-nav audit, and real auth are out of scope for the shell.
