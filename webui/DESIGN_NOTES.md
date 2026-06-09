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
| accent (links/buttons) | emerald `#0fcb8c`, link `#37cd8f` | `--mint`, kept as **secondary** |
| font | **Manrope** | `--font` (same, via Google Fonts) |
| radii | 4-10 px (crisp, not pillowy) | `--radius` 10 / `--radius-sm` 6 |

## The one deliberate divergence

Asta's primary accent is emerald green. OlmoEarth's brand (its logo, the Ai2
mark) is **pink `#F0529C`**. So the **primary accent here is pink** (logo,
"New run", the AGENT tag, active tab, send button, primary CTAs), with Asta's
emerald kept as a **secondary** (links, card icon chips, the live-status dot,
the reasoning rule). Dark teal is common ground (both Asta *and* the OlmoEarth
PRISM deck use it), so the swap feels native rather than grafted.

## Content mapping (Asta -> OlmoEarth)

- Tagline structure mirrors Asta's ("A *<role>* that *<does X>*. It uses *<scale>*...").
- Segmented tabs: Asta's *Find papers / Generate a report / Analyze data* ->
  **Run a prediction / Analyze results / Prep & configure** (the skill stages).
- Example queries are **real** agent briefs; the capability grid is the **18
  skills** from `SKILLS.md`; the transcript is the **real** `olmoearth_change_detect`
  showcase example.

## How it was checked

Built, then iterated against headless-Chromium screenshots (desktop hero, the
card grid, the transcript, and 390 px mobile) until the layout, type, and color
matched the reference and there were no console errors. The app uses an inner
scroll container (`.scroll`), so full-page captures need to scroll that element,
not the window.

## Status

- ~~Live data~~, **done**: the [FastAPI bridge](../src/olmoearth_agent/serve.py)
  (`olmoearth-agent-serve`) streams real `LeadAgent` runs over SSE, with a
  multi-turn chat + saved history and a Studio project tree.
- **AOI draw-in-chat** (`js/aoi.js`), **done**: a composer "Draw AOI" button
  opens a Leaflet map (Leaflet 1.7.1 + Leaflet.draw 1.0.4, OSM basemap, loaded
  lazily from CDN like `js/attach.js` lazy-loads pdf.js - no build step). The
  drawn rectangle/polygon is stored as a Studio area (`POST /api/areas`) and
  attached as a chip (`addAoiAttachment`) carrying its `area_id` + bbox. The
  agent can also summon the widget: when it calls `olmoearth_request_aoi`,
  `js/run.js` renders an inline "Draw the area" button that seeds a follow-up
  turn. The modal can also **reuse a saved area** (pick one of the selected
  project's existing areas via `GET /api/areas/{id}` to render + attach it,
  no duplicate). Saved areas also appear as an **"Areas" branch under each
  project** in the sidebar tree (`js/projects.js`) and are **draggable into
  the chat** (same `application/x-oe-*` drag pattern as prediction results;
  the drop in `js/attach.js` fetches the geometry and attaches the chip).
  (Leaflet is pinned to 1.7.x because Leaflet.draw 1.0.4 calls the pre-1.8
  `L.Polyline._flat`.) The bridge serves the webui `no-cache`, so edited
  modules show on a normal refresh (no hard-refresh needed).
- **In-chat result visuals** (`js/viz.js`), **done**: tool results render an
  inline visual - result rasters on a Leaflet map (auth-gated tiles via the
  `GET /api/tile` proxy, fit to extent), a trajectory chart for change
  detection, grouped bars for baseline-compare, per-class bars for evaluate,
  and a stat card for `compare_results`. Two rasters (side by side, or dragged
  in) get a **difference-map scan**: both are sampled on a grid (`GET
  /api/pixel-value`) and each cell painted by B-A progressively. The bridge
  caches proxied tiles + pixel values so re-scans are fast.
- **Comparisons panel** (`js/compares.js`), **done**: stores difference-scan
  results (real stats + diff grid) in `localStorage` and re-opens them in a modal;
  saved via an `oe:save-comparison` event so `viz.js` need not import the panel.
  It is one segment of the sidebar switch bar (below).
- **Sidebar switch bar** (`js/switcher.js`, `js/areas.js`), **done**: the rail is
  reorganized Claude-desktop style - an always-visible **Chats** list plus a 3-way
  **segmented switcher** (Projects · Comparisons · **Areas**) feeding one secondary
  pane (only one shown at a time; the active segment is persisted; Chats and the
  pane scroll independently). **Areas** flattens every AOI across *all* projects
  (label = its project), fanning out one `GET /api/projects/{id}/areas` per project
  and **filling progressively** (cached) since Studio has no single "all areas"
  endpoint; rows drag into the chat like tree areas. Projects keeps its shallow
  drill-down tree (so drag-to-attach still works); the per-project Areas branch in
  the tree remains. Built from the Claude-design sidebar spec.
- Still out of scope for the shell: light theme, full keyboard-nav audit, real auth.
- The demo GIF/MP4 in `demo/` (and `screenshots/desktop.png`) are generated by
  [`demo/record_demo.py`](demo/record_demo.py) (Playwright -> ffmpeg): re-run after UI changes.
