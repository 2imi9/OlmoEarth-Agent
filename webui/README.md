# OlmoEarth Agent — web UI

A front-end shell for the OlmoEarth Agent, styled after [Ai2 **Asta**](https://asta.allen.ai/)
(dark-teal canvas, cream text, Manrope, a centered prompt hero, a left sidebar)
and rebranded with **OlmoEarth** elements (the pink Ai2/OlmoEarth logo, the
EO/Studio content, the 15-skill catalog).

By default it's a **static mock** — no build step, no framework, no tracking —
so the chat, the projects tree, and the "what a run looks like" transcript are
illustrative sample data. Serve it with the **bridge** (`olmoearth-agent-serve`,
see [Live mode](#live-mode-the-bridge)) and it upgrades in place to the **live
agent**: a multi-turn **chat with saved history**, your **real Studio projects as
a drill-down tree** (project → model/embeddings → predictions → results), and
briefs streamed through `LeadAgent` over Server-Sent Events. Chat history is kept
client-side (localStorage), same as the key.

![OlmoEarth Agent — live demo](demo/olmoearth-agent-demo.gif)

*A brief in, the agent loop streamed out — reasoning, the `olmoearth_change_detect`
call, the result, and a plain-English answer ([MP4](demo/olmoearth-agent-demo.mp4)).
Demo runs are scripted (see [`app.js`](app.js) `runDemo`); served by the bridge, the same UI streams the real `LeadAgent`.*

![desktop](screenshots/desktop.png)

## View it

```bash
# from the repo root (port 8080 — the LLM owns 8000)
python -m http.server 8080 --directory webui
# then open http://localhost:8080
```

Or just open `webui/index.html` in a browser.

## Files

| File | What |
|---|---|
| `index.html` | markup — sidebar, prompt hero, capability grid, example transcript, footer |
| `styles.css` | the design system (tokens at `:root`) — no preprocessor |
| `app.js`     | renders the 15 skill cards + tab / example / mobile-menu interactions |
| `assets/OlmoEarth-logo.png` | the official OlmoEarth wordmark |
| `screenshots/` | reference renders (desktop, transcript, mobile) |

## Design

Palette and type were taken from Asta's *computed* styles (not guessed):
dark teal **`#032629`**, cream **`#faf2e9`**, **Manrope**, and Asta's emerald
**`#0FCB8C`** kept as a secondary accent. The **primary accent is OlmoEarth
pink `#F0529C`** (the brand) — so it reads as Asta's calm, minimal aesthetic but
unmistakably OlmoEarth. See [`DESIGN_NOTES.md`](DESIGN_NOTES.md).

Responsive: the sidebar collapses to a hamburger and the card grid reflows to a
single column below 880 px.

## Account (bring your own key)

There's no login to build — a user with a Studio assignment **already has an
OlmoEarth Studio API key** (Studio → profile → API Keys). They paste it into the
sidebar ("Connect OlmoEarth Studio"); it's kept **client-side** (localStorage).
In the **static mock** nothing is sent anywhere — the UI just unlocks the sample
projects and says *"Connected · demo"*. Under the **bridge** the same key is
forwarded per request (header `X-Olmoearth-Key`) so calls hit *their own* Studio
account, and the card switches to *"Studio connected"*. Email-based sign-in is
planned for later and out of scope here.

## Live mode (the bridge)

The static shell upgrades to the real agent via a small FastAPI bridge
(`olmoearth_agent.serve`) that serves this `webui/` **and** exposes the agent:

```bash
uv sync --extra serve                  # install fastapi + uvicorn
scripts/serve-llm.sh                   # bring up the LLM (llama.cpp, port 8000)
export OLMOEARTH_API_KEY=…             # optional; the UI key is used per-request
uv run olmoearth-agent-serve           # bridge on 127.0.0.1:8088, serves webui/
# open http://127.0.0.1:8088, paste your Studio key, run a brief
```

Endpoints:

| Route | Purpose |
|---|---|
| `GET /api/health` | the front-end probes this to flip from demo → live |
| `GET /api/projects` | your real Studio projects (`load_context`) |
| `POST /api/run` | streams `LeadAgent.run_stream` as Server-Sent Events |

The browser sends your Studio key in the `X-Olmoearth-Key` header; the bridge
forwards it per request and never stores it. Opened as a plain file — or served
by anything that isn't the bridge — `/api/health` 404s and the page stays in
demo mode. A full live run needs the LLM served (`scripts/serve-llm.sh`) and a
valid Studio key.

*Styling inspired by Ai2 Asta. This is a research-demo UI, not an official
product page.*
