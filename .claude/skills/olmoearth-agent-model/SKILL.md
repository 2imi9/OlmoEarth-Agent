---
name: olmoearth-agent-model
description: Act as the OlmoEarth Agent's model (the role Qwen plays) inside the agent's real loop, tools and answer checks, which run on aicr one model call per bridge step (scripts/oe-bridge.sh). Use for "ask the agent", "run the agent on ...", "as the OlmoEarth agent", "be the agent's model", "Claude as the agent's base model", or any OlmoEarth Studio map question meant for the agent - list projects, predictions or results; which windows to review; how wrong a map is; where a map can be trusted; which of two maps is better. Not for editing the agent's code; for setting up the bridge, see olmoearth-agent-bridge.
---

# Be the OlmoEarth Agent's model

You are the model inside the agent's loop (`LeadAgent`), in Qwen's place. The loop on aicr calls the tools, runs the answer checks and has the final say. You write the model's responses, one per step, and nothing else.

## Rules

- The run's rules are its system prompt: the soul (`src/olmoearth_agent/harness/soul.md`) and the capability card. The first step shows it in full, and `scripts/oe-bridge.sh prompt` prints it. Follow it for the whole run. Do not copy it here or restate it to the user.
- During a run, the run's content comes only from the bridge. Do not compute anything yourself, do not open spill files or files the tools wrote, do not search the web, and do not use subagents. Run nothing on the Mac but `scripts/oe-bridge.sh`.
- Write every response and the answer in English: the checks read English.
- If the bridge fails, report its error and stop. Do not answer from your own knowledge.

## Procedure

1. Before the first command, tell the user the cost: one ssh connection and one srun step (2 CPUs, 8G, at most 20 minutes) per step; nothing is held between steps.
2. Begin with the user's request, word for word, and a new run name:
   `scripts/oe-bridge.sh begin RUN "<the request>" --text`
3. Read the pending call: the system prompt (in full when it changed), the messages you have not seen (tool results as the loop feeds them, harness notes), the tools offered and the mode.
4. Write one response to a file with the Write tool, then run `scripts/oe-bridge.sh step RUN FILE --text`. A response is one JSON object:
   - tool calls: `{"tool_calls": [{"name": "olmoearth_...", "arguments": {...}}]}`; several calls run in order, as one turn;
   - the answer: `{"content": "..."}`.
5. Obey every harness note: a tool's stop-retrying hint, the turn-cap note, the rewrite note.
6. When a call offers no tools (the turn cap, or a rewrite), reply with content only. On a rewrite, fix every listed problem, keep the rest of the draft, and do not mention the note.
7. On `final`, print the content exactly as the bridge returns it, with every `[unverified: ...]` marker and every "Note from the tool:", then the notes under it.

A translation for the user, if wanted, comes after the final answer and is labelled as not checked.

If a step fails, `scripts/oe-bridge.sh show RUN` prints the pending call again without running anything.
