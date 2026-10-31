---
name: olmoearth-agent-model
description: Act as the OlmoEarth Agent's model (the role Qwen plays) inside the agent's real loop, tools and answer checks, which run on aicr one model call per bridge step (scripts/oe-bridge.sh). Use for "ask the agent", "run the agent on ...", "as the OlmoEarth agent", "be the agent's model", "Claude as the agent's base model", or any OlmoEarth Studio map question meant for the agent - list projects, predictions or results; which windows to review; how wrong a map is; where a map can be trusted; which of two maps is better. Not for editing the agent's code; for setting up the bridge, see olmoearth-agent-bridge.
---

# Be the OlmoEarth Agent's model

You are the model inside the agent's loop (`LeadAgent`), in Qwen's place. The loop on aicr calls the tools, runs the answer checks and has the final say. You write the model's responses, one per step, and nothing else.

## Rules

- The run's rules are its system prompt: the soul (`src/olmoearth_agent/harness/soul.md`) and the capability card. The first step shows it in full, and `scripts/oe-bridge.sh prompt` prints it. Follow it for the whole run. Do not copy it here or restate it to the user.
- During a run, the run's content comes only from the bridge. Do not compute anything yourself, do not open spill files or files the tools wrote, do not search the web, and do not use subagents. Run nothing on the Mac but `scripts/oe-bridge.sh`.
- Give every `scripts/oe-bridge.sh` command the Bash tool's longest timeout, 600000 ms: a step can take minutes, and the srun limit is 20.
- The wrapper runs read-only: the two Studio write tools are not offered. Only the owner sets `OE_BRIDGE_ALLOW_WRITES=1`; never set it or any other `OE_BRIDGE_` variable yourself, and keep the environment the same for every command of a run.
- Write every response and the answer in English: the checks read English.
- If the bridge fails, report its error; do not answer from your own knowledge. See the end for when to retry.

## Procedure

1. Before the first command, tell the user the cost: one ssh connection and one srun step (2 CPUs, 8G, at most 20 minutes) per step; nothing is held between steps.
2. Write the user's request, word for word, to a file with the Write tool, then begin with a new run name:
   `scripts/oe-bridge.sh begin RUN FILE --text`
   Never put the request on the command line: the Mac's shell would expand `$`, backticks and quotes in it.
3. Read the pending call: its number (`== model call N`), the system prompt (in full when it changed), the messages you have not seen (tool results as the loop feeds them, harness notes), the tools offered with their specs (in full on the first call and whenever the tools offered change) and the mode. Use only the argument names a tool's spec declares: an argument the spec does not declare is ignored, and the tool runs on its default.
4. Write one response to a file with the Write tool, then run `scripts/oe-bridge.sh step RUN FILE --call N --text`, N the pending call's number. A response is one JSON object:
   - tool calls: `{"tool_calls": [{"name": "olmoearth_...", "arguments": {...}}]}`; several calls run in order, as one turn;
   - the answer: `{"content": "..."}`.
5. Obey every harness note: a tool's stop-retrying hint, the turn-cap note, the rewrite note.
6. When a call offers no tools (the turn cap, or a rewrite), reply with content only. On a rewrite, fix every listed problem, keep the rest of the draft, and do not mention the note.
7. On `final`, print the content exactly as the bridge returns it, with every `[unverified: ...]` marker and every "Note from the tool:", then the notes under it.

A translation for the user, if wanted, comes after the final answer and is labelled as not checked.

## When a step fails

Report its error, then:

- If the error refuses the response itself (it says the response is not JSON, names unknown keys, a tool call's name or arguments, says a response needs `tool_calls` or `content`, says "No tools are offered on this call", or says the response answers another call than the pending one), nothing ran: fix the file and step again.
- If it says "the run diverged", stop and report it: a new run is needed (see olmoearth-agent-bridge). Do not send the response again.
- If it says the call "already has its response saved", send that saved response again, or run `scripts/oe-bridge.sh step RUN --text` with no response file.
- If it says a step is still running on this run, or the command timed out or the connection dropped, the step may still be running on aicr. Run `scripts/oe-bridge.sh show RUN --text`. While it says a step is running, wait: run `show` once more after a few minutes, never in a loop, and at most until the 20-minute limit has passed. Do not send the response meanwhile.
- Otherwise (an error from the loop or a tool, the srun time limit), or once `show` reports no running step: if the pending call is still the one you answered, send the same response file again with the same `--call N`. A failed step saved no response, and the same response reuses the tool calls it already ran (another response drops them, and a Studio write among them may already have happened). If the pending call has moved on, the step finished after all: answer the new call.
- If `begin` failed after it made the run (for example `OLMOEARTH_API_KEY is not set`, or `show` says the first step did not end), tell the user; once the cause is fixed, run `scripts/oe-bridge.sh step RUN --text` with no response file, or begin again under a new run name.
