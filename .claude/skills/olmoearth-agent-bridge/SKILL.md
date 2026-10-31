---
name: olmoearth-agent-bridge
description: Operate the OlmoEarth Agent's replay bridge on aicr (scripts/oe-bridge.sh, olmoearth-agent-bridge) - set up the cluster checkout and its .venv, smoke-test it, read a run directory, and know what the read-only Studio key allows. Use before the first run of olmoearth-agent-model, when a bridge step fails, or for "set up the bridge", "bridge smoke test", "where are the bridge runs", "why did the run diverge".
---

# Operate the agent bridge

How the bridge works and what it costs: `docs/CLAUDE_AS_MODEL.md`.

## State the cost first

Before any command, tell the user what it uses:

- each bridge command, `show` included: one ssh connection and one srun step on the cpu partition (2 CPUs, 8G, limit 20 minutes); on the login node only that srun runs;
- nothing is held between commands;
- disk: run directories and the tools' outputs on scratch (`RUNS`, `OUTPUTS` at the top of `scripts/oe-bridge.sh`), purged after 30 days, readable by the account only.

Never poll the cluster in a loop.

## Set up the cluster checkout (once)

- `AGENT_DIR` at the top of `scripts/oe-bridge.sh` is the agent checkout on aicr, with its `.venv` inside.
- Put the branch to test there with git. Then build the env inside an srun job, never on the login node: `cd $AGENT_DIR && ~/.local/bin/uv sync --extra inferencex --python 3.12`.
- The Studio key: `~/.config/olmoearth-agent/studio.env` on aicr, chmod 600, holding `OLMOEARTH_API_KEY=...`. The owner writes it. Never read it, print it or copy it. The wrapper sources it inside the srun step only.
- If srun refuses the QOS, run with `OE_BRIDGE_QOS=normal`: the AICR guide lists QOS normal for the cpu partition.

## Smoke test

1. `scripts/oe-bridge.sh tools` prints the tool specs.
2. Write `How many OlmoEarth projects do I have?` to a file with the Write tool, then `scripts/oe-bridge.sh begin smoke-1 FILE --text`. The brief never goes on the command line: the Mac's shell would expand `$`, backticks and quotes in it.
3. A response calling `olmoearth_load_context` with `{}`, then an answer from its result.

## The read-only key

- In the 31 Oct probe, every GET, search and aggregate request tried with the read-only key answered 200.
- Writes were never tried. A 403 on a write is expected, since the key's access level shows nowhere else; this is unverified with a read-only key.
- So `olmoearth_create_project` and `olmoearth_submit_prediction` should fail. The model gets each failure as a failed tool result, and the answer says so. It is not a bridge failure.
- With a read-only key, run every command of a run with `OE_BRIDGE_READ_ONLY=1` (it sets `OLMOEARTH_READ_ONLY=1`): the two write tools are then not offered, and the capability card says why. Changing it in the middle of a run shows as a divergence.

## A run directory

`$RUNS/RUN` holds these files:

- `brief.json`: the brief and settings;
- `responses.jsonl`: one line per model call, with the request's sha256;
- `tool_cache.jsonl`: each tool call's outcome;
- `events.jsonl`: the loop's events;
- `pending.json`: the pending call, or the final answer;
- `provenance.json`.

## When a step fails

- A failed step (an error, the srun time limit) saves no response, and `pending.json` stays as it was. Send the same response file again; that also finishes a step stopped between saving the response and writing `pending.json`. After a dropped connection, run `show RUN --text` first: if the pending call has moved on, the step finished, and the same file would answer the wrong call.
- Tool calls a failed step already ran are kept: the same response reuses them, and another response drops them. A dropped call did run; the view's `dropped_tool_calls` (a `bridge:` line with `--text`) names them, and a Studio write among them may already have happened.
- "the run diverged at call k" or "at model call k": the loop no longer makes the calls the saved responses made. This happens when the code, the environment or the cache changed between steps. Start a new run; do not edit the files.
- `OLMOEARTH_API_KEY is not set`: the env file is missing on aicr.
