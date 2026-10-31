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

Never poll the cluster in a loop. Give every `scripts/oe-bridge.sh` command the Bash tool's longest timeout, 600000 ms.

## Set up the cluster checkout (once)

- `AGENT_DIR` at the top of `scripts/oe-bridge.sh` is the agent checkout on aicr, with its `.venv` inside.
- Every setup command, git included, runs inside an srun step; on the login node nothing runs but the srun itself. Put the branch there without git on the login node and without pushing it anywhere: make a bundle on the Mac and send it over ssh's stdin, which srun forwards, into one srun step that saves it on scratch and checks it out:
  `git bundle create - BRANCH | ssh -T aicr 'srun -A p2026_0089_neu -p cpu -q interactive -n 1 -c 1 --mem=2G -t 00:10:00 bash -c "umask 077 && cd /scratch/qi_zim_neu/olmoearth_inferenceX/agent-checks/bridge && cat > branch.bundle && cd src && git fetch ../branch.bundle BRANCH && git checkout -B BRANCH FETCH_HEAD"'`
  The first time, in place of `cd src && git fetch ... && git checkout ...`: `git clone -b BRANCH branch.bundle src`.
- Then build the env in another srun step: `ssh -T aicr 'srun -A p2026_0089_neu -p cpu -q interactive -n 1 -c 2 --mem=8G -t 00:20:00 bash -c "cd /scratch/qi_zim_neu/olmoearth_inferenceX/agent-checks/bridge/src && ~/.local/bin/uv sync --extra inferencex --python 3.12"'`.
- The Studio key: `~/.config/olmoearth-agent/studio.env` on aicr, chmod 600, holding `OLMOEARTH_API_KEY=...`. The owner writes it. Never read it, print it or copy it. The wrapper sources it inside the srun step only, and stops before the bridge runs when the file is missing.
- If srun refuses the QOS, run with `OE_BRIDGE_QOS=normal`: the AICR guide lists QOS normal for the cpu partition.

## Smoke test

1. `scripts/oe-bridge.sh tools` prints the tool specs.
2. Write `How many OlmoEarth projects do I have?` to a file with the Write tool, then `scripts/oe-bridge.sh begin smoke-1 FILE --text`, or under the next free name (`smoke-2`, ...) when `smoke-1` exists. The brief never goes on the command line: the Mac's shell would expand `$`, backticks and quotes in it.
3. A response calling `olmoearth_load_context` with `{}` (`step RUN FILE --call 1 --text`), then an answer from its result (`--call 2`).

## The read-only key

- In the 31 Oct probe, every GET, search and aggregate request tried with the read-only key answered 200.
- Writes were never tried. A 403 on a write is expected, since the key's access level shows nowhere else; this is unverified with a read-only key.
- The wrapper is read-only by default: it runs the bridge with `OLMOEARTH_READ_ONLY=1`, so `olmoearth_create_project` and `olmoearth_submit_prediction` are not offered, and the capability card says why. Only the owner sets `OE_BRIDGE_ALLOW_WRITES=1`, for a key that can write. A run records the setting at `begin`, and a step with the other setting is refused.
- With writes allowed and a read-only key, the two write tools should fail with 403. The model gets each failure as a failed tool result, with Studio's refusal as its hint, and the answer says so. It is not a bridge failure.

## A run directory

`$RUNS/RUN` holds these files:

- `brief.json`: the brief and settings (the read-only switch among them);
- `responses.jsonl`: one line per model call, with the request's sha256;
- `tool_cache.jsonl`: each tool call's outcome;
- `events.jsonl`: the loop's events;
- `pending.json`: the pending call, or the final answer;
- `provenance.json`;
- `step.lock`, only while a step runs: who runs it and since when.

## When a step fails

- After a timeout or a dropped connection, the step may still be running on aicr. Run `show RUN --text`: while it says a step is running, wait (one more `show` after a few minutes, at most until the 20-minute limit), and send nothing; a second step is refused while the lock is held. If the pending call has moved on, the step finished: answer the new call.
- A failed step (an error, the srun time limit) saves no response, and `pending.json` stays as it was. Once `show` reports no running step and the same pending call, send the same response file again with the same `--call N`; that also finishes a step stopped between saving the response and writing `pending.json`. `--call N` refuses a response meant for another call, so it never answers the wrong one.
- A response the bridge refuses (not JSON, unknown keys, a tool call's name or arguments, "No tools are offered on this call", another call pending) ran nothing: fix the file and step again.
- Tool calls a failed step already ran are kept: the same response reuses them, and another response drops them. A dropped call did run; the view's `dropped_tool_calls` (a `bridge:` line with `--text`) names them, and a Studio write among them may already have happened.
- "the run diverged at call k" or "at model call k": the loop no longer makes the calls the saved responses made. This happens when the code, the environment or the cache changed between steps. Start a new run; do not edit the files.
- `studio.env is missing on aicr`, or `OLMOEARTH_API_KEY is not set`: the env file is missing or empty on aicr. If `begin` had made the run, then once the owner has written the file, run `scripts/oe-bridge.sh step RUN --text` with no response file to start it, or begin under a new run name.
