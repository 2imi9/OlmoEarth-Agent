# Claude as the agent's model

The agent's model sits behind one seam: `chat(messages, tools=..., mode=...)`
(`src/olmoearth_agent/llm/types.py`). Qwen plays it through `OlmoEarthLLM`.
Claude can play it two ways.

| | Native (`AnthropicLLM`) | Replay bridge (`olmoearth-agent-bridge`) |
|---|---|---|
| Who is the model | Claude through the Anthropic API | a Claude Code session |
| Needs | the owner's Anthropic API key, the `claude` extra | ssh to aicr, a cluster checkout, the Studio key there |
| Pays | per token, on the owner's Anthropic account | the Claude Code session; one ssh + one srun per model call |
| Where it runs | the web UI (`serve.py`) | the loop on aicr, the model on the Mac |
| The model sees | exactly the loop's request | the loop's request as text, beside its own Claude Code context |

## The native route

- `src/olmoearth_agent/llm/anthropic_client.py` (`AnthropicLLM`) has the same
  `chat()` seam and returns the same `ChatResponse`. The `claude` extra
  installs `anthropic`.
- `serve.py` wires it. In the web UI, open Settings -> LLM backend, pick
  Claude and paste a key. The key goes with each request (`X-LLM-Key`) and is
  not stored.
- The command line (`olmoearth-agent`) always builds `OlmoEarthLLM`, so it has
  no Claude switch.
- This is the clean route. The soul is Claude's system prompt, the tools are
  native tool specs, and nothing else reaches the model.

## The replay bridge

A Claude Code session writes the model's responses, one model call per
command. The agent's real loop, tools and checks (`LeadAgent.run_stream`,
unchanged) run on aicr. Nothing is held between commands: no process, no
allocation, no MCP config.

### How one step works

Each command is a fresh process on a run directory:

1. It builds a `LeadAgent` as `olmoearth-agent` does. That means the default
   registry, `StudioClient.from_env()`, a fresh `ThreadState`, and the skill
   index and memory block, kept from `begin`. The claim check is off, `local`
   is off, and the surface is the command line's.
2. `ScriptedLLM` stands in for the model. Its k-th call returns the model's
   k-th response (`responses.jsonl`), with tool-call ids `call_<k>_<j>`.
3. Each tool handler is wrapped, not `dispatch`. So the registry's argument
   check, envelopes, retry hint and failure counts run as live, and so do the
   loop's spill, records and provenance.
4. A wrapped handler serves its result from `tool_cache.jsonl` when the cache
   has it. Only a call past the cache runs, and its outcome is appended at
   once, with the model call whose response made it.
5. A cached exception is raised again with the same class name, text and
   `contract`, so its envelope is the live one.
6. `olmoearth_load_skill` runs again on every step. It sets the run's loaded
   tool groups, which a fresh process does not have.
7. The step stops at the first model call nothing answers yet. It writes that
   call to `pending.json` and cancels the loop. If the loop ends instead, the
   step writes the final answer.

Every recorded response keeps the sha256 of the request it answered: the
system prompt, the messages, the offered tools' specs and the mode. A replay
that makes another request, or calls another tool or arguments than the cache
holds, stops with "the run diverged at model call k" or "at call k". It never
goes on with a run the model did not see.

### Commands

On aicr (each a fresh process):

```bash
olmoearth-agent-bridge begin --run-dir D --brief-file brief.txt
olmoearth-agent-bridge step --run-dir D --response-file response.json
olmoearth-agent-bridge show --run-dir D --text
olmoearth-agent-bridge prompt           # the system prompt: soul + card + skill index
olmoearth-agent-bridge tools [--run-dir D]
```

`--brief` takes the text itself; `--brief-file -` and `--response-file -`
read stdin.

From the Mac, `scripts/oe-bridge.sh` runs the same commands through
`ssh aicr srun`. Nothing runs locally but ssh, and nothing on the login node
but srun. A brief or a response is a file on the Mac. It goes over ssh's
stdin, which srun forwards to the bridge, so its text passes through no
shell. Its paths are variables at the top: `AGENT_DIR`, `RUNS` and `OUTPUTS`.

A response is `{"tool_calls": [{"name": ..., "arguments": {...}}]}` or
`{"content": "..."}`. When a call offers no tools (the turn cap, the answer
checks' rewrite), the bridge refuses tool calls.

Each step prints a view. For a pending call, it holds:

- the messages the model has not seen since its last response (tool results
  as the loop feeds them, spilled when large, and the harness's notes);
- the system prompt: in full on the first call and whenever it changed, else
  its sha256;
- the tools offered and the mode.

For the end, it holds the answer, whether it was revised, marked or forced
by the turn cap, and the notes `olmoearth-agent` prints.

The skills `.claude/skills/olmoearth-agent-model` (playing the model) and
`.claude/skills/olmoearth-agent-bridge` (operations) say how to use it.

### The run directory

It is readable by its owner only (0700): its files hold the account's names
and the tools' whole results.

- `brief.json`: the brief, the history and the settings.
- `responses.jsonl`: one response per model call, with the request's sha256.
- `tool_cache.jsonl`: each handler's outcome, in call order, with the model
  call whose response made it.
- `events.jsonl`: the loop's events from the last replay.
- `pending.json`: the pending call, or the final answer.
- `provenance.json`: the provenance log from the last replay.

### A failed step

A step writes the response it was given, then the other files, only once
the loop has stopped cleanly. `pending.json` comes last.

- A step that failed (an error, the srun time limit) has saved no response,
  and `pending.json` is as it was. Send the same response again. After a
  dropped connection, check with `show` first: the step may have finished on
  the cluster, and then the same response would answer the next call.
- The tool calls it ran are in the cache already, as provisional records.
  The same response reuses them. Another response drops them from the first
  call it makes otherwise, or at its end if it makes fewer. A dropped call
  did run, so a Studio write among them may already have happened. The view
  names the dropped calls (`dropped_tool_calls`).
- A step stopped after it saved the response but before it wrote
  `pending.json` is finished by the next step, given the same response or
  none.

### Costs

- Each command, `show` included, is one ssh connection and one srun step on
  the cpu partition (2 CPUs, 8G, limit 20 minutes), plus any queue wait.
- Nothing is held between steps.
- Each step replays the run from its start. The cached tools cost nothing,
  but the checks run again and the package is imported again, so a step costs
  more as the run grows.
- A run makes one model call per tool-calling turn, one for the answer (at
  the turn cap, the forced one), and at most one for the answer checks'
  rewrite.

### Where the bridge differs from a live run

- **The model's context.** The Claude Code session also holds its own system
  prompt, the skill and its chat with the user. The agent's system prompt and
  messages reach it as command output, and the tool specs as `tools` prints
  them. Only the native route gives the model the loop's request alone.
- **Sampling.** The mode is shown but sets nothing: there is no temperature
  to set. No reasoning is passed back unless a response has `"thinking"`.
- **Results through JSON.** Tool results go through JSON once, on the live
  call too, so every step reads the same value. A tuple comes back as a
  list. The text the model reads is unchanged.
- **Run state not restored.** `olmoearth_load_context` sets
  `ThreadState.studio_context` and `olmoearth_submit_prediction` appends to
  `prediction_ids`. Both are served from the cache, so neither field is set
  on a replay. No code reads either field at this version.
- **Spill file names.** The live loop names a spilled result
  `<tool>_<uuid4 hex>.json`. During a step, the n-th spill of a run takes the
  same name every time (from the run's id and n), so the path the model read
  stays the path the run wrote.
- **Provenance times.** Each replay renews the timestamps.
  `provenance.json` is the last step's. Its run id is the bridge run's id.
- **The claim check.** It is off, as by default. A claim-check call reaching
  `ScriptedLLM` stops the step.
- **A killed step.** A handler cut off mid-call (a killed process, the srun
  time limit) is not recorded, and the next step runs it again. A call a
  failed step finished is kept, and dropped if the next response differs
  (see "A failed step").
- **Settings.** Keep the answer-check switches (`OLMOEARTH_CHECK_NUMBERS`,
  `OLMOEARTH_CHECK_ANSWER`), the soul and the code as they were at `begin`.
  A changed switch stops the step. Any other change shows as a divergence.

### Status

Written 31 October 2026. The tests in `tests/bridge/` cover:

- the replay against `LeadAgent` (the same requests, events, provenance and
  answer);
- the cache;
- a failed step, retried with the same or another response;
- divergence;
- the rewrite call;
- the spill names;
- skill loading;
- the command line.

Not checked yet:

- a run on aicr;
- the QOS on the cpu partition: the AICR guide lists `normal` there, and
  `OE_BRIDGE_QOS` overrides it;
- the queue wait;
- srun forwarding the wrapper's stdin to the bridge (Slurm's default); if it
  does not, the bridge reads an empty brief or response and refuses it;
- writes with the read-only key, which are expected to fail with 403
  (unverified with a read-only key).
