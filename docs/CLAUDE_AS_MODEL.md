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
   index, kept from `begin`. The claim check is off, `local`
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
   `contract`, and, for a refused key (`StudioAccessError`), the same
   `hint`, so its envelope is the live one. The first, live call is served
   from its record too. A record cached before the hint was kept has none,
   and replays with the generic hint it had then.
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
olmoearth-agent-bridge step --run-dir D --response-file response.json --call N
olmoearth-agent-bridge step --run-dir D   # starts a run whose first step failed
olmoearth-agent-bridge show --run-dir D --text
olmoearth-agent-bridge prompt           # the system prompt: soul + card + skill index
olmoearth-agent-bridge tools [--run-dir D]
```

`--brief` takes the text itself; `--brief-file -` and `--response-file -`
read stdin. `--call N` names the model call the response answers (the view's
`model_call`): the step is refused before anything runs when another call is
pending, so a response sent again after its call was answered never answers
the next one.

From the Mac, `scripts/oe-bridge.sh` runs the same commands through
`ssh aicr srun` (`step RUN [RESPONSE.json] [--call N] [--text]`). Nothing
runs locally but ssh, and nothing on the login node but srun. A brief or a
response is a file on the Mac. It goes over ssh's stdin, which srun forwards
to the bridge, so its text passes through no shell. Its paths are variables
at the top: `AGENT_DIR`, `RUNS` and `OUTPUTS`. A missing
`~/.config/olmoearth-agent/studio.env` on aicr stops a command before the
bridge runs, so a missing key file leaves no half-made run behind.

The wrapper is read-only by default: it runs the bridge with
`OLMOEARTH_READ_ONLY=1`, so the two Studio write tools
(`olmoearth_create_project`, `olmoearth_submit_prediction`) are not offered,
and the capability card says why. Only the owner sets
`OE_BRIDGE_ALLOW_WRITES=1`, for a key that can write, to offer them.
`olmoearth_request_aoi` is still offered: it calls no Studio endpoint
itself, and it is the web UI that saves the drawn area (answering a refused
save with 403). The run records the setting at `begin` (`brief.json`), and a
step with the other setting is refused. (Until 31 October 2026 read-only was
opt-in, with `OE_BRIDGE_READ_ONLY=1`.)

A response is `{"tool_calls": [{"name": ..., "arguments": {...}}]}` or
`{"content": "..."}`. When a call offers no tools (the turn cap, the answer
checks' rewrite), the bridge refuses tool calls.

Each step prints a view. For a pending call, it holds:

- the messages the model has not seen since its last response (tool results
  as the loop feeds them, spilled when large, and the harness's notes);
- the system prompt: in full on the first call and whenever it changed, else
  its sha256;
- the tools offered and the mode;
- the offered tools' specs (name, description, JSON Schema) on the first call
  and whenever the tools offered change (`tool_specs`). The capability card
  names required arguments only, and an argument a spec does not declare is
  passed through and ignored by its handler, so a misnamed one runs the tool
  on its default: use only the names the specs declare.

For the end, it holds the answer, whether it was revised, marked or forced
by the turn cap, and the notes `olmoearth-agent` prints. While a step holds
the run's lock, `show` adds `step_running` (since when, which Slurm job).

The skills `.claude/skills/olmoearth-agent-model` (playing the model) and
`.claude/skills/olmoearth-agent-bridge` (operations) say how to use it.

### The run directory

It is readable by its owner only (0700), also when `begin` was given an
empty directory that existed; a symlink is refused. Its files hold the
account's names and the tools' whole results.

- `brief.json`: the brief, the history and the settings.
- `responses.jsonl`: one response per model call, with the request's sha256.
- `tool_cache.jsonl`: each handler's outcome, in call order, with the model
  call whose response made it.
- `events.jsonl`: the loop's events from the last replay.
- `pending.json`: the pending call, or the final answer.
- `provenance.json`: the provenance log from the last replay.
- `step.lock`: there only while a step runs (host, pid, Slurm job, start
  time). A second step is refused while it is held. One older than 25
  minutes (srun's 20-minute limit and a margin) is a dead step's, and the
  next step takes it over. It is a file made with `O_EXCL`, not `flock`,
  since successive srun steps run on different nodes.

### A failed step

A step writes the response it was given, then the other files, only once
the loop has stopped cleanly. `pending.json` comes last.

- Give each `scripts/oe-bridge.sh` call the Bash tool's longest timeout
  (600000 ms); the srun limit is 20 minutes, so a slow step can still
  outlast it. After a timeout or a dropped connection the step may still be
  running, or waiting in the queue, on aicr: `ssh -T` has no terminal to hang
  up, so srun may go on (not tested live).
- Then run `show`. If it says a step is running (`step_running`), wait and
  run `show` again later (one more after a few minutes, never a polling
  loop, at most the 20-minute limit); do not send the response meanwhile,
  which the lock would refuse anyway. If the pending call has moved on, the
  step finished: answer the new call.
- A step that failed (an error, the srun time limit) has saved no response,
  and `pending.json` is as it was. When `show` reports no running step and
  the pending call is still the one answered, send the same response file
  again, with `--call N`.
- A step refused for its response (not JSON, unknown keys, a tool call's name
  or arguments, no `tool_calls` or `content`, "No tools are offered on this
  call", another call pending) ran nothing: fix the file and step again.
- The tool calls it ran are in the cache already, as provisional records.
  The same response reuses them. Another response drops them from the first
  call it makes otherwise, or at its end if it makes fewer. A dropped call
  did run, so a Studio write among them may already have happened. The view
  names the dropped calls (`dropped_tool_calls`).
- A step stopped after it saved the response but before it wrote
  `pending.json` is finished by the next step, given the same response or
  none.
- A step killed while it appended a record leaves a cut-off last line; the
  next step drops it (that record was never saved), so a handler cut off
  runs again and a cut-off response counts as not saved.
- If `begin` failed after it made the run (say `OLMOEARTH_API_KEY is not
  set`, or `show` says the first step did not end), fix the cause, then run
  `scripts/oe-bridge.sh step RUN --text` with no response file to start it,
  or begin again under a new run name.

### Costs

- Each command, `show` included, is one ssh connection and one srun step on
  the cpu partition (2 CPUs, 8G, limit 20 minutes, QOS `interactive` unless
  `OE_BRIDGE_QOS` says otherwise), plus any queue wait.
- Nothing is held between steps.
- Each step replays the run from its start. The cached tools cost nothing,
  but the checks run again and the package is imported again, so a step costs
  more as the run grows.
- A run makes one model call per tool-calling turn, one for the answer (at
  the turn cap, the forced one), and at most one for the answer checks'
  rewrite.

### Where the bridge differs from a live run

- **The model's context.** The Claude Code session also holds its own system
  prompt, the skill and its chat with the user. The agent's system prompt,
  messages and tool specs reach it as command output: the specs in the view
  on the first call and whenever the tools offered change. Only the native
  route gives the model the loop's request alone.
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
  time limit) is not recorded, and the next step runs it again; so is one
  whose record was cut off mid-write. A call a failed step finished is kept,
  and dropped if the next response differs (see "A failed step").
- **Settings.** Keep the answer-check switches (`OLMOEARTH_CHECK_NUMBERS`,
  `OLMOEARTH_CHECK_ANSWER`), the read-only switch (`OLMOEARTH_READ_ONLY`,
  which the wrapper sets unless `OE_BRIDGE_ALLOW_WRITES=1`), the soul and the
  code as they were at `begin`. A changed switch stops the step. Any other
  change shows as a divergence.

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
- the command line;
- the registry's `cannot` lines in the replay (the requests compared with
  `LeadAgent`'s);
- a refused key's hint through the replay (the envelope compared with
  `LeadAgent`'s, on the live call and on a later step);
- the run's lock (a second step refused while it is held, a stale one taken
  over, `show` reporting it), a response for another call refused, the
  read-only setting kept, a cut-off last line dropped, the tool specs in the
  view, and a run directory made private when it existed.

Checked live on 31 October 2026: two runs on aicr through
`scripts/oe-bridge.sh`, a Claude Code session as the model, with a read-only
Studio key and `OE_BRIDGE_READ_ONLY=1`. Each was a `begin` and six steps, one
ssh connection and one srun step per command. A `begin` before them showed
the `cannot` bug below, and both runs had its fix: the first ran with
`OE_BRIDGE_READ_ONLY` and that fix, the second also with the label
inventory and `olmoearth_scores_from_studio` merged.

- A run on aicr: both reached a final answer that passed the answer checks,
  not revised and not marked.
- The QOS: `interactive` on the cpu partition, the wrapper's default, was
  accepted. The AICR guide lists `normal` there; `OE_BRIDGE_QOS=normal` was
  not needed.
- srun forwards the wrapper's stdin to the bridge (`--brief-file -`,
  `--response-file -`).
- The queue wait: none was seen. Each step took about 5 to 9 s of wall time
  from the Mac.
- The `begin` before the two runs found that the bridge's copy of the
  registry dropped the registry's `cannot` lines: its system prompt left
  out the two write tools, as it should, and also the capability card's
  line saying why (the `prompt` command, which builds the registry
  directly, showed the line).
  The replay now keeps them, and a test compares the requests with
  `LeadAgent`'s.

Checked live later on 31 October 2026, after the review fixes: three more
runs on aicr, with the wrapper's defaults (no `OE_BRIDGE_` variable set). One
asked about "my kelp prediction" when the account held five and ended with
the model asking which; one ran the label inventory, then
`olmoearth_scores_from_studio` with no patch and `olmoearth_review_set` on
each zone; one ran `olmoearth_review_set_from_result` on a 16 x 16 grid. All
three ended with an answer that passed the answer checks, not revised and not
marked.

- The read-only default: 28 tools were offered, without the two write tools
  (the registry held 30 before the trim; it holds 25 now, so 23 are offered);
  the capability card said why; each run's `brief.json` recorded
  `read_only` true; each run directory was mode 0700.
- The tool specs: in full on the first call, "unchanged since the last call"
  after it.
- The run's lock: while the grid step ran on one compute node, `show` (its
  own srun step) said a step was still running, naming that step's job and
  node, and the same response sent again, from an srun step on another node,
  was refused by the lock; the lock file was gone once the step ended.
- `--call`: the same response sent again after its call was answered was
  refused, "nothing ran".
- With no patch passed, `olmoearth_scores_from_studio` pooled the two kelp
  zones at patch 6 and patch 11, each stated as a fact, in one call.

Not checked yet:

- writes with the read-only key, which are expected to fail with 403
  (unverified with a read-only key): the wrapper leaves the write tools out
  by default, so none was tried;
- a run with a key that can write, with `OE_BRIDGE_ALLOW_WRITES=1`;
- a step with no response file after a failed first step, the wrapper's stop
  on a missing `studio.env`, a stale lock taken over and a cut-off last line
  dropped: unit tests cover them, and no live run has needed them.
