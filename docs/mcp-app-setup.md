# MCP App: quick setup

This installs `olmoearth-agent-mcp`, the OlmoEarth Agent as an MCP App, from this repository's main branch with
[uv](https://docs.astral.sh/uv/). Claude Desktop shows its panels, in a chat and in its Code tab. Claude Code in a
terminal gets the same tools, as text.

Agents: do only these steps, in order, and ask the user where a step says so. If a step fails, stop and tell the user
what you saw.

**Done when:** a new Claude Desktop chat shows the start card, and after the user signs in its jobs are on. In
Claude Code in a terminal: `/mcp` shows `olmoearth-agent`, and Claude names the jobs.

## Rules for the agent

- Never ask for, read, print, store or write the user's Studio API key: not in the config, not in the environment,
  not in the chat. The user signs in on the page the app opens. If the user pastes a key, tell them to revoke it in
  Studio and sign in on the page.
- Do not open `~/.config/olmoearth-agent/`: it may hold the saved key.
- Never open, read, copy or edit Claude Desktop's config or its backups (`claude_desktop_config.json.bak-...`, beside
  it) yourself: each can hold the Studio key and other servers' tokens, and a file you read goes to the model whether
  you print it or not. The install command (step 3) edits the config without showing it.
- Show the user the install command (step 3) and ask before running it.
- Ask before installing uv or git. Never use sudo.
- Install only from `git+https://github.com/2imi9/OlmoEarth-Agent` (main), a release tag from v1.5.0 on, or a commit
  of it; never from PyPI or a tag up to v1.4.0: they have no MCP App. A `pip install` line in an error message does not apply here.
- The user quits and reopens the host. Do not stop its process yourself.

## Steps

### 1. Find the system and the host

```bash
uname -s
```

`Darwin` is macOS and `Linux` is Linux. Anything else (Git Bash prints `MINGW64_NT-…`) is Windows: run the PowerShell
lines in PowerShell, not Git Bash. Ask the user where they use Claude: the Claude Desktop app, in a chat or its Code
tab (panels); Claude Code in a terminal (text; it does not read Claude Desktop's config: for both, step 3 takes
`--claude-code`; for Claude Code alone, use the by-hand `claude mcp add` line at the end of step 3); or both.

### 2. Check uv and git

```bash
uv --version
git --version
```

Success: each prints a version. If uv is missing, ask the user before running uv's official installer, and say that it
runs a script from astral.sh and may edit their shell startup files to put uv on the PATH:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

`brew install uv` works too.

uv fetches this repository with git. If `git --version` fails, ask the user, then install it: on macOS
`xcode-select --install` (the user clicks Install); on Windows the user installs
[Git for Windows](https://git-scm.com/downloads/win); on Linux the user installs `git` with the system's package
manager. On Windows, git is on the PATH only in a new terminal.

### 3. Run the install command

Show the user this command, say what it does (below), and ask before running it. The same line works in PowerShell:

```bash
uvx --python 3.12 --from "olmoearth-agent[mcp,inferencex] @ git+https://github.com/2imi9/OlmoEarth-Agent" olmoearth-agent-mcp install
```

Add `--claude-code` at the end when the user also wants Claude Code in a terminal. Just after installing uv, `uvx`
may not be found until a new terminal: use its full path, in the folder the installer printed after `installing to`
(by default `.local/bin` in the home folder).

The command always writes Claude Desktop's config too, and makes the file and its folder when Claude Desktop is not
installed; its last line about quitting Claude Desktop then does not apply. To set up Claude Code alone and leave
Claude Desktop's config untouched, skip this command: find uvx's full path, run the `--help` download line, then the
`claude mcp add` line, all under "By hand" below.

What the command does:

- It downloads Python 3.12 (if missing), this repository and its packages once. That can take a few minutes; it is
  done now so that the first start fits in the time Claude waits for a server (Claude Code waits 30 seconds).
- It adds the `olmoearth-agent` entry to Claude Desktop's config, with uvx's full path (Claude Desktop does not start
  a command from your shell). It never shows the file. Every other entry and setting stays as it is; an
  `olmoearth-agent` entry already there keeps its `env` block; its command and package are written anew (a pin is
  kept only with `--ref`).
- It makes a backup beside the config first (`claude_desktop_config.json.bak-<date>-<time>`), and says where. The
  backup is a full copy, with the same keys as the config. One is left for each change; the user deletes old ones
  once everything works. Agents: do not open them.
- With `--claude-code` it runs `claude mcp add --scope user` for the same command, or prints that line when `claude`
  is not on the PATH. Claude Code in a terminal does not read Claude Desktop's config. An `olmoearth-agent` entry
  already in Claude Code is removed and added again, without its `--env` settings, and the command says so.
- It sends nothing anywhere, except uv's own download from GitHub and PyPI. It does not open the saved Studio key.

Success: it prints `Added olmoearth-agent to Claude Desktop's config:` (or `Updated olmoearth-agent in …`, or
`Already set up …`) with the file's path, then what to do next. If it stops, it says why (with the last lines of
uv's or Claude Code's output when one of them failed). A config it could not write is as it was. With `--claude-code`,
Claude Desktop's config is written first: if only the Claude Code step then fails, Claude Desktop's entry stays
written and the command says so; if Claude Code already had an `olmoearth-agent` entry and could not add the new one,
it says that the old one is gone and shows the line to run. On Windows with a Microsoft Store copy there are two
config files; if the second cannot be written, the first is already changed, and its backup is beside it.

More: `install --dry-run` only shows the file and the entry it would write; `--ref v1.5.0` pins a release or a commit;
`--config PATH` names another config file; `--no-download` skips the download. The Claude Desktop config's own place:

| System | Path |
|---|---|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

On Windows the command also writes each Microsoft Store copy of the config it finds, and names it.

#### By hand, if the command cannot run

The user, not the agent, edits the config: in Claude Desktop, Settings, Developer, Edit Config opens it (and makes it
if there is none). Find uvx's full path with `command -v uvx` (PowerShell: `(Get-Command uvx).Source`; in JSON, write
each `\` as `\\`). Download everything first, in a terminal:

```bash
uvx --python 3.12 --from "olmoearth-agent[mcp,inferencex] @ git+https://github.com/2imi9/OlmoEarth-Agent" olmoearth-agent-mcp --help
```

It prints `usage: olmoearth-agent-mcp` and ends; `--help` starts no server and makes no output folder. Then the user
adds this entry inside `mcpServers` (adding `mcpServers` if it is missing; an empty file takes the whole block), in
place of any old `olmoearth-agent` entry, and saves. An old entry may hold their key, so they never paste it into the
chat.

```json
{
  "mcpServers": {
    "olmoearth-agent": {
      "command": "/absolute/path/to/uvx",
      "args": ["--python", "3.12", "--from",
               "olmoearth-agent[mcp,inferencex] @ git+https://github.com/2imi9/OlmoEarth-Agent",
               "olmoearth-agent-mcp"]
    }
  }
}
```

For Claude Code in a terminal, by hand (show the user the line and ask before running it):

```bash
claude mcp add --scope user --transport stdio olmoearth-agent -- /absolute/path/to/uvx --python 3.12 --from "olmoearth-agent[mcp,inferencex] @ git+https://github.com/2imi9/OlmoEarth-Agent" olmoearth-agent-mcp
```

Success: `claude mcp list` lists `olmoearth-agent`.

### 4. Restart the host

The user quits Claude Desktop fully (closing the window is not enough) and opens it again. In Claude Code, start a
new session; `/mcp` shows the server.

### 5. Check

In a new chat, ask: "What can the OlmoEarth Agent do?"

Success: Claude may ask to use an OlmoEarth tool (Allow once or Always allow); then the start card shows the OlmoEarth
figure, the jobs and "Sign in first, then press a job or ask Claude in the chat." A host without panels names the
jobs in text.

### 6. Sign in (the user)

On the card, press **Open the sign-in page**. A page that the app serves on this computer (`127.0.0.1`) opens in the
browser, for 10 minutes. Paste your own Studio API key (in Studio: your profile, then API Keys). It goes to the app,
not through Claude. "Remember on this computer" is ticked by default: it keeps the key in
`~/.config/olmoearth-agent/studio.env`, readable by you alone. Untick it to sign in for this run of the app only.

Success: the card's top line shows your account and Sign out, and the jobs are on. Without panels, Claude gives the
sign-in link in the chat.

No Studio account? Press **Request access** on the card, or go to https://olmoearth.allenai.org/. Do not use someone
else's key.

## Options

Add an `env` block next to `args` (Claude Desktop: the user, in Edit Config; Claude Code: `--env KEY=value` before
`--transport`). The install command keeps an `env` block that is already in Claude Desktop's config. In Claude Code,
running it again with `--claude-code` removes the entry and adds it again without its `--env` settings: add them
again with the by-hand line (step 3, with `--env KEY=value` before `--transport`). The block:

```json
"env": {"OLMOEARTH_READ_ONLY": "1", "OLMOEARTH_OUTPUT_ROOT": "/absolute/path/to/folder"}
```

- `OLMOEARTH_READ_ONLY=1`, for a read-only key: the tools that make a project or run a model are left out. Saving an
  area from the map is still offered; with a read-only key Studio should refuse it (not tried live), and a key that
  can write saves it.
- `OLMOEARTH_OUTPUT_ROOT`: the folder the app works and writes in (default `~/olmoearth-agent-outputs`). The tools
  read files only from it, so name a new folder for this app alone, never your home folder.
- Pin a release or a commit: `olmoearth-agent[mcp,inferencex] @ git+https://github.com/2imi9/OlmoEarth-Agent@v1.5.0`
  (or `@<commit>`); the install command writes it with `install --ref v1.5.0`. Run without `--ref`, the command sets
  the entry back to main, so give the same `--ref` each time you run it. To update, pin a newer one and restart the
  host.
- Keep `--python 3.12`: on Python 3.14 the statistics package
  ([olmoearth-inferencex](https://github.com/2imi9/olmoearth_inferenceX)) does not install.

**Ai2's Asta research skills** (optional; they find and summarise scientific papers):

- Install them in Claude Code: `/plugin marketplace add allenai/asta-plugins`, then `/plugin install asta-tools`.
- They work in Claude Code only (not the chat in Claude Desktop), and some of their commands need an Ai2 login.
- Then turn Asta on in **Settings** (the text button at the bottom left of every card): the start card shows
  **Search papers with Asta**, and the app asks for Asta only when you press it.

Once installed, Claude Code may also use Asta when you ask for it in the chat;
the switch only shows or hides the app's tile.

## What runs where

- **OlmoEarth Studio** (Ai2's cloud) holds the models and the maps. The app reads them with your key, and writes to
  Studio only to make a project, save an area or run a model.
- **Your computer** runs the app: it downloads and decodes a map, runs the statistics (olmoearth-inferencex), and
  keeps the pick, the sheet and the result in `~/olmoearth-agent-outputs` (private to you when the app makes it).
- **Earth Search** (public; no key, no account) serves the Sentinel-2 pictures, only after you switch them on in the
  panel. Its catalogue gets the box around the map, rounded out to 0.1 degree, and the map's months; its file store
  gets requests for pieces of image files. Both see your IP address.
- **Claude** shows the panels and passes your presses on. Your key never goes through it, and nothing checks its
  replies against the tools.

Every panel, step by step: [mcp-app.md](mcp-app.md). The words on the panels (square, point, part, range):
[terms.md](terms.md).

## Limits

- The pictures are Sentinel-2 only, from 2015 on. A map made from other imagery, such as Landsat, is still labelled
  on Sentinel-2 pictures.
- A part is read up to 256 million values (pixels times bands); a part's file over 513 MB is refused.
- A large map can take a while to open, and the host may stop a long tool call (see Troubleshooting).
- Each part's squares are picked at random once and kept.
- Running a model and saving an area follow Studio's API spec but have not been tried live.

## Troubleshooting

| What you see | Why | Fix |
|---|---|---|
| No `olmoearth-agent` in Claude Desktop; `mcp.log` shows an error such as `spawn uvx ENOENT` | Claude Desktop cannot find `uvx` | Run the install command (step 3; with the same `--ref` if the entry is pinned): it writes uvx's absolute path |
| `Git executable not found`, or on macOS `xcrun: error`, in `mcp-server-olmoearth-agent.log` | git is missing | Install git (step 2); quit and reopen the host |
| `install` says the config is not valid JSON | The file was broken before the command ran, for example by an edit by hand; the command did not change it | The user fixes it in Edit Config, or moves it aside; then run the command again |
| Nothing changed after editing the config | The app was closed, not quit, or the edit broke the JSON | The user fixes the JSON in Edit Config; quit fully and reopen |
| Windows: still no `olmoearth-agent` after a full restart, and the JSON is valid | The Microsoft Store build may read its own copy of the config | List the copies with `Get-ChildItem $env:LOCALAPPDATA -Recurse -Filter claude_desktop_config.json -ErrorAction SilentlyContinue` and ask the user which one to edit. The install command (step 3) writes every Store copy it finds |
| The server fails on its first start, or `/mcp` shows it failed | The download took longer than the host waits | Run the install command again (step 3), with the same `--ref` if the entry is pinned; it downloads first. Then in Claude Code, set any `--env` options again (the command replaces the entry without them) and Reconnect it in `/mcp`; in Claude Desktop, quit fully and reopen. Later starts use uv's cache |
| Tools answer in text, no panels | The host shows no MCP App panels, or the app started before an update | Use Claude Desktop; quit it fully and reopen |
| A tool says it needs olmoearth-inferencex; nothing was computed | Python 3.14 or later, or the `inferencex` extra is missing | Keep `--python 3.12` and `[mcp,inferencex]` in the entry; restart |
| A card says "This step was stopped before it finished." | The host stopped the call, for example when Stop was pressed or a large map ran too long | Ask Claude to run it again, as the card says |

Logs: on macOS in `~/Library/Logs/Claude/` (Windows `%APPDATA%\Claude\logs`, Linux `~/.config/Claude/logs`).
`mcp.log` holds connection errors; `mcp-server-olmoearth-agent.log` holds this server's messages, never the key.
Agents: read only this server's lines (in `mcp.log`, the lines with `olmoearth-agent`).

## Remove

1. If a key was remembered, press **Sign out** on the start card first. Without panels, the user deletes
   `~/.config/olmoearth-agent/studio.env` or revokes the key in Studio. Agents: do not open or delete it.
2. Run the uninstall command (agents: show it and ask first), with `--claude-code` at the end for Claude Code too.
   It removes only the `olmoearth-agent` entry from Claude Desktop's config, after a backup, and never shows the
   file; it does not delete the saved key. The backups beside the config stay; each is a full copy of the config as
   it was, so it can hold keys and tokens. The user deletes them when no longer needed. Agents: do not open or delete
   them. Then the user quits Claude Desktop and reopens it.

   ```bash
   uvx --python 3.12 --from "olmoearth-agent[mcp,inferencex] @ git+https://github.com/2imi9/OlmoEarth-Agent" olmoearth-agent-mcp uninstall
   ```

   By hand: the user deletes the `olmoearth-agent` entry in Edit Config; for Claude Code,
   `claude mcp remove --scope user olmoearth-agent`.
3. `~/olmoearth-agent-outputs` holds your sheets and results; it is yours to delete or keep. Agents: do not delete it.

The start card's overview figure is the OlmoEarth docs' own, used with Ai2's permission.
