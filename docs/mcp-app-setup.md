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
- Never open, read or edit Claude Desktop's config: it can hold the Studio key and other servers' tokens, and a file
  you read goes to the model whether you print it or not. The user adds the entry (step 5a).
- Show the user the Claude Code command (step 5b) and ask before running it.
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
tab (panels, step 5a); Claude Code in a terminal (text, step 5b; it does not read Claude Desktop's config); or both
(5a and 5b).

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

### 3. Find uvx's absolute path

Claude Desktop needs the full path: it does not start the command from your shell.

```bash
command -v uvx
```

```powershell
(Get-Command uvx).Source
```

Just after installing uv, this may find nothing until a new terminal. Use the folder the installer printed after
`installing to` (by default `.local/bin` in the home folder), here and for `uvx` in step 4.

Success: a full path, such as `/Users/you/.local/bin/uvx` or `C:\Users\you\.local\bin\uvx.exe` (on Windows a drive
letter, never `/c/…`). In JSON, write each `\` as `\\`.

### 4. Download everything first

The first start downloads Python 3.12 (if missing), this repository and its packages. That can take minutes, longer
than Claude waits for a server to start (Claude Code waits 30 seconds). Do it now, in a terminal (the same line works
in PowerShell):

```bash
uvx --python 3.12 --from "olmoearth-agent[mcp,inferencex] @ git+https://github.com/2imi9/OlmoEarth-Agent" olmoearth-agent-mcp --help
```

Success: it prints `usage: olmoearth-agent-mcp [-h]` and ends. `--help` starts no server and makes no output folder.

### 5a. Claude Desktop: the user adds the entry

The config file (in Claude Desktop, Settings, Developer, Edit Config opens it, and makes it if there is none):

| System | Path |
|---|---|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

Put the path from step 3 in `command`, show the user this entry, and ask them to add it in Edit Config: inside
`mcpServers` (adding `mcpServers` if it is missing; an empty file takes the whole block), in place of any old
`olmoearth-agent` entry. An old entry may hold their key, so they never paste it into the chat. Then they save.

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

Success (the user checks): the file is still valid JSON, and every other entry is unchanged.

### 5b. Claude Code: add the server

```bash
claude mcp add --scope user --transport stdio olmoearth-agent -- /absolute/path/to/uvx --python 3.12 --from "olmoearth-agent[mcp,inferencex] @ git+https://github.com/2imi9/OlmoEarth-Agent" olmoearth-agent-mcp
```

Success: `claude mcp list` lists `olmoearth-agent`.

### 6. Restart the host

The user quits Claude Desktop fully (closing the window is not enough) and opens it again. In Claude Code, start a
new session; `/mcp` shows the server.

### 7. Check

In a new chat, ask: "What can the OlmoEarth Agent do?"

Success: Claude may ask to use an OlmoEarth tool (Allow once or Always allow); then the start card shows the OlmoEarth
figure, the jobs and "Sign in first, then press a job or ask Claude in the chat." A host without panels names the
jobs in text.

### 8. Sign in (the user)

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
`--transport`):

```json
"env": {"OLMOEARTH_READ_ONLY": "1", "OLMOEARTH_OUTPUT_ROOT": "/absolute/path/to/folder"}
```

- `OLMOEARTH_READ_ONLY=1`, for a read-only key: the tools that make a project or run a model are left out. Saving an
  area from the map is still offered; with a read-only key Studio should refuse it (not tried live), and a key that
  can write saves it.
- `OLMOEARTH_OUTPUT_ROOT`: the folder the app works and writes in (default `~/olmoearth-agent-outputs`). The tools
  read files only from it, so name a new folder for this app alone, never your home folder.
- Pin a release or a commit: `olmoearth-agent[mcp,inferencex] @ git+https://github.com/2imi9/OlmoEarth-Agent@v1.5.0`
  (or `@<commit>`). To update, pin a newer one and restart the host.
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
| No `olmoearth-agent` in Claude Desktop; `mcp.log` shows an error such as `spawn uvx ENOENT` | Claude Desktop cannot find `uvx` | Put uvx's absolute path in `command` (step 3) |
| `Git executable not found`, or on macOS `xcrun: error`, in `mcp-server-olmoearth-agent.log` | git is missing | Install git (step 2); quit and reopen the host |
| Nothing changed after editing the config | The app was closed, not quit, or the edit broke the JSON | The user fixes the JSON in Edit Config; quit fully and reopen |
| Windows: still no `olmoearth-agent` after a full restart, and the JSON is valid | The Microsoft Store build may read its own copy of the config | List the copies with `Get-ChildItem $env:LOCALAPPDATA -Recurse -Filter claude_desktop_config.json -ErrorAction SilentlyContinue` and ask the user which one to edit |
| The server fails on its first start, or `/mcp` shows it failed | The download took longer than the host waits | Run step 4 in a terminal. Then in Claude Code, Reconnect it in `/mcp`; in Claude Desktop, quit fully and reopen. Later starts use uv's cache |
| Tools answer in text, no panels | The host shows no MCP App panels, or the app started before an update | Use Claude Desktop; quit it fully and reopen |
| A tool says it needs olmoearth-inferencex; nothing was computed | Python 3.14 or later, or the `inferencex` extra is missing | Keep `--python 3.12` and `[mcp,inferencex]` in the entry; restart |
| A card says "This step was stopped before it finished." | The host stopped the call, for example when Stop was pressed or a large map ran too long | Ask Claude to run it again, as the card says |

Logs: on macOS in `~/Library/Logs/Claude/` (Windows `%APPDATA%\Claude\logs`, Linux `~/.config/Claude/logs`).
`mcp.log` holds connection errors; `mcp-server-olmoearth-agent.log` holds this server's messages, never the key.
Agents: read only this server's lines (in `mcp.log`, the lines with `olmoearth-agent`).

## Remove

1. If a key was remembered, press **Sign out** on the start card first. Without panels, the user deletes
   `~/.config/olmoearth-agent/studio.env` or revokes the key in Studio. Agents: do not open or delete it.
2. Claude Desktop: the user deletes the `olmoearth-agent` entry in Edit Config, then quits and reopens. Claude Code:
   `claude mcp remove olmoearth-agent`.
3. `~/olmoearth-agent-outputs` holds your sheets and results; it is yours to delete or keep. Agents: do not delete it.

The start card's overview figure is the OlmoEarth docs' own, used with Ai2's permission.
