#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
#
# Run one olmoearth-agent-bridge command on aicr, from the Mac.
#
# Nothing runs on this machine but ssh and shell builtins. Each command is
# one ssh connection and one srun step on the cpu partition (2 CPUs, 8G, at
# most 20 minutes); `show` without --text is a `cat` of pending.json on the
# login node instead. Nothing is held between commands. The Studio key is
# read from ~/.config/olmoearth-agent/studio.env on the cluster, inside the
# srun step, and never printed. A brief or a response goes to the cluster as
# a file over ssh's stdin, so no JSON is quoted through two shells.
#
#   scripts/oe-bridge.sh begin RUN "the brief, word for word" [MAX_TURNS] [--text]
#   scripts/oe-bridge.sh step RUN RESPONSE.json [--text]
#   scripts/oe-bridge.sh show RUN [--text]
#   scripts/oe-bridge.sh prompt
#   scripts/oe-bridge.sh tools [RUN]
#
# RUN is a short name (letters, digits, . _ -); its directory is $RUNS/RUN.
# Works with macOS's bash 3.2. See docs/CLAUDE_AS_MODEL.md.

set -euo pipefail

# --- where things are on aicr ----------------------------------------------
HOST="${OE_BRIDGE_HOST:-aicr}"
# An agent checkout with its .venv (uv sync --extra inferencex).
AGENT_DIR=/scratch/qi_zim_neu/olmoearth_inferenceX/agent-checks/bridge/src
# The run directories (scratch is purged after 30 days).
RUNS=/scratch/qi_zim_neu/olmoearth_inferenceX/agent-checks/bridge/runs
# The agent's workspace root (OLMOEARTH_OUTPUT_ROOT): spilled results, files
# the tools write, the preference memory.
OUTPUTS=/scratch/qi_zim_neu/olmoearth_inferenceX/agent-checks/bridge/outputs
# The srun step. The AICR guide lists QOS normal for the cpu partition; if
# srun refuses the QOS, run with OE_BRIDGE_QOS=normal.
QOS="${OE_BRIDGE_QOS:-interactive}"
SRUN="srun -A p2026_0089_neu -p cpu -q $QOS -n 1 -c 2 --mem=8G -t 00:20:00"

BIN="$AGENT_DIR/.venv/bin/olmoearth-agent-bridge"
INBOX="$RUNS/.inbox"
# What srun runs on the compute node: export the Studio key from the env file
# (set -a), point the workspace root at $1, then exec the bridge with the
# rest of the arguments.
INNER='set -a; . "$HOME/.config/olmoearth-agent/studio.env"; set +a; export OLMOEARTH_OUTPUT_ROOT="$1"; shift; exec "$@"'

usage() {
  printf '%s\n' \
    'usage: scripts/oe-bridge.sh begin RUN "the brief" [MAX_TURNS] [--text]' \
    '       scripts/oe-bridge.sh step RUN RESPONSE.json [--text]' \
    '       scripts/oe-bridge.sh show RUN [--text]' \
    '       scripts/oe-bridge.sh prompt' \
    '       scripts/oe-bridge.sh tools [RUN]' >&2
  exit 2
}

fail() {
  printf 'oe-bridge: %s\n' "$1" >&2
  exit 2
}

check_run() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] ||
    fail "RUN must be letters, digits, . _ - (got '$1')"
}

[[ "$QOS" =~ ^[A-Za-z0-9_-]+$ ]] || fail "OE_BRIDGE_QOS must be a QOS name"

# The remote command that runs the bridge with the given arguments in a srun
# step. Every word is quoted for the remote shell with printf %q.
bridge_cmd() {
  local cmd arg
  cmd="mkdir -p $(printf '%q' "$RUNS") $(printf '%q' "$OUTPUTS")"
  cmd+=" && $SRUN bash -c $(printf '%q' "$INNER") oe-bridge"
  cmd+=" $(printf '%q' "$OUTPUTS") $(printf '%q' "$BIN")"
  for arg in "$@"; do
    cmd+=" $(printf '%q' "$arg")"
  done
  printf '%s' "$cmd"
}

# The remote command that saves stdin to $INBOX/NAME, runs the bridge (whose
# arguments name that file), removes the file and exits with the bridge's
# status.
with_input() {
  local file="$INBOX/$1"
  shift
  printf 'mkdir -p %q && cat > %q && { %s; rc=$?; rm -f %q; exit $rc; }' \
    "$INBOX" "$file" "$(bridge_cmd "$@")" "$file"
}

[[ $# -ge 1 ]] || usage
action="$1"
shift

case "$action" in
  begin)
    [[ $# -ge 2 ]] || usage
    run="$1"
    brief="$2"
    shift 2
    check_run "$run"
    extra=()
    for arg in "$@"; do
      if [[ "$arg" == "--text" ]]; then
        extra+=(--text)
      elif [[ "$arg" =~ ^[0-9]{1,3}$ ]]; then
        extra+=(--max-turns "$arg")
      else
        fail "begin takes MAX_TURNS and --text after the brief (got '$arg')"
      fi
    done
    remote="$(with_input "$run.brief" begin --run-dir "$RUNS/$run" \
      --brief-file "$INBOX/$run.brief" ${extra[@]+"${extra[@]}"})"
    printf '%s' "$brief" | ssh -T "$HOST" "$remote"
    ;;
  step)
    [[ $# -ge 2 ]] || usage
    run="$1"
    response="$2"
    shift 2
    check_run "$run"
    [[ -r "$response" ]] || fail "cannot read the response file '$response'"
    extra=()
    for arg in "$@"; do
      [[ "$arg" == "--text" ]] ||
        fail "step takes only --text after the response file (got '$arg')"
      extra+=(--text)
    done
    remote="$(with_input "$run.response" step --run-dir "$RUNS/$run" \
      --response-file "$INBOX/$run.response" ${extra[@]+"${extra[@]}"})"
    ssh -T "$HOST" "$remote" < "$response"
    ;;
  show)
    [[ $# -ge 1 ]] || usage
    run="$1"
    check_run "$run"
    if [[ "${2:-}" == "--text" ]]; then
      ssh -T "$HOST" "$(bridge_cmd show --run-dir "$RUNS/$run" --text)" < /dev/null
    else
      ssh -T "$HOST" "cat $(printf '%q' "$RUNS/$run/pending.json")" < /dev/null
    fi
    ;;
  prompt)
    ssh -T "$HOST" "$(bridge_cmd prompt)" < /dev/null
    ;;
  tools)
    if [[ $# -ge 1 ]]; then
      check_run "$1"
      ssh -T "$HOST" "$(bridge_cmd tools --run-dir "$RUNS/$1")" < /dev/null
    else
      ssh -T "$HOST" "$(bridge_cmd tools)" < /dev/null
    fi
    ;;
  *)
    usage
    ;;
esac
