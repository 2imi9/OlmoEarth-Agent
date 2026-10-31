#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
#
# Run one olmoearth-agent-bridge command on aicr, from the Mac.
#
# Nothing runs on this machine but ssh and shell builtins. Each command is
# one ssh connection running one srun step on the cpu partition (2 CPUs, 8G,
# at most 20 minutes); on the login node nothing runs but that srun. Nothing
# is held between commands. The Studio key is read from
# ~/.config/olmoearth-agent/studio.env on the cluster, inside the srun step,
# and never printed. A brief or a response is a file here: it goes over ssh's
# stdin, which srun forwards to the bridge (--brief-file -, --response-file -),
# so its text passes through no shell, here or there.
#
#   scripts/oe-bridge.sh begin RUN BRIEF.txt [MAX_TURNS] [--text]
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
# OE_BRIDGE_READ_ONLY=1 runs the bridge with OLMOEARTH_READ_ONLY=1: the key
# is declared read-only, so the two Studio write tools are not offered. It
# changes the tools and the capability card, so keep it the same for every
# command of a run (a change shows as a divergence).
READ_ONLY="${OE_BRIDGE_READ_ONLY:-}"

BIN="$AGENT_DIR/.venv/bin/olmoearth-agent-bridge"
# What srun runs on the compute node: new files private to the account
# (umask 077: run files and outputs hold account names and tool results),
# the Studio key exported from the env file (set -a), the workspace root at
# $1 (made if missing), then the bridge with the rest of the arguments.
INNER='umask 077; set -a; . "$HOME/.config/olmoearth-agent/studio.env"; set +a; export OLMOEARTH_OUTPUT_ROOT="$1"; mkdir -p "$1"; shift; exec "$@"'

usage() {
  printf '%s\n' \
    'usage: scripts/oe-bridge.sh begin RUN BRIEF.txt [MAX_TURNS] [--text]' \
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
[[ -z "$READ_ONLY" || "$READ_ONLY" == 1 ]] ||
  fail "OE_BRIDGE_READ_ONLY must be 1 or unset"

# The remote command: one srun step that runs the bridge with the given
# arguments. Every word is quoted for the remote shell with printf %q.
bridge_cmd() {
  local cmd arg
  cmd="$SRUN bash -c $(printf '%q' "$INNER") oe-bridge"
  cmd+=" $(printf '%q' "$OUTPUTS")"
  if [[ -n "$READ_ONLY" ]]; then
    cmd+=" env OLMOEARTH_READ_ONLY=1"
  fi
  cmd+=" $(printf '%q' "$BIN")"
  for arg in "$@"; do
    cmd+=" $(printf '%q' "$arg")"
  done
  printf '%s' "$cmd"
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
    [[ -f "$brief" && -r "$brief" ]] ||
      fail "cannot read the brief file '$brief' (write the request to a file)"
    extra=()
    for arg in "$@"; do
      if [[ "$arg" == "--text" ]]; then
        extra+=(--text)
      elif [[ "$arg" =~ ^[0-9]{1,3}$ ]]; then
        extra+=(--max-turns "$arg")
      else
        fail "begin takes MAX_TURNS and --text after the brief file (got '$arg')"
      fi
    done
    ssh -T "$HOST" "$(bridge_cmd begin --run-dir "$RUNS/$run" --brief-file - \
      ${extra[@]+"${extra[@]}"})" < "$brief"
    ;;
  step)
    [[ $# -ge 2 ]] || usage
    run="$1"
    response="$2"
    shift 2
    check_run "$run"
    [[ -f "$response" && -r "$response" ]] ||
      fail "cannot read the response file '$response'"
    extra=()
    for arg in "$@"; do
      [[ "$arg" == "--text" ]] ||
        fail "step takes only --text after the response file (got '$arg')"
      extra+=(--text)
    done
    ssh -T "$HOST" "$(bridge_cmd step --run-dir "$RUNS/$run" --response-file - \
      ${extra[@]+"${extra[@]}"})" < "$response"
    ;;
  show)
    [[ $# -ge 1 ]] || usage
    run="$1"
    check_run "$run"
    extra=()
    if [[ $# -ge 2 ]]; then
      [[ "$2" == "--text" ]] || fail "show takes only --text after RUN (got '$2')"
      extra+=(--text)
    fi
    ssh -T "$HOST" "$(bridge_cmd show --run-dir "$RUNS/$run" \
      ${extra[@]+"${extra[@]}"})" < /dev/null
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
