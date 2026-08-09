# NVIDIA NemoClaw -- assessment for OlmoEarth Agent

An architectural read of [`NVIDIA/NemoClaw`](https://github.com/NVIDIA/NemoClaw)
-- an operational-safety stack for always-on agents in NVIDIA OpenShell
sandboxes -- mapped onto how the OlmoEarth Agent actually runs (a trusted,
single-user, in-process tool loop), with a fit verdict and the controls worth
porting. Same playbook as
[`docs/science-skills-assessment.md`](science-skills-assessment.md) and
[`docs/eo-skills-shortlist.md`](eo-skills-shortlist.md): research first, then
build only the highest-value honest transfer.

> **Sources.** The repo at `NVIDIA/NemoClaw` (Apache-2.0, primary language
> TypeScript, ~2,160 files, alpha) read directly -- not the docs site summary.
> Modules cited below were read in full or in part:
> `nemoclaw/src/blueprint/ssrf.ts` (endpoint validation + DNS pinning),
> `nemoclaw/src/blueprint/private-networks.ts` (CIDR block list),
> `nemoclaw-blueprint/policies/presets/*.yaml` (per-integration egress
> allowlists) + `schemas/policy-preset.schema.json`,
> `nemoclaw-blueprint/policies/presets/local-inference.yaml`,
> `nemoclaw-blueprint/scripts/seccomp-guard.js`,
> `scripts/checks/direct-credential-env.ts`, and the CLI under
> `src/commands/{sandbox,credentials,inference}/`.

## TL;DR

- **As a stack, NemoClaw is off-scale for us** -- the same call open-webui got.
  It runs an *untrusted, always-on* agent (OpenClaw / Hermes) inside an
  *OpenShell container* and enforces safety at the **OS / network layer**:
  seccomp syscall filtering, Linux capability drops, a DNS proxy, a host-side
  tool-gateway broker, and an operator-approval flow for new egress. The
  OlmoEarth Agent is a *trusted, single-user, in-process* Python library on the
  user's own machine with **no container** -- so none of the container/fleet
  machinery applies, and most of it cannot even be expressed in our runtime.
- **But three of NemoClaw's primitives port cleanly and honestly**, because we
  have two real escape hatches an attacker can lean on: (a) **env-configurable
  endpoints** (`OLMOEARTH_BASE_URL`, the hosted-LLM base URLs) that carry a real
  credential, and (b) an **opt-in code-exec subprocess** (`OLMOEARTH_RUN_PYTHON`)
  that inherits our secrets.
- **What shipped (this PR):** an in-process **egress guard** -- a per-capability
  host allowlist plus an SSRF block (private/loopback/link-local/metadata
  ranges), with NemoClaw's **enforce / audit** split -- wired into the two
  credentialed boundaries (Studio + cloud-LLM) and the two key-free fetchers
  (litsearch + HF), feeding the **provenance manifest**; and a
  **credential-scrubbed environment** for the opt-in subprocess so executed code
  can no longer read our Studio/LLM keys. Default mode is `audit` (log only), so
  turning it on never breaks a working deployment.
- **What we deliberately did NOT port:** seccomp / capability drops / process
  limits (Linux-container-only), the DNS proxy + tool-gateway broker +
  operator-approval flow (needs an untrusted process behind a proxy), and signed
  skill bundles. They are documented here as off-scale, not silently skipped.

## 1. What NemoClaw is

A reference stack for running **always-on AI agents more safely inside NVIDIA
OpenShell sandboxes**, driven by one CLI (`nemoclaw` / `nemohermes`). It targets
operators who run an agent like OpenClaw or Hermes 24/7 and need it boxed in. The
README's own framing: "guided onboarding, a hardened blueprint, routed
inference, network policy, and lifecycle management through a single CLI."

Concretely, the repo is:

- **A CLI** (`src/commands/...`, `bin/`) with verbs for the full lifecycle:
  `sandbox` (connect / exec / destroy / rebuild / recover / doctor / rotate-token
  / policy add|list|remove / hosts add|list|remove / channels ...),
  `credentials` (list / reset), and `inference` (get / set).
- **A hardened container blueprint** (`nemoclaw-blueprint/`, `Dockerfile*`): the
  network-policy presets, a private-networks CIDR list, a router pool config, and
  preload guards (`seccomp-guard.js`, `sandbox-safety-net.js`,
  `ciao-network-guard.js`).
- **A host-side tool-gateway broker** (`agents/hermes/host/tool-gateway-broker.ts`,
  `managed-tool-gateway.ts`) that mediates the agent's tool/network access from
  *outside* the sandbox.

It is an **operational-safety lens**: it assumes the agent's *capabilities* exist
and concentrates on containing them. That is orthogonal to our project's focus
(authoring in-domain EO *capabilities*), which is exactly why only the safety
primitives -- not the capabilities -- are candidates for transfer.

## 2. How NemoClaw enforces safety (the parts that matter to us)

- **SSRF / endpoint validation** (`ssrf.ts`). `validateEndpointUrl(url)` enforces
  an `http`/`https` scheme allowlist, rejects private/internal **hostnames**,
  resolves DNS and rejects any resolved **IP** in a private range, then **pins**
  the hostname to the validated IP to defeat DNS-rebinding TOCTOU. The block list
  (`private-networks.ts`) is a memoised `BlockList` built from a YAML of
  `ipv4` / `ipv6` / `names` entries, each carrying a `purpose` string "so
  reviewers can judge the block."
- **Per-integration egress allowlists** (`policies/presets/*.yaml`, validated by
  `policy-preset.schema.json`). Each preset is a named bundle of `endpoints`
  (`host` + `port` + `protocol` + **`enforcement: enforce | audit`** + method/path
  `rules`) plus the `binaries` allowed to make them. The thinking is real: the
  `huggingface` preset is **download-only** (`GET /**`) and a comment explains
  that `POST` was removed so a leaked HF token could not publish models from the
  sandbox. `local-inference.yaml` is the key precedent for us -- it must
  **explicitly allowlist** the loopback/private host-gateway IPs because the SSRF
  guard rejects them by default.
- **Credential centralization** (`scripts/checks/direct-credential-env.ts`). A CI
  check (a TypeScript AST walk) that **fails the build** on any direct
  `process.env.ANTHROPIC_API_KEY`-style read, forcing all provider credentials
  through one resolver so they cannot leak into the wrong place.
- **Sandbox hardening** (`Dockerfile`, `seccomp-guard.js`): capability drops,
  process limits, and seccomp syscall filtering -- enforced by the OpenShell
  container runtime, with Node preloads to keep libraries from crashing on
  blocked syscalls.
- **Routed inference** (`src/commands/inference/`, `router/pool-config.yaml`):
  multiple providers behind a validated router, keys resolved centrally.
- **Operator approval + auditable lifecycle**: new egress requests surface for
  human approval (`skills/.../approve-network-requests.md`); the CLI gives a
  start -> run -> terminate lifecycle with rotateable tokens.

## 3. How the OlmoEarth Agent differs

- **In-process, not containerised.** Tools are `async` Python handlers dispatched
  inside one process ([`harness/agent.py`](../src/olmoearth_agent/harness/agent.py),
  [`tools/registry.py`](../src/olmoearth_agent/tools/registry.py)). There is no
  OS boundary to attach seccomp / capabilities / a DNS proxy to. The single
  code-exec path is the **opt-in** `olmoearth_run_python` subprocess
  ([`tools/system.py`](../src/olmoearth_agent/tools/system.py)).
- **Trusted, single-user.** The operator runs the agent on their own machine with
  their own keys. The threat model is not "a hostile agent we must cage" but
  "don't let a malicious *input* (a typo'd or attacker-supplied endpoint, a
  crafted DOI, an executed snippet) turn our own credentials or process against
  us."
- **Outbound surface is small and known.** Five capabilities:
  Studio (`olmoearth.allenai.org`, but the base URL is `OLMOEARTH_BASE_URL`-
  overridable), the hosted LLMs (`api.anthropic.com` / `api.openai.com` /
  `generativelanguage.googleapis.com` / `integrate.api.nvidia.com`), the local
  LLM (loopback), litsearch (`export.arxiv.org` + `api.openalex.org`), and HF
  (`datasets-server.huggingface.co`). That small set is exactly what makes a
  static allowlist tractable for us where NemoClaw needs a dynamic policy engine.
- **We already have a provenance manifest** the agent run owns
  ([`provenance/log.py`](../src/olmoearth_agent/provenance/log.py)) -- a natural
  place to record an egress audit trail that NemoClaw lists as an open limitation.

## 4. Side-by-side

| Dimension | NemoClaw | OlmoEarth Agent | Takeaway |
|---|---|---|---|
| **Runtime** | Untrusted agent in an OpenShell container | Trusted in-process Python library | No OS boundary for us to attach kernel-level controls to. |
| **Enforcement layer** | OS / network (seccomp, caps, DNS proxy, gateway broker) | Application code (httpx call sites, subprocess env) | We can only enforce where *our* code builds a URL or spawns a child. |
| **Egress control** | Dynamic per-integration policy presets, enforce/audit | Static per-capability host allowlist, enforce/audit | Port the **model** (allowlist + enforce/audit + loopback exception), right-sized to a fixed host set. |
| **SSRF guard** | `ssrf.ts`: scheme + name + post-DNS IP checks + DNS pinning | scheme + name + IP-range checks (no DNS resolution) | Port the checks via stdlib `ipaddress`; **skip DNS pinning** -- a per-call interactive agent should not block on lookups, and the allowlist is the primary control. |
| **Credentials** | Central resolver + a CI guard against direct env reads | Per-request keys, never stored ([`serve.py`](../src/olmoearth_agent/serve.py)) | Already close; the env-read CI guard is a cheap future borrow (see below). |
| **Code exec** | seccomp + caps + process limits in-container | Opt-in subprocess, `python -I`, throwaway cwd, timeout | We cannot seccomp ourselves on Windows; **scrub secrets from the child env** is the portable analog. |
| **Provenance / audit** | Operator-approval logs; reproducibility an open limitation | First-class per-run manifest + replay | We are ahead; egress decisions slot straight into the manifest. |
| **Distribution** | Container blueprint + signed skill bundles (`skill.oms.sig`) | One Python package | Signing/fleet machinery is off-scale. |

## 5. Fit verdict

**Off-scale as a stack, by design.** NemoClaw's value is containing an untrusted,
always-on agent at the OS/network layer for a fleet operator. We have no
container, one user, and a trusted process. Adopting the blueprint / broker /
DNS-proxy / seccomp machinery would add a large, Linux-container-shaped
dependency surface for a threat model we do not have -- the same reason the
open-webui stack was declined.

**But the safety *ideas* transfer**, because our two escape hatches are real:

1. An **endpoint that carries a credential** can be pointed at an attacker
   (a malicious `OLMOEARTH_BASE_URL`, or -- if a custom-endpoint feature is ever
   added -- a hosted-LLM base URL). NemoClaw's SSRF guard + egress allowlist is
   the exact countermeasure, re-scoped to an in-process check.
2. An **opt-in executed snippet** inherits our environment, including the Studio
   and LLM keys. NemoClaw caps this at the seccomp/network layer; lacking that,
   we can still shrink the blast radius by scrubbing the secrets from the child.

Both are honest, Windows-portable, non-breaking, and advance existing tracks
(the SSRF/egress guard extends the provenance ethos and issue #54's sandbox
spec; the env scrub directly advances #54).

## 6. Adoptable patterns

| Pattern | Effort | Recommendation |
|---|---|---|
| Per-capability host **allowlist** with `enforce` / `audit` modes | M | **Built.** `security/egress.py`; default `audit` (log only) so enabling it never breaks a deployment; `OLMOEARTH_EGRESS=enforce` to block; `OLMOEARTH_EGRESS_ALLOW` to add a self-hosted host. |
| **SSRF block** of private/loopback/link-local/metadata ranges | S-M | **Built.** Port `ssrf.ts` + `private-networks.ts` intent via stdlib `ipaddress` (no YAML, no new dep). Loopback allowed only for the `llm-local` capability (cf. `local-inference.yaml`). DNS pinning intentionally skipped (documented). |
| **Egress audit trail** in provenance | S | **Built.** `ProvenanceLog.record_egress` -> host-only records in the manifest; the run now answers "what external hosts did this contact, and were any off-allowlist?" |
| **Credential scrub** for opt-in code exec | S | **Built.** `tools/system.py` passes a secret-scrubbed env to the subprocess. Honest framing: defence-in-depth, not a network sandbox. |
| CI guard against **direct credential env reads** (`direct-credential-env.ts`) | S | **Future borrow.** A small test/ruff rule asserting provider keys are read only in the one resolver path. Low effort, fits the honest-results ethos; not in this PR. |
| **Operator-approval** on a new egress host | M | **Future option, not now.** Our analog is config-time (`OLMOEARTH_EGRESS_ALLOW`). A live "approve this host?" prompt only makes sense once the agent runs unattended. |
| seccomp / capability drops / process limits / DNS proxy / gateway broker | -- | **Off-scale.** Linux-container controls with no in-process equivalent on our (Windows) host. Documented, not adopted. |

## 7. Decision -- what shipped

A single, focused security PR under one thesis: **shrink the blast radius of the
agent's two opt-in escape hatches**, informed by NemoClaw, scoped to what an
in-process agent can honestly enforce.

- **`src/olmoearth_agent/security/egress.py`** -- `validate_endpoint(url,
  capability)` + a pure `check_endpoint` classifier. Per-capability allowlist
  (`studio`, `llm-cloud`, `llm-local`, `litsearch`, `hf`); SSRF block via
  `ipaddress`; `enforce` / `audit` / `off` modes (`OLMOEARTH_EGRESS`, default
  `audit`); `OLMOEARTH_EGRESS_ALLOW` to extend.
- **Wiring** at every outbound `httpx` boundary: `StudioClient.__init__`
  (capability `studio`, guards the env-overridable base URL before the Bearer key
  is bound), `serve._llm_for_request` (capability `llm-cloud`, guards the hosted
  endpoint before the BYO key is handed over -> HTTP 403 in enforce),
  `analysis/litsearch.py` + `analysis/automate.py` fetchers (`litsearch` / `hf`).
- **Provenance** -- `ProvenanceLog.record_egress` + an `egress` section in the
  manifest; the harness records the run's Studio endpoint at run start
  (best-effort, host only).
- **`tools/system.py`** -- the opt-in subprocess now runs with a
  credential-scrubbed environment (drops `OLMOEARTH_*` / `*_API_KEY` / `*TOKEN` /
  `*SECRET` / cloud-provider keys), keeping OS-essential vars so it still launches
  on Windows.
- **Tests.** `tests/security/test_egress.py` (allow/deny, SSRF ranges incl.
  IPv6 ULA + link-local + IPv4-mapped + CGNAT, scheme reject, env-extend, mode
  handling, provenance hook), a `record_egress` provenance test, and two live
  subprocess tests proving the agent's keys are invisible to executed code while
  `PATH` survives. Live-checked: enforce mode blocks a `169.254.169.254` Studio
  base URL with an `EgressError`; the real Studio host is accepted; audit mode
  logs-but-allows.

**Honest limits, stated in code and docs:** this is not a sandbox. It guards the
URLs *our* code builds, not arbitrary egress from executed snippets or the OS; it
does not resolve DNS (no rebinding defense -- the allowlist is the control); and
the default `audit` mode only logs. It is the in-process subset of NemoClaw's
posture that we can actually enforce, and nothing more.

## 8. References

- Repo: <https://github.com/NVIDIA/NemoClaw> (read directly, 2026-06-04).
- NemoClaw modules: `nemoclaw/src/blueprint/{ssrf,private-networks}.ts`,
  `nemoclaw-blueprint/policies/presets/*.yaml`,
  `schemas/policy-preset.schema.json`,
  `nemoclaw-blueprint/policies/presets/local-inference.yaml`,
  `nemoclaw-blueprint/scripts/seccomp-guard.js`,
  `scripts/checks/direct-credential-env.ts`.
- OWASP SSRF prevention (private-range blocking, no-follow internal redirects):
  <https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html>
- Our code: [`security/egress.py`](../src/olmoearth_agent/security/egress.py),
  [`provenance/log.py`](../src/olmoearth_agent/provenance/log.py),
  [`tools/system.py`](../src/olmoearth_agent/tools/system.py),
  [`studio/client.py`](../src/olmoearth_agent/studio/client.py),
  [`serve.py`](../src/olmoearth_agent/serve.py).
- Related assessments: [`science-skills-assessment.md`](science-skills-assessment.md),
  [`eo-skills-shortlist.md`](eo-skills-shortlist.md); issue #54 (sandbox spec).
