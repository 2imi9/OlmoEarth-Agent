# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The environment a subprocess the agent starts receives: its own keys removed.

Every child process the agent starts (olmoearth-inferencex's command line,
``tools.inferencex.run_cli``) gets a copy of ``os.environ`` without the
agent's Studio, LLM and cloud credentials, so code the child runs cannot read
and send them on. One denylist for any child, here. It is
defence in depth, not a sandbox: the child keeps every other variable (PATH,
GDAL and PROJ settings, ...) and is not network-isolated.
"""

from __future__ import annotations

import os

#: Name fragments that mark a credential, matched case-insensitively.
SECRET_FRAGMENTS = (
    "API_KEY",
    "APIKEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
)

#: Name prefixes of the agent's own settings and the providers' keys.
SECRET_PREFIXES = (
    "OLMOEARTH_",
    "LLM_",
    "AWS_",
    "ANTHROPIC_",
    "OPENAI_",
    "GEMINI_",
    "HF_",
)


def scrubbed_env() -> dict[str, str]:
    """A copy of ``os.environ`` with the credential variables removed.

    Returns
    -------
    dict
        The environment for the child process.

    Examples
    --------
    >>> "OLMOEARTH_API_KEY" in scrubbed_env()
    False
    """
    env: dict[str, str] = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if upper.startswith(SECRET_PREFIXES):
            continue
        if any(fragment in upper for fragment in SECRET_FRAGMENTS):
            continue
        env[name] = value
    return env
