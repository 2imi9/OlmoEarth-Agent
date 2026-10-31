# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Settings every test shares.

The claim check (``harness/claim_check.py``) is on by default and makes one
more model call on each answer, two when it asks for a rewrite. A test that
scripts the model's calls for another behaviour would have to script those
too, so the tests run with it off; ``tests/harness/test_claim_check.py``
turns it on, and tests its default there.

``OLMOEARTH_READ_ONLY`` is unset for every test, whatever the shell running
them holds: with it on, the default registry leaves out the Studio write
tools, which the registry's tests count; the tests that want it set it.
"""

from __future__ import annotations

import pytest

from olmoearth_agent.harness.agent import CHECK_CLAIMS_ENV
from olmoearth_agent.skills.registry import READ_ONLY_ENV


@pytest.fixture(autouse=True)
def _claim_check_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CHECK_CLAIMS_ENV, "0")


@pytest.fixture(autouse=True)
def _studio_key_not_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(READ_ONLY_ENV, raising=False)
