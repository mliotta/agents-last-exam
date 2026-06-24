"""car — Common Agent Runtime as an ALE out-of-sandbox harness.

Vendored into a fork at ``ale_run/agents/car/``. ``CarDeployer`` is imported
lazily so importing this package (e.g. for the pure transcript mapper) does not
require the ALE framework to be installed.
"""
from __future__ import annotations

from .config import CarConfig

__all__ = ["CarConfig", "CarDeployer"]


def __getattr__(name: str):  # PEP 562 — lazy, framework-only deployer import
    if name == "CarDeployer":
        from .deployer import CarDeployer
        return CarDeployer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
