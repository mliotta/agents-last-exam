"""CarConfig — per-episode knobs for the CAR out-of-sandbox ALE harness.

Mirrors the ALE config convention (a standalone ``@dataclass`` with a
``name`` ClassVar + ``__post_init__`` validation; built from the experiment
yaml's ``config:`` block by ``ale_run.orchestration.factory.build_config``).

CAR is an out-of-sandbox harness: ``launch()`` shells out to ``car run-task``,
which drives the VM through the same ``vm_mcp_server`` (shell/files) +
``cua_mcp_server`` (GUI) bridges every ALE harness uses. The agent loop,
execution substrate, resume, and failure taxonomy all live in the Rust binary
(see docs/ale-harness-adapter-scope.md).

**Model ids:** ALE configs use LiteLLM ids; CAR resolves models through its own
registry/providers (no OpenRouter path). ``to_car_model_id()`` maps the LiteLLM
id → a CAR ``--model`` alias reaching the same underlying checkpoint
provider-direct. An unmapped id is a hard error (never silently mis-route a
backbone). See the mapping table in docs/ale-harness-adapter-scope.md §7.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


# LiteLLM model id (with any leading ``openrouter/`` stripped) → CAR --model alias.
# Keep in lockstep with docs/ale-harness-adapter-scope.md §7.
LITELLM_TO_CAR: dict[str, str] = {
    "anthropic/claude-opus-4-8": "claude-opus-4-8",
    "anthropic/claude-opus-4-7": "claude-opus-4-7",
    "anthropic/claude-sonnet-4.6": "claude-sonnet-4-6",
    "anthropic/claude-sonnet-4-6": "claude-sonnet-4-6",
    "anthropic/claude-haiku-4-5": "claude-haiku-4-5",
    "openai/gpt-5.5": "gpt-5.5",
    "openai/gpt-5.4": "gpt-5.4",
    "openai/gpt-5.3-codex": "gpt-5.3-codex",
    "openai/o3": "o3",
    "openai/o4-mini": "o4-mini",
    "gemini/gemini-2.5-pro": "gemini-2.5-pro",
    "vertex_ai/gemini-2.5-pro": "gemini-2.5-pro",
    "gemini/gemini-2.5-flash": "gemini-2.5-flash",
    "vertex_ai/gemini-2.5-flash": "gemini-2.5-flash",
}


def map_litellm_to_car(litellm_id: str) -> str:
    """Map an ALE LiteLLM model id → a CAR ``--model`` alias.

    Strips a leading ``openrouter/`` (CAR reaches the provider directly, so the
    route prefix is irrelevant — the *underlying checkpoint* is what must match).
    Raises on an unmapped id rather than guessing.
    """
    key = litellm_id
    if key.startswith("openrouter/"):
        key = key[len("openrouter/"):]
    car = LITELLM_TO_CAR.get(key)
    if car is None:
        raise ValueError(
            f"CarConfig: no CAR mapping for model {litellm_id!r} (normalized {key!r}). "
            f"Add it to LITELLM_TO_CAR or set config.car_model_override. "
            f"Known: {sorted(LITELLM_TO_CAR)}"
        )
    return car


@dataclass
class CarConfig:
    """Tunables for :class:`CarDeployer`."""

    name: ClassVar[str] = "car"

    model: str = "anthropic/claude-opus-4-8"
    """LiteLLM-format model id (as every ALE harness takes). Mapped to a CAR
    --model alias via :func:`map_litellm_to_car` unless ``car_model_override``."""

    car_model_override: str | None = None
    """Bypass the mapping table and pass this verbatim to ``car run-task --model``.
    Escape hatch for a CAR id not in the table; you own backbone-parity then."""

    max_turns: int = 100
    """Hard ceiling on the agent loop (``car run-task --max-turns``)."""

    prompt_mode: str = "base"
    """``base`` (clean engine baseline) or ``verify`` (completion-discipline,
    experimental). Sets ``CAR_RUNTASK_PROMPT`` for the subprocess."""

    enable_gui: bool = True
    """Wire the ``cua_mcp_server`` (GUI) bridge as a ``cua`` connector so the
    model gets ``mcp_cua_*`` tools. Disable for shell/files-only tasks."""

    resume: bool = False
    """Pass ``car run-task --resume`` so a re-run continues from the per-turn
    checkpoint (Phase 0.5). Default off; the framework owns retries."""

    car_binary: str | None = None
    """Path to the ``car`` CLI. None → ``$CAR_BINARY`` env, else ``car`` on PATH."""

    substrate_transport: str = "mcp"
    """Documentation/parity only — CAR's execution substrate is always MCP
    (the ``vm`` server). Kept so a config can assert it matches ale_claw."""

    upstream_version: str | None = None
    """Source CAR git SHA for the run-task binary (surfaced via deployer.version)."""

    def __post_init__(self) -> None:
        if self.prompt_mode not in ("base", "verify"):
            raise ValueError(
                f"CarConfig.prompt_mode={self.prompt_mode!r} not in {{base, verify}}"
            )
        if self.substrate_transport != "mcp":
            raise ValueError(
                f"CarConfig.substrate_transport={self.substrate_transport!r} — CAR's "
                "substrate is always 'mcp'; remove the override or set it to 'mcp'."
            )
        if self.max_turns <= 0:
            raise ValueError(f"CarConfig.max_turns must be > 0, got {self.max_turns}")
        # Fail fast on an unmapped model unless an explicit override is given.
        if self.car_model_override is None:
            map_litellm_to_car(self.model)

    def to_car_model_id(self) -> str:
        """The value for ``car run-task --model``."""
        if self.car_model_override is not None:
            return self.car_model_override
        return map_litellm_to_car(self.model)
