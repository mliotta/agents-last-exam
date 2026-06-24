"""CarDeployer — CAR as an ALE out-of-sandbox harness.

``launch()`` shells out to ``car run-task`` (the Rust agent loop) pointed at the
eval VM through the bundled ``vm_mcp_server`` (shell/files) + ``cua_mcp_server``
(GUI) bridges — the same bridges ale_claw uses, staged via the shared
``_bootstrap`` helpers. So the leaderboard entry exercises the exact code path as
CAR's internal benchmark; the agent loop, execution substrate, resume, and
failure taxonomy all live in the binary (docs/ale-harness-adapter-scope.md).

CAR reaches the backbone provider-direct (no OpenRouter path); the experiment's
LiteLLM ``model:`` is mapped to a CAR ``--model`` alias by :class:`CarConfig`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, ClassVar

from ale_run.base_interface import (
    AgentRunResult,
    BaseAgentDeployer,
    TrajectoryBuilder,
)
from ale_run.agents._bootstrap import (
    cua_bridge_env,
    ensure_cua_mcp_server_at,
    ensure_node_npm,
    ensure_vm_mcp_server,
    vm_bridge_env,
)

from .config import CarConfig
from .transcript_to_trajectory import find_transcript, parse_car_transcript_into, read_run_end

logger = logging.getLogger(__name__)


class CarDeployer(BaseAgentDeployer):
    """Common Agent Runtime (CAR) out-of-sandbox harness deployer."""

    default_executor: ClassVar[str] = "local"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"local", "docker"})

    # At least one provider key must be set (CAR is provider-direct: Anthropic /
    # OpenAI / Google — no OpenRouter path).
    _api_key_alternatives: ClassVar[tuple[str, ...]] = (
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
    )

    # ---- helpers -----------------------------------------------------------

    def _car_binary(self) -> str:
        cfg: CarConfig = self.config  # type: ignore[assignment]
        return cfg.car_binary or os.environ.get("CAR_BINARY") or "car"

    # ---- install -----------------------------------------------------------

    async def install(self) -> None:
        cfg: CarConfig = self.config  # type: ignore[assignment]
        car_bin = self._car_binary()
        if shutil.which(car_bin) is None and not Path(car_bin).is_file():
            raise RuntimeError(
                f"CarDeployer: `car` binary not found ({car_bin!r}). Build it "
                "(cargo build -p car-cli --release) or set config.car_binary / $CAR_BINARY."
            )
        if not any(os.environ.get(k) for k in self._api_key_alternatives):
            raise RuntimeError(
                f"CarDeployer: no provider key in env — set one of "
                f"{', '.join(self._api_key_alternatives)} (CAR is provider-direct)."
            )
        # Validate the model maps now (fail fast before VMs spin up).
        _ = cfg.to_car_model_id()
        Path(self.executor.work_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            "CarDeployer: install ok (model=%s → %s, work_dir=%s, executor=%s)",
            cfg.model, cfg.to_car_model_id(), self.executor.work_dir, self.executor.type,
        )

    # ---- launch ------------------------------------------------------------

    async def launch(self, prompt: str) -> AgentRunResult:
        cfg: CarConfig = self.config  # type: ignore[assignment]
        work_dir = Path(self.executor.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1. Stage the MCP bridges (idempotent) and build the run-task mcp config.
        #    Bridges run host-side and reach the VM's cua-server via CUA_SERVER_URL
        #    (== sandbox.endpoint) — same pattern as ale_claw. CAR's `vm` server
        #    name backs its execution substrate; `cua` becomes mcp_cua_* tools.
        node_path, _ = await ensure_node_npm()
        vm_dir = await ensure_vm_mcp_server(str(work_dir / "mcp" / "vm"))
        servers: list[dict[str, Any]] = [{
            "name": "vm",
            "command": node_path,
            "args": [os.path.join(vm_dir, "src", "index.js")],
            "env": vm_bridge_env(self.executor),
        }]
        if cfg.enable_gui:
            cua_dir = await ensure_cua_mcp_server_at(str(work_dir / "mcp" / "cua"))
            servers.append({
                "name": "cua",
                "command": node_path,
                "args": [os.path.join(cua_dir, "src", "index.js")],
                "env": cua_bridge_env(self.executor),
            })

        mcp_config_path = work_dir / "mcp.json"
        mcp_config_path.write_text(json.dumps({"servers": servers}, indent=2))
        goal_path = work_dir / "goal.txt"
        goal_path.write_text(prompt)
        transcript_path = find_transcript(work_dir)
        eventlog_path = work_dir / "eventlog.jsonl"
        stderr_path = work_dir / "car_run_task.stderr.log"

        # 2. Build the car run-task invocation.
        cmd = [
            self._car_binary(), "run-task",
            "--goal-file", str(goal_path),
            "--mcp-config", str(mcp_config_path),
            "--transcript", str(transcript_path),
            "--eventlog", str(eventlog_path),
            "--max-turns", str(cfg.max_turns),
            "--model", cfg.to_car_model_id(),
        ]
        if cfg.resume:
            cmd.append("--resume")

        env = dict(os.environ)
        if cfg.prompt_mode:
            env["CAR_RUNTASK_PROMPT"] = cfg.prompt_mode

        # 3. Drive. The episode wall budget is orchestration-owned (the executor
        #    wraps launch() in asyncio.wait_for); a cancellation kills the child.
        logger.info("CarDeployer: launch — %s", " ".join(cmd))
        t0 = time.monotonic()
        with stderr_path.open("wb") as errf:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=errf, env=env,
            )
            try:
                exit_code = await proc.wait()
            except asyncio.CancelledError:
                proc.kill()
                await proc.wait()
                raise  # framework records timeout

        # 4. Map CAR's exit code + run_end → ALE status.
        run_end = read_run_end(transcript_path) or {}
        failure_class = run_end.get("failure_class")
        # 0 completed, 1 task_max_turns → "completed" (a turn-budget finish is a
        # real outcome, not a wall-clock failure — mirrors ale_claw). 2/3 → failed.
        if exit_code in (0, 1):
            status = "completed"
            error = None
        else:
            status = "failed"
            error = run_end.get("error") or f"car run-task exit {exit_code} ({failure_class})"

        return AgentRunResult(
            status=status,
            duration_s=time.monotonic() - t0,
            transcript_path=str(transcript_path) if transcript_path.exists() else None,
            stderr_path=str(stderr_path) if stderr_path.exists() else None,
            exit_code=exit_code,
            error=error,
        )

    # ---- parse_artifacts ---------------------------------------------------

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: CarConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        if not work_dir.exists():
            builder.add_step(
                source="system",
                message=f"car: work_dir missing {work_dir}",
                extra={"reason": "no_work_dir"},
            )
            return
        try:
            parse_car_transcript_into(work_dir, builder)
        except Exception as exc:  # noqa: BLE001
            logger.exception("CarDeployer: parse_artifacts failed")
            builder.add_step(
                source="system",
                message=f"car: transcript parse failed: {type(exc).__name__}: {exc}",
                extra={"reason": "parse_error"},
            )
        builder.trajectory.extra.setdefault("car", {}).update({
            "work_dir": str(work_dir),
            "transcript_path": run_result.transcript_path,
            "run_status": run_result.status,
            "exit_code": run_result.exit_code,
        })

    # ---- version -----------------------------------------------------------

    @property
    def version(self) -> str | None:
        cfg: CarConfig = self.config  # type: ignore[assignment]
        if cfg.upstream_version:
            return cfg.upstream_version
        try:
            out = __import__("subprocess").run(
                [self._car_binary(), "--version"], capture_output=True, text=True, timeout=10,
            )
            return (out.stdout or out.stderr).strip() or None
        except Exception:  # noqa: BLE001
            return None
