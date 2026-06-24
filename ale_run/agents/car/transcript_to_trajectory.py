"""Map a ``car run-task`` JSONL transcript → ALE ATIF (ALE-v1.0) steps.

Two layers, on purpose:

* :func:`car_transcript_to_steps` — **pure**: transcript path → a list of
  normalized step dicts. No ALE import, so it is unit-testable under any Python
  (the framework needs 3.12; this lets us validate the mapping standalone).
* :func:`parse_car_transcript_into` — the ALE binding: turns those dicts into
  ``ale_run.base_interface`` pydantic objects and appends them to a
  ``TrajectoryBuilder``. Imports ALE lazily (inside the function) so importing
  this module never requires the framework.

CAR transcript event → ATIF step:
  agent_turn  → source="agent"      (message + tool_calls + token metrics)
  observation → source="environment"(observation.results, one per tool result)
  truncation / near_limit_nudge / loop_nudge → source="system" (extra-tagged)
  run_start / run_resumed → not steps (instruction + resume noted on extra)
  run_end     → not a step; surfaced on trajectory.extra["car"] (the framework
                finalizes status+reward; CAR doesn't grade)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _iter_records(transcript_path: str | Path):
    path = Path(transcript_path)
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a half-written final line (SIGTERM)
            if isinstance(rec, dict):
                yield rec


def car_transcript_to_steps(transcript_path: str | Path) -> list[dict[str, Any]]:
    """Pure transcript → normalized step dicts.

    Step dict shape:
      {"source", "message", "reasoning", "tool_calls":[{id,name,arguments}],
       "observation":{"results":[{tool_call_id,content,is_error}], "error"},
       "metrics":{"input_tokens","output_tokens"}, "extra":{}}
    """
    steps: list[dict[str, Any]] = []
    for rec in _iter_records(transcript_path):
        t = rec.get("type")
        if t == "agent_turn":
            usage = rec.get("usage") or {}
            calls = [
                {
                    "id": c.get("id") or f"call_{len(steps)}",
                    "name": c.get("name", ""),
                    "arguments": c.get("arguments") or {},
                }
                for c in (rec.get("tool_calls") or [])
            ]
            steps.append({
                "source": "agent",
                "message": rec.get("text") or None,
                "reasoning": None,
                "tool_calls": calls,
                "observation": None,
                "metrics": {
                    "input_tokens": int(usage.get("input_tokens", 0) or 0),
                    "output_tokens": int(usage.get("output_tokens", 0) or 0),
                },
                "extra": {},
            })
        elif t == "observation":
            results = [
                {
                    "tool_call_id": r.get("tool_call_id", ""),
                    "content": r.get("content", ""),
                    "is_error": bool(r.get("is_error", False)),
                }
                for r in (rec.get("results") or [])
            ]
            steps.append({
                "source": "environment",
                "message": None,
                "reasoning": None,
                "tool_calls": [],
                "observation": {"results": results, "error": None},
                "metrics": None,
                "extra": {},
            })
        elif t in ("truncation", "near_limit_nudge", "loop_nudge"):
            steps.append({
                "source": "system",
                "message": f"car: {t}",
                "reasoning": None,
                "tool_calls": [],
                "observation": None,
                "metrics": None,
                "extra": {"kind": t, **{k: v for k, v in rec.items() if k != "type"}},
            })
        # run_start / run_resumed / run_end carry no step (handled as metadata).
    return steps


def read_run_end(transcript_path: str | Path) -> dict[str, Any] | None:
    """Return the terminal run_end record (status/failure_class/...), or None."""
    run_end = None
    for rec in _iter_records(transcript_path):
        if rec.get("type") == "run_end":
            run_end = rec
    return run_end


def find_transcript(work_dir: Path) -> Path:
    """The transcript path the CAR deployer writes (``<work_dir>/transcript.jsonl``)."""
    return work_dir / "transcript.jsonl"


def parse_car_transcript_into(work_dir: Path, builder: Any) -> None:
    """ALE binding: append CAR transcript steps to a TrajectoryBuilder.

    Imports ale_run lazily so this module imports fine without the framework.
    """
    from ale_run.base_interface import (  # noqa: PLC0415 — lazy on purpose
        ContentPart,
        Observation,
        StepMetrics,
        ToolCall,
        ToolResult,
    )

    transcript = find_transcript(work_dir)
    for sd in car_transcript_to_steps(transcript):
        tool_calls = [
            ToolCall(id=c["id"], name=c["name"], arguments=c["arguments"])
            for c in sd["tool_calls"]
        ]
        observation = None
        if sd["observation"] is not None:
            observation = Observation(
                results=[
                    ToolResult(
                        tool_call_id=r["tool_call_id"],
                        content=[ContentPart(type="text", text=str(r["content"]))],
                        is_error=r["is_error"],
                    )
                    for r in sd["observation"]["results"]
                ],
                error=sd["observation"]["error"],
            )
        metrics = None
        if sd["metrics"] is not None:
            metrics = StepMetrics(
                input_tokens=sd["metrics"]["input_tokens"],
                output_tokens=sd["metrics"]["output_tokens"],
            )
        builder.add_step(
            sd["source"],
            message=sd["message"],
            reasoning=sd["reasoning"],
            tool_calls=tool_calls,
            observation=observation,
            metrics=metrics,
            extra=sd["extra"],
        )

    run_end = read_run_end(find_transcript(work_dir))
    if run_end is not None:
        builder.trajectory.extra.setdefault("car", {}).update({
            "run_status": run_end.get("status"),
            "failure_class": run_end.get("failure_class"),
            "infra_tool_errors": run_end.get("infra_tool_errors"),
            "resumed": run_end.get("resumed"),
        })
