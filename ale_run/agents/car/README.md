# CAR (Common Agent Runtime)

CAR as an **out-of-sandbox** harness. `launch()` shells out to the `car run-task`
binary, which runs the agent loop in its own process and drives the eval VM
remotely through the bundled MCP bridges — the agent loop, execution substrate,
context management, and resume all live in the (Rust) runtime.

## Architecture

```
FRAMEWORK (this deployer)                  SANDBOX
lifecycle.py                               cua-computer-server (HTTP, :5000)
  install()  -> verify `car` binary, provider key, model-map
  launch()   -> car run-task --goal-file --mcp-config --transcript --model
                  |                          ^
                  +-- vm_mcp_server (stdio) --+  shell / files / pty
                  +-- cua_mcp_server (stdio) -+  GUI (mcp_cua_* tools)
                     (CUA_SERVER_URL == sandbox.endpoint, via _bootstrap)
  parse_artifacts() <- transcript.jsonl  -> ATIF steps
```

CAR's `vm` server backs its execution substrate, so the built-in
`read_file`/`write_file`/`run_command` tools execute **inside the VM** (one
coherent filesystem + shell); `cua` is exposed as `mcp_cua_*` GUI tools. The
bridges are staged with the shared `_bootstrap` helpers and reached over
`CUA_SERVER_URL == sandbox.endpoint` — the same pattern as `ale_claw`.

## Model ids

CAR reaches providers **directly** (Anthropic / OpenAI / Google; no OpenRouter
route). The experiment's LiteLLM `model:` is mapped to a CAR `--model` alias in
`config.py` (`LITELLM_TO_CAR`); the underlying checkpoint is what must match for
"same backbone". An unmapped id is a hard error (no silent mis-route); set
`car_model_override` to pass a CAR id straight through.

## Config

`configs/agents/car.yaml`. Keys (defaults in `config.py`): `model`, `max_turns`,
`prompt_mode` (`base` | `verify`), `enable_gui`, `resume`, `car_model_override`,
`car_binary`. Auth: provider keys in the operator's shell env
(`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY` | `GEMINI_API_KEY`).

## Prereqs

The `car` binary on `$PATH` (or `config.car_binary` / `$CAR_BINARY`):
`cargo build -p car-cli --release` from the CAR repo, or a published release.

## Provenance

Maintained in the CAR repo (`bench/ale-harness/`) and vendored here via
`sync-to-fork.sh`. CAR: https://github.com/Parslee-ai/car-releases
