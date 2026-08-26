# llama-swap-priority

## Quick commands

```bash
# Run tests
.venv/bin/pytest test_proxy.py -v

# Run proxy locally
python -m llama_swap

# Run proxy with debug output
python -m llama_swap --debug

# Docker
docker compose -f docker-compose.yml up
```

## Architecture

```
Client → ProxyRouter (aiohttp, port 11434) → Docker containers (llama-server backends)
```

Key files:
- **`llama_swap/proxy/router.py:ProxyRouter`** — aiohttp server, routes requests, manages instance lifecycle, idle timers, preemption
- **`llama_swap/instance/manager.py`** — `start_instance()` / `stop_instance()` for Docker containers; `_instances` dict keyed by section name
- **`llama_swap/config.py`** — `Config`, `ModelConfig`, `ModelRegistry` (reads `presets.ini`, resolves `default-running-model`)
- **`llama_swap/preset/ini_parser.py`** — reads presets.ini, strips `version` line, returns section options
- **`llama_swap/main.py`** — CLI entry point, parses `--debug`, runs router

## Entry points

```python
# Main entry
python -m llama_swap

# Programmatic
config = Config(port=11434)
router = ProxyRouter(config)
asyncio.run(router.run())
```

## Key behaviors

### Preemption
When a request arrives with priority higher than all running instances, the proxy terminates lower-priority instances before routing. When priority is equal or lower and no free instance exists, returns HTTP 429 with `retry_after`. A 2-second cooldown after preemption prevents re-accepting requests immediately.

### Idle shutdown
`sleep-idle-seconds` from `presets.ini` starts a timer when a model has no pending requests. After the timeout elapses, the instance is stopped to free GPU memory. The timer only fires when no requests are pending and resets on each new request. Non-interactive endpoints (`/v1/models`, `/v1/props`, `/v1/metrics`) do not affect idle timers.

### Default-running model
- Add `default-running-model = <section_name>` to the `[*]` section in `presets.ini`
- The referenced model must have a `priority` field (sections without priority are skipped)
- When all instances are stopped and a default model is configured, the proxy automatically launches it
- While running as the default model, idle timeouts are disabled — the instance stays running
- If a new request arrives for a higher-priority model, the default model is preempted
- When the higher-priority request completes, the default model is relaunched automatically
- Default-running instances are tracked with `InstanceState.default_running = True`

### Bootstrap
On startup, the proxy:
1. Starts a temporary "bootstrap" instance on port 20000 with all presets loaded
2. Fetches `/models` to populate `self._models_response`
3. For each model in the registry, starts an instance on the bootstrap port, fetches `/props?model=<name>`, caches the result, then tears down
4. Tears down the bootstrap instance
5. Starts the health check loop and begins accepting client requests

### Instance lifecycle
- `start_instance(model_config)` kills any existing Docker container for the same section, then launches a new container via `docker run` with a filtered presets file (only `[*]` + target section, stripped of `priority` and `sleep-idle-seconds`)
- Container names are derived from section names: `llama_server_<safe_name>` (colons, slashes, spaces replaced with hyphens)
- `stop_instance(name)` removes the Docker container via `docker rm -f`
- Health check runs every 5s, pings `{host}:{port}/props` for `is_sleeping` boolean
- `_instances` is a module-level dict in `llama_swap/instance/manager.py` shared across all router instances

### SSE streaming
`forward_request()` checks `resp.content_type` — if `text/event-stream`, streams chunks via `web.StreamResponse`; otherwise returns JSON. Retries once on connection error if instance is still loading.

## Config

| Setting    | Default      | Notes |
|------------|-------------|-------|
| `ini_path` | `presets.ini` | Model definitions, INI format with `[section]` headers |
| `port`     | 11434       | Proxy listen port |
| `start_port` | 12000     | First port assigned to backend instances |
| `work_dir` | `.`         | Directory for presets.ini |
| `binary`   | `llama-server` | Docker entrypoint name |
| `debug`    | `False`     | Enable debug logging |

## presets.ini format

```ini
[high]
priority = 10
sleep-idle-seconds = 600
model = /models/high.gguf

[low]
priority = 1
sleep-idle-seconds = 300
model = /models/low.gguf
```

- `priority` (required) — integer, higher = higher priority; sections without this field are skipped
- `sleep-idle-seconds` (optional) — seconds of inactivity before instance stops; `0` disables it
- `model` — path to the model file
- `[*]` section provides common defaults merged into each model's filtered config
- Proxy-only fields (`priority`, `sleep-idle-seconds`) are stripped from filtered presets passed to llama-server
- `version` line at the top of the file is stripped by `_clean_ini()`

## Running locally (without Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m llama_swap
```

Docker is still required for the llama-server backend containers — the proxy spawns containers via the Docker API. You must have `nvidia-container-toolkit` and a `presets.ini` in place.

## Testing

- `test_proxy.py` — all tests (29 tests), covers config, registry, parser, instance manager, proxy router, idle timers, preemption cooldown, default-running model
- Requires `pytest-asyncio` for async tests
- Uses `aiohttp.test_utils.AioHTTPTestCase` for router tests
- Run: `.venv/bin/pytest test_proxy.py -v`

## Docker

```bash
docker compose -f docker-compose.yml up
```

- Builds from `python:3.12-slim`, copies `llama_swap/`, installs the package, runs `python -m llama_swap`
- Mounts `presets.ini` and model files into the container
- Uses `${ENV_FILE:-./stack.env}` for env vars
- GPU passthrough via nvidia runtime
- Network mode: `host` (container sees ports directly)
- Mounts `/var/run/docker.sock` so the proxy can spawn backend containers

## Gotchas

- Section names in `presets.ini` are lowercased by `ConfigParser` — if your section names are case-sensitive, account for this in tests
- `InstanceState` fields `current_priority` and `default_running` track runtime state; `config` is `None` for docker instances
- `_instances` is a module-level dict — tests that mutate it need to clean up or use fresh router instances
- Bootstrap port is hardcoded to 20000; don't change it without updating `_bootstrap_models()`
- `filter_section_presets()` and `filter_all_presets()` write temp files to `/tmp` with prefix `llama-swap-`
- Idle timers use `_idle_fired` set to prevent duplicate shutdowns; cleared on new requests
