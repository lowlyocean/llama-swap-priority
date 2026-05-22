# llama-swap-priority

## Quick commands

```bash
# Tests
.venv/bin/pytest test_proxy.py -v

# Run the proxy
python -m llama_swap
```

## Architecture

```
Client → ProxyRouter (aiohttp, port 11434) → llama-server instances (ports 12000+)
```

- **`llama_swap/proxy/router.py:ProxyRouter`** — aiohttp server, entry point, routes requests, manages instance lifecycle
- **`llama_swap/instance/manager.py`** — `start_instance()` / `stop_instance()` for llama-server backends; `_instances` dict keyed by section name
- **`llama_swap/config.py`** — `Config`, `ModelConfig`, `ModelRegistry` (reads `presets.ini`)
- **`llama_swap/preset/ini_parser.py`** — reads presets.ini, strips proxy-only fields

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
When a request arrives with priority higher than all running instances, the proxy terminates lower-priority instances before routing. When priority is equal or lower and no free instance exists, returns HTTP 429 with `retry_after`.

### Idle shutdown
`sleep-idle-seconds` from `presets.ini` starts a timer when a model has no pending requests. After the timeout elapses, the instance is stopped to free GPU memory. The timer only fires when no requests are pending and resets on each new request. Non-interactive endpoints (`/v1/models`, `/v1/props`, `/v1/metrics`) do not affect idle timers.

### Instance lifecycle
- `start_instance(model_config)` kills any existing instance for the same section, then launches `llama-server` with a filtered presets file (only `[*]` + target section, stripped of `priority` and `sleep-idle-seconds` fields)
- `stop_instance(name)` kills the process via `proc.kill()` + `asyncio.to_thread(proc.wait)`, no blocking sleep
- Health check runs every 5s, pings `{host}:{port}/props` for `is_sleeping` boolean, updates `inst.healthy`

### SSE streaming
`forward_request()` checks `resp.content_type` — if `text/event-stream`, streams chunks via `web.StreamResponse`; otherwise returns JSON. Catches `asyncio.CancelledError` and `asyncio.TimeoutError`.

## Config

| Setting    | Default      | Notes |
|------------|-------------|-------|
| `ini_path` | `presets.ini` | Model definitions, INI format with `[section]` headers |
| `port`     | 11434       | Proxy listen port |
| `start_port` | 12000     | First port assigned to backend instances |
| `work_dir` | `.`         | Directory for presets.ini |

## presets.ini format

```ini
[model_name]
priority = 1
sleep-idle-seconds = 300
model = /path/to/model.gguf
```

- `priority` (required) — integer, higher = higher priority; sections without this field are skipped
- `sleep-idle-seconds` (optional) — seconds of inactivity before instance stops; `0` disables it
- Sections with no `priority` field are skipped by `ModelRegistry`
- The `[*]` section provides common defaults merged into each model's filtered config
- Proxy-only fields (`priority`, `sleep-idle-seconds`) are stripped from filtered presets passed to llama-server

## Tests

- `test_proxy.py` — all tests, 11 tests covering config, registry, parser, instance manager, proxy router
- Requires `pytest-asyncio` for async tests
- Run: `.venv/bin/pytest test_proxy.py -v`

## Docker

```bash
docker compose -f docker-compose.yml up
```

Builds from `python:3.12-slim`, copies `llama_swap/`, installs the package, runs `python -m llama_swap`. Mounts `presets.ini` and model files into the container. Uses `${ENV_FILE:-./stack.env}` for env vars. GPU passthrough via nvidia runtime.
