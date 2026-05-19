# llama-swap-priority

A priority-based HTTP proxy/router that sits in front of multiple `llama-server` instances and routes requests to the appropriate backend based on model name and priority configuration.

## Overview

When you have multiple `llama-server` instances serving different models (possibly with different hardware priorities), this proxy provides:

- A single entry point (OpenAI-compatible API) for clients
- Automatic routing of requests to the correct backend instance based on the `model` field
- SSE (server-sent events) streaming passthrough for chat completions
- Model list discovery via `/v1/models`

## Architecture

```
Client → llama-swap-priority (proxy) → llama-server instances (backends)
```

The proxy listens for incoming HTTP requests, determines which backend model should handle the request, and forwards the request accordingly. Streaming responses are streamed back to the client in real-time.

## Installation

```bash
# Clone or copy the project
cd llama-swap-priority

# Create a virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install -e .

# Or install directly
pip install aiohttp
```

## Configuration

Create a `presets.ini` file in the project root (or set `work_dir` in config). The INI file uses section headers for each model, with a `priority` field:

```ini
[llama3-8b]
priority = 1
model = /path/to/model.gguf

[llama3-70b]
priority = 2
model = /path/to/other-model.gguf
```

Each section maps to one model. The `priority` field (integer) determines load-balancing weight — higher priority models receive proportionally more traffic.

### Config defaults

Override defaults by passing them to `Config`:

| Setting       | Default     | Description                        |
|---------------|-------------|-----------------------------------|
| `ini_path`    | `presets.ini` | Path to the model config INI file |
| `host`        | `0.0.0.0`   | Bind address                      |
| `port`        | `11434`     | Proxy listen port                 |
| `start_port`  | `12000`     | First port for backend instances  |
| `work_dir`    | `.`         | Directory for the INI file        |
| `binary`      | `llama-server` | Path to llama-server binary     |

## Usage

### From command line

```bash
python3 -m llama_swap
```

### As a module

```python
import asyncio
from llama_swap.config import Config
from llama_swap.proxy.router import ProxyRouter

config = Config(port=11434)
router = ProxyRouter(config)
asyncio.run(router.run())
```

## API

The proxy exposes an OpenAI-compatible API:

```bash
# List available models
curl http://localhost:11434/v1/models

# Chat completions (streaming)
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3-8b", "messages": [{"role": "user", "content": "Hello"}]}'

# Non-streaming completions
curl http://localhost:11434/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3-8b", "prompt": "Hello"}'
```

## Project Structure

```
llama-swap-priority/
├── llama_swap/
│   ├── __init__.py
│   ├── config.py          # Config, ModelConfig, ModelRegistry
│   ├── main.py            # Entry point
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── models.py      # URL helpers, options loader
│   │   └── request_context.py  # Request state
│   ├── instance/
│   │   ├── __init__.py
│   │   └── manager.py     # Start/stop llama-server instances
│   ├── preset/
│   │   ├── __init__.py
│   │   └── ini_parser.py  # Presets.ini reader
│   └── proxy/
│       ├── __init__.py
│       └── router.py      # ProxyRouter (aiohttp server)
├── pyproject.toml
└── README.md
```

## Requirements

- Python 3.12+
- `aiohttp` (pip installable)
- `llama-server` binary available on `$PATH`
