# llama-swap-priority

A drop-in replacement for llama.cpp's built-in router that adds **priority-based preemption** and **idle instance shutdown** on top of the existing `presets.ini` format.

## What makes this different

| Feature | llama.cpp router | llama-swap-priority |
|---------|-----------------|---------------------|
| Priority-based preemption | No | Yes — higher-priority requests preempt lower-priority instances |
| Automatic idle shutdown | No | Yes — instances stop after `sleep-idle-seconds` of inactivity, freeing GPU |
| Works with existing presets.ini | N/A | Yes — just add `priority` and optionally `sleep-idle-seconds` fields |
| SSE streaming passthrough | Yes | Yes |
| Docker support | No | Yes — GPU passthrough built in |

**In short:** If you already use llama.cpp's router mode with a `presets.ini`, you can switch to this proxy by adding two fields to your config. Higher-priority models automatically preempt lower ones, and idle instances shut down to reclaim GPU memory.

## Prerequisites

- **Docker** installed and running
- **NVIDIA GPU** with CUDA drivers and the `nvidia-container-toolkit` installed
- An existing **`presets.ini`** file from your llama.cpp router setup

## Setup

### 1. Clone and prepare

```bash
git clone <repo-url>
cd llama-swap-priority
```

### 2. Configure your presets.ini

Copy your existing `presets.ini` into the project root. Add a `priority` field (integer) to each model section. Optionally add `sleep-idle-seconds` to control when idle instances shut down:

```ini
[*]
batch = 32

[llama3-8b]
priority = 1
sleep-idle-seconds = 300
model = /models/llama3-8b.gguf

[llama3-70b]
priority = 2
sleep-idle-seconds = 600
model = /models/llama3-70b.gguf
```

**How priority works:**
- Higher number = higher priority
- When a request arrives for a higher-priority model than any running instance, the proxy terminates the lower-priority instance(s) before routing
- When priority is equal or lower and no free instance exists, returns HTTP 429 (server busy)

**How `sleep-idle-seconds` works:**
- Set to `0` to disable (instance stays running indefinitely)
- Set to a number of seconds — the instance stops after that many seconds of no requests
- Only fires when no other requests are pending for the model
- Non-interactive endpoints (`/v1/models`, `/v1/props`, `/v1/metrics`) do not affect idle timers

### 3. Configure environment

Create a `.env` file in the project root. This file controls GPU passthrough and paths:

```bash
# GPU passthrough
NVIDIA_VISIBLE_DEVICES=all
CUDA_VISIBLE_DEVICES=0,1

# Paths to your presets.ini and model files
PRESETS_PATH=/path/to/your/presets.ini
MODELS_PATH=/path/to/your/models/

# Docker image for llama.cpp backend
SERVER_IMAGE=local/llama.cpp:full-cuda
ENV_FILE=./.env
```

Key fields:
- `PRESETS_PATH` — absolute path to your `presets.ini` file (mounted read-only into the container)
- `MODELS_PATH` — absolute path to where your model files are stored (mounted read-only)
- `SERVER_IMAGE` — the Docker image that runs the actual llama.cpp server (adjust for your setup)
- `ENV_FILE` — path to the `.env` file (defaults to `./` if not set)

### 4. Run

```bash
docker compose -f docker-compose.yml up
```

The proxy listens on port **11434** (Ollama's default port).

### 5. Verify

```bash
# List available models
curl http://localhost:11434/v1/models

# Chat completion (streaming)
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3-8b", "messages": [{"role": "user", "content": "Hello"}]}'
```

Any OpenAI-compatible client can now point to `http://localhost:11434/v1/`.

## How it works at a glance

```
Client → Proxy (port 11434) → Docker containers (llama-server backends)
```

The proxy routes requests to backend containers and manages their lifecycle based on priority and idle timeouts.

### Non-interactive endpoints

These endpoints do **not** affect idle timers or preemption logic:

- `GET /v1/models` — List available models (falls back to backend if all unhealthy)
- `GET /v1/props` — Proxy to the first running instance
- `GET /v1/metrics` — Proxy to the first running instance, forwards query params

### Interactive endpoints

These trigger preemption, idle timers, and instance lifecycle management:

- `POST /v1/chat/completions` — Chat completions (streaming)
- `POST /v1/completions` — Legacy completions
- `POST /v1/embeddings` — Embeddings

## How priority preemption works

When a request arrives for a higher-priority model than any running instance, the proxy terminates the lower-priority instance(s) before routing. If priority is equal or lower and no free instance exists, it returns HTTP 429 (server busy). A 2-second cooldown prevents re-accepting requests immediately after preemption.

## Running locally (without Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m llama_swap
```

This runs the proxy directly on your machine. Docker is still required for the llama-server backend containers — the proxy will spawn containers via the Docker API. You must have the `nvidia-container-toolkit` and a `presets.ini` in place.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/models` | List available models |
| POST | `/v1/chat/completions` | Chat completions (streaming) |
| POST | `/v1/completions` | Legacy completions |
| POST | `/v1/embeddings` | Embeddings |

All paths also support the `/{model}` suffix variant.

## Requirements

- Python 3.12+ (for local development)
- Docker + Docker Compose (for production)
- NVIDIA GPU with CUDA support (for backends)
- `aiohttp` Python package
