"""Instance management — start, stop, health-check llama-server instances."""

import asyncio
import os
import re
import subprocess
import time
from configparser import ConfigParser
from dataclasses import dataclass, field
from typing import Optional

from llama_swap.config import ModelConfig


def _clean_ini(ini_path: str) -> str:
    with open(ini_path, "r") as f:
        content = f.read()
    content = re.sub(r"^version\s*=\s*\S*", "", content, flags=re.MULTILINE)
    return content


PREEMPTION_COOLDOWN_SECONDS = 2


@dataclass
class InstanceState:
    config: Optional[ModelConfig] = None
    port: int = 0
    process: Optional[subprocess.Popen] = None
    healthy: bool = False
    running: bool = False
    current_priority: int = 0
    preempted_at: float | None = None


def filter_section_presets(ini_dir: str, section_name: str) -> str:
    """Read presets.ini and write a temporary file with only the [*] and target section.

    Strips proxy-only fields like 'priority' before writing.
    """
    ini_path = os.path.join(ini_dir, "presets.ini")
    cleaned = _clean_ini(ini_path)
    parser = ConfigParser()
    parser.read_string(cleaned)
    proxy_fields = {"priority", "sleep-idle-seconds"}

    import tempfile
    tmp = tempfile.NamedTemporaryFile(
        prefix=f"llama-swap-{section_name}-",
        suffix=".ini",
        delete=False,
        dir="/tmp",
        mode="w",
    )
    # Write the [*] section if it exists (common defaults)
    # ConfigParser lowercases section names, so [*] becomes "*"
    has_star = False
    for key in parser:
        if key == "*":
            has_star = True
            break
    if has_star:
        opts = dict(parser["*"])
        opts = {k: v for k, v in opts.items() if k not in proxy_fields}
        if opts:
            tmp.write("[*]\n")
            for k, v in opts.items():
                tmp.write(f"{k} = {v}\n")
            tmp.write("\n")

    # Write the target model section
    if section_name in parser:
        opts = dict(parser[section_name])
        opts = {k: v for k, v in opts.items() if k not in proxy_fields}
        if opts:
            tmp.write(f"[{section_name}]\n")
            for k, v in opts.items():
                tmp.write(f"{k} = {v}\n")
            tmp.write("\n")

    tmp.flush()
    tmp.close()
    return tmp.name


_instances: dict[str, InstanceState] = {}


async def _start_docker_container(section_name: str, port: int, ini_path: str, debug: bool = False) -> str:
    """Start a docker container for the given model. Returns the container name."""
    safe = section_name.replace(":", "-").replace("/", "-").replace(" ", "-")
    container_name = f"llama_server_{safe}"

    if debug:
        print(f"[DEBUG] Stopping existing container: {container_name}")

    try:
        await asyncio.to_thread(
            subprocess.run,
            ["docker", "rm", "-f", "-f", container_name],
            capture_output=True,
            check=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, Exception):
        pass

    cmd = [
        "docker", "run",
        "--name", container_name,
        "--network", "host",
        "--gpus", "all",
        *[item for key in os.environ.keys() for item in ("-e", key)],
        "--restart", "unless-stopped",
        "-v", f"""{os.environ.get("MODELS_PATH", "./models/")}:/app/models/:ro""",
        "--mount", f"type=bind,source={ini_path},target=/app/presets.ini,readonly",
        "--entrypoint", "./llama-server",
        os.environ.get("SERVER_IMAGE", "local/llama.cpp:full-cuda"),
        "--models-preset", "./presets.ini",
        "--host", "0.0.0.0",
        "--port", str(port),
    ]

    if debug:
        print(f"[DEBUG] Running docker command: {' '.join(cmd)}")

    stdout = subprocess.DEVNULL if not debug else None
    stderr = subprocess.DEVNULL if not debug else subprocess.STDOUT
    proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
    return section_name


async def start_instance(
    model_config: "ModelConfig",
    debug: bool = False,
) -> InstanceState:
    """Start a llama-server instance for the given model config.

    Kills any existing instance for the same section_name first to ensure
    no lingering instances. GPU resources are freed immediately.
    """
    if debug:
        print(f"[DEBUG] start_instance: model={model_config.section_name}, port={model_config.port}")

    section_name = model_config.section_name
    safe = section_name.replace(":", "-").replace("/", "-").replace(" ", "-")
    container_name = f"llama_server_{safe}"

    # Terminate existing instance for this model if running
    existing = _instances.get(section_name)
    if existing and existing.process:
        try:
            if debug:
                print(f"[DEBUG] Removing existing container: {container_name}")
            await asyncio.to_thread(
                subprocess.run,
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                check=True,
                timeout=10,
            )
        except (subprocess.CalledProcessError, Exception):
            pass

    filtered_ini = filter_section_presets(model_config.ini_dir, section_name)
    if debug:
        print(f"[DEBUG] Filtered presets: {filtered_ini}")

    await _start_docker_container(section_name, model_config.port, filtered_ini, debug)

    await asyncio.sleep(0.5)

    state = InstanceState(
        config=model_config,
        port=model_config.port,
        process=None,
        healthy=True,
        current_priority=model_config.priority,
    )
    _instances[section_name] = state
    return state


async def stop_instance(model_name: str, debug: bool = False) -> None:
    """Stop and remove an instance. Immediately frees GPU resources."""
    if debug:
        print(f"[DEBUG] stop_instance: model={model_name}")
    safe = model_name.replace(":", "-").replace("/", "-").replace(" ", "-")
    container_name = f"llama_server_{safe}"
    inst = _instances.pop(model_name, None)
    if inst:
        try:
            if debug:
                print(f"[DEBUG] Removing container: {container_name}")
            await asyncio.to_thread(
                subprocess.run,
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                check=True,
                timeout=10,
            )
            if debug:
                print(f"[DEBUG] Container removed: {container_name}")
        except (subprocess.CalledProcessError, Exception) as e:
            if debug:
                print(f"[DEBUG] Failed to remove container {container_name}: {e}")


def get_health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/health"
