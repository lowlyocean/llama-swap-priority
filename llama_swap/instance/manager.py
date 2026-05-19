"""Instance management — start, stop, health-check llama-server instances."""

import os
import subprocess
import asyncio
from dataclasses import dataclass, field
from typing import Optional

from llama_swap.config import ModelConfig


@dataclass
class InstanceState:
    config: Optional[ModelConfig] = None
    port: int = 0
    process: Optional[subprocess.Popen] = None
    healthy: bool = False
    current_priority: int = 0


def filter_section_presets(ini_dir: str, section_name: str) -> str:
    """Read presets.ini and write a temporary file with only the [*] and target section.

    Strips proxy-only fields like 'priority' before writing.
    """
    import configparser
    ini_path = os.path.join(ini_dir, "presets.ini")
    parser = configparser.ConfigParser()
    parser.read(ini_path)
    proxy_fields = {"priority"}

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


async def start_instance(
    model_config: "ModelConfig",
) -> InstanceState:
    """Start a llama-server instance for the given model config.

    Kills any existing instance for the same section_name first to ensure
    no lingering instances. GPU resources are freed immediately.
    """
    section_name = model_config.section_name

    # Terminate existing instance for this model if running
    existing = _instances.get(section_name)
    if existing and existing.process and existing.process.returncode is None:
        existing.process.kill()
        try:
            await asyncio.wait_for(
                asyncio.to_thread(existing.process.wait), timeout=5
            )
        except asyncio.TimeoutError:
            pass

    filtered_ini = filter_section_presets(model_config.ini_dir, section_name)

    cmd = [
        "llama-server",
        "--models-presets", filtered_ini,
        "--host", "0.0.0.0",
        "--port", str(model_config.port),
    ]

    env = os.environ.copy()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    await asyncio.sleep(0.5)

    healthy = proc.returncode is None
    state = InstanceState(
        config=model_config,
        port=model_config.port,
        process=proc,
        healthy=healthy,
        current_priority=model_config.priority,
    )
    _instances[section_name] = state
    return state


async def stop_instance(model_name: str) -> None:
    """Stop and remove an instance. Immediately frees GPU resources."""
    inst = _instances.pop(model_name, None)
    if inst and inst.process and inst.process.returncode is None:
        inst.process.kill()
        try:
            await asyncio.wait_for(
                asyncio.to_thread(inst.process.wait), timeout=5
            )
        except asyncio.TimeoutError:
            pass


def get_health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/health"
