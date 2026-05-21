"""Global config and INI loader."""

import os
import re
from configparser import ConfigParser
from dataclasses import dataclass, field


@dataclass
class Config:
    ini_path: str = "presets.ini"
    host: str = "0.0.0.0"
    port: int = 11434
    idle_timeout: int = 300
    start_port: int = 12000
    work_dir: str = "."
    binary: str = "llama-server"
    debug: bool = False


@dataclass
class ModelConfig:
    section_name: str
    port: int
    ini_dir: str
    priority: int
    base_url: str = ""
    options: dict = field(default_factory=dict)
    current_priority: int = 0
    sleep_idle_seconds: int = 0
    pending_requests: int = 0


class ModelRegistry:
    """Reads presets.ini and resolves model configs from section headers."""

    def __init__(self, ini_path: str, start_port: int, work_dir: str) -> None:
        self.ini_path = ini_path
        self.start_port = start_port
        self.work_dir = work_dir
        self.parser = ConfigParser()
        self.parser.read_string(self._clean_ini(ini_path))
        self.models: list[ModelConfig] = self._load_models()

    def _clean_ini(self, ini_path: str) -> str:
        with open(ini_path, "r") as f:
            content = f.read()
        content = re.sub(r"^version\s*=\s*\S*", "", content, flags=re.MULTILINE)
        return content

    def _load_models(self) -> list[ModelConfig]:
        models = []
        for section in self.parser.sections():
            if not self.parser.has_option(section, "priority"):
                continue
            priority = int(self.parser.get(section, "priority"))
            sleep_idle = 0
            if self.parser.has_option(section, "sleep-idle-seconds"):
                sleep_idle = int(self.parser.get(section, "sleep-idle-seconds"))
            models.append(
                ModelConfig(
                    section_name=section,
                    port=0,
                    ini_dir=self.work_dir,
                    priority=priority,
                    current_priority=priority,
                    sleep_idle_seconds=sleep_idle,
                )
            )
        return models
