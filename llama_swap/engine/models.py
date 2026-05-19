"""Model config utilities."""

import os
from configparser import ConfigParser
from llama_swap.config import ModelConfig


def make_instance_url(config: ModelConfig) -> str:
    return f"http://127.0.0.1:{config.port}"


def load_ini_options(config: ModelConfig) -> dict:
    if not config.ini_dir:
        return {}
    ini_path = os.path.join(config.ini_dir, "presets.ini")
    parser = ConfigParser()
    parser.read(ini_path)
    if config.section_name in parser:
        return dict(parser[config.section_name])
    return {}
