"""Presets.ini parser for a single model."""

from configparser import ConfigParser
import os
from typing import Optional


def read_preset(ini_dir: str, section: str) -> dict:
    ini_path = os.path.join(ini_dir, "presets.ini")
    parser = ConfigParser()
    parser.read(ini_path)
    if section not in parser:
        return {}
    return dict(parser[section])


def get_model_name(ini_dir: str, section: str) -> Optional[str]:
    opts = read_preset(ini_dir, section)
    return opts.get("model", None)
