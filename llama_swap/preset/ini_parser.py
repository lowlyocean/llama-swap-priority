"""Presets.ini parser for a single model."""

import os
import re
from configparser import ConfigParser
from typing import Optional


def _clean_ini(ini_path: str) -> str:
    with open(ini_path, "r") as f:
        content = f.read()
    content = re.sub(r"^version\s*=\s*\S*", "", content, flags=re.MULTILINE)
    return content


def read_preset(ini_dir: str, section: str) -> dict:
    ini_path = os.path.join(ini_dir, "presets.ini")
    parser = ConfigParser()
    parser.read_string(_clean_ini(ini_path))
    if section not in parser:
        return {}
    return dict(parser[section])


def get_model_name(ini_dir: str, section: str) -> Optional[str]:
    opts = read_preset(ini_dir, section)
    return opts.get("model", None)
