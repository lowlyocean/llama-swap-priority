"""Tests for llama-swap-priority."""

import asyncio
import json
import pytest
import urllib.request
import threading
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
from llama_swap.config import Config, ModelConfig, ModelRegistry
from llama_swap.proxy.router import ProxyRouter
from llama_swap.instance.manager import (
    InstanceState,
    start_instance,
    stop_instance,
    get_health_url,
    filter_section_presets,
)
from llama_swap.engine.models import make_instance_url, load_ini_options
from llama_swap.preset.ini_parser import read_preset, get_model_name


class TestConfig:
    def test_config_defaults(self):
        config = Config()
        assert config.host == "0.0.0.0"
        assert config.port == 11434
        assert config.start_port == 12000
        assert config.binary == "llama-server"

    def test_config_override(self):
        config = Config(port=9999, host="127.0.0.1")
        assert config.port == 9999
        assert config.host == "127.0.0.1"


class TestModelRegistry:
    def test_load_models(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "[model_a]\npriority = 1\nmodel = /models/model_a.gguf\n\n[model_b]\npriority = 2\nmodel = /models/model_b.gguf\n"
        )
        reg = ModelRegistry(str(presets), 12000, str(tmp_path))
        assert len(reg.models) == 2
        assert reg.models[0].section_name == "model_a"
        assert reg.models[0].priority == 1
        assert reg.models[1].priority == 2

    def test_no_priority_skipped(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "[model_a]\npriority = 1\nmodel = /models/model_a.gguf\n\n[model_b]\nmodel = /models/model_b.gguf\n"
        )
        reg = ModelRegistry(str(presets), 12000, str(tmp_path))
        assert len(reg.models) == 1
        assert reg.models[0].section_name == "model_a"


class TestModelConfig:
    def test_make_instance_url(self):
        config = ModelConfig(
            section_name="test", port=12345, ini_dir=".", priority=1
        )
        assert make_instance_url(config) == "http://127.0.0.1:12345"


class TestPresetParser:
    def test_read_preset(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text("[test]\nmodel = /models/test.gguf\n")
        result = read_preset(str(tmp_path), "test")
        assert result["model"] == "/models/test.gguf"

    def test_read_preset_missing(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text("[other]\nmodel = /models/other.gguf\n")
        result = read_preset(str(tmp_path), "missing")
        assert result == {}

    def test_get_model_name(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text("[test]\nmodel = /models/test.gguf\n")
        assert get_model_name(str(tmp_path), "test") == "/models/test.gguf"


class TestInstanceManager:
    def test_filter_section_presets(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "[*]\nbatch = 32\n\n[section_a]\npriority = 1\nmodel = /models/a.gguf\n\n[section_b]\npriority = 2\nmodel = /models/b.gguf\n"
        )
        filtered = filter_section_presets(str(tmp_path), "section_a")
        import os

        assert os.path.isfile(filtered)
        content = open(filtered).read()
        assert "section_a" in content
        assert "priority" not in content
        assert "[*]" in content
        assert "batch" in content

    def test_get_health_url(self):
        assert get_health_url(12345) == "http://127.0.0.1:12345/health"


class TestProxyRouter:
    @pytest.mark.asyncio
    async def test_models_list(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "[model_a]\npriority = 1\nmodel = /models/model_a.gguf\n\n[model_b]\npriority = 2\nmodel = /models/model_b.gguf\n"
        )
        config = Config(port=0, ini_path=str(presets), work_dir=str(tmp_path))
        router = ProxyRouter(config)
        await router.register_routes()

        app = router._app
        assert app is not None

        from aiohttp.test_utils import make_mocked_request

        req = make_mocked_request("GET", "/v1/models")
        resp = await router.handle_model_list(req)
        assert resp.status == 200
        body = resp.text
        assert "model_a" in body
        assert "model_b" in body

        await router.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
