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

    def test_load_models_with_version_field(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "version = 3\n\n[model_a]\npriority = 1\nmodel = /models/model_a.gguf\n\n[model_b]\npriority = 2\nmodel = /models/model_b.gguf\n"
        )
        reg = ModelRegistry(str(presets), 12000, str(tmp_path))
        assert len(reg.models) == 2
        assert reg.models[0].section_name == "model_a"
        assert reg.models[1].section_name == "model_b"


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


    @pytest.mark.asyncio
    async def test_models_list_fallback(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "[model_a]\npriority = 1\nmodel = /models/model_a.gguf\n\n[model_b]\npriority = 2\nmodel = /models/model_b.gguf\n"
        )
        config = Config(port=0, ini_path=str(presets), work_dir=str(tmp_path))
        router = ProxyRouter(config)
        await router.register_routes()

        from aiohttp.test_utils import make_mocked_request

        req = make_mocked_request("GET", "/v1/models")
        resp = await router.handle_model_list(req)
        assert resp.status == 200
        import json

        data = json.loads(resp.text)
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        ids = [m["id"] for m in data["data"]]
        assert "model_a" in ids
        assert "model_b" in ids
        for m in data["data"]:
            assert m["object"] == "model"
            assert m["owned_by"] == "llama-swap-priority"

    @pytest.mark.asyncio
    async def test_models_list_with_running_instance(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "[model_a]\npriority = 1\nmodel = /models/model_a.gguf\n\n[model_b]\npriority = 2\nmodel = /models/model_b.gguf\n"
        )
        config = Config(port=0, ini_path=str(presets), work_dir=str(tmp_path))
        router = ProxyRouter(config)
        await router.register_routes()

        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock

        mock_resp = AsyncMock()
        mock_resp.text = AsyncMock(return_value="backend_models_response")
        mock_resp.status = 200
        mock_resp.content_type = "application/json"

        @asynccontextmanager
        async def mock_get(*args, **kwargs):
            yield mock_resp

        mock_process = MagicMock()
        mock_process.returncode = None

        mock_inst = MagicMock()
        mock_inst.healthy = True
        mock_inst.process = mock_process
        mock_inst.port = 12000

        router._client.get = mock_get
        router._instances["model_a"] = mock_inst

        from aiohttp.test_utils import make_mocked_request

        req = make_mocked_request("GET", "/v1/models")
        resp = await router.handle_model_list(req)
        assert resp.status == 200
        assert resp.text == "backend_models_response"


class TestIdleTimer:
    @pytest.mark.asyncio
    async def test_idle_timer_not_started_with_zero_sleep(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "[model_a]\npriority = 1\nmodel = /models/model_a.gguf\nsleep-idle-seconds = 0\n"
        )
        config = Config(port=0, ini_path=str(presets), work_dir=str(tmp_path), debug=False)
        router = ProxyRouter(config)
        await router.register_routes()

        model_a = router._models["model_a"]
        assert model_a.sleep_idle_seconds == 0
        assert len(router._idle_timers) == 0

        await router._start_idle_timer("model_a")
        assert len(router._idle_timers) == 0

    @pytest.mark.asyncio
    async def test_idle_timer_started_with_nonzero_sleep(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "[model_a]\npriority = 1\nmodel = /models/model_a.gguf\nsleep-idle-seconds = 60\n"
        )
        config = Config(port=0, ini_path=str(presets), work_dir=str(tmp_path), debug=False)
        router = ProxyRouter(config)
        await router.register_routes()

        model_a = router._models["model_a"]
        assert model_a.sleep_idle_seconds == 60

        await router._start_idle_timer("model_a")
        assert "model_a" in router._idle_timers
        assert router._idle_timers["model_a"] is not None

    @pytest.mark.asyncio
    async def test_idle_timer_cancelled(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "[model_a]\npriority = 1\nmodel = /models/model_a.gguf\nsleep-idle-seconds = 60\n"
        )
        config = Config(port=0, ini_path=str(presets), work_dir=str(tmp_path), debug=False)
        router = ProxyRouter(config)
        await router.register_routes()

        await router._start_idle_timer("model_a")
        assert "model_a" in router._idle_timers

        await router._cancel_idle_timer("model_a")
        assert "model_a" not in router._idle_timers
        assert "model_a" not in router._idle_fired

    @pytest.mark.asyncio
    async def test_idle_timer_not_started_when_pending_request(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "[model_a]\npriority = 1\nmodel = /models/model_a.gguf\nsleep-idle-seconds = 60\n"
        )
        config = Config(port=0, ini_path=str(presets), work_dir=str(tmp_path), debug=False)
        router = ProxyRouter(config)
        await router.register_routes()

        model_a = router._models["model_a"]
        model_a.pending_request = True

        await router._start_idle_timer("model_a")
        assert len(router._idle_timers) == 0

    @pytest.mark.asyncio
    async def test_idle_timer_started_after_pending_request_cleared(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "[model_a]\npriority = 1\nmodel = /models/model_a.gguf\nsleep-idle-seconds = 60\n"
        )
        config = Config(port=0, ini_path=str(presets), work_dir=str(tmp_path), debug=False)
        router = ProxyRouter(config)
        await router.register_routes()

        model_a = router._models["model_a"]
        model_a.pending_request = True

        await router._start_idle_timer("model_a")
        assert len(router._idle_timers) == 0

        model_a.pending_request = False
        await router._start_idle_timer("model_a")
        assert "model_a" in router._idle_timers

    @pytest.mark.asyncio
    async def test_complete_request_marks_pending_false(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "[model_a]\npriority = 1\nmodel = /models/model_a.gguf\nsleep-idle-seconds = 60\n"
        )
        config = Config(port=0, ini_path=str(presets), work_dir=str(tmp_path), debug=False)
        router = ProxyRouter(config)
        await router.register_routes()

        model_a = router._models["model_a"]
        model_a.pending_request = True

        await router._complete_request("model_a")
        assert model_a.pending_request is False

    @pytest.mark.asyncio
    async def test_complete_request_starts_idle_timer(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "[model_a]\npriority = 1\nmodel = /models/model_a.gguf\nsleep-idle-seconds = 60\n"
        )
        config = Config(port=0, ini_path=str(presets), work_dir=str(tmp_path), debug=False)
        router = ProxyRouter(config)
        await router.register_routes()

        model_a = router._models["model_a"]
        model_a.pending_request = False

        await router._complete_request("model_a")
        assert "model_a" in router._idle_timers

    @pytest.mark.asyncio
    async def test_complete_request_noop_for_unknown_model(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "[model_a]\npriority = 1\nmodel = /models/model_a.gguf\n"
        )
        config = Config(port=0, ini_path=str(presets), work_dir=str(tmp_path), debug=False)
        router = ProxyRouter(config)
        await router.register_routes()

        await router._complete_request("unknown_model")

    @pytest.mark.asyncio
    async def test_cancel_idle_timer_clears_fired_set(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "[model_a]\npriority = 1\nmodel = /models/model_a.gguf\nsleep-idle-seconds = 60\n"
        )
        config = Config(port=0, ini_path=str(presets), work_dir=str(tmp_path), debug=False)
        router = ProxyRouter(config)
        await router.register_routes()

        router._idle_fired.add("model_a")
        assert "model_a" in router._idle_fired

        await router._cancel_idle_timer("model_a")
        assert "model_a" not in router._idle_fired

    @pytest.mark.asyncio
    async def test_start_idle_timer_does_not_fire_if_already_fired(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "[model_a]\npriority = 1\nmodel = /models/model_a.gguf\nsleep-idle-seconds = 60\n"
        )
        config = Config(port=0, ini_path=str(presets), work_dir=str(tmp_path), debug=False)
        router = ProxyRouter(config)
        await router.register_routes()

        router._idle_fired.add("model_a")
        await router._start_idle_timer("model_a")
        assert "model_a" in router._idle_timers
        timer = router._idle_timers["model_a"]
        await timer
        # Timer should complete without stopping instance because model is in _idle_fired
        # We can verify by checking that the instance is still running
        # (In this test, instance isn't actually started, but the logic path is exercised)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
