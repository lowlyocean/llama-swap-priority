"""Tests for llama-swap-priority."""

import asyncio
import json
import pytest
import time
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

        # _models_response is None — should raise
        from aiohttp.test_utils import make_mocked_request

        req = make_mocked_request("GET", "/v1/models")
        with pytest.raises(AssertionError, match="bootstrap did not complete"):
            await router.handle_model_list(req)


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
        model_a.pending_requests = 1

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
        model_a.pending_requests = 1

        await router._start_idle_timer("model_a")
        assert len(router._idle_timers) == 0

        model_a.pending_requests = 0
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
        model_a.pending_requests = 1

        await router._complete_request("model_a")
        assert model_a.pending_requests == 0

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
        model_a.pending_requests = 0

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


class TestPreemptionCooldown:
    @pytest.mark.asyncio
    async def test_preempted_instance_returns_429_within_cooldown(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "[model_a]\npriority = 1\nmodel = /models/model_a.gguf\n\n[model_b]\npriority = 2\nmodel = /models/model_b.gguf\n"
        )
        config = Config(port=0, ini_path=str(presets), work_dir=str(tmp_path), debug=False)
        router = ProxyRouter(config)
        await router.register_routes()

        inst = router._instances["model_a"]
        inst.running = True
        inst.healthy = True
        inst.preempted_at = time.time()

        from unittest.mock import MagicMock

        req = MagicMock()
        req.match_info = {'model': 'model_a'}
        req.headers = {}
        req.method = 'POST'
        req.path = '/chat/completions/model_a'
        async def mock_read():
            return b'{"model": "model_a"}'

        req.read = mock_read
        req.query_string = ''
        req.query = {}
        req.version = (1, 1)
        req.content_type = None
        req.content = None
        req.transport = None
        req.app = router._app
        req._cached = None

        resp = await router.forward_request(req)
        assert resp.status == 429
        data = json.loads(resp.text)
        assert data["error"] == "server busy"


class TestDefaultRunningModel:
    @pytest.mark.asyncio
    async def test_default_running_model_parsed_from_ini(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "[high]\npriority = 10\nmodel = /models/high.gguf\n\n"
            "[low]\npriority = 1\nmodel = /models/low.gguf\n\n"
            "[*]\ndefault-running-model = low\n"
        )
        config = Config(port=0, ini_path=str(presets), work_dir=str(tmp_path))
        router = ProxyRouter(config)
        assert router._default_running_model == "low"

    @pytest.mark.asyncio
    async def test_default_running_model_not_parsed_without_priority(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "[high]\npriority = 10\nmodel = /models/high.gguf\n\n"
            "[low]\nmodel = /models/low.gguf\n\n"
            "[*]\ndefault-running-model = low\n"
        )
        config = Config(port=0, ini_path=str(presets), work_dir=str(tmp_path))
        router = ProxyRouter(config)
        # low is not in registry because it has no priority
        assert router._default_running_model is None

    @pytest.mark.asyncio
    async def test_default_running_model_idle_timer_skipped(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "[high]\npriority = 10\nmodel = /models/high.gguf\n\n"
            "[low]\npriority = 1\nmodel = /models/low.gguf\nsleep-idle-seconds = 60\n\n"
            "[*]\ndefault-running-model = low\n"
        )
        config = Config(port=0, ini_path=str(presets), work_dir=str(tmp_path), debug=False)
        router = ProxyRouter(config)
        await router.register_routes()

        low = router._models["low"]
        low_inst = router._instances["low"]
        low_inst.default_running = True
        low_inst.running = True

        await router._start_idle_timer("low")
        assert "low" not in router._idle_timers

    @pytest.mark.asyncio
    async def test_default_running_model_not_started_when_normal(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "[high]\npriority = 10\nmodel = /models/high.gguf\n\n"
            "[low]\npriority = 1\nmodel = /models/low.gguf\nsleep-idle-seconds = 60\n\n"
            "[*]\ndefault-running-model = low\n"
        )
        config = Config(port=0, ini_path=str(presets), work_dir=str(tmp_path), debug=False)
        router = ProxyRouter(config)
        await router.register_routes()

        low = router._models["low"]
        low_inst = router._instances["low"]
        low_inst.default_running = False
        low_inst.running = True

        await router._start_idle_timer("low")
        assert "low" in router._idle_timers

    @pytest.mark.asyncio
    async def test_default_running_model_stopped_on_higher_priority_request(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "[high]\npriority = 10\nmodel = /models/high.gguf\n\n"
            "[low]\npriority = 1\nmodel = /models/low.gguf\nsleep-idle-seconds = 60\n\n"
            "[*]\ndefault-running-model = low\n"
        )
        config = Config(port=0, ini_path=str(presets), work_dir=str(tmp_path), debug=False)
        router = ProxyRouter(config)
        await router.register_routes()

        high = router._models["high"]
        low = router._models["low"]
        high_inst = router._instances["high"]
        low_inst = router._instances["low"]
        high_inst.running = True
        low_inst.running = True
        low_inst.default_running = True

        from unittest.mock import MagicMock, AsyncMock
        from aiohttp import ClientSession
        from aiohttp.web_response import Response

        router._client = MagicMock(spec=ClientSession)
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.content_type = "application/json"
        mock_resp.headers = {}
        mock_resp.json = AsyncMock(return_value={})
        router._client.post = AsyncMock(return_value=mock_resp)

        req = MagicMock()
        req.match_info = {"model": "high"}
        req.headers = {}
        req.method = "POST"
        req.path = "/chat/completions/high"
        async def mock_read():
            return b'{"model": "high"}'
        req.read = mock_read
        req.query_string = ""
        req.query = {}
        req.version = (1, 1)
        req.content_type = None
        req.content = None
        req.transport = None
        req.app = router._app
        req._cached = None

        resp = await router.forward_request(req)
        assert resp.status == 200
        assert not low_inst.running
        assert not low_inst.default_running

    @pytest.mark.asyncio
    async def test_default_running_model_kept_when_same_priority(self, tmp_path):
        presets = tmp_path / "presets.ini"
        presets.write_text(
            "[high]\npriority = 10\nmodel = /models/high.gguf\n\n"
            "[low]\npriority = 1\nmodel = /models/low.gguf\nsleep-idle-seconds = 60\n\n"
            "[*]\ndefault-running-model = low\n"
        )
        config = Config(port=0, ini_path=str(presets), work_dir=str(tmp_path), debug=False)
        router = ProxyRouter(config)
        await router.register_routes()

        high = router._models["high"]
        low = router._models["low"]
        high_inst = router._instances["high"]
        low_inst = router._instances["low"]
        high_inst.running = True
        low_inst.running = True
        low_inst.loading = True
        low_inst.default_running = True

        from unittest.mock import MagicMock, AsyncMock
        from aiohttp import ClientSession

        router._client = MagicMock(spec=ClientSession)
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.content_type = "application/json"
        mock_resp.headers = {}
        mock_resp.json = AsyncMock(return_value={})
        router._client.post = AsyncMock(return_value=mock_resp)

        req = MagicMock()
        req.match_info = {"model": "low"}
        req.headers = {}
        req.method = "POST"
        req.path = "/chat/completions/low"
        async def mock_read():
            return b'{"model": "low"}'
        req.read = mock_read
        req.query_string = ""
        req.query = {}
        req.version = (1, 1)
        req.content_type = None
        req.content = None
        req.transport = None
        req.app = router._app
        req._cached = None

        resp = await router.forward_request(req)
        assert resp.status == 200
        assert low_inst.running
        assert low_inst.default_running
