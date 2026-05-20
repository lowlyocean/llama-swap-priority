"""Proxy — aiohttp server that routes requests and manages instances."""

import asyncio
import json
import signal

from aiohttp import ClientSession, web

from llama_swap.instance.manager import (
    InstanceState,
    start_instance,
    stop_instance,
    _instances,
)
from llama_swap.preset.ini_parser import read_preset
from llama_swap.config import Config, ModelConfig, ModelRegistry


class ProxyRouter:
    """Routes requests to backend instances."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.registry = ModelRegistry(
            config.ini_path, config.start_port, config.work_dir
        )
        self._models: dict[str, ModelConfig] = {}
        self._instances: dict[str, InstanceState] = {}
        self._app: web.Application | None = None
        self._client: ClientSession | None = None
        self._runner: web.AppRunner | None = None
        self._health_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None

        # Register models from registry
        port = config.start_port
        for model_cfg in self.registry.models:
            model_cfg.port = port
            model_cfg.ini_dir = config.work_dir
            model_cfg.options = read_preset(config.work_dir, model_cfg.section_name)
            self._models[model_cfg.section_name] = model_cfg
            self._instances[model_cfg.section_name] = InstanceState(
                config=model_cfg,
                port=port,
                healthy=False,
                current_priority=model_cfg.priority,
            )
            port += 1

    def _get_highest_instance_priority(self) -> int:
        """Return the highest current_priority among running instances.

        Uses self._models to get priority since docker containers don't set
        inst.config (it's None). Falls back to inst.current_priority if
        model not in _models.
        """
        max_priority = 0
        for name, inst in self._instances.items():
            if not inst.running:
                continue
            # Get priority from model config (works for docker instances)
            model_cfg = self._models.get(name)
            if model_cfg:
                pri = model_cfg.priority
            else:
                pri = inst.current_priority
            if pri > max_priority:
                max_priority = pri
        return max_priority

    def _find_instance_to_terminate(self, new_priority: int) -> str | None:
        """Find a running instance with priority lower than the new request's priority."""
        for name, inst in self._instances.items():
            if not inst.running:
                continue
            model_cfg = self._models.get(name)
            if model_cfg:
                pri = model_cfg.priority
            else:
                pri = inst.current_priority
            if pri < new_priority:
                return name
        return None

    async def _check_instance_status(self, port: int) -> bool:
        """Check if an instance is running and not sleeping.

        Returns True if the instance is healthy and actively serving.
        Checks /props endpoint for the top-level 'is_sleeping' boolean field.
        """
        try:
            async with self._client.get(
                f"http://127.0.0.1:{port}/props",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    return False
                data = await resp.json()
                is_sleeping = data.get("is_sleeping", False)
                return not is_sleeping
        except Exception:
            return False

    async def _health_check(self) -> None:
        """Periodically check all running instances and update their state.

        Uses /props endpoint to get the top-level 'is_sleeping' boolean field.
        Sets inst.running = True when the instance is up and serving.
        Sets inst.healthy = False when the instance is sleeping.
        """
        try:
            while True:
                for name, inst in list(self._instances.items()):
                    if inst.process and inst.process.returncode is not None:
                        inst.healthy = False
                    elif inst.process:
                        inst.healthy = await self._check_instance_status(inst.port)
                    else:
                        try:
                            healthy = await self._check_instance_status(inst.port)
                            inst.healthy = healthy
                            if healthy and not inst.running:
                                inst.running = True
                        except Exception:
                            inst.healthy = False
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass

    async def forward_request(self, request: web.Request) -> web.Response:
        path_model = request.match_info.get("model", None)
        body = await request.read()
        body_str = body.decode("utf-8") if body else ""

        model = path_model
        if path_model is None and body_str:
            try:
                parsed = json.loads(body_str)
                model = parsed.get("model")
            except (json.JSONDecodeError, ValueError):
                model = None

        if self.config.debug:
            print(f"[DEBUG] Incoming request: path={request.path}, model={model}")

        if model not in self._models:
            print(f"[DEBUG] Unknown model: {model}")
            return web.json_response(
                {"error": f"unknown model: {model}"}, status=404
            )

        model_config = self._models[model]
        new_priority = model_config.priority

        print(f"[DEBUG] Routing: model={model}, priority={new_priority}")

        # Preemption logic
        highest = self._get_highest_instance_priority()

        print(f"[DEBUG] Highest running priority: {highest}")

        if new_priority > highest:
            print(f"[DEBUG] Preemption: new priority {new_priority} > highest {highest}")
            to_terminate = self._find_instance_to_terminate(new_priority)
            while to_terminate:
                print(f"[DEBUG] Terminating instance: {to_terminate}")
                await stop_instance(to_terminate, debug=self.config.debug)
                inst = self._instances.get(to_terminate)
                if inst:
                    inst.running = False
                    inst.healthy = False
                highest = self._get_highest_instance_priority()
                if new_priority > highest:
                    to_terminate = self._find_instance_to_terminate(new_priority)
                else:
                    break
        elif new_priority <= highest:
            running_inst = self._instances.get(model)
            if not (running_inst and running_inst.running):
                print(f"[DEBUG] No free instance, returning 429")
                return web.json_response(
                    {
                        "error": "server busy",
                        "error_code": "try_again_later",
                        "retry_after": 5,
                    },
                    status=429,
                )

        inst = self._instances.get(model)
        if not inst or not inst.running:
            print(f"[DEBUG] Starting instance for model={model}")
            await start_instance(model_config, debug=self.config.debug)
            inst = self._instances.get(model)
            if inst:
                inst.running = True
                inst.healthy = True

        inst = self._instances.get(model)
        if inst:
            backend_host = "http://127.0.0.1:{port}".format(port=inst.port)
        else:
            backend_host = "http://127.0.0.1:{port}".format(port=model_config.port)

        print(f"[DEBUG] Forwarding to backend: {backend_host}{request.path}")

        path = request.path
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "connection", "content-length")
        }

        assert self._client is not None
        resp = await self._client.post(
            f"{backend_host}{path}",
            headers=headers,
            data=body,
        )

        print(f"[DEBUG] Backend response status: {resp.status}")

        # Stream SSE or return JSON depending on content type
        content_type = resp.content_type or ""
        if "text/event-stream" in content_type:
            resp_headers = {
                k: v
                for k, v in resp.headers.items()
                if k.lower() not in ("transfer-encoding", "connection")
            }
            stream_response = web.StreamResponse(
                status=resp.status, headers=resp_headers
            )
            await stream_response.prepare(request)
            try:
                while True:
                    chunk = await resp.content.read(8192)
                    if not chunk:
                        break
                    await stream_response.write(chunk)
                    await stream_response.drain()
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass
            finally:
                await stream_response.write_eof()
            return stream_response
        else:
            try:
                data = await resp.json()
            except Exception:
                data = await resp.read()
                data = {"error": data.decode()} if isinstance(data, bytes) else data

            print(f"[DEBUG] Returning response status: {resp.status}")
            filtered_headers = {
                k: v
                for k, v in resp.headers.items()
                if k.lower() not in ("transfer-encoding", "connection")
            }
            return web.json_response(data, status=resp.status, headers=filtered_headers)

    async def handle_model_list(self, request: web.Request) -> web.Response:
        path = request.path

        for inst in self._instances.values():
            if inst.healthy:
                backend_url = f"http://127.0.0.1:{inst.port}"
                try:
                    assert self._client is not None
                    async with self._client.get(
                        f"{backend_url}{path}",
                        headers={
                            k: v
                            for k, v in request.headers.items()
                            if k.lower() not in ("host", "connection", "content-length")
                        },
                    ) as resp:
                        data = await resp.text()
                        return web.Response(
                            text=data,
                            status=resp.status,
                            content_type=resp.content_type,
                        )
                except Exception:
                    continue

        model_list = []
        for model_cfg in self.registry.models:
            model_list.append(
                {
                    "id": model_cfg.section_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "llama-swap-priority",
                }
            )

        return web.json_response(
            {"object": "list", "data": model_list}
        )

    async def register_routes(self) -> None:
        if self._app is None:
            self._app = web.Application()
        router = self._app.router

        # GET routes
        router.add_route("GET", "/v1/models", self.handle_model_list)
        router.add_route("GET", "/v1/models/{model}", self.handle_model_list)
        router.add_route("GET", "/models", self.handle_model_list)
        router.add_route("GET", "/models/{model}", self.handle_model_list)

        # POST routes — chat completions
        router.add_route("POST", "/v1/chat/completions", self.forward_request)
        router.add_route("POST", "/v1/chat/completions/{model}", self.forward_request)
        router.add_route("POST", "/chat/completions", self.forward_request)
        router.add_route("POST", "/chat/completions/{model}", self.forward_request)

        # POST routes — completions
        router.add_route("POST", "/v1/completions", self.forward_request)
        router.add_route("POST", "/v1/completions/{model}", self.forward_request)

        # POST routes — embeddings
        router.add_route("POST", "/v1/embeddings", self.forward_request)
        router.add_route("POST", "/v1/embeddings/{model}", self.forward_request)
        router.add_route("POST", "/embeddings", self.forward_request)
        router.add_route("POST", "/embeddings/{model}", self.forward_request)

        # GET routes — props
        router.add_route("GET", "/v1/props", self.handle_model_list)
        router.add_route("GET", "/v1/props/{model}", self.handle_model_list)
        router.add_route("GET", "/props", self.handle_model_list)
        router.add_route("GET", "/props/{model}", self.handle_model_list)

        self._client = ClientSession()

    async def run(self) -> None:
        self._app = web.Application()

        self._stop_event = asyncio.Event()

        async def _shutdown_handler(sig: int) -> None:
            print(f"[PROXY] Received signal {sig}, initiating shutdown...")
            self._stop_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.ensure_future(_shutdown_handler(s)))

        # Start health check loop
        self._health_task = asyncio.create_task(self._health_check())

        await self.register_routes()

        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, self.config.host, self.config.port)
        await site.start()
        print(f"Proxy listening on {self.config.host}:{self.config.port}")
        self._runner = runner
        try:
            await self._stop_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            await self.cleanup()

    async def cleanup(self) -> None:
        print("[PROXY] Cleaning up...")
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        if self._runner:
            await self._runner.cleanup()
        if self._client:
            await self._client.close()
        for name in list(self._instances):
            await stop_instance(name, debug=self.config.debug)

    async def stop(self) -> None:
        await self.cleanup()
