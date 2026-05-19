"""llama-swap-proxy: priority-based router between llama-server instances."""

import asyncio
from llama_swap.proxy.router import ProxyRouter
from llama_swap.config import Config


async def main() -> None:
    config = Config()
    router = ProxyRouter(config)

    for model_cfg in router.registry.models:
        port = config.start_port
        router.register_model(model_cfg.section_name, port)
        port += 1

    try:
        await router.run()
    except KeyboardInterrupt:
        pass
    finally:
        await router.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
