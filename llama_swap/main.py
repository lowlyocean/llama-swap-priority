"""llama-swap-proxy: priority-based router between llama-server instances."""

import argparse
import asyncio
from llama_swap.proxy.router import ProxyRouter
from llama_swap.config import Config


async def main() -> None:
    parser = argparse.ArgumentParser(description="llama-swap-proxy")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    config = Config(debug=args.debug)
    router = ProxyRouter(config)

    port = config.start_port
    for model_cfg in router.registry.models:
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
