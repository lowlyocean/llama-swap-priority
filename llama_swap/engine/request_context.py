"""Request context — one model assignment per request."""

from dataclasses import dataclass
from llama_swap.config import ModelConfig


@dataclass
class RequestContext:
    """State for a single inbound request."""

    model: str
    config: ModelConfig
    port: int
    active_connections: int = 0
