
"""Define the configurable parameters for the agent."""
from dataclasses import dataclass, fields
from typing import Optional

from langchain_core.runnables import RunnableConfig


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    thread_id: str = "thread-1"
    model: str = "anthropic/claude-3-7-sonnet-latest"
    prompt: str = "You are a helpful AI assistant, please respond to the users query to the best of your ability!"
    workspace: str = "workspace"

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ):
        """Create a Configuration instance from a RunnableConfig object."""
        configurable = (config.get("configurable") or {}) if config else {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})