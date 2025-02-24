"""Define the configurable parameters for the agent."""

from dataclasses import dataclass, fields
from typing import Optional

from langchain_core.runnables import RunnableConfig


@dataclass(kw_only=True)
class IndexConfiguration:
    """The configuration for the agent."""
    retriever_provider: str = "pinecone"
    embedding_model: str = "openai/text-embedding-3-small"
    response_model: str = "anthropic/claude-3-5-sonnet-20240620"
    query_model: str = "anthropic/claude-3-haiku-20240307"
    user_id: str = "manoj"

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ):
        """Create a Configuration instance from a RunnableConfig object."""
        configurable = (config.get("configurable") or {}) if config else {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})