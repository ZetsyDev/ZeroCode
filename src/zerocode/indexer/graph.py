"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""

from typing import Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from zerocode.indexer.configuration import IndexConfiguration
from zerocode.indexer.state import IndexState
from zerocode.retriever.retriever import make_retriever




def index_docs(
    state: IndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
    """Asynchronously index documents in the given state using the configured retriever.

    This function takes the documents from the state, ensures they have a user ID,
    adds them to the retriever's index, and then signals for the documents to be
    deleted from the state.

    Args:
        state (IndexState): The current state containing documents and retriever.
        config (Optional[RunnableConfig]): Configuration for the indexing process.r
    """
    if not config:
        raise ValueError("Configuration required to run index_docs.")
    with make_retriever(config) as retriever:
        retriever.add_documents(state.docs)
    return {"docs": "delete"}


# Define a new graph


builder = StateGraph(IndexState, config_schema=IndexConfiguration)
builder.add_node(index_docs)
builder.add_edge(START, "index_docs")
builder.add_edge("index_docs", END)
# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = builder.compile()
graph.name = "IndexGraph"
