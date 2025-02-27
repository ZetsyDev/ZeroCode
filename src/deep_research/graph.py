from typing import Annotated, Literal, TypedDict, cast

from langchain_core.runnables import RunnableConfig

from langgraph.graph import START, END, StateGraph

from browser_use import Agent
from langchain_anthropic import ChatAnthropic
from zerocode.utils import GAIA_NORMALIZATION_PROMPT, get_message_text
from langgraph.graph.message import add_messages
from pydantic import BaseModel
llm = ChatAnthropic(model="claude-3-7-sonnet-latest")

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    query: str
    context: str

class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str


def generate_query(
    state: State, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate a search query based on the current state and configuration.

    This function analyzes the messages in the state and generates an appropriate
    search query. For the first message, it uses the user's input directly.
    For subsequent messages, it uses a language model to generate a refined query.

    Args:
        state (State): The current state containing messages and other information.
        config (RunnableConfig | None, optional): Configuration for the query generation process.

    Returns:
        dict[str, list[str]]: A dictionary with a 'queries' key containing a list of generated queries.

    Behavior:
        - If there's only one message (first user input), it uses that as the query.
        - For subsequent messages, it uses a language model to generate a refined query.
        - The function uses the configuration to set up the prompt and model for query generation.
    """
    messages = state["messages"]
    human_input = get_message_text(messages[-1])
    prompt = f"""For the following query, generate a search query that will be used to search the web.
    Query: {human_input}
    """
    llm_response = llm.with_structured_output(
            SearchQuery
        ).invoke(prompt)
    print("Query: ", llm_response.query)
    return {"query": llm_response.query}

async def search_web_browser(state: State, config: RunnableConfig):
    agent = Agent(
        task=state["query"],
        llm=llm,
    )
    history = await agent.run()
    result = history.final_result()
    return {"context": result}

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder = StateGraph(State)

def system_prompt(state: State):
    return {"messages": [{"role": "system", "content": GAIA_NORMALIZATION_PROMPT}]}

graph_builder.add_node("system", system_prompt)
graph_builder.add_node("generate_query", generate_query)
graph_builder.add_node("search_web_browser", search_web_browser)
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "generate_query")
graph_builder.add_edge("generate_query", "search_web_browser")
graph_builder.add_edge("search_web_browser", "system")
graph_builder.add_edge("system", "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()
graph.name = "Deep Research"



async def amain():
    result = await graph.ainvoke({"messages": [{"role": "user", "content": "What is the capital of France?"}]})
    return result

if __name__ == "__main__":
    import asyncio
    final_state = asyncio.run(amain())
    print(final_state)