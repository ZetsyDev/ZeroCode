from typing import Annotated

from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from utils import GAIA_NORMALIZATION_PROMPT

llm = ChatAnthropic(model="claude-3-5-sonnet-latest")


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder = StateGraph(State)

def system_prompt(state: State):
    return {"messages": [{"role": "system", "content": GAIA_NORMALIZATION_PROMPT}]}

graph_builder.add_node("system", system_prompt)
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "system")
graph_builder.add_edge("system", "chatbot") 
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()
graph.name = "Simple Chatbot"


