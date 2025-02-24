from typing import Literal
from langchain_anthropic import ChatAnthropic

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
graph = create_react_agent(llm, tools=tools)
graph.name = "reAct agent"

