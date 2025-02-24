from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

tools = [
]

graph = create_react_agent(llm, tools=tools)