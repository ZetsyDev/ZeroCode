from tools.sandbox import execute_code
from zerocode.utils import load_chat_model

from tools.markdown import read_as_markdown
from tools.browser import search_web
from zerocode.configuration import Configuration
from langgraph.prebuilt import create_react_agent

def load_tools(config: Configuration):
    tools = [read_as_markdown, execute_code, search_web]
    return tools

def create_graph(config: Configuration):
    """Create and configure the agent workflow graph."""
    tools = load_tools(config)

    # Initialize LLM with configured model
    llm = load_chat_model(config.model)
    prompt = config.prompt
    graph = create_react_agent(name="react-agent", model=llm, tools=tools, prompt=prompt)
    
    return graph
