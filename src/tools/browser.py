from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from browser_use import Agent, Browser, BrowserConfig

#TODO: Convert this to class to share the browser instance across multiple calls

@tool
async def search_web(query: str):
    """Search the web for the given query.
    
    Uses a browser agent to perform web searches and extract relevant information.
    The agent will:
    - Execute web searches based on the query
    - Visit relevant pages
    - Extract and summarize key information
    - Handle pagination and multiple results as needed
    
    Args:
        query (str): The search query to execute
        
    Returns:
        dict: Contains "context" key with the aggregated search results and 
        extracted information from relevant web pages
        
    Example:
        result = await search_web("latest news about AI")
        context = result["context"]  # Contains summarized web search findings
    """
    llm_search = ChatAnthropic(model="claude-3-5-sonnet-latest")
    config = BrowserConfig(
        headless=True,
        disable_security=True
    )
    browser = Browser(config=config)
    agent = Agent(
        browser=browser,
        task=query,
        llm=llm_search,
        use_vision=True,
        save_conversation_path="logs/conversation"
    )
    history = await agent.run(max_steps=20)
    result = history.final_result()
    await browser.close()
    return {"context": result}