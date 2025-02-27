from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from markitdown import MarkItDown
from openai import OpenAI

from zerocode.utils import GAIA_NORMALIZATION_PROMPT
from browser_use import Agent, Browser
from e2b_code_interpreter import Sandbox
from langchain.chat_models import init_chat_model
import os


client = OpenAI()

md = MarkItDown(llm_client=client, llm_model="gpt-4o")

@lru_cache(maxsize=1)
def get_browser():
    return Browser()

@tool
def read_as_markdown(file_path: str) -> str:
    """Read a document file and return the contents as markdown text.

    Uses MarkItDown to convert various document formats to markdown text, including:
    - PDF documents
    - PowerPoint presentations (.pptx)
    - Word documents (.docx)
    - Excel spreadsheets (.xlsx)
    - Images (.jpg, .png, etc.) - extracts EXIF metadata and performs OCR
    - Audio files - extracts EXIF metadata and transcribes speech
    - HTML files
    - Text formats like CSV, JSON, XML
    - ZIP archives (processes contained files)

    Forr other file types, you can use the execute_code tool to execute code to convert the file to markdown.

    Args:
        file_path: Path to the input document file

    Returns:
        str: The document contents converted to markdown text
    """
    try:
        result = md.convert(file_path)
        return result.text_content
    except:
        return f"Error converting file to markdown"

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
    agent = Agent(
        task=query,
        llm=llm_search,
        use_vision=True,
        browser=get_browser(),
    )
    history = await agent.run(max_steps=20)
    result = history.final_result()
    return {"context": result}

@tool
def execute_code(code : str) -> str:
    """Execute python code in a Jupyter notebook cell and returns any rich data (eg charts), stdout, stderr, and error.

    This function provides a sandboxed environment to safely execute arbitrary code
    and capture its output. It runs the provided code in an isolated context and
    collects stdout, stderr, execution results and any errors that occur.

    The code is executed using a Sandbox interpreter that provides:
    - Isolated execution environment
    - Captured stdout/stderr streams
    - Error handling and reporting
    - Access to execution results
    
    Args:
        code (str): The code block to execute, as a string

    Returns:
        str: A formatted string containing:
            - stdout: Standard output from code execution
            - stderr: Standard error output
            - error: Any error messages or exceptions
            - results: The execution results/return values

    Example:
        result = execute_code("print('hello')\nx = 1 + 2\nprint(x)")
        # Returns:
        # stdout: hello\n3
        # stderr: 
        # error: None
        # results: None
    """
    print(f"***Code Interpreting...\n{code}\n====")

    with Sandbox(api_key=os.getenv("E2B_API_KEY"), timeout=60) as code_interpreter:
        execution = code_interpreter.run_code(code)
    return f"stdout: {execution.logs.stdout}\nstderr: {execution.logs.stderr}\nerror: {execution.error} \nresults: {execution.results}"

tools = [read_as_markdown, execute_code]


llm = init_chat_model(model="claude-3-7-sonnet-latest")

prompt = f""" 
You are given 3 tools to answer the question.
1. read_as_markdown: to read a document file and return the contents as markdown text. Use this tool if the question is about a document to be read from file system.
2. execute_code: to execute python code. Use this tool for complex calcualations or data analysis.

{GAIA_NORMALIZATION_PROMPT}
"""
graph = create_react_agent(llm, tools=tools, prompt=prompt)
graph.name = "reAct agent"


# 3. search_web: to search the web for the given query. Use this tool if you want to lookup web for some information.