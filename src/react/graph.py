from typing import Literal
from langchain_anthropic import ChatAnthropic

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from markitdown import MarkItDown
from openai import OpenAI

from zerocode.utils import GAIA_NORMALIZATION_PROMPT

client = OpenAI()

md = MarkItDown(llm_client=client, llm_model="gpt-4o")


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
    And more file types

    Args:
        file_path: Path to the input document file

    Returns:
        str: The document contents converted to markdown text
    """
    result = md.convert(file_path)
    return result.text_content

tools = [read_as_markdown]

llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
graph = create_react_agent(llm, tools=tools, prompt=GAIA_NORMALIZATION_PROMPT)
graph.name = "reAct agent"

