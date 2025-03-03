from openai import OpenAI
from markitdown import MarkItDown
from langchain.tools import tool

client = OpenAI()

md = MarkItDown(llm_client=client, llm_model="gpt-4o")


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

    Args:
        file_path: Path to the input document file

    Returns:
        str: The document contents converted to markdown text
    """
    try:
        result = md.convert(file_path)
        if len(result.text_content) > 10000:
            return "File is too large to convert to markdown. Please use the execute_code tool to read the file."
        return result.text_content
    except:
        return "Error converting file to markdown"