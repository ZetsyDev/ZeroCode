import os
from langchain.tools import tool

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