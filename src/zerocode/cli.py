# Cli for ZeroCode local development
import typer

from zerocode.agent.utils import stream_graph_updates

app = typer.Typer()

@app.command()
def chat():
    from zerocode.agent.graph import graph
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input, graph)
        
@app.command()
def index():
    """Index content into the database."""
    from zerocode.indexer.graph import graph
    while True:
        user_input = input("Content to Index: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        graph.invoke({"docs": user_input})

@app.command()
def retrieve():
    from zerocode.retriever.graph import graph
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input, graph)