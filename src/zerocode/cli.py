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
        
