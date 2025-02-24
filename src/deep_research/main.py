if __name__ == "__main__":
    from deep_research.graph import graph
    import asyncio
    r = asyncio.run(graph.ainvoke({"topic": "The impact of AI on society"}))
    print(r)