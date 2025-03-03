import asyncio
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

load_dotenv()

from zerocode import create_graph, Configuration

agent = create_graph(Configuration())


async def main() -> None:
    inputs = {"messages": [("user", "Find me a recipe for chocolate chip cookies")]}
    result = await agent.ainvoke(
        inputs,
        config=RunnableConfig(configurable={"thread_id": uuid4()}),
    )
    result["messages"][-1].pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
