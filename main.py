import asyncio
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.mcp import MCPClient


async def main():
    # TODO: Add a check for ollama's status/health
    agent = RequirementAgent(
        llm=ChatModel.from_name("ollama:granite3.3"),
        role="AI assistant",
        instructions="Be direct, to the point, and concise in all your interactions.",
        tools=[DuckDuckGoSearchTool(), WikipediaTool()],
        memory=UnconstrainedMemory(),
    )

    response = await agent.run(input("Prompt: "))
    print(response.last_message.text)

if __name__ == "__main__":
    asyncio.run(main())
