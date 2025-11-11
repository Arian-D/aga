import asyncio
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.react import ReActAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.mcp import MCPTool, MCPClient

from mcp import StdioServerParameters, stdio_client

from pprint import pprint
from typing import List, Dict, Tuple

from functools import cache

@cache
def docker_command(oci: str) -> Tuple[str, List[str]]:
    "Get the docker command. Once."
    from shutil import which
    command = which("docker") or which("podman") or which("nerdctl")
    if not command:
        raise Exception("Containerless 🥀")
    args = [
        "run",
        "--interactive",
        "--rm",
        oci
    ]
    return (command, args)

def mcp_container(oci: str, args: List[str], env: Dict[str, str]):
    """
    Create a client of an oci mcp running on stdio
    """
    docker, docker_args = docker_command(oci)
    stdio_server = StdioServerParameters(
        command=docker,
        args=docker_args + args,
        env=env,
    )
    return stdio_client(stdio_server)

def log_event(data, event):
    pprint(event)

async def main() -> None:
    # TODO: Make these user-configurable
    tools = [
        DuckDuckGoSearchTool(),
        WikipediaTool(),
    ]
    mcps = [
        mcp_container("mcr.microsoft.com/playwright/mcp", [], {}),
        # mcp_container("docker.io/mcp/wikipedia-mcp", [], {}),
    ]
    for mcp in mcps:
        tools += await MCPTool.from_client(mcp)

    # TODO: Add a match check for ollama's status/health

    agent = ReActAgent(
        llm=ChatModel.from_name("ollama:granite3.3"),
        # role="assistant",
        # instructions="Be direct, to the point, and concise in all your interactions. Be skeptical and double check your work.",
        tools=tools,
        memory=UnconstrainedMemory(),
    )

    # TODO: Read this from stdin
    response = await agent.run(input("Prompt: ")).on("*", log_event)
    print(response.last_message.text)

if __name__ == "__main__":
    asyncio.run(main())
