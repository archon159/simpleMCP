"""
Perplexity search MCP server.

HTTP-to-stdio bridge: this server is exposed to our agent over HTTP, and
internally forwards every tool call to the official Perplexity MCP server
(`@perplexity-ai/mcp-server`, stdio transport, npx-spawned).

On startup we spawn the upstream server and keep the stdio session alive
for the lifetime of this process via FastMCP's lifespan. Each
`perplexity_search` call we receive is forwarded to the upstream
`perplexity_search` tool with the same arguments.

Requires:
  - Node.js / npx available on PATH
  - PERPLEXITY_API_KEY in secrets.env
"""

import os
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
import uvicorn

from dotenv import load_dotenv

load_dotenv("./secrets.env")

server_name = "perplexity-search"


class _UpstreamHolder:
    session: Optional[ClientSession] = None
    exit_stack: Optional[AsyncExitStack] = None


_upstream = _UpstreamHolder()


@asynccontextmanager
async def lifespan(app):
    api_key = os.environ.get("PERPLEXITY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY must be set (e.g. in secrets.env).")

    _upstream.exit_stack = AsyncExitStack()
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@perplexity-ai/mcp-server"],
        env={**os.environ, "PERPLEXITY_API_KEY": api_key},
    )
    read, write = await _upstream.exit_stack.enter_async_context(
        stdio_client(server_params)
    )
    _upstream.session = await _upstream.exit_stack.enter_async_context(
        ClientSession(read, write)
    )
    await _upstream.session.initialize()

    try:
        yield
    finally:
        if _upstream.exit_stack is not None:
            await _upstream.exit_stack.aclose()
        _upstream.exit_stack = None
        _upstream.session = None


mcp = FastMCP(server_name, lifespan=lifespan, json_response=True)


@mcp.tool()
async def perplexity_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using Perplexity.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        Concatenated text content from the upstream Perplexity MCP server
        (may be JSON or markdown depending on the upstream version).
    """
    if not _upstream.session:
        raise RuntimeError("Upstream Perplexity MCP session is not initialized.")

    result = await _upstream.session.call_tool(
        "perplexity_search",
        {"query": query, "max_results": max_results},
    )

    texts = []
    for item in result.content:
        if hasattr(item, "text") and item.text:
            texts.append(item.text)
    return "\n".join(texts) if texts else ""


def run_server(host: str = "127.0.0.1", port: int = 8003):
    print(f"[{server_name}] starting on {host}:{port}")
    app = mcp.streamable_http_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
