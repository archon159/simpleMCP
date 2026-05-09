"""
Google search MCP server (via SerpAPI's hosted MCP endpoint).

HTTP-to-HTTP proxy: this server is exposed to our agent over HTTP, and
internally forwards every tool call to the official SerpAPI MCP server at
`https://mcp.serpapi.com/{key}/mcp` (Streamable HTTP transport).

Unlike perplexity_search (which spawns a stdio subprocess via npx), the
upstream here is already an HTTP MCP endpoint, so we just open another
HTTP MCP session and relay calls.

Requires:
  - SERPAPI_API_KEY in secrets.env
"""

import os
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
import uvicorn

from dotenv import load_dotenv

load_dotenv("./secrets.env")

server_name = "google-search"


class _UpstreamHolder:
    session: Optional[ClientSession] = None
    exit_stack: Optional[AsyncExitStack] = None


_upstream = _UpstreamHolder()


@asynccontextmanager
async def lifespan(app):
    api_key = os.environ.get("SERPAPI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY must be set (e.g. in secrets.env).")

    upstream_url = f"https://mcp.serpapi.com/{api_key}/mcp"

    _upstream.exit_stack = AsyncExitStack()
    read, write, _ = await _upstream.exit_stack.enter_async_context(
        streamable_http_client(upstream_url)
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
async def google_search(query: str, count: int = 5) -> str:
    """
    Search the web using Google via SerpAPI's hosted MCP server.

    Args:
        query: Search query string.
        count: Maximum number of organic results to return.

    Returns:
        Concatenated text content from the upstream SerpAPI MCP server
        (typically a JSON blob with organic_results, etc.).
    """
    if not _upstream.session:
        raise RuntimeError("Upstream SerpAPI MCP session is not initialized.")

    result = await _upstream.session.call_tool(
        "search",
        {
            "params": {"q": query, "engine": "google", "num": count},
            "mode": "compact",
        },
    )

    texts = []
    for item in result.content:
        if hasattr(item, "text") and item.text:
            texts.append(item.text)
    return "\n".join(texts) if texts else ""


def run_server(host: str = "127.0.0.1", port: int = 8004):
    print(f"[{server_name}] starting on {host}:{port}")
    app = mcp.streamable_http_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
