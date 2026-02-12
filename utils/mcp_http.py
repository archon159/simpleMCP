import asyncio
from typing import Dict, Any, List, Tuple

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class MultiMcp:
    """
    Manage multiple mcp servers:
      - Aggregate list_tools
      - Route call_tool to correct server.
    """
    def __init__(self, url_map: Dict[str, str], enabled: List[str]):
        self.url_map = url_map
        self.enabled = enabled # Enabled Servers
        self._contexts: List[Any] = []
        self.sessions: Dict[str, ClientSession] = {}

    async def __aenter__(self):
        """
        Open the http stream with mcp servers.
        """
        for name in self.enabled:
            url = self.url_map[name]

            transport_ctx = streamable_http_client(url)
            read_stream, write_stream, _ = await transport_ctx.__aenter__()
            self._contexts.append(transport_ctx)

            session_ctx = ClientSession(read_stream, write_stream)
            session = await session_ctx.__aenter__()
            self._contexts.append(session_ctx)

            await session.initialize()
            self.sessions[name] = session

        return self

    async def __aexit__(self, exc_type, exc, tb):
        """
        Close the http streams.
        """
        for ctx in reversed(self._contexts):
            try:
                await ctx.__aexit__(exc_type, exc, tb)
            except Exception:
                pass

    async def list_tools(self, server: str):
        return await self.sessions[server].list_tools()

    async def call_tool(self, server: str, tool_name: str, args: dict):
        return await self.sessions[server].call_tool(tool_name, args)