from mcp.server.fastmcp import FastMCP

server_name = "custom-server"
mcp = FastMCP(server_name, json_response=True)

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two given integers"""
    return a + b


@mcp.tool()
def get_weather() -> str:
    """Return current weather."""
    weather = "Sunny"
    return weather

if __name__=="__main__":
    print("MCP Server Running...")
    mcp.run(transport="streamable-http")