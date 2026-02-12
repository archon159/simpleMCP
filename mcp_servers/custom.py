from mcp.server.fastmcp import FastMCP
import uvicorn

server_name = "custom-server"
mcp = FastMCP(server_name, json_response=True, stateless_http=True)

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two given integers"""
    return a + b

@mcp.tool()
def get_weather() -> str:
    """Return current weather."""
    return "Sunny"

def run_server(host: str = "127.0.0.1", port: int = 8001):
    print(f"[{server_name}] starting on {host}:{port}")

    app = mcp.streamable_http_app()
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server()