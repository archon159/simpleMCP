"""
Standalone custom_add MCP server exposed publicly via ngrok.

Designed to run on a personal laptop so OpenAI's Responses API (with the
"mcp" tool type) can reach it directly.

Setup:
    pip install mcp uvicorn pyngrok

    # ngrok requires a (free) authtoken:
    #   https://dashboard.ngrok.com/get-started/your-authtoken
    # Either set the env var NGROK_AUTHTOKEN before running, or once globally:
    #   ngrok config add-authtoken <token>

Run:
    python public_custom_mcp.py

The script prints the public URL to copy into your OpenAI tool config.
Press Ctrl+C to shut everything down.
"""

import os

from mcp.server.fastmcp import FastMCP
import uvicorn
from pyngrok import ngrok, conf


PORT = 8001

mcp = FastMCP("custom-server", json_response=True, stateless_http=True)


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two given integers"""
    return a + b


def main():
    token = os.environ.get("NGROK_AUTHTOKEN")
    if token:
        conf.get_default().auth_token = token

    # host_header="rewrite" makes ngrok rewrite the incoming Host header to
    # match the upstream (127.0.0.1:PORT) so uvicorn/starlette accepts it.
    public_tunnel = ngrok.connect(PORT, "http", host_header="rewrite")
    public_url = public_tunnel.public_url
    mcp_url = f"{public_url}/mcp"

    print()
    print("=" * 70)
    print(f"  Local URL:   http://127.0.0.1:{PORT}/mcp")
    print(f"  Public URL:  {mcp_url}")
    print("=" * 70)
    print()
    print("  Drop this into the OpenAI Responses 'mcp' tool config:")
    print(f'    {{"type": "mcp", "server_label": "custom", "server_url": "{mcp_url}", "require_approval": "never"}}')
    print()
    print("  Press Ctrl+C to stop.\n")

    try:
        uvicorn.run(
            mcp.streamable_http_app(),
            host="127.0.0.1",
            port=PORT,
            log_level="warning",
        )
    except KeyboardInterrupt:
        pass
    finally:
        ngrok.disconnect(public_url)
        ngrok.kill()


if __name__ == "__main__":
    main()
