# simpleMCP

Simple MCP client and server playground. Supports HuggingFace local models and API-based models (OpenAI, Anthropic), three OpenAI API modes (Chat Completions, Responses, Responses + remote MCP URL), and three patterns for adding MCP servers.

## Usage

### 1. Create `secrets.env`

```
touch ./secrets.env
```

Drop in whichever keys you need:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
BRAVE_API_KEY=...
PERPLEXITY_API_KEY=pplx-...
SERPAPI_API_KEY=...
MODEL_DIR=/path/to/hf_models/   # optional, default ../hf_models/
```

### 2. Run the local MCP servers

```bash
python3 run_mcp_servers.py
```

Auto-discovers files in `mcp_servers/` and starts each on its declared port (`PORTS` dict in `run_mcp_servers.py`).

### 3. Run the client

```bash
# HuggingFace local model (default routing)
python3 mcp_client.py -m "Where does the computer scientist Seungeon Lee work?"

# OpenAI API (routes by model name prefix)
python3 mcp_client.py --model gpt-4o -m "..."

# Anthropic API
python3 mcp_client.py --model claude-3-5-sonnet-20241022 -m "..."
```

## Backend routing

The backend is selected automatically by the `--model` value:

| Model name prefix | Backend |
|---|---|
| `gpt-`, `o1-`, `o3-`, `o4-`, `chatgpt-` | OpenAI API |
| `claude-` | Anthropic API |
| anything else | HuggingFace local (loaded from `MODEL_DIR/<model>`) |

## OpenAI API modes (`--openai_api`)

Three switchable modes for OpenAI models — useful for comparing how each handles tool calling end to end. Logs in `cur.log` show the wire-level differences.

| Mode | Who calls the MCP tool | Rounds the client makes | Local MCP servers needed |
|---|---|---|---|
| `chat_completions` (default) | Client (us) | Multi-round loop | Yes |
| `responses` | Client (us) | Multi-round loop | Yes |
| `responses_url` | **OpenAI server** (directly via `mcp` tool type) | **1 call** | **No** |

Examples:

```bash
# Chat Completions (default)
python3 mcp_client.py --model gpt-4o -m "..."

# Responses API with function tools
python3 mcp_client.py --model gpt-4o --openai_api responses -m "..."

# Responses API + server-side MCP (OpenAI talks to the MCP URL directly).
# The URL must be reachable from OpenAI's servers (use a public hosted MCP
# or expose a local one via a tunnel).
python3 mcp_client.py --model gpt-4o \
  --openai_api responses_url \
  --mcp_url "https://mcp.serpapi.com/<KEY>/mcp" \
  --mcp_label google \
  -m "..."
```

For exposing your laptop's MCP server publicly so `responses_url` can hit it, see [`public_custom_mcp.py`](public_custom_mcp.py) — runs the `custom_add` tool locally and tunnels it via ngrok in one command.

## Selecting which MCP servers to enable

Two ways:

```python
# default (top of mcp_client.py)
ENABLED_SERVERS = ["custom"]
```

```bash
# CLI override (comma-separated)
python3 mcp_client.py --enabled custom,perplexity_search -m "..."
```

CLI takes precedence; otherwise the constant is used.

## Adding a new MCP server

Each `mcp_servers/<name>.py` is a standalone HTTP MCP server that defines its tools and exposes a `run_server(host, port)`. Three concrete patterns are already in the repo — copy whichever matches your case:

| Pattern | Use when… | Reference file |
|---|---|---|
| **A. Native HTTP (no upstream MCP)** | You want to wrap a REST API yourself with FastMCP. | [`mcp_servers/brave_search.py`](mcp_servers/brave_search.py) — also [`mcp_servers/custom.py`](mcp_servers/custom.py) for the simplest possible case |
| **B. Bridge to a stdio-based upstream MCP** | The vendor ships an npm/npx MCP server (stdio only). Spawn it as a subprocess in `lifespan`, forward calls. | [`mcp_servers/perplexity_search.py`](mcp_servers/perplexity_search.py) |
| **C. Proxy to a hosted HTTP MCP** | The vendor already provides an HTTP MCP endpoint. Just open another HTTP MCP session in `lifespan` and relay calls. | [`mcp_servers/google_search.py`](mcp_servers/google_search.py) (SerpAPI) |

After adding the new file:
1. Register the port in `run_mcp_servers.py`'s `PORTS` dict.
2. Add an entry to `MCP_URLS` and (optionally) `TOOL_ALLOWLIST` in `mcp_client.py`.
3. Restart `run_mcp_servers.py` (the existing process is a snapshot).

Tool names are exposed to the model in `<server>__<tool>` form (double-underscore separator) so they pass OpenAI's `^[a-zA-Z0-9_-]+$` constraint.

## Other knobs

| Argument | Default | Notes |
|---|---|---|
| `--model` | `Llama-3.1-8B-Instruct` | HF model dir name or API model ID |
| `--device` | `cuda:0` | HuggingFace only |
| `--dtype` | `auto` | HuggingFace only |
| `--temperature` | `0.0` | Sampling temperature (ignored on reasoning models) |
| `--seed` | `0` | Random seed (ignored on reasoning models / Anthropic) |
| `--max_tool_rounds` | `6` | Max LLM rounds in a single query (multi-turn tool calling supported) |
| `--writing_mode` | off | HF only. Logs the model's raw output **with special tokens** (e.g., `<\|python_tag\|>`, `<tool_call>`) so you can see the native tool-calling format. Agent behavior is unchanged. |
| `--enable_thinking` | off | HF only. Passes `enable_thinking=True` to `apply_chat_template` (Qwen3 thinking mode). |
| `--openai_api` | `chat_completions` | OpenAI mode: `chat_completions` / `responses` / `responses_url` |
| `--mcp_url` | — | Required for `responses_url` mode |
| `--mcp_label` | `custom` | `server_label` exposed to OpenAI in `responses_url` mode |
| `--enabled` | `ENABLED_SERVERS` | Comma-separated list of MCP servers to enable |
| `--system_message` | `""` | Extra system prompt beyond the tool-calling preamble |
| `-m`, `--user_message` | — | User query |

## Output format parsing (HF backends)

Different open-source models emit tool calls in different native formats. `utils/misc.py:parse_output` recognizes:

- **Llama 3.1**: `<|python_tag|>{"name": ..., "parameters": ...}`
- **Qwen 2.5 / 3 / Hermes**: `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
- **Qwen 3.5**: `<tool_call><function=name><parameter=key>val</parameter></function></tool_call>`

Add new branches to `parse_output` if you bring in another model family.

## Requirements

```
python    == 3.10.19
mcp       == 1.26.0
numpy     == 1.26.4
torch     == 2.2.2
transformers == 4.57.1
openai    >= 1.0.0
anthropic >= 0.20.0
python-dotenv
uvicorn
fastmcp / mcp[server]      # for FastMCP
pyngrok                    # only for public_custom_mcp.py
```

For Perplexity (`mcp_servers/perplexity_search.py`) you also need Node.js / `npx` on PATH — it spawns `@perplexity-ai/mcp-server` automatically.
