# simpleMCP
Simple MCP client and server implementation supporting HuggingFace local models and API-based models (OpenAI, Anthropic).

### Usage

1. Create "secrets.env" and add any required API keys.
```
touch ./secrets.env
```

Example `secrets.env`:
```
BRAVE_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

2. Run MCP servers. This will run all MCP servers defined in "mcp_servers" directory.
```
python3 run_mcp_servers.py
```

3. Run the MCP client.
```
# HuggingFace local model (default)
python3 mcp_client.py -m "Where does the computer scientist Seungeon Lee work?"

# OpenAI API
python3 mcp_client.py --model gpt-4o -m "Where does the computer scientist Seungeon Lee work?"

# Anthropic API
python3 mcp_client.py --model claude-3-5-sonnet-20241022 -m "Where does the computer scientist Seungeon Lee work?"
```

### Model Routing

The backend is selected automatically based on the model name:

| Model name prefix | Backend |
|---|---|
| `gpt-`, `o1-`, `o3-`, `o4-`, `chatgpt-` | OpenAI API |
| `claude-` | Anthropic API |
| anything else | HuggingFace local |

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `Llama-3.1-8B-Instruct` | Model name or API model ID |
| `--device` | `cuda:0` | Device (HuggingFace only) |
| `--dtype` | `auto` | Model dtype (HuggingFace only) |
| `--temperature` | `0.0` | Sampling temperature |
| `--seed` | `0` | Random seed |
| `--system_message` | `""` | Additional system message |
| `-m`, `--user_message` | — | User query |

### Requirements

python == 3.10.19

mcp == 1.26.0

numpy == 1.26.4

torch == 2.2.2

transformers == 4.57.1

openai >= 1.0.0

anthropic >= 0.20.0
