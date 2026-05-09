import asyncio
import os

from dotenv import load_dotenv
load_dotenv("./secrets.env")

from utils.config import LLMConfig, McpConfig
from utils.backend import create_backend
from utils.mcp_http import MultiMcp
from utils.misc import apply_allowlist, to_llm_tools
from utils.agent_loop import run_agent

import argparse
from pathlib import Path
from utils.logger import create_logger

MCP_URLS = {
    "custom": "http://localhost:8001/mcp",
    "brave_search": "http://localhost:8002/mcp",
    "perplexity_search": "http://localhost:8003/mcp",
    "google_search": "http://localhost:8004/mcp",
}

TOOL_ALLOWLIST = {
    "custom": {
        "add"
    },
    "brave_search": {
        "brave_web_search"
    },
    "perplexity_search": {
        "perplexity_search"
    },
    "google_search": {
        "google_search"
    },
}

# Default servers to enable. Override at runtime with --enabled custom,brave_search,...
ENABLED_SERVERS = ["custom"]

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "../hf_models/"))

async def build_tools(mcp: MultiMcp, mcp_cfg: McpConfig):
    all_tools = []
    for server in mcp_cfg.enabled:
        tools_resp = await mcp.list_tools(server)
        tools = tools_resp.tools

        allow = None
        if mcp_cfg.allowlist is not None:
            allow = mcp_cfg.allowlist.get(server)

        tools = apply_allowlist(tools, allow)
        prefix = server if mcp_cfg.prefix_tools else None
        all_tools.extend(to_llm_tools(tools, prefix=prefix))
    return all_tools

def parse_arguments(return_default: bool = False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0', help="Device to run (HuggingFace only)")
    parser.add_argument('--dtype', type=str, default='auto', help="Model dtype (HuggingFace only)")
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3.5-35B-A3B', help='Model name (HuggingFace local) or API model ID (e.g. gpt-4o, claude-3-5-sonnet-20241022)')
    parser.add_argument('--max_tool_rounds', type=int, default=6, help='Maximum number of LLM rounds (each round may invoke one tool or produce the final answer).')
    parser.add_argument('--writing_mode', action='store_true', help='HuggingFace only: also log the model output with special tokens kept (e.g. <|python_tag|>) for inspection. Agent behavior is unchanged.')
    parser.add_argument('--enable_thinking', action='store_true', help='HuggingFace only: pass enable_thinking=True to apply_chat_template (e.g. Qwen3 thinking mode). Default False.')
    parser.add_argument('--openai_api', type=str, choices=['chat_completions', 'responses', 'responses_url'], default='chat_completions', help='Which OpenAI API mode to use (only for OpenAI models). responses_url: OpenAI server connects to the MCP server directly via --mcp_url. Default: chat_completions.')
    parser.add_argument('--mcp_url', type=str, default=None, help='Public URL of the MCP server (required for --openai_api=responses_url).')
    parser.add_argument('--mcp_label', type=str, default='custom', help='server_label exposed to OpenAI in responses_url mode.')
    parser.add_argument('--enabled', type=str, default=None, help=f'Comma-separated list of MCP servers to enable. Default: {",".join(ENABLED_SERVERS)}.')

    parser.add_argument(
        '--system_message',
        type=str,
        default='',
        help='Additional system message beyond tool calling.',
    )

    parser.add_argument(
        '-m', '--user_message',
        type=str,
        default='Introduce yourself in one sentence.',
        help='User message',
    )

    if return_default:
        return parser.parse_args([])
    return parser.parse_args()

async def main():
    logger = create_logger()

    args = parse_arguments()

    llm_cfg = LLMConfig(
        temperature=args.temperature,
        max_new_tokens=256,
        max_tool_rounds=args.max_tool_rounds,
    )

    logger.info(f'[User Question] {args.user_message}')

    backend = create_backend(
        args.model,
        model_dir=MODEL_DIR,
        device=args.device,
        dtype=args.dtype,
        logger=logger,
        writing_mode=args.writing_mode,
        enable_thinking=args.enable_thinking,
        openai_api=args.openai_api,
        mcp_url=args.mcp_url,
        mcp_label=args.mcp_label,
    )

    if args.openai_api == "responses_url":
        # OpenAI server talks to the MCP server directly. No local MCP
        # connection, no client-side tool routing -- the backend makes a
        # single Responses API call and returns the final answer.
        answer = backend.run(
            system_message=args.system_message,
            user_message=args.user_message,
            max_new_tokens=llm_cfg.max_new_tokens,
            temperature=llm_cfg.temperature,
            seed=args.seed,
            logger=logger,
        )
    else:
        enabled = [s.strip() for s in args.enabled.split(",")] if args.enabled else list(ENABLED_SERVERS)
        mcp_cfg = McpConfig(
            url_map=MCP_URLS,
            enabled=enabled,
            allowlist=TOOL_ALLOWLIST,
            prefix_tools=True,
        )

        async with MultiMcp(mcp_cfg.url_map, mcp_cfg.enabled) as mcp:
            llm_tools = await build_tools(mcp, mcp_cfg)

            logger.info(f'Available Tools:\n{llm_tools}\n')

            answer = await run_agent(
                backend=backend,
                mcp=mcp,
                llm_tools=llm_tools,
                system_message=args.system_message,
                user_message=args.user_message,
                temperature=llm_cfg.temperature,
                max_new_tokens=llm_cfg.max_new_tokens,
                max_tool_rounds=llm_cfg.max_tool_rounds,
                seed=args.seed,
                logger=logger,
            )

    logger.info(f'Final Answer:\n{answer}')

if __name__ == "__main__":
    asyncio.run(main())
