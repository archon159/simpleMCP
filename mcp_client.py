import asyncio

from utils.config import LLMConfig, McpConfig
from utils.hf_model import load_hf_model
from utils.mcp_http import MultiMcp
from utils.misc import apply_allowlist, to_llm_tools
from utils.agent_loop import run_agent

import argparse
from pathlib import Path
from utils.logger import create_logger

MCP_URLS = {
    "custom": "http://localhost:8000/mcp",
    "brave_search": "http://0.0.0.0:8080/mcp",
}

TOOL_ALLOWLIST = {
    "custom": {
        "add"
    },
    "brave_search": {
        "brave_web_search"
    },
}

MODEL_DIR = Path("../hf_models/")

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

    parser.add_argument('--device', type=str, default='cuda:0', help="Device to run")
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature')
    parser.add_argument('--model', type=str, default='Llama-3.1-8B-Instruct', help='Model')

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
        model_id=MODEL_DIR / args.model,
        temperature=args.temperature,
        max_new_tokens=256,
        max_tool_rounds=6,
    )
    mcp_cfg = McpConfig(
        url_map=MCP_URLS,
        enabled=["custom", "brave_search"],
        allowlist=TOOL_ALLOWLIST,
        prefix_tools=True,
    )
    
    logger.info(f'[User Question] {args.user_message}')

    logger.info('Loading LLM...\n')
    hf = load_hf_model(llm_cfg.model_id, dtype="auto", device=args.device)

    async with MultiMcp(mcp_cfg.url_map, mcp_cfg.enabled) as mcp:
        llm_tools = await build_tools(mcp, mcp_cfg)
        
        logger.info(f'Available Tools:\n{llm_tools}\n')
        
        answer = await run_agent(
            hf=hf,
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