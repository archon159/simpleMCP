import json
from typing import Any, Dict, List

from utils.hf_model import HFModel, generate_from_messages
from utils.prompting import build_chat_messages
from utils.misc import split_prefixed, extract_tool_result_text, parse_output, preprocess_by_schema


async def run_agent(
    *,
    hf: HFModel,
    mcp,  # MultiMcp
    llm_tools: List[Dict[str, Any]],
    system_message: str,
    user_message: str,
    temperature: float,
    max_new_tokens: int,
    max_tool_rounds: int,
    seed: int = 0,
    logger: Any = None,
) -> str:
    rounds = 0
    tool_result = None
    
    while True:
        rounds += 1
        if rounds > max_tool_rounds:
            raise RuntimeError("Tool calling rounds exceeded.")

        logger.info(f"[ROUND {rounds}] START\n")
            
        messages = build_chat_messages(
            system_message=system_message,
            user_message=user_message,
            tools=llm_tools,
            tool_result=tool_result,
        )
        
        logger.info(f"[ROUND {rounds}] LLM Input:\n{messages}\n")

        raw = generate_from_messages(
            hf,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            logger=logger,
        )
        
        logger.info(f"[ROUND {rounds}] Raw LLM Output:\n{raw}\n")

        if tool_result is not None:
            return raw.strip()

        response_type, tool_name, tool_args = parse_output(raw)

        if response_type == "final":
            return (tool_name or "").strip()

        # response_type == "call"
        assert tool_name is not None and tool_args is not None

        server, raw_tool = split_prefixed(tool_name)

        tool_schema = None
        for t in llm_tools:
            if t["function"]["name"] == tool_name:
                tool_schema = t["function"]["parameters"]
                break

        if tool_schema:
            tool_args = preprocess_by_schema(tool_args, tool_schema)
        
        logger.info(f"[ROUND {rounds}] TOOL: {tool_name}")
        logger.info(f"[ROUND {rounds}] ARGS: {json.dumps(tool_args, ensure_ascii=False)}")

        result = await mcp.call_tool(server, raw_tool, tool_args)
        dump = result.model_dump()

        result_text = extract_tool_result_text(dump)

        logger.info(f"[ROUND {rounds}] RESULT:\n{result_text}\n")

        tool_result = result_text