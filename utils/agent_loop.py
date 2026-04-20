import json
from typing import Any, Dict, List

from utils.hf_model import HFModel, generate_from_messages
from utils.prompting import build_initial_messages
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
    messages = build_initial_messages(
        system_message=system_message,
        user_message=user_message,
    )

    for round_num in range(1, max_tool_rounds + 1):
        logger.info(f"[ROUND {round_num}] START\n")
        logger.info(f"[ROUND {round_num}] LLM Input:\n{messages}\n")

        raw = generate_from_messages(
            hf,
            messages,
            tools=llm_tools,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            logger=logger,
        )

        logger.info(f"[ROUND {round_num}] Raw LLM Output:\n{raw}\n")

        response_type, tool_name, tool_args = parse_output(raw)

        if response_type == "final":
            return (tool_name or "").strip()

        # tool call
        assert tool_name is not None and tool_args is not None

        server, raw_tool = split_prefixed(tool_name)

        tool_schema = None
        for t in llm_tools:
            if t["function"]["name"] == tool_name:
                tool_schema = t["function"]["parameters"]
                break

        if tool_schema:
            tool_args = preprocess_by_schema(tool_args, tool_schema)

        logger.info(f"[ROUND {round_num}] TOOL: {tool_name}")
        logger.info(f"[ROUND {round_num}] ARGS: {json.dumps(tool_args, ensure_ascii=False)}")

        result = await mcp.call_tool(server, raw_tool, tool_args)
        result_text = extract_tool_result_text(result.model_dump())

        logger.info(f"[ROUND {round_num}] RESULT:\n{result_text}\n")

        # Append assistant tool call and tool result to history, then return final answer
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"call_{round_num}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": tool_args,  # dict, not json string (template encodes it)
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": f"call_{round_num}",
            "content": result_text,
        })

        raw_final = generate_from_messages(
            hf,
            messages,
            tools=None,  # no tools needed — just summarize the result
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            logger=logger,
        )

        logger.info(f"[ROUND {round_num + 1}] Raw LLM Output:\n{raw_final}\n")
        return raw_final.strip()

    raise RuntimeError("Tool calling rounds exceeded.")
