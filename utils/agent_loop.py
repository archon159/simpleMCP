import json
from typing import Any, Dict, List

from utils.prompting import build_initial_messages
from utils.misc import split_prefixed, extract_tool_result_text, preprocess_by_schema


async def run_agent(
    *,
    backend,
    mcp,
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

        response = backend.complete(
            messages,
            llm_tools,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            logger=logger,
        )

        if response.tool_call is None:
            return response.content or ""

        tc = response.tool_call
        server, raw_tool = split_prefixed(tc.name)

        tool_schema = next(
            (t["function"]["parameters"] for t in llm_tools if t["function"]["name"] == tc.name),
            None,
        )
        tool_args = preprocess_by_schema(tc.args, tool_schema) if tool_schema else tc.args

        logger.info(f"[ROUND {round_num}] TOOL: {tc.name}")
        logger.info(f"[ROUND {round_num}] ARGS: {json.dumps(tool_args, ensure_ascii=False)}")

        result = await mcp.call_tool(server, raw_tool, tool_args)
        result_text = extract_tool_result_text(result.model_dump())

        logger.info(f"[ROUND {round_num}] RESULT:\n{result_text}\n")

        messages.append(backend.build_tool_call_message(tc))
        messages.append(backend.build_tool_result_message(tc, result_text))

        final = backend.complete(
            messages,
            tools=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            logger=logger,
        )
        return final.content or ""

    raise RuntimeError("Tool calling rounds exceeded.")
