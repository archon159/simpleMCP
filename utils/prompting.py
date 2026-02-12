import json
from typing import Any, Dict, List, Optional


SYSTEM_TOOL_PREAMBLE = """When you receive a tool call response, use the output to format an answer to the original user question.

You are a helpful assistant with tool calling capabilities.
"""

USER_TOOL_REQUEST_TEMPLATE = """Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt. Exclude the unnessary arguments.

Respond in the format {{"name": function name, "parameters": dictionary of argument name and its value}}. Do not use variables.

{functions}

Question: {question}
"""


def build_chat_messages(
    *,
    system_message: str,
    user_message: str,
    tools: List[Dict[str, Any]],
    tool_result: Optional[str] = None,
) -> List[Dict[str, str]]:

    sys = (system_message.strip() + "\n\n" + SYSTEM_TOOL_PREAMBLE).strip()

    msgs: List[Dict[str, str]] = [{"role": "system", "content": sys}]

    if tool_result is None:
        functions_block = "\n".join(json.dumps(t["function"], ensure_ascii=False) for t in tools)
        user_content = USER_TOOL_REQUEST_TEMPLATE.format(
            functions=functions_block,
            question=user_message,
        )
        msgs.append({"role": "user", "content": user_content})
    else:
        msgs.append({"role": "user", "content": user_message})
        msgs.append({"role": "tool", "content": tool_result})


    return msgs