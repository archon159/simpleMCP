from typing import Any, Dict, List


SYSTEM_TOOL_PREAMBLE = "You are a helpful assistant with tool calling capabilities."


def build_initial_messages(
    *,
    system_message: str,
    user_message: str,
) -> List[Dict[str, Any]]:
    sys = (system_message.strip() + "\n\n" + SYSTEM_TOOL_PREAMBLE).strip()
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user_message},
    ]