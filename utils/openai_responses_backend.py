import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI

from utils.backend import ToolCall, ChatResponse


def _is_reasoning_model(name: str) -> bool:
    return name.startswith(("o1-", "o3-", "o4-", "gpt-5"))


@dataclass
class OpenAIResponsesBackend:
    """
    OpenAI Responses API backend.

    Differences from Chat Completions:
      - Uses `input` (a heterogeneous item list) instead of `messages`.
      - Tool schema is flat: {"type": "function", "name": ..., "parameters": ...}
        rather than nested under a "function" key.
      - Tool calls and results are items, not role-based messages:
            {"type": "function_call",        "call_id", "name", "arguments"}
            {"type": "function_call_output", "call_id", "output"}
      - Output is a list of typed items (function_call / message / reasoning ...);
        we scan for function_call first, then fall back to text.
    """

    model: str
    _client: OpenAI = field(
        default_factory=lambda: OpenAI(max_retries=3, timeout=60.0),
        init=False,
        repr=False,
    )

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int,  # Responses API does not accept a seed; ignored
        logger: Any,
    ) -> ChatResponse:
        responses_tools = [
            {
                "type": "function",
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "parameters": t["function"]["parameters"],
            }
            for t in (tools or [])
        ]

        kwargs: Dict[str, Any] = dict(
            model=self.model,
            input=messages,
            max_output_tokens=max_new_tokens,
        )
        if not _is_reasoning_model(self.model):
            kwargs["temperature"] = temperature
        if responses_tools:
            kwargs["tools"] = responses_tools

        resp = self._client.responses.create(**kwargs)
        logger.info(f"Raw API Response:\n{resp.output}\n")

        for item in resp.output:
            if item.type == "function_call":
                return ChatResponse(
                    content=None,
                    tool_call=ToolCall(
                        id=item.call_id,
                        name=item.name,
                        args=json.loads(item.arguments),
                    ),
                )

        return ChatResponse(content=(resp.output_text or "").strip(), tool_call=None)

    def build_tool_call_message(self, tc: ToolCall) -> Dict[str, Any]:
        return {
            "type": "function_call",
            "call_id": tc.id,
            "name": tc.name,
            "arguments": json.dumps(tc.args, ensure_ascii=False),
        }

    def build_tool_result_message(self, tc: ToolCall, result: str) -> Dict[str, Any]:
        return {
            "type": "function_call_output",
            "call_id": tc.id,
            "output": result,
        }
