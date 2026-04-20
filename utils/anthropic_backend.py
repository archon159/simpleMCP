from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anthropic

from utils.backend import ToolCall, ChatResponse


@dataclass
class AnthropicBackend:
    model: str
    _client: anthropic.Anthropic = field(
        default_factory=anthropic.Anthropic, init=False, repr=False
    )

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int,  # not supported by Anthropic; ignored
        logger: Any,
    ) -> ChatResponse:
        system = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        non_system = [m for m in messages if m["role"] != "system"]

        anthropic_tools = [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"],
            }
            for t in (tools or [])
        ]

        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_new_tokens,
            temperature=temperature,
            system=system,
            messages=non_system,
        )
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        resp = self._client.messages.create(**kwargs)
        logger.info(f"Raw API Response:\n{resp.content}\n")

        for block in resp.content:
            if block.type == "tool_use":
                return ChatResponse(
                    content=None,
                    tool_call=ToolCall(id=block.id, name=block.name, args=block.input),
                )

        text = next(
            (block.text for block in resp.content if block.type == "text"), ""
        )
        return ChatResponse(content=text.strip(), tool_call=None)

    def build_tool_call_message(self, tc: ToolCall) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args}],
        }

    def build_tool_result_message(self, tc: ToolCall, result: str) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tc.id, "content": result}],
        }
