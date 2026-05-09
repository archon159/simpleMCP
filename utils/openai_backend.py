import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI

from utils.backend import ToolCall, ChatResponse


@dataclass
class OpenAIBackend:
    model: str
    _client: OpenAI = field(default_factory=OpenAI, init=False, repr=False)

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int,
        logger: Any,
    ) -> ChatResponse:
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
        )
        if tools:
            kwargs["tools"] = tools

        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        logger.info(f"Raw API Response:\n{msg}\n")

        if msg.tool_calls:
            tc = msg.tool_calls[0]
            return ChatResponse(
                content=None,
                tool_call=ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    args=json.loads(tc.function.arguments),
                ),
            )

        return ChatResponse(content=(msg.content or "").strip(), tool_call=None)

    def build_tool_call_message(self, tc: ToolCall) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.args, ensure_ascii=False),
                },
            }],
        }

    def build_tool_result_message(self, tc: ToolCall, result: str) -> Dict[str, Any]:
        return {"role": "tool", "tool_call_id": tc.id, "content": result}
