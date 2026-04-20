from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from utils.hf_model import HFModel, generate_from_messages
from utils.misc import parse_output
from utils.backend import ToolCall, ChatResponse


@dataclass
class HFBackend:
    hf: HFModel

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
        raw = generate_from_messages(
            self.hf,
            messages,
            tools=tools,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            logger=logger,
        )
        logger.info(f"Raw LLM Output:\n{raw}\n")

        response_type, tool_name, tool_args = parse_output(raw)
        if response_type == "final":
            return ChatResponse(content=(tool_name or "").strip(), tool_call=None)

        return ChatResponse(
            content=None,
            tool_call=ToolCall(id="call_0", name=tool_name, args=tool_args),
        )

    def build_tool_call_message(self, tc: ToolCall) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.args,  # dict; Llama template encodes it
                },
            }],
        }

    def build_tool_result_message(self, tc: ToolCall, result: str) -> Dict[str, Any]:
        return {"role": "tool", "tool_call_id": tc.id, "content": result}
