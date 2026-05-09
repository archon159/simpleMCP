import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI


def _is_reasoning_model(name: str) -> bool:
    return name.startswith(("o1-", "o3-", "o4-", "gpt-5"))


@dataclass
class OpenAIResponsesUrlBackend:
    """
    OpenAI Responses API where the model talks to an MCP server directly
    via its public URL (the `mcp` tool type).

    The OpenAI server itself connects to the MCP server at `mcp_url`,
    enumerates its tools, calls them, and feeds the results back to the
    model -- all server-side. We make a single API call and receive the
    final answer; there is no client-side tool routing loop.

    Because of this, the local MCP context (`MultiMcp`) and `agent_loop`
    are bypassed entirely when this backend is used.
    """

    model: str
    mcp_url: str
    mcp_label: str = "custom"
    allowed_tools: Optional[List[str]] = None
    require_approval: str = "never"

    _client: OpenAI = field(
        default_factory=lambda: OpenAI(max_retries=3, timeout=120.0),
        init=False,
        repr=False,
    )

    def run(
        self,
        *,
        system_message: str,
        user_message: str,
        max_new_tokens: int,
        temperature: float,
        seed: int,  # Responses API does not accept a seed
        logger: Any,
    ) -> str:
        input_msgs: List[Dict[str, Any]] = []
        if system_message.strip():
            input_msgs.append({"role": "system", "content": system_message.strip()})
        input_msgs.append({"role": "user", "content": user_message})

        mcp_tool: Dict[str, Any] = {
            "type": "mcp",
            "server_label": self.mcp_label,
            "server_url": self.mcp_url,
            "require_approval": self.require_approval,
        }
        if self.allowed_tools is not None:
            mcp_tool["allowed_tools"] = self.allowed_tools

        kwargs: Dict[str, Any] = dict(
            model=self.model,
            input=input_msgs,
            max_output_tokens=max_new_tokens,
            tools=[mcp_tool],
        )
        if not _is_reasoning_model(self.model):
            kwargs["temperature"] = temperature

        logger.info(
            f"[ROUND 1] LLM Input:\n{json.dumps(input_msgs, indent=2, ensure_ascii=False)}\n"
        )
        logger.info(
            f"[ROUND 1] MCP Tool Spec:\n{json.dumps(mcp_tool, indent=2, ensure_ascii=False)}\n"
        )

        resp = self._client.responses.create(**kwargs)
        logger.info(f"Raw API Response (full output items):\n{resp.output}\n")

        return (resp.output_text or "").strip()
