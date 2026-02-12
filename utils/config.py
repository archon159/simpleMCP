from dataclasses import dataclass
from typing import Dict, List

@dataclass(frozen=True)
class LLMConfig:
    model_id: str
    temperature: float = 0.0
    max_new_tokens: int = 256
    max_tool_rounds: int = 8

@dataclass(frozen=True)
class McpConfig:
    # key: server name, value: mcp url
    url_map: Dict[str, str]
    enabled: List[str]
    # tool candidate list
    allowlist: Dict[str, set] | None = None
    # use tool name as prefix
    prefix_tools: bool = True