from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ChatResponse:
    content: Optional[str]
    tool_call: Optional[ToolCall]


def _is_openai_model(name: str) -> bool:
    return name.startswith(("gpt-", "o1-", "o3-", "o4-", "chatgpt-"))


def _is_anthropic_model(name: str) -> bool:
    return name.startswith("claude-")


def create_backend(
    model_name: str,
    *,
    model_dir: Path,
    device: str,
    dtype: str,
    logger: Any = None,
) -> Any:
    if _is_openai_model(model_name):
        from utils.openai_backend import OpenAIBackend
        return OpenAIBackend(model_name)

    if _is_anthropic_model(model_name):
        from utils.anthropic_backend import AnthropicBackend
        return AnthropicBackend(model_name)

    # HuggingFace local model
    if logger:
        logger.info("Loading LLM...\n")
    from utils.hf_model import load_hf_model
    from utils.hf_backend import HFBackend
    hf = load_hf_model(model_dir / model_name, dtype=dtype, device=device)
    return HFBackend(hf)
